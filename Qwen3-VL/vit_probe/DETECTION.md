# 计划 1：独立 ViT + 查询式检测

当前实现：原图 → 原版预处理 → 最后一层 pre-merger ViT 特征图 → LayerNorm/维度投影 + 二维位置编码 → object queries + 6 层检测 decoder → 分类/框回归。

它是使用单尺度 ViT 特征的 DETR 风格诊断模型：没有额外视觉 encoder、FPN、Merger 或 LLM；没有裁剪、额外检测器或训练时随机缩放。检测部分从头学习，训练包含匈牙利匹配、背景类别、分类交叉熵、L1、GIoU，以及各 decoder 层辅助损失。

## 已验证与尚待验证

- 用户服务器已经通过独立 ViT 的真实权重一致性验收：torch 2.6.0+cu124、transformers 5.19.0、FP32/SDPA，800×800 样本误差为 0。
- 新检测代码本地使用 torch 2.8.0+cpu、transformers 5.19.0 验证小型真实 Qwen 模型，包括冻结/微调梯度、不同尺寸图片、空标注、坐标回转、匹配、COCO 指标、检查点和 CLI。
- 用户已在服务器完成冻结 ViT 的四图、50 轮拟合检查：loss 从 75.5847 降到 41.9179，但 AP50/AR 始终为 0；检测部分有梯度，ViT 无梯度。该检查尚未通过，不能据此判断 ViT 缺失小目标信息。
- 缓存特征诊断脚本仅在本地小型真实 Qwen 模型上测试，尚未在服务器完整 3.06 亿参数 ViT 上验证其拟合效果、显存和速度；当前没有完整数据集的检测结果。
- 5 轮默认训练只用于初步观察。随机检测头尚未收敛时，低 AP 不能证明 ViT 丢失了小目标信息；冻结/微调差异也不能定位 Merger/LLM 的瓶颈。

## 服务器环境与数据

所有命令在 `/data/lz/pythonProjects/qwen3-vl-vit/Qwen3-VL` 执行。

```bash
conda activate qwen3-vl
cd /data/lz/pythonProjects/qwen3-vl-vit/Qwen3-VL
python -m pip install -r vit_probe/requirements-detection.txt
```

配置文件是 `vit_probe/configs/aitod_detection.json`：

| 项目 | 路径或初始值 |
| --- | --- |
| 独立 ViT | `/data/lz/models/Qwen3-VL-4B-ViT` |
| 训练图片 | `/data/wrj/AI-TOD/aitod/images/train` |
| 训练标注 | `/data/wrj/AI-TOD/aitod/annotations/aitod_train.json` |
| 验证图片 | `/data/wrj/AI-TOD/aitod/images/val` |
| 验证标注 | `/data/wrj/AI-TOD/aitod/annotations/aitod_val.json` |
| 精度 | BF16 autocast；参数和优化器保持 FP32 |
| batch / 梯度累积 | 2 / 4，有效 batch=8（最后不足一组时按实际批次处理） |
| 检测部分 / ViT 学习率 | 1e-4 / 1e-5 |
| 隐藏维度 / decoder 层数 | 256 / 6 |
| queries | `0` 表示由训练集最大非 crowd/ignore 目标数向上取整到 50 的倍数，至少 300 |

使用指定的一张卡；按照当前显存快照，可以从物理 GPU 4 开始。它只是执行前的快照，不构成服务器资源预留。

```bash
export CUDA_VISIBLE_DEVICES=4
```

此后程序里的 `--device cuda:0` 就是物理 GPU 4。不要同时改成 `cuda:4`。

## 1. CPU 数据预检

```bash
python -m vit_probe.inspect_data \
  --config vit_probe/configs/aitod_detection.json
```

结果写入 `vit_probe/artifacts/data_audit.json`。检查训练/验证类别 ID 与名字、所有图片路径、样本尺寸、标注有效性、目标数、空图、拆分重叠和标注文件 SHA256。尺寸预检只抽读代表图片，训练时会逐图验证实际尺寸。无法仅凭目录名判断是不是 AI-TOD-v2；已有 `test_v1_1.0` 文件名更不能证明整个数据目录的版本。当前先按实际 train/val 标注做实验，论文使用前需要确认数据来源/版本。

queries 的自动值仅由训练集决定，正式 A/B 训练应使用同一完整训练集，因此会得到相同值。不截断 GT。若密集图片导致查询数很大，注意服务器实际运行的显存和速度；不能通过丢弃目标来悄悄缩小实验。

## 2. 少量图片拟合检查

先确认冻结 ViT 的检测部分能学习。这个运行在同一小训练子集上训练和评估，所有结果明确标记 `TRAINING_SUBSET_DEBUG_ONLY`，不能当作泛化性能。

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.train \
  --config vit_probe/configs/aitod_detection.json \
  --mode frozen --device cuda:0 \
  --overfit --limit-train 4 --epochs 50 \
  --batch-size 2 --accumulation-steps 1 \
  --output-dir vit_probe/runs/overfit_frozen_4
```

看 `epoch_*.json` 中的 loss/AP 趋势，以及 `visualizations_last/` 中左侧 GT、右侧预测框。右侧仅绘制分数≥0.2的最多100个框；该阈值不影响 AP。4 张图、batch=2、累积=1、50 轮只有 **100 次参数更新**，原训练的余弦学习率也会在这 50 轮内降到零。这只是调试预算，不保证拟合成功；没有拟合好就先排查坐标、匹配与收敛，再启动完整实验。

### 四图 AP 为零时：缓存冻结特征诊断

使用以下独立脚本进一步定位问题。保持同一检测头结构与损失，按配置的 seed 选取同一小训练子集。ViT 使用 FP32、eval 模式提取一次原始图像特征，随后释放 ViT 权重，仅训练检测部分。标签只用于损失和评估，不参与特征生成。

诊断设置为 **FP32、dropout=0、恒定学习率、明确的更新步数**。这些设置一起改变，因此本次用于排查能否拟合，不能用于证明单个因素是失败原因，也不替代正式冻结/微调对照。它从随机检测头重新开始，不接着原 50 轮检查点训练。

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.overfit_cached \
  --config vit_probe/configs/aitod_detection.json \
  --limit-train 4 --steps 2000 --batch-size 2 \
  --eval-every 100 --log-every 100 --device cuda:0 \
  --output-dir vit_probe/runs/overfit_cached_frozen_4
```

2000 指优化器更新次数，不是轮数；这是新的诊断预算，不保证必然拟合。脚本要求新的空输出目录，保留原运行结果。仅适用于 1–32 张图的冻结诊断，不用于全数据集训练。

| 输出 | 用途 |
| --- | --- |
| `run_config.json` | 记录图片 ID、实际配置、ViT 权重哈希、初始检测头哈希、缓存大小和与原训练的设置差异 |
| `train_steps.jsonl` | 按更新步记录总损失、最终层 CE/L1/GIoU、学习率、裁剪前梯度范数、各模块梯度范数和批次目标数 |
| `step_000000.json`、`step_000100.json` 等 | 初始和定期评估结果，包含 COCO 指标与逐图定位诊断 |
| `visualizations_last/`、`predictions_last.json` | 最近一次评估的图像与预测，显示阈值仍不影响 AP |
| `head_last.pt`、`head_best.pt` | 只存检测头、优化器和配置，避免重复写入冻结 ViT；不能传给原 `vit_probe.evaluate`，本脚本暂不支持恢复训练 |
| `status.json` | 运行状态、实际更新次数、最佳 AP、首次有目标批次的模块梯度检查步数 |

总损失包含全部 decoder 辅助损失；单独的 `ce/l1/giou` 是最终层的未加权损失，不能直接相加得到总损失。定期评估固定遍历整个小子集，更适合比较趋势；训练日志来自当前批次。

逐图 `per_image` 字段的含义：

- `best_iou_mean`：每个 GT 与所有预测框的最大 IoU，再对 GT 求均值。长期接近 0 说明框仍未定位到目标附近。
- `gt_fraction_with_any_query_iou50`：有任一预测框达到 IoU≥0.5 的 GT 比例。它忽略类别、分数，允许复用预测框，**不是检测召回率**。
- `matched_class_accuracy`：匈牙利匹配后的类别准确率，预测为背景也算错。
- `matched_center_error_px`：匹配框与 GT 中心的平均像素距离，越小越好。
- `mean_background_probability`：全部 queries 的平均背景概率；多数 query 本来就应为背景，不能单独把高值判定为失败。

若框有重叠但 AP 仍为 0，需要结合匹配类别、分数排序与 COCO 指标检查；若框长期不靠近目标，则优先检查回归、匹配和学习过程。空标注图片的框回归梯度为零是正常现象，脚本在首次有目标的批次检查投影、分类和框回归模块梯度。训练集拟合成功后仍需完整数据集的独立验证。

再做一次微调反向传播与显存检查：

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.train \
  --config vit_probe/configs/aitod_detection.json \
  --mode finetune --device cuda:0 \
  --limit-train 4 --limit-val 4 --epochs 1 \
  --output-dir vit_probe/runs/smoke_finetune_4
```

这里应看到 `head_nonzero_grad=true`、`backbone_nonzero_grad=true`；冻结运行中后者必须是 false。显存记录在 `train.peak_gpu_memory_GiB`。小子集可能对应较少 queries；正式训练还要观察完整配置首批次的显存。

## 3. 正式的 5 轮初步对照

两个实验分别从同一导出 ViT 和同一随机检测头初始化出发；B 不从 A 的已训练头开始。相同 seed、数据顺序、头结构、预处理、有效 batch 和训练轮数；学习率与优化器状态分别记录。`run_config.json` 中的 `head_initial_sha256` 应相同。

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.train \
  --config vit_probe/configs/aitod_detection.json \
  --mode frozen --device cuda:0 \
  --output-dir vit_probe/runs/frozen_5ep
```

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.train \
  --config vit_probe/configs/aitod_detection.json \
  --mode finetune --device cuda:0 \
  --output-dir vit_probe/runs/finetune_5ep
```

如出现显存不足，先停止本次失败运行，将**两组**都改成 `--batch-size 1 --accumulation-steps 8`，并使用新的输出目录。微调默认启用非重入 gradient checkpointing。不要仅为微调组缩小输入分辨率，否则改变了比较条件。

每组保存：`run_config.json`、逐轮结果、最后一轮预测和可视化、`last.pt`、按验证集 AP@maxDets100 选出的 `best.pt`、`status.json`。检查点包含整个模型和优化器，正式微调可能产生数 GB 文件；它们已加入忽略规则，不应上传 Git。

中断后按原配置恢复（总轮数不可改变）：

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.train \
  --config vit_probe/configs/aitod_detection.json \
  --mode finetune --device cuda:0 \
  --output-dir vit_probe/runs/finetune_5ep \
  --resume vit_probe/runs/finetune_5ep/last.pt
```

## 4. 指标和最后的 test 评估

- 指标为 **COCO 风格 bbox AP**，IoU=0.50:0.05:0.95，数值范围0–1。`0.2` 即20%。
- 同时报告标准 `maxDets=100` 和密集目标设置 `maxDets=queries`（queries>100时）。后者必须标明容量，不能混作标准 COCO AP。
- `AP_small` 使用 COCO 面积范围，**不是 AI-TOD 的官方 very-tiny/tiny/small/medium 分组**。
- 预测按原图宽高还原；每个查询取最高的非背景类别，不做 NMS、不按显示阈值筛选 AP 输入。记录每类别 AP；没有 GT 的分组/类别用 null，不伪造0分。
- val 只用于模型选择；调参期间不使用 test。后续确认 test 标注与训练集版本、类别相容后，再执行：

```bash
CUDA_VISIBLE_DEVICES=4 python -m vit_probe.evaluate \
  --checkpoint vit_probe/runs/finetune_5ep/best.pt \
  --images /data/wrj/AI-TOD/aitod/images/test \
  --annotations /data/wrj/AI-TOD/aitod/annotations/aitod_test_v1_1.0.json \
  --output-dir vit_probe/runs/finetune_5ep_test \
  --device cuda:0
```

版本是否一致不能只靠类别校验判断，首次 test 前需人工确认数据来源。

## 开发检查

```bash
python -m pytest vit_probe/tests -q
```

结构参考：[DETR 官方论文与代码](https://github.com/facebookresearch/detr)。本实现自行连接已导出的 Qwen ViT，并不复现官方 DETR 全部训练配置或性能。
