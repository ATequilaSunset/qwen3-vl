# 计划 1：独立 ViT + 查询式检测

当前实现：原图 → 原版预处理 → 最后一层 pre-merger ViT 特征图 → LayerNorm/维度投影 + 二维位置编码 → object queries + 6 层检测 decoder → 分类/框回归。

它是使用单尺度 ViT 特征的 DETR 风格诊断模型：没有额外视觉 encoder、FPN、Merger 或 LLM；没有裁剪、额外检测器或训练时随机缩放。检测部分从头学习，训练包含匈牙利匹配、背景类别、分类交叉熵、L1、GIoU，以及各 decoder 层辅助损失。

## 已验证与尚待验证

- 用户服务器已经通过独立 ViT 的真实权重一致性验收：torch 2.6.0+cu124、transformers 5.19.0、FP32/SDPA，800×800 样本误差为 0。
- 新检测代码本地使用 torch 2.8.0+cpu、transformers 5.19.0 验证小型真实 Qwen 模型，包括冻结/微调梯度、不同尺寸图片、空标注、坐标回转、匹配、COCO 指标、检查点和 CLI。
- 尚未在服务器的完整 3.06 亿参数 ViT 上运行新检测训练；没有 AI-TOD 检测结果，也没有已验证的 A100 训练显存/速度估计。
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

看 `epoch_*.json` 中的 loss/AP 趋势，以及 `visualizations_last/` 中左侧 GT、右侧预测框。右侧仅绘制分数≥0.2的最多100个框；该阈值不影响 AP。50 轮×4图只是一个可延长的调试预算，不保证必然拟合成功；没有拟合好就先排查坐标、匹配与收敛，再启动完整实验。

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
