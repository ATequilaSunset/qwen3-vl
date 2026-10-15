# COCO 检测链路验证

目的：检验“Qwen3-VL 独立 pre-merger ViT + 原有查询式检测头”能否学习普通场景的类别和位置。
保留原 ViT、空间还原、300 起的 query 数、6 层 decoder、匹配和损失；类别由标注自动变为 80 类。
不使用 AI-TOD 的已训练分类头。缓存拟合从新的随机检测头开始。

COCO 有小目标，不能仅换数据集名字就认为已经控制了尺度。
准备脚本选取 4 张简单训练图片：每图 1–10 个目标、每个可见标注框宽高均至少 32 原图像素、无 crowd/ignore。
只筛选整张图片，保留选中图片的全部标注和 80 类映射，不把未保留的目标错误地变成背景。
该子集用于排障，有意偏向容易样本，不代表 COCO 全集能力；32px 是原图框边长筛选，不是官方面积分组。

## 文件及环境

图片使用用户给定位置。其他路径是标准布局假设，需要服务器预检：

```text
/home/wrj/AITOD_Cloud/datasets/coco/
  train2017/
  val2017/                             # 第二阶段需要
  annotations/instances_train2017.json # 第一阶段就需要
  annotations/instances_val2017.json   # 第二阶段需要
```

ViT 默认沿用 `/data/lz/models/Qwen3-VL-4B-ViT`，需要 `vit.safetensors`、`vit_config.json` 及预处理配置。
若已换服务器，使用准备命令的 `--vit-path /实际导出目录`；不要填完整 VLM 权重目录。

更新压缩包包含完整 `vit_probe/`。可以在已有 `Qwen3-VL` 目录下解压到新的 `coco_probe_validation` 目录，
进入该目录后执行命令，从而不覆盖已有代码。所有命令都要求当前目录下能看到 `vit_probe/`。
使用已经能够运行 Qwen3-VL 的 `qwen3-vl` 环境；新增准备脚本只需 Pillow 和标准库。
运行检测所需依赖与上一轮相同，见 `requirements-detection.txt`。

## 1. 准备 4 张普通目标图片

```bash
python -m vit_probe.prepare_coco \
  --output-dir vit_probe/artifacts/coco_sanity
```

不需要 val2017。若训练标注不在默认位置，加 `--train-annotations /实际位置/instances_train2017.json`。
也支持 `--train-images`、`--val-images`、`--val-annotations`。
结果为 `coco_overfit.json`、`coco_pilot.json`、完整子集标注、`prepare_report.json` 和 `ground_truth/` 标注图。
先检查标注图中框与目标是否对应，以及 `vit_weights_present` 是否为 true。
准备脚本没有检查所有训练图片，只检查选中的 4 张；正式训练加载时还会检查所选图片。
输出目录要求不存在或为空；修改路径或样本设置后使用新的目录。

## 2. 冻结 ViT 的 4 图拟合

GPU 4 沿用此前指定编号，启动前确认该卡仍适合使用。它在程序内映射为 cuda:0。

```bash
CUDA_VISIBLE_DEVICES=4 python -u -m vit_probe.overfit_cached \
  --config vit_probe/artifacts/coco_sanity/coco_overfit.json \
  --limit-train 4 --steps 2000 --batch-size 2 \
  --eval-every 100 --log-every 100 --device cuda:0 \
  --output-dir "vit_probe/runs/coco_overfit_4_$(date +%Y%m%d_%H%M%S)"
```

保持上一轮缓存诊断设置：FP32、dropout=0、恒定学习率、明确 2000 次更新。
这与常规训练设置不同，作用是检查可拟合性；2000 次不是保证收敛的理论界限。

看 `step_*.json` 的 `evaluation.coco_maxDets_100.AP50`、`per_image` 和 `visualizations_last/`。
数值范围 0–1。建议的工程检查目标是训练子集 AP50 达到约 0.8 以上、每张图的绝大多数目标类别与位置正确、
匹配 IoU 和类别准确率随训练明显提高。这是人为设置的排障参考，不是论文标准或架构正确性的证明。
总 loss 下降、偶尔一个正确框或 AP50 刚刚非零，都不足以证明拟合通过。
图上展示阈值为 0.2；AP 计算保留低分预测，因此不能仅凭图上没有框就判定 AP 为零。

若 4 图仍失败，先用相同配置 `--limit-train 1 --steps 2000 --batch-size 1` 并更换输出目录，降低记忆难度。
再结合已有逐图诊断检查：坐标/类别映射、定位/分类哪个未学会、背景竞争、梯度和学习率。
单图或四图拟合成功只证明能记住这些样本，尚不能证明特征具有可泛化的检测信息。

## 3. 独立验证：先小规模试训

仅在拟合检查通过后进行。使用原始 train2017 随机抽取 1000 张，val2017 独立抽取 200 张。
不沿用简单图片筛选，不从四图记忆检查点继续。80 类可能覆盖不均，报告缺失类别的 null，
这仍是训练趋势检查，不是 COCO 正式排行榜结果。

先检查完整 train/val 路径和类别：

```bash
python -m vit_probe.inspect_data \
  --config vit_probe/artifacts/coco_sanity/coco_pilot.json \
  --output vit_probe/artifacts/coco_sanity/pilot_data_audit.json
```

没有 val2017/对应标注时先停在四图检查；不要用训练集指标冒充独立验证。
两组均从相同 ViT、相同随机头初始化，使用相同样本与训练预算。

```bash
CUDA_VISIBLE_DEVICES=4 python -u -m vit_probe.train \
  --config vit_probe/artifacts/coco_sanity/coco_pilot.json \
  --mode frozen --limit-train 1000 --limit-val 200 --epochs 10 \
  --device cuda:0 \
  --output-dir "vit_probe/runs/coco_pilot_frozen_$(date +%Y%m%d_%H%M%S)"
```

如果冻结组不足以判断，再运行微调组（需要更多显存）：

```bash
CUDA_VISIBLE_DEVICES=4 python -u -m vit_probe.train \
  --config vit_probe/artifacts/coco_sanity/coco_pilot.json \
  --mode finetune --limit-train 1000 --limit-val 200 --epochs 10 \
  --device cuda:0 \
  --output-dir "vit_probe/runs/coco_pilot_finetune_$(date +%Y%m%d_%H%M%S)"
```

10 轮只有约 1250 次参数更新（batch=2、累积=4），用来检查趋势，不能直接作为淘汰路线的依据。
若 loss/验证指标仍持续改善，需要按预算重新规划更长训练。原版 DETR 的官方训练使用长周期：
https://github.com/facebookresearch/detr#training 。本项目不是原版 DETR，不能直接套用其性能或收敛轮数。
显存不足时两组都用 `--batch-size 1 --accumulation-steps 8` 保持有效 batch；保留相同预处理分辨率。

## 怎样决定后续路线

| 观察 | 支持的结论与下一步 |
| --- | --- |
| COCO 四图拟合成功、独立验证也能检测 | 基本链路可用；回到 AI-TOD 排查尺度、遥感域差异、特征分辨率和头的适配 |
| 四图成功，独立验证差 | 存在记忆能力，但泛化未证明；检查样本量、训练预算及特征适配 |
| 冻结差、微调明显改善 | 更支持当前冻结特征与检测任务适配不足，不能据此称视觉信息已丢失 |
| 单图/四图长期无法拟合 | 优先排查检测实现和优化；需要时用成熟检测头或已知有效特征作对照，再考虑换路线 |

COCO 与 AI-TOD 同时存在尺度、类别、密度和图像域等差异。COCO 成功而 AI-TOD 失败，
并不能单独定位“微小目标信息在哪一层丢失”。后续仍需尺度/阶段对照。

## 本地检查与边界

新增开发检查覆盖 80 类非连续 ID、整图筛选保留全部标注、train/val 不重叠、缺标注拒绝运行、
理想框 AP=1 以及小型真实 Qwen ViT 的 80 类缓存训练入口。
本地合成图片用于验证代码与评估，不用于宣称真实 COCO 检测性能。
当前会话无法访问服务器数据、完整 ViT 或 GPU，真实拟合效果需要以上服务器运行结果确认。

本次本地验证：torch 2.8.0+cpu、transformers 5.19.0，`python -m pytest vit_probe/tests -q` 共 19 项通过。
