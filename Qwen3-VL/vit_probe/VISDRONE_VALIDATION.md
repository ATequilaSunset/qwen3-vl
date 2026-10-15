# VisDrone MOT 逐帧检测诊断

沿用已在 COCO 训练图片上拟合成功的独立 ViT + 查询式检测头，进一步检查无人机航拍场景。
只做单帧检测；不加入时序模型、关联模块或轨迹损失。轨迹 ID 仅保存在转换标注中。
本实现输出的是自定义 COCO 风格逐帧 bbox 指标，不是 VisDrone 官方 MOT/DET 排行榜指标。

## 更新位置

完整压缩包内是 `vit_probe/`，合并到仓库内层 `Qwen3-VL/vit_probe/`（与 infer.py 同级）。
可继续使用当前实验分支；如果要单独保存这一组工作，可从已更新的 COCO 分支创建
`experiment/vit-visdrone-sanity`。本次交付没有自动推送 GitHub。

本更新除了新增 `prepare_visdrone.py`，还更新了 data.py、criterion.py、evaluate.py、overfit_cached.py。
必须更新这些文件，不能只复制转换脚本。既有 COCO/AI-TOD 文件没有显式 ignore_regions_xywh 字段时，
原有数据、匹配、损失和评估行为不变；缓存诊断多输出尺度统计。

## 数据和标注合同

默认根目录：`/data/lz/VisDrone2019-MOT-train`。
需要 `sequences/<序列名称>/<数字帧号>.jpg` 以及 `annotations/<序列名称>.txt`。
例：sequences/uav0000001_00000_v/0000001.jpg 对应 annotations/uav0000001_00000_v.txt 中帧号 1。

MOT 每行 10 列：frame_index,target_id,left,top,width,height,score,category,truncation,occlusion。
8 列的 DET 标注会报错，不允许错位读取。
坐标按提供的 left/top/width/height 解释，不额外减 1；越界框裁到图像边界并记录。
无效有效目标框会报错；空的忽略区域记录后跳过。

正样本要求 score=1 且 category 在 1..10；截断、遮挡和微小尺寸不作为删除正样本的依据。
类别词表固定为 pedestrian、people、bicycle、car、van、truck、tricycle、awning-tricycle、bus、motor。
不同 MOT 发布版本中可能只有其中五类存在有效标注，以 prepare_report.json 的实际统计为准。
不存在 GT 的类别返回 null。不能把这项十类词表的帧检测诊断当作官方五类 MOT 评测。

score=0 或 category=0/11 的框进入显式忽略区域，不作为普通背景或正目标：

- 训练时，预测框与任一忽略区域的交面积 / 预测框可见面积 >=0.5 时，不对该未匹配 query 施加背景分类损失。
- 与有效 GT 匹配的 query 始终保留监督，即使与忽略区域重叠。
- 评估时将每个忽略区域展开成每个类别的 COCO crowd 框，允许 COCOeval 忽略区域内的重复预测。
  COCO crowd 的匹配阈值随评估 IoU 阈值变化。这是明确记录的自定义适配，不宣称等价于 VisDrone 官方规则。
- 输入图片不做遮黑或裁剪；标注可视化中橙色为忽略区域，绿色为有效目标。

官方格式依据：https://aiskyeye.com/evaluate/results-format_2021/
官方数据集：https://github.com/VisDrone/VisDrone-Dataset

## 1. 准备数据

在服务器内层 Qwen3-VL 目录及 qwen3-vl 环境运行：

```bash
python -m vit_probe.prepare_visdrone \
  --root /data/lz/VisDrone2019-MOT-train \
  --output-dir vit_probe/artifacts/visdrone_sanity
```

默认每 30 帧采样一次（1、31、61……），只读取图片尺寸，不复制图片，不更改源数据。
如果提供 --val-root /实际/VisDrone2019-MOT-val，使用指定验证集，并检查视频分组不重叠。
未提供验证集时，在当前训练发布集内固定 seed=42，按视频分组留出约 20% 做内部验证。
uavNNNNNNN_* 的不同片段保守地归入同一视频组，其余名称按完整序列名分组。
不能仅凭不同帧号就认为训练和验证独立；后续正式泛化结论还需核实视频来源。
只有一个视频组时仍能生成拟合配置，但不生成 pilot 配置，避免伪造独立验证。

从训练组中优先选取不同视频的 4 帧；每帧 1–100 个有效目标，保留每帧全部有效框和忽略区域。
--max-objects 只筛选整帧，不截断目标；--frame-stride 与 --images 可显式调整。
没有按目标大小筛选，因而与先前专门挑选简单大目标的 COCO 子集不是严格单变量对照。

输出：

| 文件 | 内容 |
| --- | --- |
| prepare_report.json | 原始标注哈希、视频分组、类别/忽略区域统计、所选 4 帧及 ViT 权重路径 |
| ground_truth/ | 4 张训练图片的 GT 与忽略区域可视化 |
| overfit_annotations.json / visdrone_overfit.json | 固定 4 帧标注与拟合配置 |
| train_annotations.json / val_annotations.json | 按视频分组的采样数据，供后续独立验证 |
| visdrone_pilot.json | 仅存在独立验证分组时生成 |

先看 GT 框是否正确，并确认 vit_weights_present=true。
ViT 默认沿用 /data/lz/models/Qwen3-VL-4B-ViT，可用 --vit-path 覆盖。
若目录结构不符，准备脚本会报告缺失位置；不要套用 COCO 的 prepare_coco.py。

## 2. 4 图冻结拟合

沿用 GPU 4（启动前检查占用）、FP32、dropout=0、恒定学习率和 2000 次参数更新：

```bash
CUDA_VISIBLE_DEVICES=4 python -u -m vit_probe.overfit_cached \
  --config vit_probe/artifacts/visdrone_sanity/visdrone_overfit.json \
  --limit-train 4 --steps 2000 --batch-size 2 \
  --eval-every 100 --log-every 100 --device cuda:0 \
  --output-dir "vit_probe/runs/visdrone_overfit_4_$(date +%Y%m%d_%H%M%S)"
```

检测头重新初始化为 10 类；不加载 COCO 已训练检测头。ViT 保持同一导出权重。
比较 AP50、预测图、matched_class_accuracy、matched_center_error_px 和 matched_iou_mean。
每个 step_*.json 的 per_image 中还记录 processed_wh、processed_box_short_side_px、
size_groups_diagnostic：按预处理后框短边 <8、8–16、16–32、>=32 像素分组。
这些组给出 GT 数、最大 IoU 和“任一 query 达到 IoU 0.5 的 GT 比例”，忽略类别/分数并允许复用框；
它们是定位诊断，不是正式 AP/召回率，也不是 VisDrone 官方目标尺寸分组。

若训练图上的航拍大目标可拟合、小目标难拟合，尺度相关瓶颈的假设更值得检查；
若各尺度都差，还需要检查遥感域差异、密度、类别和检测头适配。
跨数据集结果不是单一变量控制，不能据此定位信息在 ViT 某层丢失。

## 3. 可选后续独立验证

仅当四图拟合通过并且存在 visdrone_pilot.json 时进行：

```bash
python -m vit_probe.inspect_data \
  --config vit_probe/artifacts/visdrone_sanity/visdrone_pilot.json \
  --output vit_probe/artifacts/visdrone_sanity/pilot_data_audit.json

CUDA_VISIBLE_DEVICES=4 python -u -m vit_probe.train \
  --config vit_probe/artifacts/visdrone_sanity/visdrone_pilot.json \
  --mode frozen --limit-train 1000 --limit-val 200 --epochs 10 \
  --device cuda:0 \
  --output-dir "vit_probe/runs/visdrone_pilot_frozen_$(date +%Y%m%d_%H%M%S)"
```

最多取 1000/200 帧，实际数量以日志为准。这是小规模趋势验证；训练不足不能直接作为换路线的依据。
指标使用 COCO 风格 bbox AP，0–1 取值，同时有 maxDets100 和可用时的 queries 容量版本。

## 已验证范围

CPU 小型真实 Qwen ViT：22 项测试通过，包含既有 COCO/AI-TOD 检查、MOT 十列解析、
小目标保留、忽略区域背景梯度、有效匹配优先、忽略区域预测不计误检、区域外误检降低 AP、
同一视频不同片段不泄漏、重复帧文件名正确处理、四帧缓存训练和预处理后尺度统计。
本次未访问服务器真实 VisDrone 数据、完整 ViT 权重或 GPU；真实效果需运行确认。
