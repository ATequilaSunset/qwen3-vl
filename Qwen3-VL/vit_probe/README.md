# Qwen3-VL 独立 ViT：提取与检测实验

第一阶段完成 **提取主 Merger 之前的 ViT，以及验证提取是否正确**。
第二阶段已加入 object queries、检测 decoder、分类/回归头和训练评估入口，
使用说明见 [DETECTION.md](DETECTION.md)。真实权重的一致性验收通过后再进入第二阶段。
普通场景 COCO 架构排障见 [COCO_VALIDATION.md](COCO_VALIDATION.md)，先做简单图片拟合，再做独立验证。
现有 `infer.py` 和历史实验均不需要改动。

## 1. 提取边界

保留原始图像预处理、Patch Embedding、学习式位置编码、视觉 RoPE 和所有 ViT blocks。
复用 Transformers 的原版视觉前向实现，将主 Merger 替换为 `Identity`，
并关闭 DeepStack 的旁路 Merger。没有 LLM、语言输出头或额外检测器。

`spatial_merge_size=2` **仍然保留在配置中**：预处理和位置编码依赖这个分组顺序。
它不代表这个独立模块仍在合并 token。输出 token 数是 `H_grid * W_grid`，没有除以 4。
导出的权重中没有主 Merger 或 DeepStack Merger 参数。

输入必须是原版 image processor 生成的 `pixel_values` 和 `image_grid_thw`。
输出含两种表示：

| 字段 | 形状与含义 |
| --- | --- |
| `tokens` | `[所有图像的 patch 数之和, hidden_size]`，保持原模型的分组顺序 |
| `feature_maps` | 一个列表，每张图对应 `[hidden_size, H_grid, W_grid]`，已还原正常行列顺序 |
| `image_grid_thw` | 每张图的 `[1, H_grid, W_grid]` |

不要直接对 `tokens` 执行 `reshape(H,W,C)`：原处理器按 2×2 分组排列，直接 reshape 会错位。
本阶段只接受静态图像（T=1）。网格对应的是**预处理后的图像**；后续检测框需要按缩放比例映射回原图。

## 2. 环境

Python >= 3.10；本实现固定使用 `transformers==5.19.0`，对应仓库已有推理记录。
模型必须是本地、未量化、safetensors 格式的 dense Qwen3-VL 检查点。
加载权重时严格检查所有保留的参数；缺少参数会报错，不允许随机初始化后继续实验。

已使用 CPU 环境 `torch==2.8.0+cpu`、`torchvision==0.23.0+cpu` 验证。
已有服务器记录是 `torch==2.6.0+cu124`、`torchvision==0.21.0+cu124`；
该 CUDA 组合需要在服务器上实测。注意 2.6.0 的 CPU 构建与这里固定的 Transformers
组合在本次验证中发生了 accelerator 初始化错误，因此 CPU 验证请采用 2.8.0/0.23.0。

先在项目内层目录（含 `infer.py` 的 `Qwen3-VL`）运行。
使用已有可运行的 Qwen 环境时，先查看版本，再安装本目录的轻量依赖：

```bash
python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
python -m pip install -r vit_probe/requirements.txt
```

如果需要在无 GPU 的电脑创建独立验证环境，可使用：

```bash
conda create -n qwen-vit-check python=3.11 -y
conda activate qwen-vit-check
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r vit_probe/requirements.txt
```

这两段环境方案按实际情况选择一个。导出可在 CPU 上完成，但仍需原始视觉权重和足够内存。
无需在本地电脑下载完整模型；可以在已有权重的服务器上执行下面两步。

## 3. 导出独立 ViT

下面是仓库实验记录中的服务器路径，请按实际目录调整：

```bash
python -m vit_probe.extract \
  --model-path /data/lz/models/Qwen3-VL-4B-Instruct \
  --output-dir /data/lz/models/Qwen3-VL-4B-ViT \
  --dtype float32
```

PowerShell 下把一条命令写在同一行，不使用上述 Bash 的行末反斜杠。

默认保存 float32，不对原始低精度权重额外有损量化；文件会比 bf16 大。
脚本只读取视觉参数；即使视觉与 LLM 权重位于同一个 shard，也不会加载其中的 LLM 张量。
输出目录必须不存在或为空。

导出文件包括：`vit.safetensors`、`vit_config.json`、`extraction.json` 和图像预处理配置。
导出格式由本目录的 `PreMergerViT.from_export` 加载，不能当作完整 Qwen VLM 交给 AutoModel 加载。
`extraction.json` 会记录来源、参数量、版本、精度和实际读取的 shard。

## 4. 检查导出前后的特征一致性

```bash
python -m vit_probe.verify \
  --model-path /data/lz/models/Qwen3-VL-4B-Instruct \
  --vit-path /data/lz/models/Qwen3-VL-4B-ViT \
  --images /data/wrj/AI-TOD/aitod/images/test/1dc20e8b1.png \
  --device cuda:0 \
  --dtype float32 \
  --output vit_probe/artifacts/consistency.json
```

该图片路径来自历史实验，请确认图片实际存在，也可替换为任意本地 RGB 图片。
`--images` 可以接多个路径。首次用一张图，控制显存和排查范围。
无 GPU 时使用 `--device cpu --dtype float32`，真实 4B 视觉编码器会运行得较慢。

检查脚本会：

1. 分别加载原始与导出的预处理器，确认输入张量完全一致。
2. 加载**原版视觉模块（保留全部 Merger）**，在主 Merger 输入位置设置 hook。
3. 释放参考模型，再加载导出的独立 ViT，比较输出特征。
4. 保存误差、容差、图像缩放比例、网格和特征图形状。超出容差时返回非零退出码。

参考路径加载原始视觉模块，不加载 LLM。脚本验证视觉阶段的等价性，不验证后续语言生成。
默认 `atol=rtol=1e-5`，两条路径使用相同 device、dtype 和 attention backend。
如果失败，先排查版本、预处理、权重和运算设置，不应直接放宽容差。
需要结合原始图像尺度解释结果；这里默认 min/max pixels 沿用 `infer.py` 的 65536/1048576。

验收看 JSON 中 `passed: true`，并检查网格、实际缩放和误差是否符合预期。
此时才能进入检测头阶段；这里的 PASS 不代表已经获得检测性能或证明小目标信息没有丢失。

## 5. 后续接检测头的接口

```python
from transformers import AutoImageProcessor
from vit_probe.backbone import PreMergerViT

vit = PreMergerViT.from_export(vit_dir).to(device)
processor = AutoImageProcessor.from_pretrained(vit_dir, local_files_only=True)
inputs = processor(images=images, return_tensors="pt").to(device)
features = vit(**inputs)
feature_map = features.feature_maps[0]  # [C,H,W]
```

前向接口保留梯度。实验 A 冻结 ViT 参数、训练检测头；实验 B 更新 ViT 和检测头。
冻结可用 `vit.requires_grad_(False); vit.eval()`；微调可用 `vit.requires_grad_(True); vit.train()`。
检测头、匹配损失、AI-TOD 数据预检与 COCO 风格 mAP 评估现已提供，见 [DETECTION.md](DETECTION.md)。

## 6. 离线开发检查

不下载真实 Qwen 权重或 AI-TOD 数据，构造小型的原版 Qwen3-VL 检查点进行验证：

```bash
python -m pip install pytest
python -m pytest vit_probe/tests -q
```

覆盖原模型 hook 对照、eager/SDPA、导出再加载、异形多图空间顺序、分片读取且缺省 LLM shard、
微调/冻结梯度、缺失权重拒绝加载，以及人为破坏权重后检查脚本是否正确失败。
这些离线检查不替代真实权重上的服务器验收。

实现依据：
- https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
