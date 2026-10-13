随机抽样 20 张，种子 `861235812`。异常样本直接跳过，原始回答和已发生的重试记录均保留。

不同提示词的补充结果保存在 `anomalies/`，不计入正常批次统计。

| 样本 | 异常现象 | 原始回答 |
|---|---|---|
| `P2339__1.0__1686___1389.png` | 坐标超出规定的 0–1000 范围；达到生成 token 上限，JSON 未完整输出；生成未正常结束，JSON 不完整或达到生成时限 | [查看](/data/lz/pythonProjects/qwen3-vl/Qwen3-VL/outputs/exp1/responses/P2339__1.0__1686___1389_attempt1_response.txt) |
| `1920__2265_1725.png` | 坐标超出规定的 0–1000 范围；生成未正常结束，JSON 不完整或达到生成时限 | [查看](/data/lz/pythonProjects/qwen3-vl/Qwen3-VL/outputs/exp1/responses/1920__2265_1725_attempt1_response.txt) |
| `2552__1800_1800.png` | 坐标超出规定的 0–1000 范围；生成未正常结束，JSON 不完整或达到生成时限 | [查看](/data/lz/pythonProjects/qwen3-vl/Qwen3-VL/outputs/exp1/responses/2552__1800_1800_attempt1_response.txt) |
| `0000313_00201_d_0000437__0_0.png` | 生成未正常结束，JSON 不完整或达到生成时限 | [查看](/data/lz/pythonProjects/qwen3-vl/Qwen3-VL/outputs/exp1/responses/0000313_00201_d_0000437__0_0_attempt1_response.txt) |
| `9999984_00000_d_0000152__600_0.png` | 生成未正常结束，JSON 不完整或达到生成时限 | [查看](/data/lz/pythonProjects/qwen3-vl/Qwen3-VL/outputs/exp1/responses/9999984_00000_d_0000152__600_0_attempt1_response.txt) |
