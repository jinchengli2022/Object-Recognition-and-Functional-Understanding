# 大模型语义辅助实验目录

本目录用于把大模型/VLM 用在指标2.3任务中“它能稳定完成的部分”：RGB 物体语义识别、候选可供性判断、文本级功能描述。

本目录不会修改本地大模型文件。默认只读引用：

```text
/home/amadeus/research_proj/vidaff/mllm/Qwen3.5-2B
```

## 目录内容

- `configs/qwen_semantic_config.yaml`：本地 Qwen 模型路径、测试集 manifest、affordance 词表
- `prompts/qwen_semantic_prompt_zh.txt`：语义识别与功能判断提示词
- `scripts/qwen_semantic_eval.py`：调用本地 Qwen，对 RGB 图像输出类别和候选 affordance
- `scripts/summarize_semantic_results.py`：统计语义类别准确率
- `docs/method_comparison.md`：本项目方法、大模型基线、目标识别/功能理解模块对比表
- `docs/model_capability_matrix.md`：大模型能做和不能做的任务边界表
- `outputs/`：运行结果输出目录

## 运行方式

本次使用已有 conda 环境 `vidaff_qwen` 运行，不修改该环境，也不写入 Qwen 模型目录。

小样本验证：

```bash
cd /home/amadeus/project_dev/指标2.3/中期验收/指标验收/大模型语义辅助
CUDA_VISIBLE_DEVICES=0 conda run -n vidaff_qwen python scripts/qwen_semantic_eval.py --limit 10 --device cuda
```

完整跑 226 张 RGB 图像：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n vidaff_qwen python scripts/qwen_semantic_eval.py --device cuda --output-dir outputs/qwen3_5_2b_semantic_eval_closedset_full
```

完整结果已写入：

```text
outputs/qwen3_5_2b_semantic_eval_closedset_full/
```

包括：

- `semantic_predictions.tsv`
- `summary.json`


## 本地 Qwen 实测结果

同一批目标识别测试图像上，本地 Qwen3.5-2B 只进行 RGB 语义类别识别和文本级 affordance 先验生成，不输出 mask，也不输出三维功能区域。

| 方法 | 输入 | 输出 | 结果 |
|---|---|---|---:|
| 本地 Qwen3.5-2B 语义辅助（封闭类别） | RGB 图像 + 候选类别表 | 物体类别 + 候选 affordance + 文本描述 | 206 / 226 = 91.15% |
| 本项目目标识别模型 | RGB 图像 | 物体类别 + 2D mask | 225 / 226 = 99.56% |

按目标识别阶段的类别正确性比较，本项目目标识别模型相对封闭类别 Qwen 语义基线绝对提升约 8.41 个百分点。Qwen 的主要误差来自项目类别定义与通用语义类别不完全一致，例如将 `bag` 类中的医疗箱判断为 `medicine_box`，将 `open_container` 类中的铁锅、水杯、盆栽等判断为更具体的 `kettle`、`bottle` 或 `plate`。

## 评价口径

该实验只评价大模型在语义层面的能力：

- RGB 图像物体类别识别；
- 候选可供性列表生成；
- 文本级功能描述。

它不评价：

- 2D mask IoU；
- 三维点云点级可供性区域；
- 距离容忍 aIoU。

因此，大模型语义结果不能直接替代指标2.3完整结果。完整指标仍使用：

```text
目标识别准确率 × 功能理解准确率
```

当前项目完整方法结果为：

| 指标 | 结果 |
|---|---:|
| 目标识别准确率 | 225 / 226 = 99.56% |
| 功能理解准确率 | 857 / 1000 = 85.70% |
| 指标2.3综合准确率 | 85.3208% |

## 文档表格

方法对比见：

```text
docs/method_comparison.md
```

大模型能力边界见：

```text
docs/model_capability_matrix.md
```
