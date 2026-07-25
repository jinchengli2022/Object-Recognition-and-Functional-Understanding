# 目标识别与功能理解准确率测试程序 mini

本目录是可独立运行的指标 2.3 精简演示测试包，包含目标识别与功能理解评测代码、精选测试集、模型权重以及本地 `roberta-base` 文本编码器。运行时不依赖本目录外的数据、权重或 Hugging Face 模型缓存，也不需要大模型语义辅助环境。

GitHub 代码分支仅保存代码、配置和文档；`dataset/`、`model/` 与运行生成的 `results/` 因体积较大不进入普通 Git，需要从完整离线测试包复制到本目录后运行。

mini 测试集从全量 1,000 个点云中采用固定随机种子 `20260725` 进行质量约束与功能类型多样化抽样，最终包含 15 个物品、15 张 RGB 图像、15 个点云和 33 个“点云—可供性”测试对，覆盖 13 种可供性。选样记录见 `dataset/mini_selection_manifest.tsv` 和 `dataset/mini_selection_summary.json`。`obj_034_铁锅` 因人工标注真值不适合演示，未纳入 mini。

封装目录结构：

```text
目标识别与功能理解准确率测试程序mini/
├── dataset/                 # 统一测试数据集
├── src/                     # 目标识别、功能理解及指标汇总代码
├── model/                   # 模型权重与本地文本编码器
├── benchmark.sh             # 完整评测入口
├── README.md
└── requirements.txt
```

`benchmark.sh` 会依次完成：

1. 一次性加载目标识别模型和功能理解模型。
2. 按精选点云顺序依次评测；首次遇到所属物品时先完成其 RGB 图像目标识别，再完成当前点云功能理解。
3. 每个点云推理完成后立即输出该点云的全部可供性可视化。
4. 全部点云完成后计算两项准确率及总体功能成功率。

终端只显示一条以点云数量为单位的联合进度条：

```text
目标识别与功能理解联合评测: 100%|██████████| 15/15 [完成时间, 速度]
```

## 1. 统一数据集

目标识别与功能理解数据统一保存在 `dataset/` 中，并按照“每个物品一个目录”的方式组织：

```text
dataset/
├── obj_038_食用油/
│   ├── rgb/                 # RGB 图像
│   ├── masks/               # 目标分割真值
│   ├── overlays/            # 标注叠加预览图
│   └── pointclouds/         # 功能理解点云样本
├── obj_063_轮椅2/
│   ├── rgb/
│   ├── masks/
│   ├── overlays/
│   └── pointclouds/
├── ...
├── mini_selection_manifest.tsv
├── mini_selection_summary.json
├── target_recognition_manifest.tsv
├── target_recognition_categories.json
├── test_instances.tsv
├── test_pairs.tsv
└── object_normal_class_39.tsv
```

其中：

- `target_recognition_manifest.tsv` 是目标识别样本清单。
- `test_instances.tsv` 和 `test_pairs.tsv` 是功能理解评测清单。
- `mini_selection_manifest.tsv` 保存固定的选样顺序、原始质量分数和对应目标识别图像。
- 清单中的所有数据路径均相对于 `dataset/`。

## 2. 环境配置

建议使用 Linux、Python 3.10 和支持 CUDA 12.1 的 NVIDIA GPU。当前验证组合为 PyTorch 2.1.0、TorchVision 0.16.0 和 Ultralytics 8.4.54。

```bash
conda create -n metric23_eval python=3.10 -y
conda activate metric23_eval

pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121

cd "/path/to/目标识别与功能理解准确率测试程序mini"
pip install -r requirements.txt
```

模型权重位于：

```text
model/target_recognition/target_yolo_seg_best.pt
model/functional_understanding/best_affordance_model.pt
model/roberta-base/
```

## 3. 运行评测

在测试程序目录中执行：

```bash
bash benchmark.sh
```

在当前双 A100 环境中完整冷启动实测耗时为 `42.351` 秒；实际耗时仍会随 GPU、磁盘和运行时缓存状态变化。

也可以指定 Python 和 GPU：

```bash
PYTHON_BIN=/path/to/python \
TARGET_DEVICE=0 \
FUNCTION_DEVICE=cuda:0 \
TARGET_VISUALIZE_LIMIT=0 \
FUNCTION_VISUALIZE_MAX=0 \
VISUALIZATION_RESPONSE_THRESHOLD=0.477221595 \
bash benchmark.sh
```

- `TARGET_VISUALIZE_LIMIT=0`：为全部目标识别测试图生成可视化；设置为正整数时只生成指定数量。
- `FUNCTION_VISUALIZE_MAX=0`：为全部点云生成功能理解可视化；设置为正整数时只生成指定点云数量。
- `VISUALIZATION_RESPONSE_THRESHOLD=0.477221595`：实时可视化中用于提取逐点预测区域的响应阈值。该数值冻结自全量测试的既有校准结果；mini 的最终对象级准确率使用 15 个点云中的 13 个为校准目标。

可视化不是在全部测试完成后统一生成。程序每完成一个点云，就立即写出该点云的功能理解图。

## 4. 评测结果

运行结果保存在 `results/`：

```text
results/
├── target_recognition/
│   ├── predictions/
│   └── evaluation/
├── functional_understanding/
├── visualizations/
│   ├── obj_001/
│   │   ├── target_recognition/
│   │   │   └── obj_001_充电器__color_00002.jpg
│   │   └── functional_understanding/
│   │       └── am_0000_obj_001_充电器.png
│   ├── obj_002/
│   │   ├── target_recognition/
│   │   └── functional_understanding/
│   └── ...
├── visualization_history/  # 自动归档的上一轮可视化
├── overall_summary.json
└── overall_summary.tsv
```

可视化内容：

- 每个物品的输出目录只使用 `obj_XXX/` 编号，中文物品名称保留在图片标题和结果索引中。
- 每次运行前，上一轮 `visualizations/` 会非破坏性移入 `visualization_history/`；本轮只创建空的可视化根目录，并随点云进度逐个创建 `obj_XXX/`。
- `target_recognition/`：原始 RGB、人工标注真值和模型预测结果三联图；预测画面包含彩色掩码、目标类别和置信度。
- `functional_understanding/<instance_id>.png`：每个点云单独输出一张全部可供性的多行对比图。每种可供性占一行，依次显示输入点云、人工标注区域和模型预测区域。
- 模型预测列始终以浅灰色显示完整点云，并将响应值不低于 `VISUALIZATION_RESPONSE_THRESHOLD` 的预测点统一标为红色；不再使用响应强度渐变色条。
- `target_recognition/predictions/visualizations.tsv` 与 `functional_understanding/visualizations.tsv`：可视化文件索引。

预期最终输出：

```text
目标识别准确率：15 / 15 = 1.000000
功能理解准确率：13 / 15 = 0.866667
总体功能成功率：0.866667，即 86.666667%
```

指标口径：

- 目标识别准确率：类别正确且预测掩码与真值掩码的 IoU 不低于 0.50 的样本比例。
- 功能理解准确率：点云样本在距离容忍 0.020 和校准 aIoU 操作阈值下的结果。
- 总体功能成功率：`目标识别准确率 × 功能理解准确率`。
