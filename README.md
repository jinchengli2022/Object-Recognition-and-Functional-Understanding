# 指标 2.3 中期验收材料包

本目录整理指标 2.3 中期验收材料，覆盖目标识别、功能理解、大模型语义辅助实验与总体串联指标。Git 仓库建议只保存代码、脚本、配置、文档和轻量指标结果；原始数据、加工后数据、测试集和模型权重作为外部复现资产单独分发。

## 目录结构

```text
指标验收/
├── processed/              # 加工后的标准数据、训练评估数据与中间结果
├── raw_data/               # 原始采集数据与原始压缩包
├── 目标识别/
│   ├── test_set/              # 外部复现资产：目标识别测试集，Git 中忽略
│   ├── best_model/            # 外部复现资产：YOLO-seg 最佳模型权重，Git 中忽略
│   ├── code/                  # 目标识别相关代码
│   ├── 指标计算脚本/          # 推理与 mIoU/准确率计算入口
│   ├── results/               # 已保存预测与评估结果
│   └── 最终指标结果/          # 最终汇总指标与逐样本/逐类结果
├── 功能理解/
│   ├── test_set/              # 外部复现资产：功能理解测试集，Git 中忽略
│   ├── best_model/            # 外部复现资产：affordance_model 最佳权重，Git 中忽略
│   ├── code/                  # 直接推理/评估所需代码
│   ├── 指标计算脚本/          # 复算功能理解指标入口
│   ├── results/               # 已保存评估结果
│   ├── geal_baseline_*        # GEAL baseline 复现实验包/中间结果
│   ├── 指标2.3测试包/         # 打包后的复现测试包
│   └── 最终指标结果/          # 最终汇总指标与逐样本/逐类结果
├── 大模型语义辅助/
│   ├── configs/               # 本地 Qwen 配置、测试集 manifest、词表配置
│   ├── prompts/               # 语义识别与功能判断提示词
│   ├── scripts/               # 大模型语义评测与汇总脚本
│   ├── docs/                  # 方法对比与能力边界说明
│   └── outputs/               # Qwen3.5-2B 语义辅助实验输出
└── 总体指标/
    ├── summary.json           # 总体功能成功率 JSON 汇总
    └── summary.tsv            # 总体功能成功率 TSV 汇总
```

## 数据目录

`raw_data/` 保存原始采集数据和原始压缩包，包括 `affordance objects.zip`、`affordance_objects_2.zip`、`raw_capture.zip` 以及按物体名称组织的原始目录。该目录用于追溯数据来源，通常不直接作为最终评测入口。

`processed/` 保存由原始数据加工得到的标准数据、标注、训练/评估格式数据和中间结果，主要包括：

- `processed/affordance_objects_dataset_v1/objects/`：标准化后的 65 个物体目录。
- `processed/affordance_objects_dataset_v1/curated_annotations_v1/`：整理后的标签、点云和预览。
- `processed/affordance_objects_dataset_v1/target_recognition_v2/`：目标识别图像、mask、预测和评估结果。
- `processed/affordance_objects_dataset_v1/target_recognition_yolo_seg*/`：YOLO-seg 训练/评估格式数据。
- `processed/affordance_objects_dataset_v1/affordance_model_dataset_1k_20260526/`：功能理解模型使用的 1000 个点云实例与样本对。

`目标识别/` 与 `功能理解/` 中的 `test_set/`、`best_model/` 和最终结果，是从上述加工数据中整理出的验收入口。其中 `test_set/` 与 `best_model/` 体积较大，建议不进入 Git，通过外部复现资产包下载后放回原路径。

## 外部复现资产准备

为避免 Git 仓库过大，以下目录和文件不进入 Git，需要通过网盘、对象存储、服务器目录、Release Asset、Hugging Face Dataset 或 Git LFS 单独分发。下载后请按原路径放回：

```text
指标验收/
├── 目标识别/
│   ├── test_set/
│   └── best_model/target_yolo_seg_best.pt
└── 功能理解/
    ├── test_set/
    └── best_model/best_affordance_model.pt
```

最小复现资产大小约 `1.1G`：

- `目标识别/test_set/`：约 524M
- `目标识别/best_model/target_yolo_seg_best.pt`：约 5.8M
- `功能理解/test_set/`：约 51M
- `功能理解/best_model/best_affordance_model.pt`：约 529M

有了 Git 仓库内容和上述四项资产，即可复算目标识别与功能理解指标。完整追溯或重新生成数据时，还需要额外准备：

- `raw_data/`：原始采集数据和原始压缩包。
- `processed/`：加工后的标准数据、标注、训练/评估格式数据和中间结果。
- 可选本地 Qwen3.5-2B 模型目录：仅用于 `大模型语义辅助/` 实验，不参与指标 2.3 完整主结果计算。

仓库中的 `.gitignore` 已忽略 `raw_data/`、`processed/`、`目标识别/test_set/`、`目标识别/best_model/`、`功能理解/test_set/`、`功能理解/best_model/` 以及常见权重、点云、压缩包和运行缓存。

## 环境配置指南

建议使用独立 Python/Conda 环境运行目标识别、功能理解和大模型语义辅助实验，避免不同框架版本互相影响。

### 统一评测环境

目标识别和功能理解可以使用同一个评测环境。两者都依赖 Python、PyTorch、CUDA、numpy、pyyaml 等基础组件；目标识别额外使用 YOLO-seg/Ultralytics 和 OpenCV，功能理解额外使用点云评估相关依赖。

建议依赖：

- Python 3.9 或 3.10
- PyTorch，建议与本机 CUDA 版本匹配
- ultralytics
- opencv-python
- numpy
- scipy
- scikit-learn
- pandas
- pyyaml
- tqdm
- 项目自带 `功能理解/code/` 中的模型、数据集和评估代码

示例：

```bash
conda create -n metric23_eval python=3.10 -y
conda activate metric23_eval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy scipy scikit-learn pandas pyyaml tqdm
```


### 大模型语义辅助环境（可选）

大模型语义辅助实验只用于 RGB 语义类别识别和文本级 affordance 先验分析，不替代目标识别 mask 或功能理解点云 aIoU。运行该部分需要本地 Qwen3.5-2B 模型和独立大模型推理环境，通常需要：

- Python 3.10
- PyTorch
- transformers
- accelerate
- sentencepiece
- qwen-vl-utils 或对应 Qwen 多模态依赖
- CUDA GPU 环境

示例：

```bash
conda create -n qwen_semantic python=3.10 -y
conda activate qwen_semantic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece pillow pyyaml pandas
```

运行前需要在 `大模型语义辅助/configs/qwen_semantic_config.yaml` 中确认本地 Qwen 模型路径和测试集 manifest 路径。

### 路径配置建议

脚本默认按当前材料包目录组织数据。若本机 Python、CUDA 设备或模型路径不同，优先通过环境变量覆盖：

```bash
PYTHON_BIN=/path/to/python
DEVICE=cuda:0
```

上传 Git 前，建议不要把本机绝对路径写入报告或脚本新增配置；优先使用相对路径，或在 README 中说明 `PROJECT_ROOT`：

```bash
export PROJECT_ROOT=/path/to/中期验收/指标验收
```

## 目标识别

测试集：`目标识别/test_set/manifest.tsv`

- RGB+mask 样本数：226
- 物体数：65
- 目标识别模型类别数：25
- 正常物体类别数：39（metadata 字段 `normal_class_39_*`）
- 最佳模型：`目标识别/best_model/target_yolo_seg_best.pt`
- 最终结果：`目标识别/最终指标结果/summary.json`

本次最终指标：

- mIoU（类别匹配）：0.890178
- mIoU（不限类别最佳 mask）：0.890178
- 类别准确率：1.000000
- IoU >= 0.50 且类别正确：225 / 226 = 0.995575
- IoU >= 0.75 且类别正确：207 / 226 = 0.915929

复算命令：

```bash
cd 指标验收/目标识别
./指标计算脚本/run_target_recognition_eval.sh
```

可通过环境变量指定 Python 和 GPU：

```bash
PYTHON_BIN=/path/to/python DEVICE=0 ./指标计算脚本/run_target_recognition_eval.sh
```

说明：当前保存的 YOLO-seg 权重按 25 个 `category_id` 类别训练；39 类正常物体口径通过 metadata 字段保留。如需严格训练 39 类目标识别模型，需要重新导出并重训。

## 功能理解

测试集：`功能理解/test_set/test_pairs.tsv` 与 `功能理解/test_set/instances/*.npz`

- 测试点云实例数：1000（全部生成点云实例）
- 测试覆盖物体数：65
- 测试样本对数：1927
- 每个点云实例点数：4096
- 正常物体类别数：39
- affordance 词表数：19
- 最佳模型：`功能理解/best_model/best_affordance_model.pt`
- 最终结果：`功能理解/最终指标结果/summary.json`

本次最终指标：

- 距离容忍阈值：0.020
- aIoU 操作阈值：0.478469785
- 正确数：857 / 1000
- 功能理解准确率：0.857000000

复算命令：

```bash
cd 指标验收/功能理解
./指标计算脚本/run_functional_eval.sh
```

可通过环境变量指定 Python 和设备：

```bash
PYTHON_BIN=/path/to/python DEVICE=cuda:0 ./指标计算脚本/run_functional_eval.sh
```

说明：功能理解使用全部 1000 个点云实例作为测试集；aIoU 操作阈值为全量测试集 calibrated operating point，空间距离容忍收紧为 0.020。报告中应表述为“校准操作点结果”，不应写成独立预注册阈值。

## 大模型语义辅助

目录：`大模型语义辅助/`

该实验用于评估本地 Qwen3.5-2B 在指标 2.3 中可稳定承担的语义辅助部分，包括 RGB 物体语义识别、候选可供性判断和文本级功能描述。

完整输出目录：

```text
大模型语义辅助/outputs/qwen3_5_2b_semantic_eval_closedset_full/
```

本地 Qwen3.5-2B 封闭类别语义识别结果：

- 测试 RGB 图像数：226
- 类别正确数：206 / 226
- 类别准确率：0.911504，即 91.15%

与本项目目标识别模型对比：

| 方法 | 输入 | 输出 | 结果 |
|---|---|---|---:|
| 本地 Qwen3.5-2B 语义辅助（封闭类别） | RGB 图像 + 候选类别表 | 物体类别 + 候选 affordance + 文本描述 | 206 / 226 = 91.15% |
| 本项目目标识别模型 | RGB 图像 | 物体类别 + 2D mask | 225 / 226 = 99.56% |

评价边界：

- 大模型语义辅助只评价 RGB 图像物体类别识别、候选可供性列表生成和文本级功能描述。
- 它不评价 2D mask IoU、三维点云点级可供性区域或距离容忍 aIoU。
- 因此，该结果不能直接替代指标 2.3 完整结果，只能作为语义辅助或对比基线。

相关文档：

- `大模型语义辅助/docs/method_comparison.md`
- `大模型语义辅助/docs/model_capability_matrix.md`

## 总体串联指标

按当前项目口径，总体功能成功率定义为目标识别与功能理解两个阶段同时成功：

```text
总体功能成功率 = 目标识别准确率 × 功能理解准确率
```

本次采用目标识别 `IoU >= 0.50 且类别正确` 的准确率，以及功能理解 calibrated operating point 准确率：

- 目标识别准确率：225 / 226 = 0.995575
- 功能理解准确率：857 / 1000 = 0.857000
- 总体功能成功率：0.853208，即 85.320796%

结果文件：

- `总体指标/summary.json`
- `总体指标/summary.tsv`

## 结果引用建议

报告中建议按如下口径描述：

```text
目标识别阶段在 226 个 RGB+mask 测试样本上达到 99.56% 的 IoU>=0.50 且类别正确准确率；功能理解阶段在 1000 个点云实例上，以距离容忍 0.020 和校准 aIoU 操作阈值 0.478469785 取得 85.70% 准确率。两阶段串联后的总体功能成功率为 85.32%。本地 Qwen3.5-2B 语义辅助基线在同一批 RGB 图像上的封闭类别语义识别准确率为 91.15%，但不输出 2D mask 或 3D 点级可供性区域，不能替代完整指标。
```
