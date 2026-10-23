# YOLO 装甲目标模型导出、推理与性能基准测试

本项目提供一套可复现的流程，用于将 YOLO26 Pose 模型导出为多种部署格式，在相同视频输入上执行推理，并比较不同运行时的性能表现。

## 项目简介

本项目包含以下功能：

- 将训练好的 Ultralytics YOLO Pose 权重导出为 **ONNX**、**TorchScript** 和 **OpenVINO** 格式。
- 使用原始 PyTorch 权重及导出后的模型进行视频推理。
- 在本地保存带检测标注的输出视频，并收集推理统计数据。
- 比较模型文件大小、推理延迟、吞吐率和检测数量。
- 从各格式的推理结果中提取相同帧，便于进行可视化检查。

本项目使用自定义的 YOLO26 Pose 模型，包含六个类别：`B1`、`B2`、`B3`、`R1`、`R2` 和 `R3`。

## 支持的模型格式

| 格式 | 模型产物 | 推理运行时 |
|---|---|---|
| PyTorch | `weights/best.pt` | Ultralytics / PyTorch |
| ONNX | `models/best.onnx` | ONNX Runtime |
| TorchScript | `models/best.torchscript` | PyTorch |
| OpenVINO | `models/best_openvino_model/` | OpenVINO Runtime |

当前基准测试不包含 TensorRT。

## 项目目录

```text
yolo-armor-deployment-benchmark/
├── README.md
├── .gitignore
├── requirements.txt
├── export.py
├── inference.py
├── benchmark.py
├── extract_frames.py
├── weights/
│   └── best.pt
├── models/
│   ├── best.onnx
│   ├── best.torchscript
│   └── best_openvino_model/
├── videos/
│   └── original.mp4          # 本地输入视频，不由 Git 跟踪
├── results/
│   ├── pt_output.mp4         # 本地生成的视频，不由 Git 跟踪
│   ├── onnx_output.mp4
│   ├── torchscript_output.mp4
│   └── openvino_output.mp4
├── images/
│   ├── pt_frame_199.png
│   ├── onnx_frame_199.png
│   ├── torchscript_frame_199.png
│   └── openvino_frame_199.png
└── reports/
    ├── benchmark_results.csv
    └── comparison.md
```

输入视频和推理生成的视频会被排除在版本控制之外。运行推理前，请将有权使用的测试视频放在 `videos/original.mp4`。只有在拥有相应再分发权限时，才可以将模型权重和导出模型文件提交到公开仓库。

## 测试环境

以下版本信息来自本项目的一次基准测试运行记录：

| 组件 | 版本或设置 |
|---|---|
| 操作系统 | Windows |
| Python | 3.10.22 |
| PyTorch | 2.14.1+cpu |
| Ultralytics | 8.4.177 |
| OpenCV | 5.0.0 |
| NumPy | 2.2.6 |
| ONNX | 1.23.2 |
| ONNX Runtime | 1.23.2 |
| onnxslim | 0.1.98 |
| OpenVINO | 2026.4.1 |
| 推理设备 | CPU |
| 测试环境是否可用 CUDA | 否 |

以上信息仅描述本次测试所使用的环境，不代表所有平台和软件版本组合都能保证兼容。

## 安装依赖

建议使用 Python 3.10，并创建或激活独立的虚拟环境。例如，使用 Conda：

```powershell
conda create -n yolo-export python=3.10 -y
conda activate yolo-export
python -m pip install -r requirements.txt
```

如果需要根据目标 CPU/GPU 环境选择特定的 PyTorch 版本，请先按照 PyTorch 官方安装说明安装合适的构建版本，再安装项目的其余依赖。

检查必要文件是否存在：

```powershell
Test-Path .\weights\best.pt
Test-Path .\videos\original.mp4
```

两条命令都应返回 `True`。

## 推理与导出参数

所有格式的基准测试使用相同的推理设置：

| 参数 | 数值 | 说明 |
|---|---:|---|
| 置信度阈值 | `0.31` | 保留预测结果所使用的最低置信度 |
| IoU 阈值 | `0.70` | 非极大值抑制（NMS）的重叠阈值 |
| 输入图像尺寸 | `640` | 推理与导出使用的图像尺寸 |
| 导出批次大小 | `1` | 导出时的 batch 大小 |
| 设备 | `cpu` | 本次记录的基准测试设备 |
| 预热帧数 | `10` | 计算推理延迟统计时排除的初始帧数 |

这些参数在项目脚本中配置。复现实验或比较模型格式时，应保持各格式的设置一致。

## 导出模型

在项目根目录运行：

```powershell
python .\export.py
```

导出脚本使用 Ultralytics 的导出接口，并依次尝试导出三种格式。预期生成以下模型产物：

- `models/best.onnx`
- `models/best.torchscript`
- `models/best_openvino_model/`

导出配置为：输入尺寸 `640`、批次大小 `1`、使用 CPU 导出、启用全精度（`half=False`），并使用固定输入尺寸（`dynamic=False`）。

模型成功导出并不代表它一定能正常推理。导出完成后，应分别执行对应格式的推理命令，验证模型能够实际加载并运行。

## 执行推理

请在项目根目录中分别执行以下命令。每条命令都会读取 `videos/original.mp4`，并在 `results/` 目录中保存带标注的输出视频。

### PyTorch 原始权重

```powershell
python .\inference.py --model weights/best.pt --format pt --output results/pt_output.mp4
```

### ONNX

```powershell
python .\inference.py --model models/best.onnx --format onnx --output results/onnx_output.mp4
```

### TorchScript

```powershell
python .\inference.py --model models/best.torchscript --format torchscript --output results/torchscript_output.mp4
```

### OpenVINO

```powershell
python .\inference.py --model models/best_openvino_model --format openvino --output results/openvino_output.mp4
```

推理脚本会输出统计信息，并保存可视化推理结果。视频标注包括类别、置信度，以及模型支持时的姿态/关键点标注。

## 运行完整基准测试

如需按顺序测试四种格式并重新生成 CSV 报告，请运行：

```powershell
python .\benchmark.py
```

该命令会重新运行四种格式的推理，并生成：

```text
reports/benchmark_results.csv
```

运行过程中也会重新生成输出视频。由于输出视频已在 `.gitignore` 中排除，因此这些视频只保存在本地，不会被 Git 提交。如需查看已有的性能数据，可直接打开 `reports/benchmark_results.csv`，无需重新运行推理。

## 性能测试结果

下表来自测试 CPU 环境中的一次完整基准测试。系统负载、进程状态和软件版本都可能影响测试结果；在做部署决策前，建议在目标机器上重新测试。

| 指标 | PyTorch（`.pt`） | ONNX | TorchScript | OpenVINO |
|---|---:|---:|---:|---:|
| 模型大小（MB） | 5.619 | 9.724 | 9.946 | 9.878 |
| 平均推理延迟（ms/帧） | 110.9508 | 146.8839 | 169.6756 | 102.1914 |
| P50 延迟（ms/帧） | 102.7358 | 137.1991 | 159.2164 | 42.1727 |
| P95 延迟（ms/帧） | 154.9637 | 207.0827 | 224.8806 | 368.6794 |
| 吞吐率（FPS，根据平均延迟计算） | 9.0130 | 6.8081 | 5.8936 | 9.7856 |
| 处理帧中的检测总数 | 747 | 769 | 769 | 771 |
| 输入视频帧数 | 483 | 483 | 483 | 483 |
| 运行设备 | CPU | CPU | CPU | CPU |

详细分析见 [`reports/comparison.md`](reports/comparison.md)，原始性能数据见 [`reports/benchmark_results.csv`](reports/benchmark_results.csv)。

### 结果分析

- **模型大小：** 本次测试中，`.pt` 权重文件最小。导出为其他格式不一定会缩小模型体积。
- **平均延迟：** 在本次 CPU 测试配置下，OpenVINO 的平均推理延迟最低。TorchScript 和 ONNX 的平均延迟均高于 PyTorch 基线。
- **延迟波动：** OpenVINO 的 P95 延迟明显高于 P50 延迟。因此，较低的中位延迟并不意味着每一帧都能快速处理。实时应用还需要关注尾部延迟。
- **检测数量：** 不同格式返回的检测总数略有差异。检测数量更多并不自动代表检测准确率更高。
- **结果范围：** 本报告记录的是脚本测得的模型推理延迟，不应将其直接视为完整应用流程能够保证的端到端帧率，也不能替代在带标注验证集上进行的精度评估。

## 推理结果可视化对比

辅助脚本会从四种格式生成的输出视频中提取帧索引 `199`。由于帧索引从 `0` 开始，这对应第 200 帧。

```powershell
python .\extract_frames.py
```

脚本会将图片写入 `images/`：

- `pt_frame_199.png`
- `onnx_frame_199.png`
- `torchscript_frame_199.png`
- `openvino_frame_199.png`

如果希望这些图片显示在公开的 GitHub 仓库页面上，需要将 PNG 截图提交到 Git。MP4 输出视频仍会被排除在 Git 之外。

### PyTorch

![PyTorch 推理结果](images/pt_frame_199.png)

### ONNX

![ONNX 推理结果](images/onnx_frame_199.png)

### TorchScript

![TorchScript 推理结果](images/torchscript_frame_199.png)

### OpenVINO

![OpenVINO 推理结果](images/openvino_frame_199.png)

如果某张图片尚未生成，请先确保四种推理输出视频均已生成，再运行截图提取命令。

## 常见问题与排查

### 依赖安装失败

软件包索引或 TLS/网络问题可能导致安装中断。请检查网络连接和软件包索引配置后重试。不要通过关闭 TLS 证书验证来绕过此问题。

### 导出模型的检测数量不同

数值精度、算子实现、运行时行为或后处理差异，都可能造成预测结果略有不同。可以对相同帧逐一比较边界框、类别 ID、置信度和关键点。如有带标注的验证集，应进一步比较 Precision、Recall 和 mAP 等指标。

### 无法使用 GPU

本次测试环境使用的是 CPU 版 PyTorch，且 CUDA 不可用。因此，本报告中的性能数据均为 CPU 测试结果，不能代表 NVIDIA GPU 上的推理性能。

## 结论

本项目将模型导出为 ONNX、TorchScript 和 OpenVINO 格式，并在相同视频输入上完成了各格式的推理验证。在记录的 CPU 基准测试中，OpenVINO 的平均推理延迟最低、根据平均延迟计算的吞吐率最高；原始 `.pt` 权重文件的体积最小。不同格式的检测数量和延迟分布存在小幅差异。

上述结果仅适用于本次测试的模型权重、软件版本、输入视频和 CPU 环境。选择实际部署方案时，还应考虑目标硬件、运行时兼容性、尾部延迟，以及基于带标注验证集计算的准确率指标。

## 数据与模型许可

本仓库不包含输入视频。公开发布模型权重、导出模型文件或推理截图前，请确认你拥有再分发相关模型及源数据的权限。请勿上传受限制的数据或包含私人信息的文件。