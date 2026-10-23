from pathlib import Path
import shutil
import tempfile

import torch
from ultralytics import YOLO


# ============================================================
# 项目路径与统一导出参数
# ============================================================

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "models" / "fp16"

IMG_SIZE = 640
BATCH_SIZE = 1
DYNAMIC = False


def remove_existing(path: Path) -> None:
    """删除同名旧产物，避免旧文件被误认为本次导出结果。"""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def locate_artifact(fmt: str, returned_path, work_dir: Path) -> Path:
    """从 Ultralytics 的返回值或默认文件名定位导出产物。"""
    candidates = []

    if returned_path:
        path = Path(str(returned_path))
        if not path.is_absolute():
            path = (work_dir / path).resolve()
        candidates.append(path)

    default_names = {
        "onnx": work_dir / "best_fp16.onnx",
        "torchscript": work_dir / "best_fp16.torchscript",
        "openvino": work_dir / "best_fp16_openvino_model",
    }
    candidates.append(default_names[fmt])

    for path in candidates:
        if not path.exists():
            continue

        # 某些导出流程可能直接返回 OpenVINO 的 XML 文件。
        if fmt == "openvino" and path.is_file() and path.suffix.lower() == ".xml":
            return path.parent

        return path

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"没有找到 {fmt} 导出产物。已检查：{checked}"
    )


def export_one(
    fmt: str,
    temp_weights: Path,
    work_dir: Path,
    destination: Path,
    device,
) -> bool:
    """导出一种 FP16 格式；成功返回 True，失败返回 False。"""
    print(f"\n{'=' * 68}")
    print(f"开始导出 FP16：{fmt}")
    print(f"设备：{device}")

    options = {
        "format": fmt,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "device": device,
        "quantize": 16,
        "dynamic": DYNAMIC,
    }

    # simplify 是 ONNX 导出选项，不传给其他格式。
    if fmt == "onnx":
        options["simplify"] = True

    try:
        model = YOLO(str(temp_weights))
        returned_path = model.export(**options)

        artifact = locate_artifact(fmt, returned_path, work_dir)

        # 只有在新产物成功生成并找到后，才替换目标位置的旧文件。
        if artifact.resolve() != destination.resolve():
            remove_existing(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(artifact), str(destination))

        if not destination.exists():
            raise FileNotFoundError(f"导出目标不存在：{destination}")

        print(f"[成功] {fmt} -> {destination}")
        return True

    except Exception as exc:
        print(f"[失败] {fmt}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    if not WEIGHTS.is_file():
        print(f"[错误] 找不到权重文件：{WEIGHTS}")
        print("请确认原始模型位于 weights/best.pt。")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"源模型：{WEIGHTS}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"输入尺寸：{IMG_SIZE}")
    print(f"Batch：{BATCH_SIZE}")
    print(f"Dynamic：{DYNAMIC}")
    print(f"PyTorch CUDA available：{torch.cuda.is_available()}")

    # 使用临时目录和单独的模型文件名，避免覆盖 FP32 导出产物。
    with tempfile.TemporaryDirectory(prefix="yolo_fp16_export_") as temp:
        work_dir = Path(temp)
        temp_weights = work_dir / "best_fp16.pt"
        shutil.copy2(WEIGHTS, temp_weights)

        jobs = [
            (
                "onnx",
                "cpu",
                OUTPUT_DIR / "best_fp16.onnx",
            ),
            (
                "openvino",
                "cpu",
                OUTPUT_DIR / "best_fp16_openvino_model",
            ),
        ]

        results = {}
        for fmt, device, destination in jobs:
            results[fmt] = export_one(
                fmt=fmt,
                temp_weights=temp_weights,
                work_dir=work_dir,
                destination=destination,
                device=device,
            )

        # Ultralytics 的 TorchScript FP16 导出要求 CUDA/GPU。
        # 当前 CPU 环境下不强行导出，避免生成无法可靠运行的文件。
        if torch.cuda.is_available():
            results["torchscript"] = export_one(
                fmt="torchscript",
                temp_weights=temp_weights,
                work_dir=work_dir,
                destination=OUTPUT_DIR / "best_fp16.torchscript",
                device=0,
            )
        else:
            results["torchscript"] = None
            print(
                "\n[跳过] TorchScript FP16：当前 PyTorch 未检测到 CUDA。"
                "\n官方导出要求 GPU；现有 best.torchscript 可继续作为 FP32 对照组。"
            )

    print(f"\n{'=' * 68}")
    print("FP16 导出结果汇总：")
    for fmt, result in results.items():
        if result is True:
            status = "成功"
        elif result is False:
            status = "失败"
        else:
            status = "跳过（当前环境不支持）"
        print(f"- {fmt}: {status}")

    failed = [fmt for fmt, result in results.items() if result is False]
    if failed:
        print("\n部分格式导出失败。请先处理错误信息，再进行推理和性能测试。")
        return 2

    print("\n已完成当前环境支持的 FP16 导出。")
    print("注意：只有实际生成的 FP16 文件才能纳入 FP16 推理对比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
