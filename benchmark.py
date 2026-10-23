
from datetime import datetime
import argparse
import csv
from pathlib import Path

from inference import (
    CONF_THRESHOLD,
    DEVICE,
    IMG_SIZE,
    IOU_THRESHOLD,
    WARMUP_FRAMES,
    run_inference,
)

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "videos" / "original.mp4"
REPORT = ROOT / "reports" / "benchmark_results.csv"

# model_id：唯一标识
# format：传递给 inference.py 的运行时格式
# precision：模型精度标记
# path：模型文件或目录
# output：推理视频的保存路径
MODELS = [
    {
        "model_id": "pt_fp32",
        "format": "pt",
        "precision": "FP32",
        "path": ROOT / "weights" / "best.pt",
        "output": ROOT / "results" / "pt_output.mp4",
    },
    {
        "model_id": "onnx_fp32",
        "format": "onnx",
        "precision": "FP32",
        "path": ROOT / "models" / "best.onnx",
        "output": ROOT / "results" / "onnx_output.mp4",
    },
    {
        "model_id": "onnx_fp16",
        "format": "onnx",
        "precision": "FP16",
        "path": ROOT / "models" / "fp16" / "best_fp16.onnx",
        "output": ROOT / "results" / "onnx_fp16_output.mp4",
    },
    {
        "model_id": "torchscript_fp32",
        "format": "torchscript",
        "precision": "FP32",
        "path": ROOT / "models" / "best.torchscript",
        "output": ROOT / "results" / "torchscript_output.mp4",
    },
    {
        "model_id": "openvino_fp32",
        "format": "openvino",
        "precision": "FP32",
        "path": ROOT / "models" / "best_openvino_model",
        "output": ROOT / "results" / "openvino_output.mp4",
    },
    {
        "model_id": "openvino_fp16",
        "format": "openvino",
        "precision": "FP16",
        "path": (
            ROOT / "models" / "fp16" / "best_fp16_openvino_model"
        ),
        "output": ROOT / "results" / "openvino_fp16_output.mp4",
    },
]

FIELDS = [
    "timestamp",
    "status",
    "model_id",
    "precision",
    "model_format",
    "model_path",
    "model_size_mb",
    "source_video",
    "output_video",
    "confidence_threshold",
    "iou_threshold",
    "imgsz",
    "device",
    "frames",
    "warmup_frames_excluded",
    "measured_frames",
    "total_detections",
    "avg_detections_per_frame",
    "avg_detection_confidence",
    "avg_inference_ms",
    "p50_inference_ms",
    "p95_inference_ms",
    "inference_fps",
    "wall_time_seconds",
    "error",
]


def relative_path(path: Path) -> str:
    """将项目内路径转换为便于记录的相对路径。"""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def get_model_size_mb(path: Path) -> float:
    """计算单文件模型或模型目录的总大小。"""
    if path.is_file():
        total_bytes = path.stat().st_size
    elif path.is_dir():
        total_bytes = sum(
            file.stat().st_size
            for file in path.rglob("*")
            if file.is_file()
        )
    else:
        return 0.0

    return round(total_bytes / (1024 ** 2), 3)


def save_csv(rows: list) -> None:
    """保存当前所有模型的测试记录。"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="比较 YOLO 模型不同导出格式及 FP16/FP32 的 CPU 推理性能。"
    )
    parser.parse_args()

    if not VIDEO.is_file():
        print(f"[错误] 找不到测试视频：{VIDEO}")
        return 2

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    rows = []
    success_count = 0
    failed_count = 0
    skipped_count = 0

    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("YOLO 模型格式与精度对比")
    print(f"测试视频：{relative_path(VIDEO)}")
    print(f"置信度阈值：{CONF_THRESHOLD}")
    print(f"IoU 阈值：{IOU_THRESHOLD}")
    print(f"输入尺寸：{IMG_SIZE}")
    print(f"设备：{DEVICE}")
    print(f"预热帧数：{WARMUP_FRAMES}")
    print("=" * 68)

    for item in MODELS:
        model_id = item["model_id"]
        fmt = item["format"]
        precision = item["precision"]
        model_path = item["path"]
        output_path = item["output"]

        print(
            f"\n{'=' * 68}\n"
            f"模型：{model_id}\n"
            f"格式：{fmt} | 精度：{precision}\n"
            f"路径：{relative_path(model_path)}"
        )

        if not model_path.exists():
            print(f"[跳过] 模型不存在：{model_path}")
            skipped_count += 1
            continue

        try:
            row = run_inference(
                model_path,
                fmt,
                VIDEO,
                output_path,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                iou=IOU_THRESHOLD,
                warmup_frames=WARMUP_FRAMES,
                device=DEVICE,
            )

            # 补充实验标识，便于区分相同运行时格式的不同精度。
            row["timestamp"] = timestamp
            row["model_id"] = model_id
            row["precision"] = precision
            row["model_format"] = fmt
            row["model_path"] = relative_path(model_path)
            row["source_video"] = relative_path(VIDEO)
            row["output_video"] = relative_path(output_path)

            if row.get("model_size_mb") is None:
                row["model_size_mb"] = get_model_size_mb(model_path)

            # 兼容缺少平均耗时字段的旧统计记录。
            if (
                row.get("avg_inference_ms") in (None, "")
                and row.get("inference_fps")
            ):
                row["avg_inference_ms"] = round(
                    1000.0 / float(row["inference_fps"]), 4
                )

            row.setdefault("error", "")
            row.setdefault("status", "OK")
            rows.append(row)

            if row["status"] == "OK":
                success_count += 1
                print(f"[成功] {model_id}")
            else:
                failed_count += 1
                print(f"[失败] {model_id}: {row.get('error', '')}")

        except Exception as exc:
            print(f"[异常] {model_id}: {type(exc).__name__}: {exc}")

            rows.append({
                "timestamp": timestamp,
                "status": "FAILED",
                "model_id": model_id,
                "precision": precision,
                "model_format": fmt,
                "model_path": relative_path(model_path),
                "model_size_mb": get_model_size_mb(model_path),
                "source_video": relative_path(VIDEO),
                "output_video": relative_path(output_path),
                "confidence_threshold": CONF_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
                "imgsz": IMG_SIZE,
                "device": DEVICE,
                "error": f"{type(exc).__name__}: {exc}",
            })
            failed_count += 1

        # 每完成一个模型就保存，避免中途出错后丢失已完成记录。
        save_csv(rows)

    print("\n" + "=" * 68)
    print("基准测试结束")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"跳过：{skipped_count}")
    print(f"CSV 报告：{relative_path(REPORT)}")
    print("=" * 68)

    if failed_count > 0 or skipped_count > 0:
        return 1

    if success_count != len(MODELS):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
