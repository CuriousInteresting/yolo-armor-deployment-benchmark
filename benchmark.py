from datetime import datetime
import csv
from pathlib import Path

from inference import (
    CONF_THRESHOLD, DEVICE, IMG_SIZE, IOU_THRESHOLD, WARMUP_FRAMES,
    run_inference,
)

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "videos" / "original.mp4"
REPORT = ROOT / "reports" / "benchmark_results.csv"
MODELS = [
    ("pt", ROOT / "weights" / "best.pt"),
    ("onnx", ROOT / "models" / "best.onnx"),
    ("torchscript", ROOT / "models" / "best.torchscript"),
    ("openvino", ROOT / "models" / "best_openvino_model"),
]
FIELDS = [
    "timestamp", "status", "model_format", "model_path", "model_size_mb",
    "source_video", "output_video", "confidence_threshold", "iou_threshold",
    "imgsz", "device", "frames", "warmup_frames_excluded", "measured_frames",
    "total_detections", "avg_detections_per_frame", "avg_detection_confidence",
    "avg_inference_ms", "p50_inference_ms", "p95_inference_ms",
    "inference_fps", "wall_time_seconds", "error",
]


def save_csv(rows):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not VIDEO.is_file():
        print(f"[错误] 找不到测试视频：{VIDEO}")
        return 1

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    success = 0
    print(
        f"固定参数：conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}, "
        f"imgsz={IMG_SIZE}, device={DEVICE}, warmup={WARMUP_FRAMES}"
    )

    for fmt, model_path in MODELS:
        if not model_path.exists():
            print(f"[跳过] {fmt}：没有找到 {model_path}")
            continue
        output = ROOT / "results" / f"{fmt}_output.mp4"
        try:
            row = run_inference(
                model_path, fmt, VIDEO, output,
                conf=CONF_THRESHOLD, imgsz=IMG_SIZE, iou=IOU_THRESHOLD,
                warmup_frames=WARMUP_FRAMES, device=DEVICE,
            )
            row["timestamp"] = timestamp
            rows.append(row)
            success += 1
        except Exception as exc:
            print(f"[失败] {fmt}: {type(exc).__name__}: {exc}")
            rows.append({
                "timestamp": timestamp,
                "status": "FAILED",
                "model_format": fmt,
                "model_path": str(model_path),
                "source_video": str(VIDEO),
                "output_video": str(output),
                "confidence_threshold": CONF_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
                "imgsz": IMG_SIZE,
                "device": DEVICE,
                "error": f"{type(exc).__name__}: {exc}",
            })
        save_csv(rows)

    if not rows:
        print("没有可测试的模型。先运行 python export.py。")
        return 2
    print(f"\n成功完成 {success}/{len(rows)} 个模型。")
    print(f"对比表已保存：{REPORT}")
    return 0 if success == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
