from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CONF_THRESHOLD = 0.31
IOU_THRESHOLD = 0.70
IMG_SIZE = 640
WARMUP_FRAMES = 10
DEVICE = "cpu"


def resolve_path(value) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def size_mb(path: Path) -> float:
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    else:
        size = 0
    return round(size / (1024 * 1024), 3)


def run_inference(
    model_path, model_format, source_path, output_path,
    conf=CONF_THRESHOLD, imgsz=IMG_SIZE, iou=IOU_THRESHOLD,
    warmup_frames=WARMUP_FRAMES, device=DEVICE,
) -> dict:
    model_path = resolve_path(model_path)
    source_path = resolve_path(source_path)
    output_path = resolve_path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"视频不存在：{source_path}")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf 和 iou 必须在 0 到 1 之间。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{model_format}] 加载模型：{model_path}")
    model = YOLO(str(model_path), task="pose")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("无法读取视频分辨率。")

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    frames = 0
    detections = 0
    confidence_sum = 0.0
    confidence_count = 0
    latencies_ms = []
    wall_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            results = model.predict(
                source=frame, imgsz=imgsz, conf=conf, iou=iou,
                device=device, verbose=False, save=False,
            )
            latency = (time.perf_counter() - start) * 1000
            result = results[0]

            if result.boxes is not None:
                n = len(result.boxes)
                detections += n
                if n:
                    scores = result.boxes.conf.detach().cpu().numpy()
                    confidence_sum += float(scores.sum())
                    confidence_count += int(scores.size)

            if frames >= warmup_frames:
                latencies_ms.append(latency)

            writer.write(result.plot())
            frames += 1
            if frames % 100 == 0:
                print(f"已处理 {frames} 帧")

    finally:
        cap.release()
        writer.release()

    wall_seconds = time.perf_counter() - wall_start
    if frames == 0:
        raise RuntimeError("视频没有可读取的帧。")
    if not latencies_ms:
        raise RuntimeError(
            f"视频共 {frames} 帧，少于预热帧数 {warmup_frames}；"
            "请使用 --warmup-frames 0 或调小预热帧数。"
        )

    avg_ms = statistics.mean(latencies_ms)
    metrics = {
        "status": "OK",
        "model_format": model_format,
        "model_path": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "model_size_mb": size_mb(model_path),
        "source_video": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
        "output_video": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "imgsz": imgsz,
        "device": device,
        "frames": frames,
        "warmup_frames_excluded": warmup_frames,
        "measured_frames": len(latencies_ms),
        "total_detections": detections,
        "avg_detections_per_frame": round(detections / frames, 4),
        "avg_detection_confidence": round(confidence_sum / confidence_count, 4) if confidence_count else None,
        "avg_inference_ms": round(avg_ms, 4),
        "p50_inference_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "p95_inference_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "inference_fps": round(1000 / avg_ms, 4) if avg_ms > 0 else 0.0,
        "wall_time_seconds": round(wall_seconds, 3),
        "error": "",
    }
    print("\n推理统计：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Pose 模型视频推理与计时")
    parser.add_argument("--model", default="weights/best.pt", help="模型文件或 OpenVINO 模型目录")
    parser.add_argument("--format", default="pt", choices=["pt", "onnx", "torchscript", "openvino"])
    parser.add_argument("--source", default="videos/original.mp4", help="输入视频")
    parser.add_argument("--output", default="results/pt_output.mp4", help="输出视频")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--warmup-frames", type=int, default=WARMUP_FRAMES)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model,
        model_format=args.format,
        source_path=args.source,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
        warmup_frames=args.warmup_frames,
        device=args.device,
    )
