
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "images"

# OpenCV 的帧索引从 0 开始。
# FRAME_INDEX = 199 表示视频中的第 200 帧。
FRAME_INDEX = 199
FRAME_NUMBER = FRAME_INDEX + 1

# 六种模型使用相同的帧索引进行比较。
VIDEOS = {
    "pt": "pt_output.mp4",
    "onnx": "onnx_output.mp4",
    "onnx_fp16": "onnx_fp16_output.mp4",
    "torchscript": "torchscript_output.mp4",
    "openvino": "openvino_output.mp4",
    "openvino_fp16": "openvino_fp16_output.mp4",
}


def extract_frame(model_id: str, filename: str) -> bool:
    video_path = RESULTS_DIR / filename

    if not video_path.is_file():
        print(f"[FAIL] Video does not exist: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[FAIL] Cannot open video: {video_path}")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    success, frame = cap.read()
    cap.release()

    if not success:
        print(
            f"[FAIL] Cannot read frame {FRAME_NUMBER} "
            f"from {filename}"
        )
        return False

    # 文件名采用人类习惯的帧编号：第 200 帧写为 frame_200。
    image_path = (
        IMAGES_DIR / f"{model_id}_frame_{FRAME_NUMBER}.png"
    )

    if not cv2.imwrite(str(image_path), frame):
        print(f"[FAIL] Cannot write image: {image_path}")
        return False

    print(f"[OK] {image_path}")
    return True


def main() -> int:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    print(f"Extracting frame {FRAME_NUMBER} from each video.")
    print(f"Output directory: {IMAGES_DIR}")

    for model_id, filename in VIDEOS.items():
        if extract_frame(model_id, filename):
            success_count += 1
        else:
            failure_count += 1

    print("\nFrame extraction summary")
    print(f"Success: {success_count}/{len(VIDEOS)}")
    print(f"Failed: {failure_count}/{len(VIDEOS)}")

    if failure_count:
        print(
            "Check that all six result videos exist "
            "and can be opened."
        )
        return 1

    print("Frame extraction completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())