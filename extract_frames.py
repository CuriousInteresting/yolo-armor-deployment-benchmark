from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "images"
FRAME_INDEX = 199  # Zero-based index: the 200th frame.

VIDEOS = {
    "pt": "pt_output.mp4",
    "onnx": "onnx_output.mp4",
    "torchscript": "torchscript_output.mp4",
    "openvino": "openvino_output.mp4",
}


def main() -> int:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for model_format, filename in VIDEOS.items():
        video_path = RESULTS_DIR / filename
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[FAIL] Cannot open video: {video_path}")
            failures += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
        success, frame = cap.read()
        cap.release()

        if not success:
            print(f"[FAIL] Cannot read frame {FRAME_INDEX} from {filename}")
            failures += 1
            continue

        image_path = IMAGES_DIR / f"{model_format}_frame_{FRAME_INDEX}.png"
        if not cv2.imwrite(str(image_path), frame):
            print(f"[FAIL] Cannot write image: {image_path}")
            failures += 1
            continue

        print(f"[OK] {image_path}")

    if failures:
        print(f"Finished with {failures} failure(s). Confirm that all result videos exist.")
        return 1

    print("Frame extraction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
