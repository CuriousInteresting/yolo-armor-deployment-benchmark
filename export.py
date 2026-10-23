
from pathlib import Path
import shutil

from ultralytics import YOLO


# ============================================================
# 1. 项目路径与导出配置
# ============================================================

ROOT = Path(__file__).resolve().parent

WEIGHTS = ROOT / "weights" / "best.pt"
MODELS = ROOT / "models"

# =========================
# 导出参数配置
# =========================

IMG_SIZE = 640
BATCH_SIZE = 1

DEVICE = "cpu"

HALF = False
DYNAMIC = False

# 每种格式：
# format: Ultralytics 导出格式名称
# source: Ultralytics 通常生成的原始导出位置
# destination: 最终整理的位置
EXPORTS = {
    "onnx": {
        "format": "onnx",
        "source": ROOT / "weights" / "best.onnx",
        "destination": MODELS / "best.onnx",
    },
    "torchscript": {
        "format": "torchscript",
        "source": ROOT / "weights" / "best.torchscript",
        "destination": MODELS / "best.torchscript",
    },
    "openvino": {
        "format": "openvino",
        "source": ROOT / "weights" / "best_openvino_model",
        "destination": MODELS / "best_openvino_model",
    },
}


# ============================================================
# 2. 文件管理工具
# ============================================================

def remove_existing(path: Path) -> None:
    """删除指定的旧导出文件或目录。"""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def locate_artifact(fmt: str, returned_path) -> Path:
    """
    定位本次导出生成的文件或目录。

    优先检查 Ultralytics 返回的路径，
    再检查配置中的预期导出路径。
    """
    candidates = []

    if returned_path:
        returned = Path(str(returned_path))

        if returned.is_absolute():
            candidates.append(returned)
        else:
            # 同时兼容相对于当前工作目录、
            # 相对于项目目录的路径。
            candidates.append((Path.cwd() / returned).resolve())
            candidates.append((ROOT / returned).resolve())

    candidates.append(EXPORTS[fmt]["source"])

    # 去重，保持检查顺序
    unique_candidates = []
    seen = set()

    for path in candidates:
        key = str(path.resolve())

        if key not in seen:
            seen.add(key)
            unique_candidates.append(path)

    for path in unique_candidates:
        if not path.exists():
            continue

        # OpenVINO 返回 XML 文件时，
        # 使用包含 XML 和 BIN 的整个目录。
        if (
            fmt == "openvino"
            and path.is_file()
            and path.suffix.lower() == ".xml"
        ):
            return path.parent

        return path

    checked = "\n".join(str(path) for path in unique_candidates)

    raise FileNotFoundError(
        f"找不到 {fmt} 导出产物。\n已检查路径：\n{checked}"
    )


def validate_artifact(fmt: str, path: Path) -> None:
    """检查导出产物是否具备基本的文件结构。"""

    if not path.exists():
        raise FileNotFoundError(f"导出产物不存在：{path}")

    if fmt in ("onnx", "torchscript"):
        if not path.is_file():
            raise FileNotFoundError(
                f"{fmt} 应当生成模型文件，但得到的不是文件：{path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(f"导出文件为空：{path}")

    elif fmt == "openvino":
        if not path.is_dir():
            raise RuntimeError(
                f"OpenVINO 导出产物应为目录：{path}"
            )

        xml_files = list(path.glob("*.xml"))
        bin_files = list(path.glob("*.bin"))

        if not xml_files or not bin_files:
            raise RuntimeError(
                "OpenVINO 目录缺少 XML 或 BIN 文件："
                f"{path}"
            )


def get_size_mb(path: Path) -> float:
    """计算模型文件或模型目录的大小，单位为 MiB。"""

    if path.is_file():
        total_bytes = path.stat().st_size
    else:
        total_bytes = sum(
            file.stat().st_size
            for file in path.rglob("*")
            if file.is_file()
        )

    return total_bytes / (1024 * 1024)


# ============================================================
# 3. 主导出流程
# ============================================================

def main() -> int:
    if not WEIGHTS.is_file():
        print(f"[错误] 找不到原始模型：{WEIGHTS}")
        print(
            "请将第一次作业的 best.pt "
            "放入本项目的 weights 目录。"
        )
        return 1

    MODELS.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("YOLO 装甲板 Pose 模型格式导出")
    print(f"项目目录：{ROOT}")
    print(f"原始权重：{WEIGHTS}")
    print(f"输入尺寸：{IMG_SIZE}")
    print(f"导出设备：{DEVICE}")
    print("=" * 64)

    errors = []
    successes = []

    for name, config in EXPORTS.items():
        fmt = config["format"]
        source = config["source"]
        destination = config["destination"]

        print(f"\n{'=' * 64}")
        print(f"开始导出：{name}")

        try:
            # 清理当前格式的旧产物，防止误用旧文件。
            remove_existing(source)
            remove_existing(destination)

            # 每种格式均从原始 PT 权重独立导出。
            model = YOLO(str(WEIGHTS))

            returned_path = model.export(
                format=fmt,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                device=DEVICE,
                half=HALF,
                dynamic=DYNAMIC,
            )

            artifact = locate_artifact(fmt, returned_path)

            # 整理至统一的 models 目录。
            if artifact.resolve() != destination.resolve():
                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                remove_existing(destination)

                shutil.move(
                    str(artifact),
                    str(destination),
                )

            # 验证最终目标。
            validate_artifact(fmt, destination)

            size_mb = get_size_mb(destination)

            print(f"[成功] 格式：{name}")
            print(f"最终路径：{destination}")
            print(f"模型大小：{size_mb:.2f} MiB")

            successes.append(name)

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append((name, message))

            print(f"[失败] 格式：{name}")
            print(f"错误信息：{message}")
            print("继续尝试其他格式。")

    # ========================================================
    # 4. 汇总
    # ========================================================

    print(f"\n{'=' * 64}")
    print("模型导出汇总")
    print(f"成功：{len(successes)} / {len(EXPORTS)}")

    for name in successes:
        print(f"  [成功] {name}")

    for name, message in errors:
        print(f"  [失败] {name}: {message}")

    print(f"\n模型目录：{MODELS}")

    if errors:
        print(
            "\n部分格式导出失败。"
            "请检查上方错误信息及对应依赖。"
        )
        return 2

    print("\n三种格式均已生成并通过基本文件检查。")
    print("注意：仍需运行推理测试，才能确认模型可正常使用。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
