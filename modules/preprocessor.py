"""
Phase 1 preprocessor: scan, thumbnail, burst grouping, blur detection, face detection.
No file copying, no AI scoring, no selection — pure analysis only.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
SKILL_DIR = Path(__file__).parent.parent
YUNET_MODEL_PATH = SKILL_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class ImageInfo:
    path: Path
    mtime: float
    thumb_path: Optional[Path] = None
    blur_score: float = 0.0
    is_blurry: bool = False
    burst_group: int = -1          # -1 = singleton, >=0 = burst group index
    portrait_class: str = "unknown"  # "portrait", "group", "non-portrait", "unknown"
    face_count: int = 0
    face_boxes: list = field(default_factory=list)  # [{x,y,w,h,confidence}, ...]


def scan_images(input_dir: Path) -> list[ImageInfo]:
    """Scan input_dir for supported image files, sorted by mtime."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")

    images = []
    for p in input_dir.iterdir():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                mtime = p.stat().st_mtime
                images.append(ImageInfo(path=p, mtime=mtime))
            except OSError as e:
                print(f"  [warn] Cannot stat {p.name}: {e}")

    images.sort(key=lambda x: x.mtime)
    print(f"[scan] Found {len(images)} image(s) in {input_dir}")
    return images


def generate_thumbnails(images: list[ImageInfo], thumb_dir: Path, max_size: int = 1024) -> None:
    """Generate thumbnails using macOS sips. Skips existing thumbnails."""
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    total = len(images)
    print(f"[thumb] Generating thumbnails (max {max_size}px) into {thumb_dir} ...")

    for i, img in enumerate(images, 1):
        thumb_path = thumb_dir / img.path.name
        # HEIC → convert to jpg thumb
        if img.path.suffix.lower() == ".heic":
            thumb_path = thumb_dir / (img.path.stem + ".jpg")

        img.thumb_path = thumb_path

        if thumb_path.exists():
            print(f"  [{i}/{total}] skip (exists): {img.path.name}")
            continue

        print(f"  [{i}/{total}] {img.path.name} → {thumb_path.name}")
        try:
            cmd = ["sips", "-Z", str(max_size), str(img.path), "--out", str(thumb_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"    [warn] sips failed: {result.stderr.strip()}")
                img.thumb_path = None
        except subprocess.TimeoutExpired:
            print(f"    [warn] sips timed out for {img.path.name}")
            img.thumb_path = None
        except FileNotFoundError:
            print("    [error] sips not found — macOS only")
            img.thumb_path = None

    print(f"[thumb] Done.")


def group_bursts(images: list[ImageInfo], window_seconds: float = 2.0) -> int:
    """
    Cluster images into burst groups by mtime proximity.
    Images within window_seconds of the previous image belong to the same burst.
    Singletons get burst_group = -1.
    Returns number of burst groups found.
    """
    if not images:
        return 0

    group_idx = 0
    run_start = 0  # index of first image in current run

    for i in range(1, len(images)):
        gap = images[i].mtime - images[i - 1].mtime
        if gap <= window_seconds:
            # extend current run — assign group to the run-start image too
            if images[i - 1].burst_group == -1:
                images[i - 1].burst_group = group_idx
            images[i].burst_group = group_idx
        else:
            # finalize previous run
            if images[i - 1].burst_group != -1:
                group_idx += 1
            run_start = i

    # finalize last run
    if images and images[-1].burst_group != -1:
        group_idx += 1

    burst_count = group_idx
    singletons = sum(1 for img in images if img.burst_group == -1)
    bursts = sum(1 for img in images if img.burst_group >= 0)
    print(f"[burst] {bursts} burst image(s) in {burst_count} group(s), {singletons} singleton(s)")
    return burst_count


def compute_blur_score(image_path: Path) -> float:
    """
    Compute Laplacian variance as blur score.
    Higher = sharper. Returns 0.0 on failure or if cv2 unavailable.
    """
    if not _CV2_AVAILABLE:
        return 0.0
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(variance)
    except Exception as e:
        print(f"    [warn] blur score failed for {image_path.name}: {e}")
        return 0.0


def filter_blurry(images: list[ImageInfo], threshold: float = 80.0) -> None:
    """
    Compute blur scores and mark images below threshold as blurry.
    Prefers thumbnail path for speed; falls back to original.
    """
    if not _CV2_AVAILABLE:
        print("[blur] cv2 not available — skipping blur detection")
        return

    total = len(images)
    print(f"[blur] Computing blur scores (threshold={threshold}) ...")

    blurry_count = 0
    for i, img in enumerate(images, 1):
        source = img.thumb_path if (img.thumb_path and img.thumb_path.exists()) else img.path
        score = compute_blur_score(source)
        img.blur_score = score
        img.is_blurry = score < threshold and score > 0.0
        if img.is_blurry:
            blurry_count += 1
        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] processed")

    print(f"[blur] {blurry_count}/{total} image(s) marked blurry")


def filter_by_quality(images: list[ImageInfo]) -> list[ImageInfo]:
    """Return non-blurry images (for passing to AI evaluation)."""
    result = [img for img in images if not img.is_blurry]
    excluded = len(images) - len(result)
    if excluded:
        print(f"[quality] Excluded {excluded} blurry image(s); {len(result)} sent to AI")
    return result


def detect_faces(images: list[ImageInfo], config: dict = None) -> None:
    """
    Run YuNet face detection on thumbnails (or originals).
    Populates face_count, face_boxes, and portrait_class.
    """
    if not _CV2_AVAILABLE:
        print("[face] cv2 not available — skipping face detection")
        return

    det_cfg = (config or {}).get("detector", {})
    if "model" in det_cfg:
        model_path = Path(det_cfg["model"])
        if not model_path.is_absolute():
            model_path = SKILL_DIR / model_path
    else:
        model_path = YUNET_MODEL_PATH
    score_threshold = det_cfg.get("score_threshold", 0.6)
    nms_threshold = det_cfg.get("nms_threshold", 0.3)
    top_k = det_cfg.get("top_k", 5000)

    if not model_path.exists():
        print(f"[face] YuNet model not found at {model_path} — skipping face detection")
        return

    print(f"[face] Running YuNet face detection ...")

    try:
        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
    except Exception as e:
        print(f"[face] Failed to load YuNet model: {e}")
        return

    total = len(images)
    for i, img in enumerate(images, 1):
        source = img.thumb_path if (img.thumb_path and img.thumb_path.exists()) else img.path
        try:
            frame = cv2.imread(str(source))
            if frame is None:
                img.portrait_class = "unknown"
                continue

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            if faces is None or len(faces) == 0:
                img.face_count = 0
                img.portrait_class = "non-portrait"
            else:
                img.face_count = len(faces)
                img.face_boxes = [
                    {
                        "x": int(f[0]),
                        "y": int(f[1]),
                        "w": int(f[2]),
                        "h": int(f[3]),
                        "confidence": float(f[14]),
                    }
                    for f in faces
                ]
                img.portrait_class = _classify_portrait(faces, w, h)

        except Exception as e:
            print(f"    [warn] face detection failed for {source.name}: {e}")
            img.portrait_class = "unknown"

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] processed")

    portraits = sum(1 for img in images if img.portrait_class == "portrait")
    groups = sum(1 for img in images if img.portrait_class == "group")
    non_portraits = sum(1 for img in images if img.portrait_class == "non-portrait")
    print(f"[face] portrait={portraits}, group={groups}, non-portrait={non_portraits}")


def _classify_portrait(faces, img_w: int, img_h: int) -> str:
    """0 faces → non-portrait, 2+ faces → group, 1 face → portrait."""
    if len(faces) == 0:
        return "non-portrait"
    if len(faces) >= 2:
        return "group"
    return "portrait"
