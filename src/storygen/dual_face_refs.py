from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DualFaceRefs:
    ref_a_path: str | None = None
    ref_b_path: str | None = None
    group_style_reference_path: str | None = None
    metadata: dict[str, Any] | None = None


def _try_load_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _face_detector():
    cv2 = _try_load_cv2()
    if cv2 is None:
        return None, None
    cascade = cv2.CascadeClassifier(getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return cv2, None
    return cv2, cascade


def _crop_with_margin(image, box, margin: float = 0.25):
    x, y, w, h = box
    ih, iw = image.shape[:2]
    dx = int(w * margin)
    dy = int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(iw, x + w + dx)
    y1 = min(ih, y + h + dy)
    return image[y0:y1, x0:x1]


def _fallback_upper_half_crop(image, which: int):
    ih, iw = image.shape[:2]
    half_w = iw // 2
    x0 = 0 if which == 0 else half_w
    x1 = half_w if which == 0 else iw
    return image[: max(1, int(ih * 0.65)), x0:x1]


def split_dual_face_refs(anchor_image_path: str, output_dir: str) -> DualFaceRefs:
    cv2, cascade = _face_detector()
    if cv2 is None:
        return DualFaceRefs(metadata={"reason": "opencv_unavailable"})

    image = cv2.imread(anchor_image_path)
    if image is None:
        return DualFaceRefs(metadata={"reason": "image_unreadable"})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style_reference_path = out_dir / "group_style_reference.png"
    cv2.imwrite(str(style_reference_path), image)

    faces = []
    if cascade is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        faces = sorted(detected, key=lambda box: (box[0], box[1], -(box[2] * box[3])))[:2]

    crops = []
    for idx in range(2):
        if idx < len(faces):
            crop = _crop_with_margin(image, faces[idx])
        else:
            crop = _fallback_upper_half_crop(image, idx)
        ref_path = out_dir / f"ref_{'A' if idx == 0 else 'B'}.png"
        cv2.imwrite(str(ref_path), crop)
        crops.append(str(ref_path))

    return DualFaceRefs(
        ref_a_path=crops[0],
        ref_b_path=crops[1],
        group_style_reference_path=str(style_reference_path),
        metadata={"face_count": len(faces), "anchor_image_path": anchor_image_path},
    )
