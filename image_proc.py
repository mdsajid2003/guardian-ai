"""
image_proc.py — Guardian AI: Image pre-processing for Claude Vision
Uses Pillow only (no OpenCV) for Streamlit Cloud compatibility.
"""

import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
MIN_DIMENSION = 1000   # upscale if longest side is smaller
MAX_DIMENSION = 4000   # cap to avoid token / memory overload
JPEG_QUALITY   = 95


# ── Core helpers ──────────────────────────────────────────────────────────────

def _resize_to_bounds(img: Image.Image) -> Image.Image:
    """Scale image so the longest side stays within [MIN_DIMENSION, MAX_DIMENSION]."""
    w, h = img.size
    longest = max(w, h)

    if longest < MIN_DIMENSION:
        scale = MIN_DIMENSION / longest
    elif longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
    else:
        return img

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def preprocess_image(file_bytes: bytes) -> bytes:
    """
    Enhance an image for financial document OCR / Claude Vision:
      1. Convert to RGB
      2. Resize to safe bounds
      3. Boost contrast  (×1.5)
      4. Boost sharpness (×2.0)
      5. Return JPEG bytes
    Falls back to the original bytes on any error.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))

        # Normalise mode
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        img = _resize_to_bounds(img)

        # Enhance contrast
        img = ImageEnhance.Contrast(img).enhance(1.5)
        # Enhance sharpness
        img = ImageEnhance.Sharpness(img).enhance(2.0)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY)
        return out.getvalue()

    except Exception:
        return file_bytes   # graceful degradation


def image_to_base64(file_bytes: bytes) -> str:
    """Preprocess then return base64-encoded JPEG string."""
    processed = preprocess_image(file_bytes)
    return base64.standard_b64encode(processed).decode("utf-8")


def get_thumbnail(file_bytes: bytes, max_side: int = 200) -> bytes:
    """Return a small JPEG thumbnail for preview widgets."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        if img.mode not in ("RGB",):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=80)
        return out.getvalue()
    except Exception:
        return file_bytes


def detect_document_orientation(file_bytes: bytes) -> int:
    """
    Return estimated rotation needed (0, 90, 180, 270 degrees).
    Simple heuristic: if height > width × 1.4 → likely portrait (0°).
    Returns 0 for most financial docs (no correction needed).
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        # Landscape documents that look portrait → might be rotated 90°
        if w > h * 1.4:
            return 90
        return 0
    except Exception:
        return 0


def auto_deskew(file_bytes: bytes) -> bytes:
    """
    Light deskew using a mild unsharp mask — full deskew needs OpenCV
    which is heavy; this is a Streamlit Cloud-safe approximation.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY)
        return out.getvalue()
    except Exception:
        return file_bytes
