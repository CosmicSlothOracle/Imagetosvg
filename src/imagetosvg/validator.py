from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass(slots=True)
class ValidationReport:
    ssim: float
    mse: float


def validate_similarity(original: np.ndarray, svg_path: Path) -> ValidationReport | None:
    try:
        import cairosvg
    except Exception:
        return None

    try:
        png_bytes = cairosvg.svg2png(url=str(svg_path))
    except Exception:
        return None

    raster = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if raster is None:
        return None

    orig_bgr = original[:, :, :3] if original.ndim == 3 and original.shape[2] == 4 else original
    raster_resized = cv2.resize(raster, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_AREA)

    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    rast_gray = cv2.cvtColor(raster_resized, cv2.COLOR_BGR2GRAY)
    score = float(ssim(orig_gray, rast_gray, data_range=255))
    mse_value = float(np.mean((orig_bgr.astype(np.float32) - raster_resized.astype(np.float32)) ** 2))
    return ValidationReport(ssim=score, mse=mse_value)
