from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


def preprocess_image(image: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return enhanced image, edge map, and adaptive-threshold detail map."""
    bgr, alpha = _split_alpha(image)

    denoised = cv2.bilateralFilter(
        bgr,
        d=9,
        sigmaColor=float(config.denoise_sigma),
        sigmaSpace=float(config.denoise_sigma),
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_norm, a, b]), cv2.COLOR_LAB2BGR)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=config.canny_low, threshold2=config.canny_high)
    if config.edge_dilate_iterations > 0:
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=config.edge_dilate_iterations)

    detail_map = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    if alpha is not None and config.preserve_alpha:
        sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2BGRA)
        sharpened[:, :, 3] = alpha

    return sharpened, edges, detail_map


def _split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, None
