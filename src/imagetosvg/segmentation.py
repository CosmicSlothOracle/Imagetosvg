from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass(slots=True)
class SegmentationResult:
    quantized: np.ndarray
    palette: np.ndarray
    labels: np.ndarray


def segment_colors(image: np.ndarray, config: PipelineConfig) -> SegmentationResult:
    bgr = image[:, :, :3] if image.shape[2] == 4 else image
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    if config.use_slic:
        labels_slic = _slic_labels(lab, config.slic_segments(), config.slic_compactness)
        merged = _merge_small_superpixels(labels_slic, bgr, config.min_region_area)
        reduced = _quantize_superpixels(merged, bgr, config.max_colors())
    else:
        reduced = _kmeans_labels(lab, config.max_colors())

    reduced = _merge_tiny_label_regions(reduced, bgr, config.min_region_area)
    palette, quantized = _labels_to_palette_image(reduced, bgr)
    return SegmentationResult(quantized=quantized, palette=palette, labels=reduced)


def _slic_labels(lab_img: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    try:
        from skimage.segmentation import slic
    except Exception:
        return _kmeans_labels(lab_img, max(8, n_segments // 8))

    labels = slic(
        lab_img,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        convert2lab=False,
        channel_axis=-1,
    )
    return labels.astype(np.int32)


def _kmeans_labels(lab_img: np.ndarray, num_colors: int) -> np.ndarray:
    pixels = lab_img.reshape((-1, 3)).astype(np.float32)
    k = max(1, min(num_colors, len(pixels), int(np.unique(pixels, axis=0).shape[0])))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, attempts=4, flags=cv2.KMEANS_PP_CENTERS)
    return labels.flatten().reshape(lab_img.shape[:2]).astype(np.int32)


def _merge_small_superpixels(labels: np.ndarray, bgr: np.ndarray, min_area: int) -> np.ndarray:
    merged = labels.copy()
    ids, counts = np.unique(merged, return_counts=True)
    for sid, cnt in zip(ids.tolist(), counts.tolist()):
        if cnt >= min_area:
            continue
        mask = (merged == sid).astype(np.uint8)
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        neighbor_ids = merged[dilated.astype(bool)]
        neighbor_ids = neighbor_ids[neighbor_ids != sid]
        if neighbor_ids.size == 0:
            continue
        target = int(np.bincount(neighbor_ids).argmax())
        merged[merged == sid] = target
    return merged


def _quantize_superpixels(labels: np.ndarray, bgr: np.ndarray, max_colors: int) -> np.ndarray:
    h, w = labels.shape
    unique_ids = np.unique(labels)
    means = []
    for sid in unique_ids:
        px = bgr[labels == sid]
        means.append(px.mean(axis=0))
    means_arr = np.array(means, dtype=np.float32)

    k = max(1, min(max_colors, means_arr.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.15)
    _, lbl, centers = cv2.kmeans(means_arr, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)

    label_map = {int(sid): int(cluster) for sid, cluster in zip(unique_ids.tolist(), lbl.flatten().tolist())}
    reduced = np.zeros((h, w), dtype=np.int32)
    for sid in unique_ids:
        reduced[labels == sid] = label_map[int(sid)]
    return reduced


def _merge_tiny_label_regions(labels: np.ndarray, bgr: np.ndarray, min_area: int) -> np.ndarray:
    merged = labels.copy()
    ids, counts = np.unique(merged, return_counts=True)
    for lid, cnt in zip(ids.tolist(), counts.tolist()):
        if cnt >= min_area:
            continue
        mask = (merged == lid).astype(np.uint8)
        border = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) - mask
        neighbors = merged[border.astype(bool)]
        neighbors = neighbors[neighbors != lid]
        if neighbors.size == 0:
            continue
        merged[merged == lid] = int(np.bincount(neighbors).argmax())
    return merged


def _labels_to_palette_image(labels: np.ndarray, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(labels)
    palette = np.zeros((int(ids.max()) + 1, 3), dtype=np.uint8)
    quantized = np.zeros_like(bgr)
    for lid in ids:
        pixels = bgr[labels == lid]
        color = np.clip(np.median(pixels, axis=0), 0, 255).astype(np.uint8)
        palette[int(lid)] = color
        quantized[labels == lid] = color
    return palette, quantized
