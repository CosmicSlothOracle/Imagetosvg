from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetailPreset, PipelineConfig
from .segmentation import SegmentationResult


@dataclass(slots=True)
class PathLayer:
    name: str
    paths: list[str]
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None
    opacity: float = 1.0


@dataclass(slots=True)
class TraceResult:
    layers: list[PathLayer]


def trace_layers(
    segmented: SegmentationResult,
    edges: np.ndarray,
    detail_map: np.ndarray,
    config: PipelineConfig,
) -> TraceResult:
    layers: list[PathLayer] = []

    label_ids, counts = np.unique(segmented.labels, return_counts=True)
    color_items = sorted(zip(label_ids.tolist(), counts.tolist()), key=lambda t: t[1], reverse=True)

    for label_id, pixel_count in color_items:
        if pixel_count < config.min_region_area:
            continue
        mask = np.where(segmented.labels == label_id, 255, 0).astype(np.uint8)
        color = segmented.palette[label_id]
        paths = _contours_to_svg_paths(mask, config.simplification_ratio(), min_area=config.min_region_area)
        if not paths:
            continue
        layers.append(
            PathLayer(
                name=f"color_{int(label_id):03d}",
                paths=paths,
                fill=f"rgb({int(color[2])},{int(color[1])},{int(color[0])})",
            )
        )

    edge_paths = _contours_to_svg_paths(edges, config.simplification_ratio() * 0.6, min_area=10)
    if edge_paths:
        layers.append(
            PathLayer(
                name="edge_layer",
                paths=edge_paths,
                fill="none",
                stroke="rgb(24,24,24)",
                stroke_width=0.35 if config.detail != DetailPreset.LOW else 0.5,
                opacity=0.45,
            )
        )

    detail_paths = _contours_to_svg_paths(detail_map, config.simplification_ratio() * 0.45, min_area=6)
    if detail_paths:
        layers.append(
            PathLayer(
                name="detail_layer",
                paths=detail_paths,
                fill="none",
                stroke="rgb(12,12,12)",
                stroke_width=0.22 if config.detail == DetailPreset.ULTRA else 0.28,
                opacity=0.25,
            )
        )

    return TraceResult(layers=layers)


def _contours_to_svg_paths(mask: np.ndarray, simplification: float, min_area: int) -> list[str]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    paths: list[str] = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = max(0.1, simplification * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        d = _path_from_contour(approx)

        child_idx = hierarchy[idx][2]
        while child_idx != -1:
            child = contours[child_idx]
            child_area = cv2.contourArea(child)
            if child_area >= min_area:
                child_eps = max(0.1, simplification * cv2.arcLength(child, True))
                child_approx = cv2.approxPolyDP(child, child_eps, True)
                if len(child_approx) >= 3:
                    d += " " + _path_from_contour(child_approx)
            child_idx = hierarchy[child_idx][0]

        paths.append(d)
    return paths


def _path_from_contour(contour: np.ndarray) -> str:
    points = contour[:, 0, :]
    commands = [f"M {points[0][0]} {points[0][1]}"]
    commands.extend(f"L {pt[0]} {pt[1]}" for pt in points[1:])
    commands.append("Z")
    return " ".join(commands)
