from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .io import write_text
from .preprocess import preprocess_image
from .segmentation import segment_colors
from .svg_builder import build_svg
from .tracing import trace_layers
from .validator import ValidationReport, validate_similarity

logger = logging.getLogger(__name__)


def optimize_and_render(image, source: Path, config: PipelineConfig, output_path: Path) -> tuple[str, ValidationReport | None]:
    candidates = [config]
    if config.auto_iterate and config.validate_similarity:
        candidates.extend(
            [
                replace(config, max_colors_high=max(config.max_colors_high + 8, config.max_colors()), simplification_high=config.simplification_high * 0.8),
                replace(config, canny_low=max(10, config.canny_low - 8), canny_high=min(240, config.canny_high + 20), adaptive_c=max(1, config.adaptive_c - 1)),
            ]
        )

    best_svg = ""
    best_report: ValidationReport | None = None
    best_score = -1.0

    for idx, cand in enumerate(candidates, start=1):
        enhanced, edges, detail_map = preprocess_image(image, cand)
        segmented = segment_colors(enhanced, cand)
        trace = trace_layers(segmented, edges, detail_map, cand)
        svg_text = build_svg(trace, (image.shape[1], image.shape[0]), cand, source=source)
        write_text(output_path, svg_text)

        report = validate_similarity(image, output_path) if cand.validate_similarity else None
        score = report.ssim if report else 0.0

        logger.info("candidate=%s ssim=%s", idx, f"{score:.4f}" if report else "n/a")

        if report is None:
            if best_svg == "":
                best_svg, best_report, best_score = svg_text, None, 0.0
            continue

        if score > best_score:
            best_svg, best_report, best_score = svg_text, report, score

        if score >= cand.target_ssim():
            best_svg, best_report = svg_text, report
            break

    if best_svg:
        write_text(output_path, best_svg)

    return best_svg, best_report
