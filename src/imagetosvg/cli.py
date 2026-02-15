from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DetailPreset, PipelineConfig
from .pipeline import VectorizationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-fidelity raster to layered SVG vectorizer")
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for SVG files")
    parser.add_argument("--detail", choices=[d.value for d in DetailPreset], default=DetailPreset.HIGH.value)
    parser.add_argument("--no-validate", action="store_true", help="Disable similarity validation")
    parser.add_argument("--no-auto-iterate", action="store_true", help="Disable parameter auto-iteration")
    parser.add_argument("--min-region-area", type=int, default=20)
    parser.add_argument("--max-colors", type=int, help="Override selected detail preset color count")
    parser.add_argument("--disable-slic", action="store_true", help="Use KMeans-only segmentation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = PipelineConfig(
        detail=DetailPreset(args.detail),
        output_dir=args.output_dir,
        validate_similarity=not args.no_validate,
        auto_iterate=not args.no_auto_iterate,
        min_region_area=args.min_region_area,
        use_slic=not args.disable_slic,
    )

    if args.max_colors is not None:
        value = max(1, int(args.max_colors))
        config.max_colors_low = value
        config.max_colors_high = value
        config.max_colors_ultra = value

    results = VectorizationPipeline(config).run(args.input)

    for res in results:
        if res.report is None:
            print(f"[OK] {res.source} -> {res.output} | validation=skipped")
        else:
            print(f"[OK] {res.source} -> {res.output} | SSIM={res.report.ssim:.4f} | MSE={res.report.mse:.2f}")


if __name__ == "__main__":
    main()
