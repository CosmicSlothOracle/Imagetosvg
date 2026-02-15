from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .io import collect_inputs, ensure_output_paths, read_image
from .optimizer import optimize_and_render
from .validator import ValidationReport

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    source: Path
    output: Path
    report: ValidationReport | None


class VectorizationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config.validated()

    def run(self, input_path: Path) -> list[PipelineResult]:
        images = collect_inputs(input_path)
        targets = ensure_output_paths(images, self.config.output_dir)
        results: list[PipelineResult] = []

        for image_path in images:
            logger.info("Vectorizing %s", image_path)
            image = read_image(image_path)
            output_path = targets[image_path]
            _, report = optimize_and_render(image, image_path, self.config, output_path)
            results.append(PipelineResult(source=image_path, output=output_path, report=report))

        return results
