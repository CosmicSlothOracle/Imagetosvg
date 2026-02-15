from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DetailPreset(str, Enum):
    LOW = "low"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass(slots=True)
class PipelineConfig:
    detail: DetailPreset = DetailPreset.HIGH
    output_dir: Path = Path("output")
    preserve_alpha: bool = True
    min_region_area: int = 20

    max_colors_low: int = 18
    max_colors_high: int = 36
    max_colors_ultra: int = 72

    use_slic: bool = True
    slic_segments_low: int = 180
    slic_segments_high: int = 350
    slic_segments_ultra: int = 700
    slic_compactness: float = 11.0

    canny_low: int = 30
    canny_high: int = 130
    edge_dilate_iterations: int = 1

    denoise_sigma: float = 28.0
    adaptive_block_size: int = 35
    adaptive_c: int = 3

    simplification_low: float = 0.0026
    simplification_high: float = 0.0012
    simplification_ultra: float = 0.00045

    embed_metadata: bool = True
    validate_similarity: bool = True
    auto_iterate: bool = True
    min_ssim_low: float = 0.78
    min_ssim_high: float = 0.86
    min_ssim_ultra: float = 0.91

    def max_colors(self) -> int:
        return {
            DetailPreset.LOW: self.max_colors_low,
            DetailPreset.HIGH: self.max_colors_high,
            DetailPreset.ULTRA: self.max_colors_ultra,
        }[self.detail]

    def slic_segments(self) -> int:
        return {
            DetailPreset.LOW: self.slic_segments_low,
            DetailPreset.HIGH: self.slic_segments_high,
            DetailPreset.ULTRA: self.slic_segments_ultra,
        }[self.detail]

    def simplification_ratio(self) -> float:
        return {
            DetailPreset.LOW: self.simplification_low,
            DetailPreset.HIGH: self.simplification_high,
            DetailPreset.ULTRA: self.simplification_ultra,
        }[self.detail]

    def target_ssim(self) -> float:
        return {
            DetailPreset.LOW: self.min_ssim_low,
            DetailPreset.HIGH: self.min_ssim_high,
            DetailPreset.ULTRA: self.min_ssim_ultra,
        }[self.detail]

    def validated(self) -> PipelineConfig:
        self.min_region_area = max(1, self.min_region_area)
        self.edge_dilate_iterations = max(0, self.edge_dilate_iterations)
        self.denoise_sigma = max(1.0, self.denoise_sigma)
        self.slic_compactness = max(0.1, self.slic_compactness)
        self.slic_segments_low = max(20, self.slic_segments_low)
        self.slic_segments_high = max(20, self.slic_segments_high)
        self.slic_segments_ultra = max(20, self.slic_segments_ultra)

        block = self.adaptive_block_size
        if block < 3:
            block = 3
        if block % 2 == 0:
            block += 1
        self.adaptive_block_size = block

        return self
