"""High-fidelity raster-to-SVG conversion toolkit."""

from .config import DetailPreset, PipelineConfig

__all__ = ["PipelineConfig", "DetailPreset", "VectorizationPipeline"]


def __getattr__(name: str):
    if name == "VectorizationPipeline":
        from .pipeline import VectorizationPipeline

        return VectorizationPipeline
    raise AttributeError(name)
