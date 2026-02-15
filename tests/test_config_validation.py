from imagetosvg.config import PipelineConfig


def test_pipeline_config_validated_normalizes_values() -> None:
    cfg = PipelineConfig(min_region_area=0, adaptive_block_size=10, denoise_sigma=0.5, edge_dilate_iterations=-1)
    cfg.validated()
    assert cfg.min_region_area == 1
    assert cfg.adaptive_block_size == 11
    assert cfg.denoise_sigma >= 1.0
    assert cfg.edge_dilate_iterations == 0
