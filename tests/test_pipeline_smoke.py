from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from imagetosvg.config import DetailPreset, PipelineConfig
from imagetosvg.pipeline import VectorizationPipeline


def test_pipeline_smoke(tmp_path: Path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (58, 58), (0, 255, 0), -1)
    cv2.circle(image, (32, 32), 14, (0, 0, 255), -1)

    src = tmp_path / "sample.png"
    cv2.imwrite(str(src), image)

    config = PipelineConfig(detail=DetailPreset.HIGH, output_dir=tmp_path / "out", validate_similarity=False)
    results = VectorizationPipeline(config).run(src)

    assert len(results) == 1
    assert results[0].output.exists()
    text = results[0].output.read_text(encoding="utf-8")
    assert "<svg" in text and "edge_layer" in text and "detail_layer" in text
