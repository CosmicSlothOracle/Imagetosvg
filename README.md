# Imagetosvg — Production Raster → Layered SVG Pipeline

Imagetosvg is a modular, fidelity-first vector reconstruction system that converts raster images (PNG/JPEG/WebP) into layered, editable SVG with three explicit structural groups:

- **Region fills** (`color_*`)
- **Edge strokes** (`edge_layer`)
- **High-frequency detail strokes** (`detail_layer`)

## Features

- CLI for single-image and batch directory processing
- OpenCV preprocessing (denoise, contrast normalize, edge extraction, adaptive detail map)
- Hybrid segmentation: **SLIC superpixels + KMeans merge** (or KMeans-only fallback)
- Contour vectorization with hole-preserving topology (`RETR_CCOMP`, `fill-rule=evenodd`)
- SVG metadata embedding and browser-friendly output
- Validation by SVG render-back and SSIM/MSE scoring
- **Auto-iteration** mode: retries parameter candidates until target fidelity threshold is reached
- Presets: `low`, `high`, `ultra`

## Project Layout

```text
.
├── assets/
│   └── test_images/
├── configs/
│   └── default.yaml
├── output/
│   └── golden/
├── src/imagetosvg/
├── tests/
├── requirements.txt
└── README.md
```

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## CLI Usage

Single image:

```bash
PYTHONPATH=src python -m imagetosvg input/photo.png --detail ultra --output-dir output
```

Batch directory:

```bash
PYTHONPATH=src python -m imagetosvg ./assets/test_images --detail high --output-dir output
```

### Important flags

- `--detail {low,high,ultra}`
- `--output-dir <path>`
- `--no-validate`
- `--no-auto-iterate`
- `--disable-slic`
- `--max-colors <int>`
- `--min-region-area <int>`
- `--debug`

## Pipeline

1. Read image and normalize alpha-aware input.
2. Preprocess with bilateral denoise + CLAHE + sharpen.
3. Build edge map + adaptive detail map.
4. Segment into color regions (SLIC superpixels + KMeans merge).
5. Convert masks/contours into layered SVG paths.
6. Validate via render-back SSIM/MSE.
7. If enabled, auto-iterate config candidates until target SSIM for preset is met.

## Presets

- **low**: fast, compact
- **high**: balanced default
- **ultra**: high structural detail, strict target SSIM

## Validation / Quality

Validation uses CairoSVG render-back and compares against source image:

- SSIM (structural similarity)
- MSE (pixel error)

Preset target thresholds in config:

- low: `0.78`
- high: `0.86`
- ultra: `0.91`

## Test assets

- `assets/test_images/synthetic_checker.svg` sample fixture
- `output/golden/synthetic_checker_expected.svg` golden placeholder structure

## Notes

- GPU is optional; current implementation is deterministic CPU-first.
- For strict archival runs, use `--detail ultra` with validation enabled.
