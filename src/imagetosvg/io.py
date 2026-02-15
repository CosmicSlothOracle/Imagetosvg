from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = [p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No supported images found in {input_path}")
    return sorted(files)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_output_paths(images: Iterable[Path], output_dir: Path) -> dict[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    images_list = list(images)
    stems: dict[str, int] = {}
    for img in images_list:
        stems[img.stem] = stems.get(img.stem, 0) + 1

    mapping: dict[Path, Path] = {}
    for img in images_list:
        if stems[img.stem] == 1:
            mapping[img] = output_dir / f"{img.stem}.svg"
        else:
            parent = img.parent.name or "root"
            mapping[img] = output_dir / f"{parent}_{img.stem}.svg"

    return mapping
