from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def arrays() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((48, 60, 16), dtype=np.uint16)
    labels[12:36, 15:38, 3:13] = 4
    labels[4:8, 12:40, 3:13] = 203
    mri = np.arange(labels.size, dtype=np.float32).reshape(labels.shape) + 1
    return labels, mri


def volumes() -> tuple[sitk.Image, sitk.Image]:
    pair = tuple(sitk.GetImageFromArray(array) for array in arrays())
    for image in pair:
        image.SetSpacing((2.0, 1.0, 1.0))
        image.SetOrigin((1000.0, -120.0, 50.0))
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    return pair


def write_sources(root: Path) -> None:
    labels, mri = volumes()
    for name, image in (("masks", labels), ("images", mri)):
        directory = root / name
        directory.mkdir(parents=True)
        sitk.WriteImage(image, str(directory / "synthetic_case.mha"))
