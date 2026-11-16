"""Read and crop paired volumes on their native grid."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import numpy as np
import SimpleITK as sitk

from scripts.data.common import file_sha256
from scripts.data.spider_labels import semantic_from_instance


def check_grid(labels: sitk.Image, mri: sitk.Image) -> None:
    if labels.GetDimension() != 3 or mri.GetDimension() != 3:
        raise ValueError("Expected two 3D volumes")
    if labels.GetNumberOfComponentsPerPixel() != 1 or mri.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected scalar volumes")
    if labels.GetSize() != mri.GetSize():
        raise ValueError("MRI and labels have different sizes")
    for name in ("Spacing", "Origin", "Direction"):
        first, second = getattr(labels, f"Get{name}")(), getattr(mri, f"Get{name}")()
        if not np.allclose(first, second, rtol=0, atol=1e-4):
            raise ValueError(f"MRI and labels differ in {name.lower()}")


def read_pair(data_root: Path, series: str) -> tuple[sitk.Image, sitk.Image, dict[str, str]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", series):
        raise ValueError("Use a series name without paths or an extension")
    paths = {"labels": data_root / "masks" / f"{series}.mha",
             "mri": data_root / "images" / f"{series}.mha"}
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    labels, mri = [sitk.ReadImage(str(paths[key])) for key in ("labels", "mri")]
    check_grid(labels, mri)
    labels, mri = [sitk.DICOMOrient(image, "RAS") for image in (labels, mri)]
    check_grid(labels, mri)
    semantic_from_instance(sitk.GetArrayViewFromImage(labels))
    if not np.isfinite(sitk.GetArrayViewFromImage(mri)).all():
        raise ValueError("MRI contains nonfinite values")
    return labels, mri, hashes


def crop_pair(
    labels: sitk.Image,
    mri: sitk.Image,
    target_instance: int,
    context_instances: Sequence[int],
    half_width_mm: float = 20.0,
) -> tuple[sitk.Image, sitk.Image]:
    check_grid(labels, mri)
    if sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(labels.GetDirection()) != "RAS":
        raise ValueError("Orient the pair to RAS before cropping")
    if not np.isfinite(half_width_mm) or half_width_mm <= 0:
        raise ValueError("half_width_mm must be positive")
    array = sitk.GetArrayViewFromImage(labels)
    target = np.argwhere(array == target_instance)
    if target.size == 0:
        raise ValueError(f"Missing target instance: {target_instance}")
    selected = np.argwhere(np.isin(array, [*context_instances, target_instance]))
    spacing = np.asarray(labels.GetSpacing()[::-1])
    lower = np.maximum(selected.min(0) - np.ceil([8 / spacing[0], 25 / spacing[1], 0]).astype(int), 0)
    upper = np.minimum(selected.max(0) + np.ceil([8 / spacing[0], 35 / spacing[1], 0]).astype(int) + 1, array.shape)
    # 只截取原生薄块，不提高空间分辨率。
    middle = int(np.rint(target[:, 2].mean()))
    radius = int(np.ceil(half_width_mm / spacing[2]))
    lower[2], upper[2] = max(0, middle - radius), min(array.shape[2], middle + radius + 1)
    size, start = (upper - lower)[::-1].astype(int).tolist(), lower[::-1].astype(int).tolist()
    return tuple(sitk.RegionOfInterest(image, size, start) for image in (labels, mri))


def write_volume(path: Path, array: np.ndarray, reference: sitk.Image) -> None:
    if path.exists():
        raise FileExistsError(path)
    if array.shape != tuple(reversed(reference.GetSize())):
        raise ValueError("Array shape does not match the reference grid")
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))
    restored = sitk.ReadImage(str(path))
    check_grid(reference, restored)
    if not np.array_equal(sitk.GetArrayFromImage(restored), array):
        raise ValueError(f"Volume round-trip failed: {path.name}")


def verify_sources(data_root: Path, series: str, hashes: dict[str, str]) -> None:
    paths = {"labels": data_root / "masks" / f"{series}.mha",
             "mri": data_root / "images" / f"{series}.mha"}
    if any(file_sha256(paths[key]) != digest for key, digest in hashes.items()):
        raise RuntimeError("A source file changed during processing")
