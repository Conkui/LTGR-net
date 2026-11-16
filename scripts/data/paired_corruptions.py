"""Paired perturbations for the display example."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, shift
from scipy.spatial import cKDTree


def paired_corruptions(
    labels: np.ndarray,
    mri: np.ndarray,
    spacing_zyx: tuple[float, ...],
    seed: int = 20261116,
    target: int = 4,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    if labels.ndim != 3 or labels.shape != mri.shape or any(size == 0 for size in labels.shape):
        raise ValueError("Expected nonempty paired 3D arrays")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isfinite(mri).all():
        raise ValueError("Expected integer labels and finite MRI values")
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive values")
    if not isinstance(target, (int, np.integer)) or not 1 <= target <= 25:
        raise ValueError("target must be a vertebral instance ID")
    points = np.argwhere(labels == target)
    if len(points) == 0:
        raise ValueError("Missing target instance")
    mri = np.asarray(mri, dtype=np.float32)
    center = points.mean(0)
    z, y, x = np.indices(labels.shape, dtype=np.float32)
    zz, yy, xx = [(grid - value) * step for grid, value, step in zip((z, y, x), center, spacing)]

    hole = ((zz + 3) / 12) ** 2 + ((yy - 3) / 11) ** 2 + (xx / 13) ** 2 < 1
    missing_labels, missing_mri = labels.copy(), mri.copy()
    missing_labels[hole], missing_mri[hole] = 0, 0.0

    envelope = np.exp(-0.5 * ((zz / 25) ** 2 + (yy / 29) ** 2 + (xx / 20) ** 2))
    phase = float(np.random.default_rng(seed).uniform(-0.2, 0.2))
    coordinates = [z + 4 * envelope * np.sin(yy / 15 + phase) / spacing[0],
                   y + 11 * envelope * np.cos(zz / 21) / spacing[1], x]
    # MRI 与标签共用坐标，插值阶数不同。
    warped_labels = map_coordinates(labels, coordinates, order=0, mode="nearest", prefilter=False)
    warped_mri = map_coordinates(mri, coordinates, order=1, mode="nearest", prefilter=False)

    source = labels == target
    displacement = (0.0, 12.0 / float(spacing[1]), 0.0)
    destination = shift(source.astype(np.uint8), displacement, order=0,
                        mode="constant", cval=0, prefilter=False) > 0
    translated_mri = shift(mri, displacement, order=1, mode="constant", cval=0, prefilter=False)
    moved_labels, moved_mri = labels.copy(), mri.copy()
    moved_labels[source], moved_mri[source] = 0, 0.0
    moved_labels[destination], moved_mri[destination] = target, translated_mri[destination]

    anterior_edge = (points[:, 1].max() - center[1]) * spacing[1]
    lump = ((zz - 2) / 10) ** 2 + ((yy - anterior_edge) / 11) ** 2 + (xx / 12) ** 2 < 1
    mixed_labels, mixed_mri = warped_labels.copy(), warped_mri.copy()
    mixed_labels[hole], mixed_mri[hole] = 0, 0.0
    added = lump & (mixed_labels == 0)
    if np.any(added):
        # 额外区域的强度取自同一实例。
        nearest = cKDTree(points * spacing).query(np.argwhere(added) * spacing)[1]
        mixed_labels[added] = target
        mixed_mri[added] = mri[tuple(points[nearest].T)]

    pairs = {"B": (missing_labels, missing_mri), "D": (warped_labels, warped_mri),
             "E": (moved_labels, moved_mri), "F": (mixed_labels, mixed_mri)}
    for key, (lab, image) in pairs.items():
        if not np.any(lab != labels):
            raise ValueError(f"Perturbation {key} made no label change; check crop and spacing")
        if not np.isfinite(image).all():
            raise ValueError(f"Perturbation {key} produced nonfinite MRI values")
    recipe = {
        "seed": seed, "target_instance": target, "phase": phase,
        "translation_mm": 12.0,
        "overwritten_other_label_voxels": int(np.count_nonzero(destination & (labels != 0) & ~source)),
        "donor_added_voxels": int(added.sum()), "fill_value": 0.0,
        "mri_interpolation": "linear; nearest donor copy",
        "label_interpolation": "nearest", "clinical_simulation": False,
    }
    return pairs, recipe
