"""Draw paired MRI and label examples."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import SimpleITK as sitk

from scripts.data.common import create_output_dir, write_json
from scripts.data.demo_options import parse_options
from scripts.data.paired_corruptions import paired_corruptions
from scripts.data.spider_io import crop_pair, read_pair, verify_sources, write_volume
from scripts.data.spider_labels import semantic_from_instance

TITLES = {"B": "B  Missing region", "D": "D  Deformation",
          "E": "E  Misalignment", "F": "F  Mixed"}


def save_plate(
    path: Path,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    slice_index: int,
    spacing_zyx: tuple[float, ...],
    window: tuple[float, float],
    overlay: bool = False,
) -> None:
    figure, axes = plt.subplots(1, len(pairs), figsize=(4.1 * len(pairs), 6.2), squeeze=False)
    cmap = ListedColormap(["#4b9ac7", "#f0ad50", "#61b49e"])
    for axis, (key, (labels, mri)) in zip(axes.flat, pairs.items()):
        extent = [0, mri.shape[1] * spacing_zyx[1], 0, mri.shape[0] * spacing_zyx[0]]
        axis.imshow(mri[:, :, slice_index], cmap="gray", origin="lower", interpolation="nearest",
                    extent=extent, vmin=window[0], vmax=window[1])
        if overlay:
            semantic = semantic_from_instance(labels[:, :, slice_index])
            axis.imshow(np.ma.masked_where(semantic == 0, semantic), cmap=cmap, vmin=1, vmax=3,
                        alpha=0.5, origin="lower", interpolation="nearest", extent=extent)
        axis.set_title(TITLES.get(key, "A  Original MRI"), fontsize=15, pad=10)
        length = 20 if extent[1] >= 30 else 5
        axis.plot([3, 3 + length], [3, 3], color="white", linewidth=3)
        axis.text(3 + length / 2, 5, f"{length} mm", ha="center", color="white", fontsize=10,
                  bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1})
        axis.set_axis_off()
    figure.text(0.5, 0.065, "Same native sagittal slice and window; anterior to the right.", ha="center", fontsize=10)
    figure.text(0.5, 0.025, "Artificial perturbations; not clinical cases or model predictions.", ha="center", fontsize=10)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.12, top=0.9, wspace=0.025)
    try:
        for extension in (".png", ".pdf", ".svg"):
            figure.savefig(path.with_suffix(extension), dpi=300)
    finally:
        plt.close(figure)


def main() -> None:
    options = parse_options("Paired MRI/label display", "outputs/figures/spider_ssl_paired_mri_demo")
    output = create_output_dir(options.output_dir, options.data_root)
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    records = []
    for index, series in enumerate(options.series):
        reference, mri, hashes = read_pair(options.data_root, series)
        reference, mri = crop_pair(reference, mri, options.target, options.context, options.half_width_mm)
        labels = sitk.GetArrayFromImage(reference)
        image = sitk.GetArrayFromImage(mri).astype(np.float32)
        spacing = reference.GetSpacing()[::-1]
        slice_index = int(np.rint(np.argwhere(labels == options.target)[:, 2].mean()))
        percentiles = (0.5, 99.5)
        # percentiles = (1.0, 99.0)  # 手动核对灰度窗时可切换。
        window = tuple(float(value) for value in np.percentile(image[:, :, slice_index], percentiles))
        if window[1] <= window[0]:
            window = (window[0], window[0] + 1.0)
        pairs, recipe = paired_corruptions(labels, image, spacing, options.seed + index, options.target)
        statistics = {}
        for key, (lab, corrupted) in {"A": (labels, image), **pairs}.items():
            write_volume(output / f"{series}_{key}_labels.nii.gz", lab, reference)
            write_volume(output / f"{series}_{key}_mri.nii.gz", corrupted, reference)
            if key != "A":
                statistics[key] = {"label_changed_voxels": int(np.count_nonzero(lab != labels)),
                                   "mri_changed_voxels": int(np.count_nonzero(corrupted != image)),
                                   "slice_label_changes": int(np.count_nonzero(lab[:, :, slice_index] != labels[:, :, slice_index]))}
        for overlay in (False, True):
            suffix = "paired_overlay" if overlay else "mri_BDEF"
            save_plate(output / f"{series}_{suffix}", pairs, slice_index, spacing, window, overlay)
        save_plate(output / f"{series}_original", {"A": (labels, image)}, slice_index, spacing, window)
        verify_sources(options.data_root, series, hashes)
        records.append({"series": series, "source_sha256": hashes,
                        "shape_zyx": labels.shape, "spacing_xyz_mm": reference.GetSpacing(),
                        "origin_xyz_mm": reference.GetOrigin(), "direction": reference.GetDirection(),
                        "slice_index_x": slice_index, "window": window,
                        "recipe": recipe, "statistics": statistics})
        print(f"Saved {series}")
    write_json(output / "metadata.json", {
        "command": ["python", "-m", "scripts.figures.spider_ssl_paired_mri_demo", *sys.argv[1:]],
        "target_instance": options.target, "context_instances": options.context,
        "half_width_mm": options.half_width_mm, "seed": options.seed,
        "training_dataset": False, "model_predictions": False,
        "clinical_simulation": False, "resampling": False, "cases": records,
    })


if __name__ == "__main__":
    main()
