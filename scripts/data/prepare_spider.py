"""Export a native-grid crop for selected development series."""
from __future__ import annotations

import sys

import SimpleITK as sitk

from scripts.data.common import create_output_dir, write_json
from scripts.data.demo_options import parse_options
from scripts.data.spider_io import crop_pair, read_pair, verify_sources, write_volume


def main() -> None:
    options = parse_options("Prepare paired SPIDER crops", "processed/spider_demo")
    output = create_output_dir(options.output_dir, options.data_root)
    records = []
    for series in options.series:
        labels, mri, hashes = read_pair(options.data_root, series)
        labels, mri = crop_pair(labels, mri, options.target, options.context, options.half_width_mm)
        for key, image in (("labels", labels), ("mri", mri)):
            write_volume(output / f"{series}_{key}.nii.gz", sitk.GetArrayFromImage(image), labels)
        verify_sources(options.data_root, series, hashes)
        records.append({"series": series, "source_sha256": hashes,
                        "shape_zyx": list(reversed(labels.GetSize())),
                        "spacing_xyz_mm": labels.GetSpacing(), "origin_xyz_mm": labels.GetOrigin(),
                        "direction": labels.GetDirection()})
        print(f"Saved {series}")
    write_json(output / "metadata.json", {
        "command": ["python", "-m", "scripts.data.prepare_spider", *sys.argv[1:]],
        "target_instance": options.target, "context_instances": options.context,
        "half_width_mm": options.half_width_mm, "seed": options.seed,
        "training_dataset": False, "resampling": False, "cases": records,
    })


if __name__ == "__main__":
    main()
