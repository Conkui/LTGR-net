from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from scripts.data.common import create_output_dir, file_sha256, write_json
from scripts.data.spider_io import check_grid, crop_pair, read_pair, verify_sources, write_volume
from scripts.data.spider_labels import semantic_from_instance
from tests.fixtures import volumes, write_sources


class PreprocessingTests(unittest.TestCase):
    def test_semantic_mapping_preserves_instances(self) -> None:
        source = np.array([0, 1, 25, 100, 201, 225], dtype=np.uint16)
        before = source.copy()
        np.testing.assert_array_equal(semantic_from_instance(source), [0, 1, 1, 3, 2, 2])
        np.testing.assert_array_equal(source, before)

    def test_unknown_or_fractional_labels_are_rejected(self) -> None:
        for value in (np.array([26]), np.array([200]), np.array([-1]), np.array([1.2])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                semantic_from_instance(value)

    def test_origin_check_uses_absolute_tolerance(self) -> None:
        labels, mri = volumes()
        mri.SetOrigin((1000.001, -120.0, 50.0))
        with self.assertRaises(ValueError):
            check_grid(labels, mri)

    def test_size_and_spacing_mismatch_are_rejected(self) -> None:
        labels, mri = volumes()
        mri.SetSpacing((2.0, 1.01, 1.0))
        with self.assertRaises(ValueError):
            check_grid(labels, mri)
        with self.assertRaises(ValueError):
            check_grid(labels, sitk.Image([4, 5, 6], sitk.sitkFloat32))

    def test_crop_preserves_native_grid_and_physical_position(self) -> None:
        labels, mri = volumes()
        cropped, image = crop_pair(labels, mri, 4, (4, 203), half_width_mm=4.0)
        check_grid(cropped, image)
        self.assertEqual(cropped.GetSize()[0], 5)
        self.assertEqual(cropped.GetSpacing(), labels.GetSpacing())
        self.assertEqual(cropped.GetDirection(), labels.GetDirection())
        start = labels.TransformPhysicalPointToIndex(cropped.GetOrigin())
        original = sitk.GetArrayFromImage(labels)
        sx, sy, sz = start
        nx, ny, nz = cropped.GetSize()
        np.testing.assert_array_equal(sitk.GetArrayFromImage(cropped), original[sz:sz + nz, sy:sy + ny, sx:sx + nx])

    def test_crop_requires_present_target_and_ras(self) -> None:
        labels, mri = volumes()
        with self.assertRaises(ValueError):
            crop_pair(labels, mri, 24, (24,))
        labels, mri = [sitk.DICOMOrient(image, "LPS") for image in (labels, mri)]
        with self.assertRaises(ValueError):
            crop_pair(labels, mri, 4, (4,))

    def test_read_write_round_trip_and_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_sources(root / "source")
            labels, mri, hashes = read_pair(root / "source", "synthetic_case")
            check_grid(labels, mri)
            target = root / "labels.nii.gz"
            write_volume(target, sitk.GetArrayFromImage(labels), labels)
            with self.assertRaises(FileExistsError):
                write_volume(target, sitk.GetArrayFromImage(labels), labels)
            verify_sources(root / "source", "synthetic_case", hashes)
            source = root / "source/masks/synthetic_case.mha"
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaises(RuntimeError):
                verify_sources(root / "source", "synthetic_case", hashes)

    def test_bad_series_name_is_rejected_before_file_access(self) -> None:
        for name in ("../case", "/case", "case.mha", "case\\secret"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_pair(Path("missing"), name)

    def test_outputs_cannot_overlap_sources_or_existing_results(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            for output in (source, source / "derived", root):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    create_output_dir(output, source)
            output = create_output_dir(root / "derived", source)
            self.assertTrue((output / "说明.txt").is_file())
            with self.assertRaises(FileExistsError):
                create_output_dir(output, source)

    def test_failed_json_write_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            write_json(path, {"ok": True})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                write_json(path, {"bad": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(file_sha256(path), hashlib.sha256(original).hexdigest())
            self.assertEqual([item.name for item in path.parent.iterdir()], ["metadata.json"])


if __name__ == "__main__":
    unittest.main()
