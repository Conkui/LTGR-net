from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import shift

from scripts.data.paired_corruptions import paired_corruptions
from tests.fixtures import arrays


class PairedCorruptionTests(unittest.TestCase):
    def test_repeatable_and_inputs_unchanged(self) -> None:
        labels, mri = arrays()
        original_labels, original_mri = labels.copy(), mri.copy()
        first, recipe = paired_corruptions(labels, mri, (1.0, 1.0, 2.0), seed=17)
        second, repeated = paired_corruptions(labels, mri, (1.0, 1.0, 2.0), seed=17)
        self.assertEqual(recipe, repeated)
        for key in first:
            for left, right in zip(first[key], second[key]):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(first[key][0].dtype, labels.dtype)
            self.assertTrue(set(np.unique(first[key][0])) <= set(np.unique(labels)))
        np.testing.assert_array_equal(labels, original_labels)
        np.testing.assert_array_equal(mri, original_mri)

    def test_seed_changes_the_warp(self) -> None:
        labels, mri = arrays()
        first, _ = paired_corruptions(labels, mri, (1.0, 1.0, 2.0), seed=1)
        second, _ = paired_corruptions(labels, mri, (1.0, 1.0, 2.0), seed=2)
        self.assertTrue(np.any(first["D"][1] != second["D"][1]))

    def test_missing_region_has_matching_zero_fill(self) -> None:
        labels, mri = arrays()
        pairs, _ = paired_corruptions(labels, mri, (1.0, 1.0, 2.0))
        removed = (labels > 0) & (pairs["B"][0] == 0)
        self.assertTrue(np.any(removed))
        self.assertTrue(np.all(pairs["B"][1][removed] == 0))

    def test_translated_mri_and_instance_use_the_same_sampling(self) -> None:
        labels, mri = arrays()
        pairs, _ = paired_corruptions(labels, mri, (1.0, 1.0, 2.0))
        expected_mask = shift((labels == 4).astype(np.uint8), (0, 12, 0), order=0,
                              mode="constant", cval=0, prefilter=False) > 0
        expected_mri = shift(mri, (0, 12, 0), order=1, mode="constant", cval=0, prefilter=False)
        np.testing.assert_array_equal(pairs["E"][0] == 4, expected_mask)
        np.testing.assert_allclose(pairs["E"][1][expected_mask], expected_mri[expected_mask])
        vacated = (labels == 4) & ~expected_mask
        self.assertTrue(np.all(pairs["E"][1][vacated] == 0))

    def test_constant_mri_survives_smooth_sampling(self) -> None:
        labels, mri = arrays()
        pairs, _ = paired_corruptions(labels, np.full_like(mri, 17), (1.0, 1.0, 2.0))
        np.testing.assert_allclose(pairs["D"][1], 17)

    def test_invalid_inputs_fail(self) -> None:
        labels, mri = arrays()
        invalid = [
            (labels, mri, (0.0, 1.0, 2.0)),
            (labels, mri, (float("nan"), 1.0, 2.0)),
            (labels, mri, (1.0, 2.0)),
            (labels[0], mri[0], (1.0, 1.0, 2.0)),
            (labels, mri[:, :-1], (1.0, 1.0, 2.0)),
            (labels.astype(np.float32), mri, (1.0, 1.0, 2.0)),
            (labels, mri * np.nan, (1.0, 1.0, 2.0)),
            (np.zeros_like(labels), mri, (1.0, 1.0, 2.0)),
        ]
        for lab, image, spacing in invalid:
            with self.subTest(shape=lab.shape, spacing=spacing), self.assertRaises(ValueError):
                paired_corruptions(lab, image, spacing)


if __name__ == "__main__":
    unittest.main()
