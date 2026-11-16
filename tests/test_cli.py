from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.data.common import file_sha256
from scripts.data.demo_options import parse_options
from tests.fixtures import write_sources

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_explicit_case_selection_is_required(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            parse_options("test", "unused", [])
        self.assertEqual(result.exception.code, 2)

    def test_duplicate_and_path_names_are_rejected(self) -> None:
        for names in (["same", "same"], ["../case"]):
            with self.subTest(names=names), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_options("test", "unused", ["--series", *names])

    def test_entrypoints_run_without_project_data(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            write_sources(source)
            before = {path: file_sha256(path) for path in source.rglob("*.mha")}
            for module, name in (("scripts.data.prepare_spider", "crops"),
                                 ("scripts.figures.spider_ssl_paired_mri_demo", "plates")):
                output = root / name
                command = [sys.executable, "-B", "-m", module, "--data-root", str(source),
                           "--series", "synthetic_case", "--output-dir", str(output), "--seed", "17"]
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                        encoding="utf-8", errors="replace", timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                self.assertFalse(metadata["training_dataset"])
                self.assertEqual(metadata["cases"][0]["series"], "synthetic_case")
                self.assertEqual(len(metadata["cases"]), 1)
                self.assertTrue((output / "说明.txt").exists())
                if name == "plates":
                    self.assertEqual(len(list(output.glob("*.nii.gz"))), 10)
                    self.assertEqual(len(list(output.glob("*.png"))), 3)
                    for stats in metadata["cases"][0]["statistics"].values():
                        self.assertGreater(stats["label_changed_voxels"], 0)
            self.assertEqual(before, {path: file_sha256(path) for path in source.rglob("*.mha")})


if __name__ == "__main__":
    unittest.main()
