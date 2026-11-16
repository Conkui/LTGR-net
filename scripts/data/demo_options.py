"""Command-line options shared by the examples."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DemoOptions:
    data_root: Path
    output_dir: Path
    series: tuple[str, ...]
    target: int
    context: tuple[int, ...]
    seed: int
    half_width_mm: float


def parse_options(description: str, default_output: str, argv: Sequence[str] | None = None) -> DemoOptions:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/spider_demo.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(default_output))
    parser.add_argument("--series", nargs="+")
    parser.add_argument("--target-instance", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--half-width-mm", type=float)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    series = args.series if args.series is not None else config["series"]
    if not isinstance(series, list) or not series:
        parser.error("select development cases with --series")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in series):
        parser.error("--series expects names without paths or extensions")
    if len(set(series)) != len(series):
        parser.error("duplicate series names")
    target = args.target_instance if args.target_instance is not None else config["target_instance"]
    seed = args.seed if args.seed is not None else config["seed"]
    width = args.half_width_mm if args.half_width_mm is not None else config["half_width_mm"]
    context = config["context_instances"]
    if type(target) is not int or not 1 <= target <= 25:
        parser.error("target_instance must be an integer from 1 to 25")
    if type(seed) is not int or seed < 0:
        parser.error("seed must be a nonnegative integer")
    if not isinstance(width, (int, float)) or not np.isfinite(width) or width <= 0:
        parser.error("half_width_mm must be positive")
    if not isinstance(context, list) or any(type(value) is not int for value in context):
        parser.error("context_instances must be a list of integer IDs")
    data_root = args.data_root if args.data_root is not None else ROOT / config["data_root"]
    return DemoOptions(data_root.resolve(), args.output_dir.resolve(), tuple(series),
                       target, tuple(context), seed, float(width))
