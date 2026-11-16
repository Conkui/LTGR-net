"""Shared file helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_output_dir(path: Path, data_root: Path) -> Path:
    path, data_root = path.resolve(), data_root.resolve()
    if path == data_root or data_root in path.parents or path in data_root.parents:
        raise ValueError("Output must be separate from the source data")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "说明.txt").write_text(
        "用途：本次数据处理或配对展示的派生文件。\n"
        "来源：显式选定的输入序列；原始文件不修改。\n"
        "不属于训练数据合同，不作为健康真值、临床模拟或模型结果。\n"
        "复现命令、种子和输入哈希见 metadata.json。\n",
        encoding="utf-8",
    )
    return path
