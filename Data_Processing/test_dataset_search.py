# 扫描整个数据文件夹，生成一个可复用的 _index.json 文件索引（统计文件、后缀、大小等）

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class FileRecord:
    rel: str          # relative path (posix)
    name: str
    suffix: str
    parent: str
    size: int
    mtime: float      # unix timestamp


def build_directory_index(
    root: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    include_hidden: bool = False,
    exclude_dirs: Optional[List[str]] = None,
    include_ext: Optional[List[str]] = None,   # e.g. [".png", ".jpg"]
) -> Path:
    """
    Scan all folders/files under root and write a reusable index JSON to disk.

    Returns the path of the written index file.
    """
    root = Path(root).resolve()
    if output_path is None:
        output_path = root / "_index.json"
    else:
        output_path = Path(output_path).resolve()

    exclude_dirs = set(exclude_dirs or [])
    include_ext_set = set(e.lower() for e in (include_ext or []))

    folders: List[str] = []
    files: List[str] = []
    records: List[FileRecord] = []
    ext_hist: Dict[str, int] = {}

    for p in root.rglob("*"):
        # skip hidden
        if (not include_hidden) and p.name.startswith("."):
            continue

        # skip excluded directories by component name
        if any(part in exclude_dirs for part in p.parts):
            continue

        rel = p.relative_to(root).as_posix()

        if p.is_dir():
            folders.append(rel)
            continue

        if not p.is_file():
            continue

        # filter by extension if requested
        suffix = p.suffix.lower()
        if include_ext_set and suffix not in include_ext_set:
            continue

        st = p.stat()
        files.append(rel)
        records.append(
            FileRecord(
                rel=rel,
                name=p.name,
                suffix=suffix,
                parent=p.parent.relative_to(root).as_posix(),
                size=int(st.st_size),
                mtime=float(st.st_mtime),
            )
        )
        ext_hist[suffix] = ext_hist.get(suffix, 0) + 1

    folders.sort()
    files.sort()
    records.sort(key=lambda r: r.rel)

    payload = {
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "include_hidden": include_hidden,
            "exclude_dirs": sorted(exclude_dirs),
            "include_ext": sorted(include_ext_set),
        },
        "summary": {
            "num_folders": len(folders),
            "num_files": len(files),
            "extensions": dict(sorted(ext_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "folders": folders,
        "files": files,
        "file_records": [asdict(r) for r in records],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return output_path


def load_directory_index(index_path: Union[str, Path]) -> dict:
    """Load the saved index JSON."""
    index_path = Path(index_path)
    return json.loads(index_path.read_text(encoding="utf-8"))


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    idx_path = build_directory_index(
       
        root= r"D:\0248_data_check\Test dataset\COMP0248_test_013",
        output_path=r"D:\0248_data_check\_index.json",
        exclude_dirs=["__pycache__", ".git", "weights", "runs"],
        include_ext=None,          # e.g. [".png"] if you only want png
        include_hidden=False,
    )
    print("Index saved to:", idx_path)