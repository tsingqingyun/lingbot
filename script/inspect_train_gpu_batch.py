#!/usr/bin/env python3
"""
Static scan: GPU memory limits / pre-allocation patterns and batch-related settings
in wan_va/train.py and selected config .py files.

Usage:
  python script/inspect_train_gpu_batch.py
  python script/inspect_train_gpu_batch.py --repo-root /path/to/lingbot-va
  python script/inspect_train_gpu_batch.py \\
    --extra /path/to/va_robocasa_train_cfg.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


# (label, regex) — line-based search
GPU_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("torch.cuda.set_per_process_memory_fraction", re.compile(r"set_per_process_memory_fraction")),
    ("CUDA memory fraction / limit APIs", re.compile(r"memory_fraction|max_split_size_mb|caching_allocator")),
    ("torch.cuda.empty_cache", re.compile(r"torch\.cuda\.empty_cache\s*\(")),
    ("torch.cuda memory stats", re.compile(r"torch\.cuda\.memory_(allocated|reserved|stats)")),
    ("gc.collect", re.compile(r"gc\.collect\s*\(")),
    ("PYTORCH_CUDA_ALLOC_CONF mention", re.compile(r"PYTORCH_CUDA_ALLOC_CONF|PYTORCH_ALLOC_CONF")),
]

BATCH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("DataLoader batch_size=", re.compile(r"DataLoader\s*\([^)]*batch_size\s*=")),
    ("config.batch_size", re.compile(r"batch_size\s*=\s*config\.batch_size")),
    ("cfg batch_size assignment", re.compile(r"\w+\.batch_size\s*=")),
    ("gradient_accumulation_steps", re.compile(r"gradient_accumulation_steps")),
]


def scan_file(path: Path, patterns: list[tuple[str, re.Pattern[str]]]) -> dict[str, list[int]]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: dict[str, list[int]] = {}
    for label, rx in patterns:
        for i, line in enumerate(text, start=1):
            if rx.search(line):
                hits.setdefault(label, []).append(i)
    return hits


def print_hits(rel: str, hits: dict[str, list[int]]) -> None:
    if not hits:
        print(f"  (no matches) — {rel}")
        return
    for label, lines in sorted(hits.items(), key=lambda x: x[0]):
        print(f"  [{label}] lines {lines}")


def extract_assignments(path: Path, var_names: tuple[str, ...]) -> dict[str, list[str]]:
    """Rough parse: obj.var = <rhs> on one line."""
    out: dict[str, list[str]] = {v: [] for v in var_names}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.split("#", 1)[0].strip()
        for v in var_names:
            m = re.match(rf"^[\w.]+\.{re.escape(v)}\s*=\s*(.+?)\s*$", stripped)
            if m:
                out[v].append(m.group(1).strip())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan train.py / configs for GPU + batch settings.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="lingbot-va repo root (default: parent of script/)",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="Additional .py file to scan (can repeat)",
    )
    args = parser.parse_args()

    repo = args.repo_root or Path(__file__).resolve().parent.parent
    train_py = repo / "wan_va" / "train.py"
    robocasa_cfg = repo / "wan_va" / "configs" / "va_robocasa_cfg.py"
    robocasa_train_cfg = repo / "wan_va" / "configs" / "va_robocasa_train_cfg.py"
    run_sh = repo / "script" / "run_va_posttrain.sh"

    files_gpu = [train_py, *args.extra]
    files_batch = [train_py, robocasa_cfg, robocasa_train_cfg, *args.extra]

    print("=== Environment (current shell; training may set more in .sh) ===")
    for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "CUDA_VISIBLE_DEVICES"):
        print(f"  {k}={os.environ.get(k, '<unset>')}")

    if run_sh.is_file():
        print(f"\n=== {run_sh.relative_to(repo)} (relevant lines) ===")
        for i, line in enumerate(run_sh.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if re.search(r"PYTORCH_CUDA|ALLOC|CUDA", line):
                print(f"  L{i}: {line.strip()}")

    missing = [p for p in (train_py, robocasa_cfg, robocasa_train_cfg) if not p.is_file()]
    if missing:
        print("\nMissing expected files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)

    print(f"\n=== GPU-related patterns in {train_py.name} ===")
    if train_py.is_file():
        print_hits(str(train_py.relative_to(repo)), scan_file(train_py, GPU_PATTERNS))
    for extra in args.extra:
        if extra.is_file():
            print(f"\n=== GPU-related patterns in {extra} ===")
            print_hits(str(extra), scan_file(extra, GPU_PATTERNS))

    print("\n=== Batch-related patterns ===")
    for f in files_batch:
        if not f.is_file():
            continue
        try:
            rel = str(f.relative_to(repo))
        except ValueError:
            rel = str(f)
        print(f"\n-- {rel}")
        print_hits(rel, scan_file(f, BATCH_PATTERNS))

    if robocasa_train_cfg.is_file():
        print("\n=== Parsed assignments (robocasa train cfg) ===")
        assigns = extract_assignments(
            robocasa_train_cfg,
            ("batch_size", "gradient_accumulation_steps", "load_worker", "gc_interval"),
        )
        for k, vals in assigns.items():
            print(f"  {k}: {vals if vals else '<not assigned on one line>'}")

    print(
        "\nNote: va_robocasa_cfg.py usually holds model/data shape; "
        "batch_size is typically in va_robocasa_train_cfg.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
