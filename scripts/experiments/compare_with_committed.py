"""Compare a re-run against the committed results, metric by metric.

``git diff`` is the wrong tool for this: every re-run cell differs from its
committed copy in its timestamp, its commit SHA and its wall-clock fit time, so
a diff marks all of them as changed whether or not a single number moved. This
compares everything *except* those fields.

Three forms::

    # cells: results/E3_cells against the copy moved aside before the re-run
    # (results/E3_cells.committed), or against git HEAD if there is no copy
    python scripts/experiments/compare_with_committed.py E3
    python scripts/experiments/compare_with_committed.py E2 E3

    # cells written somewhere else, e.g. E6 with --output-root
    python scripts/experiments/compare_with_committed.py \\
        --new ../e6_rerun/results/E6_cells --ref results/E6_cells

    # one result table against its committed version in git HEAD
    python scripts/experiments/compare_with_committed.py --csv results/E3_runs.csv

Cells present only in the re-run (or only in the reference) are counted, not
treated as differences: a partial re-run is a smaller comparison, not a failed
one. Exit status is 1 if any compared value differs.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"

#: Fields that record *when and where* a cell ran rather than what it found.
VOLATILE = {"fit_seconds", "cell_seconds", "git_sha", "git_dirty",
            "timestamp_utc", "recorded_utc"}


def git_show(path: Path) -> bytes | None:
    """The committed bytes of ``path`` at HEAD, or None if it is not tracked."""
    rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=PROJECT_ROOT,
                         capture_output=True)
    return out.stdout if out.returncode == 0 else None


def shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def cell_rows(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload.get("rows", []))
    return frame.drop(columns=[c for c in frame.columns if c in VOLATILE])


def compare_frames(new: pd.DataFrame, ref: pd.DataFrame) -> dict[str, float]:
    """Columns whose values differ, mapped to the largest absolute gap (NaN if
    the column is not numeric). Empty when the frames agree exactly."""
    if new.shape != ref.shape or list(new.columns) != list(ref.columns):
        return {"<shape or columns>": float("nan")}
    out = {}
    for col in new.columns:
        a, b = new[col].reset_index(drop=True), ref[col].reset_index(drop=True)
        if a.equals(b):
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            out[col] = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
        else:
            out[col] = float("nan")
    return out


def compare_cells(new_dir: Path, ref_dir: Path | None) -> bool:
    label = shown(ref_dir) if ref_dir is not None else "git HEAD"
    print(f"\n{shown(new_dir)}  vs  {label}")
    if not new_dir.is_dir():
        print("  no re-run cells found")
        return True

    identical, differing, unmatched = 0, {}, 0
    for path in sorted(new_dir.glob("*.json")):
        if ref_dir is not None:
            ref_path = ref_dir / path.name
            raw = ref_path.read_bytes() if ref_path.exists() else None
        else:
            raw = git_show(path)
        if raw is None:
            unmatched += 1
            continue
        new = cell_rows(json.loads(path.read_text(encoding="utf-8")))
        ref = cell_rows(json.loads(raw))
        gaps = compare_frames(new, ref)
        if gaps:
            differing[path.stem] = gaps
        else:
            identical += 1

    n_ref = len(list(ref_dir.glob("*.json"))) if ref_dir is not None else None
    print(f"  compared {identical + len(differing)} cells: {identical} identical, "
          f"{len(differing)} differ; {unmatched} with no committed counterpart"
          + (f"; reference holds {n_ref}" if n_ref is not None else ""))
    for cell, gaps in differing.items():
        detail = ", ".join(f"{c} (max |d| {g:.3g})" if g == g else c
                           for c, g in gaps.items())
        print(f"  DIFFERS  {cell}: {detail}")
    return not differing


def compare_csv(path: Path) -> bool:
    print(f"\n{shown(path)}  vs  git HEAD")
    raw = git_show(path)
    if raw is None:
        print("  not tracked in git; nothing to compare against")
        return True
    new = pd.read_csv(path, compression="infer")
    ref = pd.read_csv(io.BytesIO(raw),
                      compression="gzip" if path.suffix == ".gz" else None)
    drop = [c for c in new.columns if c in VOLATILE]
    new, ref = new.drop(columns=drop), ref.drop(columns=drop, errors="ignore")
    keys = [c for c in new.columns if not pd.api.types.is_float_dtype(new[c])]
    if keys and set(keys) <= set(ref.columns):
        new = new.sort_values(keys, kind="stable").reset_index(drop=True)
        ref = ref.sort_values(keys, kind="stable").reset_index(drop=True)
    gaps = compare_frames(new, ref)
    if not gaps:
        print(f"  identical: {len(new)} rows, {len(new.columns)} columns "
              f"(ignoring {', '.join(drop) or 'nothing'})")
        return True
    for col, gap in gaps.items():
        print(f"  DIFFERS  {col}" + (f" (max |d| {gap:.3g})" if gap == gap else ""))
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare a re-run with the committed results.")
    ap.add_argument("experiments", nargs="*", help="E2, E3 or E6: compare results/<E>_cells")
    ap.add_argument("--new", type=Path, help="a directory of re-run cell JSONs")
    ap.add_argument("--ref", type=Path, help="the directory to compare --new against")
    ap.add_argument("--csv", type=Path, action="append", default=[],
                    help="a result table to compare with its git HEAD version")
    args = ap.parse_args()

    if not (args.experiments or args.new or args.csv):
        ap.error("name an experiment, or pass --new or --csv")

    ok = True
    for exp in args.experiments:
        new_dir = RESULTS / f"{exp}_cells"
        aside = RESULTS / f"{exp}_cells.committed"
        ok &= compare_cells(new_dir, aside if aside.is_dir() else None)
    if args.new:
        ok &= compare_cells(args.new, args.ref)
    for path in args.csv:
        ok &= compare_csv(path)

    print("\nverdict:", "MATCH" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
