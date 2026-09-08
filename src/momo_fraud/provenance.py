"""Machine-readable record of what actually produced a result.

``requirements.txt`` pins what *should* be installed; this records what *was*.
The distinction matters the first time a number cannot be reproduced: without
it there is no way to tell a genuine methodological difference from a
dependency that moved underneath the study.

Nothing here changes any result. It is provenance only, written alongside the
artifacts it describes.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import constants as C

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Packages whose version can change a number. Anything not installed is
#: recorded as ``None`` rather than omitted, so a missing dependency is visible
#: in the record instead of being indistinguishable from one that was never
#: checked.
TRACKED_PACKAGES = [
    "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm",
    "imbalanced-learn", "shap", "scipy", "matplotlib", "seaborn",
]


def package_versions(packages: list[str] | None = None) -> dict[str, str | None]:
    """Resolved versions of the packages that can move a result."""
    out: dict[str, str | None] = {}
    for name in packages or TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_state() -> dict[str, str | bool | None]:
    """Commit SHA and whether the tree was dirty when this ran.

    ``dirty`` is the field that matters. A SHA alone implies the recorded commit
    produced the artifact; if uncommitted changes were present, it did not.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                text=True, timeout=15, check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def environment() -> dict:
    """Interpreter, platform and package state."""
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def dataset_record() -> dict:
    """Dataset identity, from the provenance file notebook 00 writes.

    Falls back to the asserted constants when that file is absent -- those are
    checked against the real file on every load by ``data.validate``, so they
    are not merely declarative.
    """
    path = PROJECT_ROOT / "results" / "00_dataset_provenance.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": "ealaxi/paysim1",
        "csv_sha256": None,
        "n_transactions": C.N_ROWS,
        "n_fraud": C.N_FRAUD,
        "prevalence": C.PREVALENCE,
        "note": "00_dataset_provenance.json absent; values from asserted constants.",
    }


def experiment_record(
    *,
    name: str,
    seed: int = C.RANDOM_SEED,
    split: dict | None = None,
    model: dict | None = None,
    csl: dict | None = None,
    cost_ratio: float | dict | None = None,
    thresholds: dict | None = None,
    evaluation: dict | None = None,
    notes: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble one experiment's full provenance record.

    Every field is optional because not every experiment has every dimension --
    a feature-importance run has no CSL configuration. Absent fields are written
    as ``None`` rather than dropped, so the record shows what was not applicable
    as distinct from what was not recorded.
    """
    return {
        "experiment": name,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": environment(),
        "dataset": dataset_record(),
        "seed": seed,
        "split": split,
        "model": model,
        "csl": csl,
        "cost_ratio": cost_ratio,
        "thresholds": thresholds,
        "evaluation": evaluation,
        "notes": notes,
        **(extra or {}),
    }


def write(record: dict, name: str, *, directory: Path | None = None) -> Path:
    """Persist a record next to the artifacts it describes."""
    directory = directory or (PROJECT_ROOT / "results")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
