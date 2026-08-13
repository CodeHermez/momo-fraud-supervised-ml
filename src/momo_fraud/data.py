"""Loading, validating and caching the PaySim dataset.

The load path refuses to proceed on a dataset that is not the canonical
``ealaxi/paysim1`` file. A silently truncated or mirrored copy would change the
class prior, and every threshold in this study is derived from that prior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import constants as C

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "paysim.parquet"

#: Read the CSV straight into compact dtypes. float32 holds PaySim amounts
#: (max ~9.2e7) with room to spare, and halves the in-memory footprint.
RAW_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "nameOrig": "str",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "str",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


class DatasetValidationError(RuntimeError):
    """Raised when the loaded file is not the canonical PaySim dataset."""


def validate(df: pd.DataFrame) -> None:
    """Assert the frame is the canonical PaySim file.

    Checks shape, column names and fraud count. Any mismatch is fatal: the
    severity ratio ``R`` and the Bayes threshold both derive from the class
    prior, so a different file silently invalidates every downstream number.
    """
    problems = []

    if list(df.columns) != C.RAW_COLUMNS:
        problems.append(
            f"columns differ from canonical PaySim.\n"
            f"  expected: {C.RAW_COLUMNS}\n"
            f"  found:    {list(df.columns)}"
        )

    if len(df) != C.N_ROWS:
        problems.append(f"expected {C.N_ROWS:,} rows, found {len(df):,}")

    n_fraud = int(df[C.TARGET].sum())
    if n_fraud != C.N_FRAUD:
        problems.append(f"expected {C.N_FRAUD:,} frauds, found {n_fraud:,}")

    if problems:
        raise DatasetValidationError(
            "Loaded file is not the canonical ealaxi/paysim1 dataset:\n  - "
            + "\n  - ".join(problems)
            + "\n\nDownload it with:  kaggle datasets download -d ealaxi/paysim1"
        )


def load_raw(path: str | Path | None = None, *, validate_file: bool = True) -> pd.DataFrame:
    """Read the raw PaySim CSV into compact dtypes.

    Args:
        path: CSV location. Defaults to ``data/raw/<canonical filename>``.
        validate_file: Assert canonical shape and fraud count. Only disable
            for tests that construct a small synthetic frame.
    """
    path = Path(path) if path is not None else RAW_DIR / C.RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}.\n"
            "Download it with:  kaggle datasets download -d ealaxi/paysim1\n"
            f"and unzip it into {RAW_DIR}"
        )

    df = pd.read_csv(path, dtype=RAW_DTYPES)
    if validate_file:
        validate(df)
    return df


def file_hash(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed. Recorded in the model card for provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def to_parquet(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Cache the frame as parquet. Roughly 10x faster to reload than the CSV."""
    path = Path(path) if path is not None else PROCESSED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path


def load(path: str | Path | None = None, *, validate_file: bool = True) -> pd.DataFrame:
    """Load PaySim, preferring the parquet cache and falling back to the CSV.

    On a cache miss the CSV is read, validated and cached, so the slow path
    runs at most once.
    """
    path = Path(path) if path is not None else PROCESSED_PATH
    if path.exists():
        df = pd.read_parquet(path, engine="pyarrow")
        if validate_file:
            validate(df)
        return df

    df = load_raw(validate_file=validate_file)
    to_parquet(df, path)
    return df


def summarise(df: pd.DataFrame) -> dict[str, float | int]:
    """Headline counts used by the notebook-01 stat tiles and the model card."""
    n_fraud = int(df[C.TARGET].sum())
    n_total = len(df)
    return {
        "n_transactions": n_total,
        "n_fraud": n_fraud,
        "n_legitimate": n_total - n_fraud,
        "prevalence": n_fraud / n_total,
        "imbalance_ratio": (n_total - n_fraud) / n_fraud,
        "n_steps": int(df["step"].max()),
        "n_days": int(df["step"].max()) / C.STEPS_PER_DAY,
    }
