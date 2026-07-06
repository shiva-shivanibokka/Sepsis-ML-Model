"""Central configuration: paths, columns, constants, Colab/local auto-detection.

The whole project reads its paths and column lists from here, so the *same*
notebooks and scripts run unchanged on Google Colab or a local machine.

- On Colab: mounts Google Drive and points at a Drive folder.
- Locally: uses the repository root, so a ``Sepsis_Dataset.csv`` placed in the
  repo is picked up automatically with no path editing.

Override any path with an environment variable (see below).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Column / label constants ------------------------------------------------
PATIENT_ID_COL: str = "Patient_ID"
RAW_TARGET_COL: str = "SepsisLabel"   # per-hour label in the raw time-series
TARGET_COL: str = "Sepsis"            # per-patient label after aggregation
CLASS_POS: str = "Sepsis"             # positive class
CLASS_NEG: str = "No sepsis"          # negative class

DEMOGRAPHIC_COLS: list[str] = ["Age", "Gender"]

# The 8 vital signs + 26 lab values aggregated per patient (34 total). Admin
# columns (Unit1, Unit2, HospAdmTime) and the raw ICULOS are handled separately.
VITAL_AND_LAB_COLS: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
# Base aggregations passed to pandas .agg(); the _range feature is derived after.
AGG_FUNCS: list[str] = ["min", "max", "mean", "std"]

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20

# --- File names --------------------------------------------------------------
RAW_CSV_NAME: str = "Sepsis_Dataset.csv"


def _running_in_colab() -> bool:
    """True if executing inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401  (import is the probe)

        return True
    except ImportError:
        return False


IN_COLAB: bool = _running_in_colab()


def _default_data_dir() -> Path:
    """Directory holding the raw dataset and receiving generated artifacts.

    Precedence:
      1. SEPSIS_DATA_DIR env var (explicit override, works everywhere)
      2. Colab -> the Drive folder below (mounted on demand)
      3. Local -> the repository root (two parents up from this file)
    """
    override = os.environ.get("SEPSIS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    if IN_COLAB:
        return Path("/content/drive/MyDrive/Colab Notebooks/Sepsis ML Project")

    # src/sepsis_icu/config.py -> repo root is two parents up.
    return Path(__file__).resolve().parents[2]


def mount_drive_if_colab() -> None:
    """Mount Google Drive when on Colab; no-op locally. Safe to call repeatedly."""
    if IN_COLAB:
        from google.colab import drive  # imported lazily so local runs never need it

        drive.mount("/content/drive")


# --- Resolved paths ----------------------------------------------------------
DATA_DIR: Path = _default_data_dir()
RAW_CSV: Path = DATA_DIR / RAW_CSV_NAME

# Generated intermediates and model artifacts live under artifacts/ so they
# never clutter the repo root and are easy to .gitignore.
ARTIFACTS_DIR: Path = Path(
    os.environ.get("SEPSIS_ARTIFACTS_DIR", DATA_DIR / "artifacts")
)

PATIENT_CSV: Path = ARTIFACTS_DIR / "patient_data.csv"  # one row per patient
X_TRAIN_CSV: Path = ARTIFACTS_DIR / "X_train.csv"
X_TEST_CSV: Path = ARTIFACTS_DIR / "X_test.csv"
Y_TRAIN_CSV: Path = ARTIFACTS_DIR / "y_train.csv"
Y_TEST_CSV: Path = ARTIFACTS_DIR / "y_test.csv"

MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
EXAMPLES_PATH: Path = ARTIFACTS_DIR / "examples.json"  # real samples for the demo UI


def ensure_artifacts_dir() -> Path:
    """Create the artifacts directory if needed and return it."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def describe() -> str:
    """Human-readable summary of the resolved environment — handy in notebooks."""
    return (
        f"Environment : {'Google Colab' if IN_COLAB else 'local'}\n"
        f"Data dir    : {DATA_DIR}\n"
        f"Raw CSV     : {RAW_CSV}  (exists: {RAW_CSV.exists()})\n"
        f"Artifacts   : {ARTIFACTS_DIR}"
    )
