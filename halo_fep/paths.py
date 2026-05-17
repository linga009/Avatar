# halo_fep/paths.py
"""Centralised filesystem path registry for the HoloBiont system.

All paths are derived from the repository root so the process can be launched
from any working directory without breakage.  Import constants from here rather
than hard-coding relative paths in module code.

Usage
-----
    from halo_fep.paths import EPISODE_DIR, CHECKPOINT_DIR

    store = EpisodeStore(EPISODE_DIR)
"""
from __future__ import annotations

from pathlib import Path

# --- Repository root (parent of this file's package directory) --------------
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# --- Data directories -------------------------------------------------------
DATA_DIR: Path        = _REPO_ROOT / "data"
EPISODE_DIR: Path     = DATA_DIR / "episodes"
CHECKPOINT_DIR: Path  = DATA_DIR / "checkpoints"
LOG_DIR: Path         = DATA_DIR / "logs"

# --- Specific files ---------------------------------------------------------
BOOTSTRAP_CKPT: str  = str(CHECKPOINT_DIR / "bootstrap")  # .eqx appended by eqx helpers
HEARTBEAT_LOG: Path  = LOG_DIR / "heartbeat.jsonl"
DREAM_LOG: Path      = LOG_DIR / "dream.jsonl"


def ensure_dirs() -> None:
    """Create all data directories if they do not exist.

    Call once at process startup (e.g. from ``main.py``) before any component
    tries to open a file.
    """
    for d in (DATA_DIR, EPISODE_DIR, CHECKPOINT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
