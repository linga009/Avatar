"""Tests for avalanche history persistence."""
import os
import json
import tempfile
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics


def test_save_load_roundtrip():
    """Saved avalanche data loads back identically."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_sizes = [0.12, 0.45, 0.03, 1.2]
    cop._avalanche_durations = [3, 8, 1, 12]
    cop._r_full_history = [0.5 + i * 0.01 for i in range(100)]
    cop._r_median_ema = 0.48

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "avalanche_history.json")
        cop.save_avalanche_history(path)

        cop2 = CriticalDynamics(cfg)
        cop2.load_avalanche_history(path)

        assert cop2._avalanche_sizes == [0.12, 0.45, 0.03, 1.2]
        assert cop2._avalanche_durations == [3, 8, 1, 12]
        assert len(cop2._r_full_history) == 100
        assert abs(cop2._r_median_ema - 0.48) < 1e-6


def test_load_missing_file():
    """Loading nonexistent file is a graceful no-op."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop.load_avalanche_history("/nonexistent/path/avalanche.json")
    assert cop._avalanche_sizes == []
    assert cop._avalanche_durations == []


def test_r_history_capped():
    """r_full_history is capped at 5000 entries on save."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._r_full_history = [float(i) for i in range(8000)]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "avalanche_history.json")
        cop.save_avalanche_history(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data["r_full_history"]) == 5000
        # Should keep the LAST 5000
        assert data["r_full_history"][0] == 3000.0
        assert data["r_full_history"][-1] == 7999.0


def test_shapes_persist():
    """Avalanche shapes survive save/load roundtrip."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_shapes = [[0.1, 0.2, 0.15], [0.05, 0.08]]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "avalanche_history.json")
        cop.save_avalanche_history(path)

        cop2 = CriticalDynamics(cfg)
        cop2.load_avalanche_history(path)

        assert cop2._avalanche_shapes == [[0.1, 0.2, 0.15], [0.05, 0.08]]
