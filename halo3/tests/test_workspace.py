"""Tests for Global Workspace ignition and broadcast."""
import pytest
from halo3.psyche.workspace import GlobalWorkspace


def test_ignition_at_r_threshold():
    """r >= 0.5 from dark state -> ignited. Hysteresis sustains down to 0.4."""
    ws = GlobalWorkspace()
    # r=0.55 above threshold -> ignite
    r1 = ws.update(r_mean=0.55, current_topic="test", emotion="curiosity")
    assert r1["is_ignited"] is True
    assert r1["just_ignited"] is True

    # r=0.45 still above sustain threshold (0.4) -> stays ignited
    r2 = ws.update(r_mean=0.45, current_topic="test", emotion="curiosity")
    assert r2["is_ignited"] is True
    assert r2["just_ignited"] is False

    # r=0.35 below sustain threshold -> dark
    r3 = ws.update(r_mean=0.35, current_topic="test", emotion="curiosity")
    assert r3["is_ignited"] is False
    assert r3["just_darkened"] is True


def test_ignition_below_threshold():
    """r=0.45 from dark state -> stays dark (below 0.5 threshold)."""
    ws = GlobalWorkspace()
    r1 = ws.update(r_mean=0.45, current_topic="test", emotion="curiosity")
    assert r1["is_ignited"] is False
    assert r1["just_ignited"] is False


def test_sensory_novelty_tips_ignition():
    """r=0.48 alone won't ignite, but +0.05*0.9 sensory novelty tips it over."""
    ws = GlobalWorkspace()
    # Without novelty: 0.48 < 0.5 -> dark
    r1 = ws.update(r_mean=0.48, current_topic="test", emotion="curiosity",
                   sensory_novelty=0.0)
    assert r1["is_ignited"] is False

    # New workspace - with novelty: 0.48 + 0.05*0.9 = 0.525 >= 0.5 -> ignited
    ws2 = GlobalWorkspace()
    r2 = ws2.update(r_mean=0.48, current_topic="test", emotion="curiosity",
                    sensory_novelty=0.9)
    assert r2["is_ignited"] is True


def test_unity_scales_broadcast_intensity():
    """High unity -> higher broadcast intensity than low unity at same r."""
    ws_high = GlobalWorkspace()
    r_high = ws_high.update(r_mean=0.7, current_topic="test", emotion="curiosity",
                            unity=0.9)

    ws_low = GlobalWorkspace()
    r_low = ws_low.update(r_mean=0.7, current_topic="test", emotion="curiosity",
                          unity=0.2)

    assert r_high["broadcast_intensity"] > r_low["broadcast_intensity"]
    # High unity: 0.7 * (0.5 + 0.5*0.9) = 0.7 * 0.95 = 0.665
    assert abs(r_high["broadcast_intensity"] - 0.665) < 0.01
    # Low unity: 0.7 * (0.5 + 0.5*0.2) = 0.7 * 0.6 = 0.42
    assert abs(r_low["broadcast_intensity"] - 0.42) < 0.01


def test_transition_sharpness_from_chi():
    """High chi before ignition -> high transition_sharpness; low chi -> low."""
    # Sharp transition: feed high chi, then cross threshold
    ws = GlobalWorkspace()
    for _ in range(3):
        ws.update(r_mean=0.3, current_topic="test", emotion="curiosity",
                  chi_norm=0.6)
    r1 = ws.update(r_mean=0.55, current_topic="test", emotion="curiosity",
                   chi_norm=0.5)
    assert r1["just_ignited"] is True
    assert r1["transition_sharpness"] >= 0.5  # max of recent chi was 0.6

    # Quiet transition: feed low chi, then cross threshold
    ws2 = GlobalWorkspace()
    for _ in range(3):
        ws2.update(r_mean=0.3, current_topic="test", emotion="curiosity",
                   chi_norm=0.1)
    r2 = ws2.update(r_mean=0.55, current_topic="test", emotion="curiosity",
                    chi_norm=0.15)
    assert r2["just_ignited"] is True
    assert r2["transition_sharpness"] < 0.2


def test_hysteresis_prevents_flicker():
    """r oscillating 0.45-0.55 -> ignites once, stays ignited throughout."""
    ws = GlobalWorkspace()
    # First cross: ignite
    ws.update(r_mean=0.55, current_topic="test", emotion="curiosity")
    assert ws.is_ignited is True

    # Oscillate: 0.45, 0.55, 0.45, 0.55 - all above sustain (0.4)
    for r in [0.45, 0.55, 0.45, 0.55]:
        ws.update(r_mean=r, current_topic="test", emotion="curiosity")
        assert ws.is_ignited is True, f"Should stay ignited at r={r}"
