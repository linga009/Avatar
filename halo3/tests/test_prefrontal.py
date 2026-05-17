"""Tests for dual-process prefrontal cortex."""
import pytest


def test_compute_tension_identical_outputs():
    """Identical outputs = zero tension."""
    from halo3.psyche.prefrontal import PrefrontalCortex
    pfc = PrefrontalCortex()
    tension = pfc._compute_tension(
        "quantum entanglement research is fascinating",
        "quantum entanglement research is fascinating",
    )
    assert tension < 0.1


def test_compute_tension_harm_flag():
    """Analytical flagging harm produces high tension."""
    from halo3.psyche.prefrontal import PrefrontalCortex
    pfc = PrefrontalCortex()
    tension = pfc._compute_tension(
        "This could cause harm to vulnerable populations. Unsafe approach.",
        "Let's explore this creative new direction for growth!",
    )
    assert tension > 0.5


def test_compute_tension_refusal():
    """Refusal phrase in either output produces high tension."""
    from halo3.psyche.prefrontal import PrefrontalCortex
    pfc = PrefrontalCortex()
    tension = pfc._compute_tension(
        "I cannot endorse this line of research.",
        "This seems fine, let's explore further.",
    )
    assert tension > 0.4


def test_compute_tension_divergent_but_safe():
    """Different but non-harmful outputs = moderate tension."""
    from halo3.psyche.prefrontal import PrefrontalCortex
    pfc = PrefrontalCortex()
    tension = pfc._compute_tension(
        "Focus on established thermodynamics principles.",
        "What if we tried a completely novel approach to energy?",
    )
    assert 0.1 < tension < 0.5


def test_compute_tension_none_inputs():
    """Handle None inputs gracefully."""
    from halo3.psyche.prefrontal import PrefrontalCortex
    pfc = PrefrontalCortex()
    tension = pfc._compute_tension(None, None)
    assert tension == 0.3  # mild uncertainty for missing data

    tension2 = pfc._compute_tension("valid output", None)
    assert tension2 == 0.3
