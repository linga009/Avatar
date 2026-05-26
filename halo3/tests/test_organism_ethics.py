"""Tests for ethical tension integration in organism tick loop."""
import pytest
from unittest.mock import patch, MagicMock


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_ethical_tension_in_result(_mock):
    """Ethical tension should appear in tick result."""
    from halo3.psyche.organism import Organism

    org = Organism(seed_topics=["quantum physics", "biology", "ethics"])
    org.prefrontal._ollama_available = False  # avoid network calls

    result = org.tick(
        r_mean=0.5, fe_delta=-0.1,
        texts=["some research finding"],
        current_query="quantum physics",
        carry_norm=1.0,
    )

    assert "ethical_tension" in result
    assert isinstance(result["ethical_tension"], float)


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_high_tension_amplifies_intensity(_mock):
    """High ethical tension should increase emotional intensity."""
    from halo3.psyche.organism import Organism

    org = Organism(seed_topics=["quantum physics", "biology"])
    org.prefrontal._ollama_available = False

    # Baseline tick
    result1 = org.tick(
        r_mean=0.5, fe_delta=-0.1,
        texts=["normal finding"],
        current_query="quantum physics",
        carry_norm=1.0,
    )
    base_intensity = result1["intensity"]

    # Now set high tension
    org.prefrontal._ethical_tension = 0.8

    result2 = org.tick(
        r_mean=0.5, fe_delta=-0.1,
        texts=["normal finding"],
        current_query="quantum physics",
        carry_norm=1.0,
    )

    # Intensity should be higher with ethical tension
    assert result2["ethical_tension"] == 0.8
    assert result2["intensity"] >= base_intensity


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_high_tension_reported(_mock):
    """High ethical tension should be present in result with reasonable intensity."""
    from halo3.psyche.organism import Organism

    org = Organism(seed_topics=["dangerous topic", "biology"])
    org.prefrontal._ollama_available = False
    org.prefrontal._ethical_tension = 0.7

    result = org.tick(
        r_mean=0.5, fe_delta=0.0,
        texts=["some content"],
        current_query="dangerous topic",
        carry_norm=1.0,
    )

    # COP handles affect natively — tension is reported but no longer
    # overrides emotion. Check that ethical_tension is present and
    # intensity is reasonable.
    assert result["ethical_tension"] == 0.7
    assert 0.0 <= result["intensity"] <= 1.0


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_zero_tension_no_interference(_mock):
    """Zero ethical tension should not change behavior."""
    from halo3.psyche.organism import Organism

    org = Organism(seed_topics=["quantum physics", "biology"])
    org.prefrontal._ollama_available = False
    org.prefrontal._ethical_tension = 0.0

    # Use r_mean <= 0.6 to avoid triggering dialectic (which overwrites tension)
    result = org.tick(
        r_mean=0.5, fe_delta=-0.2,
        texts=["great finding about photosynthesis"],
        current_query="biology",
        carry_norm=1.0,
    )

    assert result["ethical_tension"] == 0.0
    # Normal behavior — should not be forced to anxiety
    assert result["emotion"] != "anxiety" or result["intensity"] < 0.5
