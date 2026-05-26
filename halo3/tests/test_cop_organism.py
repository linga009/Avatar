"""Tests for COP-wired organism."""
from unittest.mock import patch


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_organism_tick_returns_cop_fields(_mock):
    from halo3.psyche.organism import Organism
    org = Organism(seed_topics=["quantum physics", "biology"])
    org.prefrontal._ollama_available = False

    result = org.tick(
        r_mean=0.5, fe_delta=-0.01,
        texts=["some finding"], current_query="quantum physics",
        carry_norm=1.0,
    )

    assert "chi" in result
    assert "tau" in result
    assert "unity" in result
    assert "K" in result
    assert isinstance(result["chi"], float)
    assert isinstance(result["K"], float)


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_organism_coupling_mod_is_absolute_K(_mock):
    from halo3.psyche.organism import Organism
    org = Organism(seed_topics=["quantum physics"])
    org.prefrontal._ollama_available = False

    result = org.tick(
        r_mean=0.5, fe_delta=-0.01,
        texts=["finding"], current_query="quantum physics",
        carry_norm=1.0,
    )

    K = result["coupling_mod"]
    assert 0.01 <= K <= 3.0, f"coupling_mod={K} should be absolute K"


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_organism_emotion_still_valid(_mock):
    from halo3.psyche.organism import Organism
    org = Organism(seed_topics=["quantum physics"])
    org.prefrontal._ollama_available = False

    valid = {"satisfaction", "pride", "curiosity", "boredom", "anxiety", "frustration"}
    for _ in range(10):
        result = org.tick(
            r_mean=0.5, fe_delta=-0.01,
            texts=["finding"], current_query="quantum physics",
            carry_norm=1.0,
        )
        assert result["emotion"] in valid


@patch("halo3.psyche.prefrontal._call_ollama", return_value=None)
def test_organism_needs_dream(_mock):
    from halo3.psyche.organism import Organism
    org = Organism(seed_topics=["quantum physics"])
    org.prefrontal._ollama_available = False

    result = org.tick(
        r_mean=0.5, fe_delta=-0.01,
        texts=["finding"], current_query="quantum physics",
        carry_norm=1.0,
    )
    assert isinstance(result["needs_dream"], bool)
