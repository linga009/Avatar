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


def test_timeout_is_5_seconds():
    from halo3.psyche.prefrontal import TIMEOUT
    assert TIMEOUT == 5


def test_clean_query_rejects_prompt_scaffolding():
    """Prompt format strings like '### Instruction' must not leak as queries."""
    from halo3.psyche.prefrontal import _clean_query
    assert _clean_query("### Instruction") is None
    assert _clean_query("### Response:") is None
    assert _clean_query("## Some heading") is None
    assert _clean_query("# Title") is None
    # Valid queries pass through
    assert _clean_query("quantum entanglement experiments") is not None


def test_clean_query_rejects_meta_queries():
    """Meta-query descriptions must be rejected — these describe queries, not actual queries."""
    from halo3.psyche.prefrontal import _clean_query
    # Exact patterns from production logs (June 14-17 2026)
    assert _clean_query("web search query for resonance 0.45") is None
    assert _clean_query("The web search query for resonance 0.45 would") is None
    assert _clean_query("[web search query for resonance 0.45]") is None
    # Other meta-query variants
    assert _clean_query("a web search query about quantum computing") is None
    assert _clean_query("search query for tensor networks") is None
    assert _clean_query("the search query for this topic would be") is None
    assert _clean_query("I would search for quantum error correction") is None
    # Conditional/predictive language
    assert _clean_query("resonance experiments would be interesting") is None
    assert _clean_query("quantum computing should be explored") is None
    # "Just search for" meta pattern (from production tick 73)
    assert _clean_query("Just search for resonance 0.33") is None
    assert _clean_query("just search quantum computing") is None
    assert _clean_query("simply search for AI agents") is None
    assert _clean_query('Use the search query "Resonance 0.34" to find') is None
    # JSON array output (from production tick 72)
    assert _clean_query('["resonance 0.34 search query", "resonance 0."]') is None
    assert _clean_query('[{"query": "test"}]') is None
    # FineWeb text leak — sentence fragments (from production tick 79-90)
    assert _clean_query("Virginia has been a university English instructor") is None
    assert _clean_query("The algorithm is designed for parallel computing") is None
    assert _clean_query("Studies have shown significant improvements") is None
    assert _clean_query("Researchers were investigating the effects") is None
    assert _clean_query("This can be applied to many domains") is None
    # Valid queries still pass
    assert _clean_query("quantum error correction 2026") is not None
    assert _clean_query("tensor networks machine learning") is not None
    assert _clean_query("Bohmian mechanics pilot wave") is not None
