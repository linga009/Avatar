from unittest.mock import patch, MagicMock
from halo_fep.intellect.llm_bridge import LLMBridge, LLMResponse, parse_llm_output


def test_parse_search():
    r = parse_llm_output("SEARCH: active inference tutorial")
    assert r.action == "SEARCH"
    assert r.content == "active inference tutorial"


def test_parse_goal():
    r = parse_llm_output("GOAL: understand consciousness")
    assert r.action == "GOAL"
    assert r.content == "understand consciousness"


def test_parse_idle():
    r = parse_llm_output("IDLE")
    assert r.action == "IDLE"
    assert r.content == ""


def test_parse_learn():
    r = parse_llm_output("LEARN: free energy is minimized by prediction")
    assert r.action == "LEARN"


def test_parse_unknown_defaults_idle():
    r = parse_llm_output("random garbage response")
    assert r.action == "IDLE"


def test_bridge_not_loaded_initially():
    bridge = LLMBridge()
    assert not bridge.is_loaded


def test_bridge_think_raises_when_not_loaded():
    bridge = LLMBridge()
    try:
        bridge.think("test prompt")
        assert False, "Should have raised"
    except RuntimeError:
        pass
