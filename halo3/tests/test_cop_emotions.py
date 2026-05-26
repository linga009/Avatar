"""Tests for COP-derived emotions."""
from halo3.psyche.emotions import EmotionState


def test_emotion_returns_tuple():
    es = EmotionState()
    emotion, intensity = es.update(r_mean=0.5, fe_delta=-0.01, chi_norm=0.5)
    assert isinstance(emotion, str)
    assert isinstance(intensity, float)


def test_emotion_labels_valid():
    es = EmotionState()
    valid = {"satisfaction", "pride", "curiosity", "boredom", "anxiety", "frustration"}
    for r in [0.2, 0.4, 0.5, 0.6, 0.8]:
        for chi in [0.1, 0.5, 0.9]:
            emotion, _ = es.update(r_mean=r, fe_delta=-0.01, chi_norm=chi)
            assert emotion in valid, f"Invalid emotion '{emotion}' for r={r}, chi={chi}"


def test_high_r_low_chi_resolving_is_satisfaction():
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2)
    emotion, _ = es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2)
    assert emotion == "satisfaction"


def test_high_r_high_chi_resolving_is_pride():
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.6)
    emotion, _ = es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.6)
    assert emotion == "pride"


def test_mid_r_is_curiosity():
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7)
    emotion, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7)
    assert emotion == "curiosity"


def test_frustration_overrides():
    es = EmotionState()
    emotion, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
                           consecutive_failures=5)
    assert emotion == "frustration"


def test_intensity_range():
    es = EmotionState()
    for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        _, intensity = es.update(r_mean=r, fe_delta=-0.01, chi_norm=0.5)
        assert 0.0 <= intensity <= 1.0


def test_sensory_novelty_amplifies():
    es = EmotionState()
    _, i1 = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, sensory_novelty=0.0)
    es2 = EmotionState()
    _, i2 = es2.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, sensory_novelty=0.9)
    assert i2 >= i1 - 0.1
