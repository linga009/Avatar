"""Tests for COP-derived emotions."""
from halo3.psyche.emotions import EmotionState


def test_emotion_returns_tuple():
    es = EmotionState()
    emotion, _, intensity = es.update(r_mean=0.5, fe_delta=-0.01, chi_norm=0.5)
    assert isinstance(emotion, str)
    assert isinstance(intensity, float)


def test_emotion_labels_valid():
    es = EmotionState()
    valid = {"satisfaction", "pride", "curiosity", "boredom", "anxiety", "frustration"}
    for r in [0.2, 0.4, 0.5, 0.6, 0.8]:
        for chi in [0.1, 0.5, 0.9]:
            emotion, _, _ = es.update(r_mean=r, fe_delta=-0.01, chi_norm=chi)
            assert emotion in valid, f"Invalid emotion '{emotion}' for r={r}, chi={chi}"


def test_high_r_low_chi_resolving_is_satisfaction():
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2)
    emotion, _, _ = es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2)
    assert emotion == "satisfaction"


def test_high_r_high_chi_resolving_is_pride():
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.6)
    emotion, _, _ = es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.6)
    assert emotion == "pride"


def test_mid_r_is_curiosity():
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7)
    emotion, _, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7)
    assert emotion == "curiosity"


def test_frustration_overrides():
    es = EmotionState()
    emotion, _qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
                                       consecutive_failures=5)
    assert emotion == "frustration"


def test_intensity_range():
    es = EmotionState()
    for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        _, _, intensity = es.update(r_mean=r, fe_delta=-0.01, chi_norm=0.5)
        assert 0.0 <= intensity <= 1.0


def test_sensory_novelty_amplifies():
    es = EmotionState()
    _, _, i1 = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, sensory_novelty=0.0)
    es2 = EmotionState()
    _, _, i2 = es2.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, sensory_novelty=0.9)
    assert i2 >= i1 - 0.1


# --- New tests for qualifier, mood, body_event ---

def test_emotion_returns_triple():
    """update() must return (emotion, qualifier, intensity) — a 3-tuple."""
    es = EmotionState()
    result = es.update(r_mean=0.5, fe_delta=-0.01, chi_norm=0.5)
    assert len(result) == 3
    emotion, qualifier, intensity = result
    assert isinstance(emotion, str)
    assert isinstance(qualifier, str)
    assert isinstance(intensity, float)


def test_curiosity_qualifier_burning():
    """chi>0.6 and dF_dt<-50 → qualifier 'burning'."""
    es = EmotionState()
    # Drive to curiosity region: mid r, high chi
    for _ in range(5):
        es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7)
    emotion, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7,
                                      dF_dt=-100.0)
    assert emotion == "curiosity"
    assert qualifier == "burning"


def test_curiosity_qualifier_restless():
    """chi<0.3 → qualifier 'restless'."""
    es = EmotionState()
    # curiosity with low chi: mid r, low chi
    for _ in range(5):
        es.update(r_mean=0.45, fe_delta=0.0, chi_norm=0.2)
    emotion, qualifier, _ = es.update(r_mean=0.45, fe_delta=0.0, chi_norm=0.2)
    assert emotion == "curiosity"
    assert qualifier == "restless"


def test_satisfaction_qualifier_deep():
    """unity>0.7 → satisfaction qualifier 'deep'."""
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2)
    emotion, qualifier, _ = es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2,
                                      unity=0.8)
    assert emotion == "satisfaction"
    assert qualifier == "deep"


def test_mood_from_ignition():
    """is_ignited=True (but not just_ignited) → mood 'clarity'."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, is_ignited=True)
    assert es.mood == "clarity"


def test_mood_threshold():
    """Not ignited, chi>0.4 → mood 'threshold'."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, is_ignited=False,
              just_ignited=False)
    assert es.mood == "threshold"


def test_body_event_release():
    """avalanche_just_ended=True → body_event 'release'."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
              avalanche_just_ended=True)
    assert es.body_event == "release"


def test_body_event_clears():
    """body_event set on tick N is cleared (empty) on tick N+1 without trigger."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, avalanche_just_ended=True)
    assert es.body_event == "release"
    # Next tick — no trigger
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5)
    assert es.body_event == ""


def test_frustration_qualifier_growing():
    """dF_dt<0 → frustration qualifier 'growing'."""
    es = EmotionState()
    emotion, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
                                      consecutive_failures=5, dF_dt=-10.0)
    assert emotion == "frustration"
    assert qualifier == "growing"


def test_frustration_qualifier_futile():
    """dF_dt>=0 → frustration qualifier 'futile'."""
    es = EmotionState()
    emotion, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
                                      consecutive_failures=5, dF_dt=5.0)
    assert emotion == "frustration"
    assert qualifier == "futile"
