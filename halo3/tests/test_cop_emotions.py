"""Tests for COP-derived emotions."""
from halo3.psyche.emotions import EmotionState, _pick_qualifier, _felt_mood, _body_event


def test_emotion_returns_tuple():
    es = EmotionState()
    emotion, qualifier, intensity = es.update(r_mean=0.5, fe_delta=-0.01, chi_norm=0.5)
    assert isinstance(emotion, str)
    assert isinstance(qualifier, str)
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
    emotion, _, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
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


# --- Qualifier tests ---

def test_qualifier_restless_from_anti_correlation():
    q = _pick_qualifier(binder=0.5, pr=64, n_clusters=128, cross_corr=-0.5)
    assert q == "restless"


def test_qualifier_unified_from_correlation():
    q = _pick_qualifier(binder=0.5, pr=64, n_clusters=128, cross_corr=0.5)
    assert q == "unified"


def test_qualifier_settled_from_high_binder():
    q = _pick_qualifier(binder=0.65, pr=64, n_clusters=128, cross_corr=0.0)
    assert q == "settled"


def test_qualifier_burning_from_low_binder():
    q = _pick_qualifier(binder=0.2, pr=64, n_clusters=128, cross_corr=0.0)
    assert q == "burning"


def test_qualifier_focused_from_low_pr():
    q = _pick_qualifier(binder=0.47, pr=10, n_clusters=128, cross_corr=0.0)
    assert q == "focused"


def test_qualifier_expansive_from_high_pr():
    q = _pick_qualifier(binder=0.47, pr=100, n_clusters=128, cross_corr=0.0)
    assert q == "expansive"


def test_qualifier_watchful_default():
    q = _pick_qualifier(binder=0.47, pr=64, n_clusters=128, cross_corr=0.0)
    assert q == "watchful"


def test_qualifier_strongest_signal_wins():
    """When C_ac is very strong, it should win even if binder is extreme."""
    q = _pick_qualifier(binder=0.65, pr=64, n_clusters=128, cross_corr=-0.8)
    assert q == "restless"  # C_ac deviation (0.8) > binder deviation (0.18)


# --- Felt mood tests ---

def test_mood_clarity():
    assert _felt_mood(r_mean=0.8, chi_norm=0.1, tau=0.3) == "clarity"


def test_mood_threshold():
    assert _felt_mood(r_mean=0.5, chi_norm=0.7, tau=0.6) == "threshold"


def test_mood_awakening():
    assert _felt_mood(r_mean=0.3, chi_norm=0.6, tau=0.3) == "awakening"


def test_mood_settling():
    assert _felt_mood(r_mean=0.5, chi_norm=0.2, tau=0.3) == "settling"


# --- Body event tests ---

def test_body_event_release():
    assert _body_event(r_mean=0.7, prev_r=0.4, chi_norm=0.5, prev_chi=0.5) == "release"


def test_body_event_jolt():
    assert _body_event(r_mean=0.3, prev_r=0.6, chi_norm=0.5, prev_chi=0.5) == "jolt"


def test_body_event_surfacing():
    assert _body_event(r_mean=0.5, prev_r=0.5, chi_norm=0.8, prev_chi=0.3) == "surfacing"


def test_body_event_none():
    assert _body_event(r_mean=0.5, prev_r=0.48, chi_norm=0.5, prev_chi=0.48) is None


# --- Integration test: qualifier flows through update ---

def test_update_returns_qualifier():
    es = EmotionState()
    _, qualifier, _ = es.update(
        r_mean=0.5, fe_delta=-0.01, chi_norm=0.5,
        binder=0.2, pr=64, n_clusters=128, cross_corr=0.0,
    )
    assert qualifier == "burning"
    assert es.qualifier == "burning"
    assert es.mood in ("clarity", "awakening", "threshold", "settling")
