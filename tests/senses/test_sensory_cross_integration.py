"""Tests for sensory cross-integration into drives, emotions, consciousness."""
from halo3.psyche.drives import DriveState


def test_sensory_arousal_increases_fatigue():
    d = DriveState()
    d.update(r_mean=0.5, fe_delta=0.0, sensory_arousal=0.0)
    fatigue_low = d.fatigue
    d2 = DriveState()
    d2.update(r_mean=0.5, fe_delta=0.0, sensory_arousal=1.0)
    fatigue_high = d2.fatigue
    assert fatigue_high > fatigue_low, "High sensory arousal should increase fatigue"


def test_sensory_novelty_boosts_curiosity():
    # Use r_mean=0.3 so base curiosity is low (not capped at 1.0)
    d = DriveState()
    d.update(r_mean=0.3, fe_delta=0.0, sensory_novelty=0.0)
    cur_low = d.curiosity
    d2 = DriveState()
    d2.update(r_mean=0.3, fe_delta=0.0, sensory_novelty=0.9)
    cur_high = d2.curiosity
    assert cur_high > cur_low, "High sensory novelty should boost curiosity"


def test_sensory_arousal_dampens_starvation():
    # Start with high starvation, no perception failure so starvation decays naturally
    d = DriveState(starvation=0.8)
    d.update(r_mean=0.5, fe_delta=0.0, perception_failed=False, sensory_arousal=0.5)
    # Without sensory arousal: starvation = 0.8 - 0.3 = 0.5
    # With sensory arousal > 0.3: additional -0.1 → 0.4
    assert d.starvation < 0.5, "Sensory arousal should dampen starvation further"


from halo3.psyche.emotions import EmotionState


def test_sensory_novelty_amplifies_surprise():
    e = EmotionState()
    for _ in range(5):
        e.update(r_mean=0.5, fe_delta=0.1)
    e2 = EmotionState()
    for _ in range(5):
        e2.update(r_mean=0.5, fe_delta=0.1)
    _, i_low = e.update(r_mean=0.4, fe_delta=0.5, sensory_novelty=0.0)
    _, i_high = e2.update(r_mean=0.4, fe_delta=0.5, sensory_novelty=0.95)
    assert i_high >= i_low, "High sensory novelty should amplify intensity"


def test_speech_detected_nudges_valence():
    e = EmotionState()
    for _ in range(3):
        e.update(r_mean=0.5, fe_delta=0.0)
    v_before = e._valence
    e.update(r_mean=0.5, fe_delta=0.0, speech_detected=True)
    assert e._valence >= v_before, "Speech detection should nudge valence positive"
