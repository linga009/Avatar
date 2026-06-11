"""Tests for the cerebellum forward model."""
import math
import numpy as np
import pytest
from halo3.config import Halo3Config
from halo3.cerebellum import (
    FastPredictor, LearnedPredictor, ExperienceBuffer, Cerebellum,
    CerebellumState, Prediction, encode_action,
    ACTION_STAY, ACTION_EXPLORE, ACTION_EXPLOIT, ACTION_REST,
    N_INPUT_DIMS, N_OUTPUT_DIMS, N_STATE_DIMS, ACTION_DIM,
)


# ── FastPredictor ────────────────────────────────────────────────────────

class TestFastPredictor:
    def test_predict_r_returns_trajectory(self):
        fp = FastPredictor()
        traj = fp.predict_r(0.5, K=0.3, omega_std=0.03, n_ticks=5)
        assert len(traj) == 6  # initial + 5 steps
        assert all(0 < r < 1 for r in traj)

    def test_predict_r_stays_bounded(self):
        fp = FastPredictor()
        # Very high coupling — should not exceed 1.0
        traj = fp.predict_r(0.9, K=5.0, omega_std=0.03, n_ticks=20)
        assert all(0 < r < 1 for r in traj)
        # Very low coupling
        traj = fp.predict_r(0.1, K=0.001, omega_std=0.3, n_ticks=20)
        assert all(0 < r < 1 for r in traj)

    def test_high_K_increases_r(self):
        fp = FastPredictor()
        traj = fp.predict_r(0.3, K=1.0, omega_std=0.03, n_ticks=10)
        # With K well above K_c, r should increase
        assert traj[-1] > traj[0]

    def test_low_K_decreases_r(self):
        fp = FastPredictor()
        # K below K_c ≈ 0.048 for omega_std=0.03
        traj = fp.predict_r(0.5, K=0.01, omega_std=0.03, n_ticks=10)
        # Should decay toward 0
        assert traj[-1] < traj[0]

    def test_predict_chi_from_trajectory(self):
        fp = FastPredictor()
        # Varying trajectory should have positive chi
        traj = [0.3, 0.35, 0.4, 0.45, 0.5, 0.48]
        chi = fp.predict_chi(traj)
        assert chi > 0

    def test_predict_chi_constant_trajectory_zero(self):
        fp = FastPredictor()
        traj = [0.5, 0.5, 0.5, 0.5, 0.5]
        chi = fp.predict_chi(traj)
        assert chi == 0.0

    def test_predict_chi_short_trajectory(self):
        fp = FastPredictor()
        assert fp.predict_chi([0.5]) == 0.0
        assert fp.predict_chi([0.5, 0.6]) == 0.0

    def test_full_prediction(self):
        fp = FastPredictor()
        state = CerebellumState(r=0.4, r_a=0.3, r_c=0.5)
        action = encode_action(ACTION_EXPLORE)
        pred = fp.predict(state, action, horizon=5)
        assert isinstance(pred, Prediction)
        assert len(pred.r_trajectory) == 6
        assert pred.confidence == 1.0


# ── LearnedPredictor ─────────────────────────────────────────────────────

class TestLearnedPredictor:
    def test_init_shapes(self):
        lp = LearnedPredictor()
        assert lp.W1.shape == (N_INPUT_DIMS, 64)
        assert lp.W2.shape == (64, 64)
        assert lp.W3.shape == (64, N_OUTPUT_DIMS)

    def test_predict_returns_prediction(self):
        lp = LearnedPredictor()
        state = CerebellumState()
        action = encode_action(ACTION_STAY)
        pred = lp.predict(state, action)
        assert isinstance(pred, Prediction)
        assert 0 <= pred.avalanche_prob <= 1  # sigmoid

    def test_train_reduces_loss(self):
        np.random.seed(42)
        lp = LearnedPredictor(lr=0.01)
        # Generate synthetic data
        B = 64
        states = np.random.randn(B, N_STATE_DIMS).astype(np.float32) * 0.1
        actions = np.zeros((B, ACTION_DIM), dtype=np.float32)
        actions[:, ACTION_STAY] = 1.0
        # Target: delta_r = 0.1 * r (simple linear relationship)
        targets = np.zeros((B, N_OUTPUT_DIMS), dtype=np.float32)
        targets[:, 0] = states[:, 0] * 0.1

        loss_first = lp.train_batch(states, actions, targets)
        for _ in range(20):
            loss = lp.train_batch(states, actions, targets)
        assert loss < loss_first  # should decrease

    def test_save_load_roundtrip(self, tmp_path):
        lp = LearnedPredictor()
        lp.n_training_samples = 42
        path = str(tmp_path / "test_mlp.npz")
        lp.save(path)

        lp2 = LearnedPredictor()
        assert lp2.load(path)
        assert lp2.n_training_samples == 42
        np.testing.assert_array_equal(lp.W1, lp2.W1)

    def test_load_missing_returns_false(self):
        lp = LearnedPredictor()
        assert not lp.load("/nonexistent/path.npz")


# ── ExperienceBuffer ─────────────────────────────────────────────────────

class TestExperienceBuffer:
    def test_empty_buffer(self):
        buf = ExperienceBuffer(capacity=100, horizon=3)
        assert len(buf) == 0
        assert buf.sample_batch(32) is None

    def test_transitions_complete_after_horizon(self):
        buf = ExperienceBuffer(capacity=100, horizon=3)
        state = CerebellumState(r=0.5)
        action = encode_action(ACTION_STAY)

        # Record 4 ticks — first transition completes at tick 3
        for i in range(4):
            s = CerebellumState(r=0.5 + i * 0.05)
            buf.record(s, action)

        assert len(buf) >= 1

    def test_capacity_limit(self):
        buf = ExperienceBuffer(capacity=10, horizon=1)
        state = CerebellumState()
        action = encode_action(ACTION_STAY)
        for _ in range(50):
            buf.record(state, action)
        assert len(buf) <= 10

    def test_sample_batch_shapes(self):
        buf = ExperienceBuffer(capacity=100, horizon=1)
        action = encode_action(ACTION_STAY)
        for i in range(50):
            s = CerebellumState(r=0.3 + i * 0.01)
            buf.record(s, action)

        batch = buf.sample_batch(16)
        assert batch is not None
        states, actions, targets = batch
        assert states.shape == (16, N_STATE_DIMS)
        assert actions.shape == (16, ACTION_DIM)
        assert targets.shape == (16, N_OUTPUT_DIMS)


# ── Action encoding ──────────────────────────────────────────────────────

class TestActionEncoding:
    def test_one_hot(self):
        for i in range(4):
            vec = encode_action(i)
            assert vec[i] == 1.0
            assert sum(vec[:4]) == 1.0

    def test_k_delta(self):
        vec = encode_action(ACTION_STAY, k_delta=(0.1, -0.2, 0.05))
        assert vec[4] == pytest.approx(0.1)
        assert vec[5] == pytest.approx(-0.2)
        assert vec[6] == pytest.approx(0.05)

    def test_action_dim(self):
        vec = encode_action(ACTION_EXPLORE)
        assert len(vec) == ACTION_DIM


# ── CerebellumState ─────────────────────────────────────────────────────

class TestCerebellumState:
    def test_to_array_roundtrip(self):
        s = CerebellumState(r=0.42, chi=0.7, hunger=0.8)
        arr = s.to_array()
        assert len(arr) == N_STATE_DIMS
        s2 = CerebellumState.from_array(arr)
        assert s2.r == pytest.approx(0.42)
        assert s2.chi == pytest.approx(0.7)
        assert s2.hunger == pytest.approx(0.8)


# ── Cerebellum (facade) ─────────────────────────────────────────────────

class TestCerebellum:
    def test_confidence_starts_low(self):
        cfg = Halo3Config(enable_cerebellum=True)
        cb = Cerebellum(cfg)
        cb.learned.n_training_samples = 0  # ensure fresh
        assert cb.confidence < 0.3

    def test_confidence_increases_with_samples(self):
        cfg = Halo3Config(enable_cerebellum=True, cerebellum_confidence_scale=100.0)
        cb = Cerebellum(cfg)
        cb.learned.n_training_samples = 0
        c0 = cb.confidence
        cb.learned.n_training_samples = 200
        c200 = cb.confidence
        assert c200 > c0

    def test_predict_blends(self):
        cfg = Halo3Config(enable_cerebellum=True, cerebellum_confidence_scale=100.0)
        cb = Cerebellum(cfg)
        cb.learned.n_training_samples = 0  # ensure fresh
        state = CerebellumState(r=0.4, r_a=0.3, r_c=0.5)
        action = encode_action(ACTION_EXPLORE)

        # With no training, should return fast prediction
        pred = cb.predict(state, action)
        assert pred.confidence < 0.05  # no samples = near-zero confidence
        assert len(pred.r_trajectory) > 0

    def test_record_builds_buffer(self):
        cfg = Halo3Config(enable_cerebellum=True)
        cb = Cerebellum(cfg)
        action = encode_action(ACTION_STAY)
        for i in range(10):
            s = CerebellumState(r=0.4 + i * 0.02)
            cb.record(s, action)
        assert len(cb.buffer) > 0

    def test_train_with_sufficient_data(self):
        cfg = Halo3Config(enable_cerebellum=True, cerebellum_train_steps=5)
        cb = Cerebellum(cfg)
        action = encode_action(ACTION_STAY)
        for i in range(100):
            s = CerebellumState(r=0.3 + 0.004 * i, chi=0.1 * (i % 10))
            cb.record(s, action)
        loss = cb.train(n_steps=5, batch_size=16)
        assert loss is not None
        assert loss >= 0

    def test_predict_soc_trajectory(self):
        cfg = Halo3Config(enable_cerebellum=True)
        cb = Cerebellum(cfg)
        traj = cb.predict_soc_trajectory(r_now=0.4, K_proposed=0.15, omega_std=0.03)
        assert len(traj) == 6
        assert all(0 < r < 1 for r in traj)
