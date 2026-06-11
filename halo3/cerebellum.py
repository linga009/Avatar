"""Cerebellum — forward model for action consequence prediction.

Two-layer architecture:
  Layer 1 (Fast): Linearized Kuramoto mean-field ODE. Zero cost, no training.
  Layer 2 (Slow): 5K-param MLP trained on experience during dreams.

The cerebellum predicts (delta_r, delta_chi, delta_hunger, delta_curiosity,
delta_satiation, avalanche_prob) for a proposed action, allowing the organism
to internally rehearse before committing.

All computation is pure Python/NumPy — zero VRAM.
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from halo3.config import Halo3Config

log = logging.getLogger(__name__)

# ── Action encoding ──────────────────────────────────────────────────────

ACTION_STAY = 0
ACTION_EXPLORE = 1
ACTION_EXPLOIT = 2
ACTION_REST = 3
N_ACTION_TYPES = 4
N_K_DELTA = 3  # aa, cc, cross
ACTION_DIM = N_ACTION_TYPES + N_K_DELTA  # 7


def encode_action(action_type: int, k_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Encode an action as a vector [one-hot(4) + k_delta(3)] = 7 dims."""
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    vec[action_type] = 1.0
    vec[N_ACTION_TYPES:] = k_delta
    return vec


# ── State representation ─────────────────────────────────────────────────

N_STATE_DIMS = 11  # r, r_a, r_c, chi, tau, K_aa, K_cc, K_cross, hunger, curiosity, satiation
N_INPUT_DIMS = N_STATE_DIMS + ACTION_DIM  # 18
N_OUTPUT_DIMS = 6  # delta_r, delta_chi, delta_hunger, delta_curiosity, delta_satiation, avalanche_prob


@dataclass
class CerebellumState:
    """Snapshot of relevant state for prediction."""
    r: float = 0.5
    r_a: float = 0.5
    r_c: float = 0.5
    chi: float = 0.0
    tau: float = 0.0
    K_aa: float = 0.10
    K_cc: float = 0.50
    K_cross: float = 0.15
    hunger: float = 0.5
    curiosity: float = 0.5
    satiation: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.r, self.r_a, self.r_c, self.chi, self.tau,
            self.K_aa, self.K_cc, self.K_cross,
            self.hunger, self.curiosity, self.satiation,
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> CerebellumState:
        return cls(
            r=float(arr[0]), r_a=float(arr[1]), r_c=float(arr[2]),
            chi=float(arr[3]), tau=float(arr[4]),
            K_aa=float(arr[5]), K_cc=float(arr[6]), K_cross=float(arr[7]),
            hunger=float(arr[8]), curiosity=float(arr[9]), satiation=float(arr[10]),
        )


@dataclass
class Prediction:
    """Output of the cerebellum forward model."""
    delta_r: float = 0.0
    delta_chi: float = 0.0
    delta_hunger: float = 0.0
    delta_curiosity: float = 0.0
    delta_satiation: float = 0.0
    avalanche_prob: float = 0.0
    r_trajectory: list[float] = field(default_factory=list)
    confidence: float = 0.0


# ── Layer 1: Fast Physics Predictor ──────────────────────────────────────

class FastPredictor:
    """Linearized Kuramoto mean-field forward model — no learned parameters."""

    def predict_r(self, r_now: float, K: float, omega_std: float,
                  n_ticks: int = 5, dt: float = 0.1) -> list[float]:
        """Predict r trajectory using Kuramoto mean-field relaxation.

        Near the critical point K_c = 2/(pi*g(0)):
        - K > K_c: r relaxes toward r_∞ = sqrt(1 - K_c/K)
        - K < K_c: r decays toward 0

        Relaxation rate ~ |K - K_c| (faster far from critical point).
        Uses simple exponential relaxation — stable for all parameters.
        """
        r = max(1e-6, min(1.0 - 1e-6, r_now))
        g0 = 1.0 / (omega_std * math.sqrt(2 * math.pi) + 1e-8)
        K_c = 2.0 / (math.pi * g0 + 1e-8)
        trajectory = [r]
        for _ in range(n_ticks):
            if K > K_c:
                r_inf = math.sqrt(max(0, 1 - K_c / K))
                rate = (K - K_c) * 0.5  # relaxation rate
            else:
                r_inf = 0.0
                rate = (K_c - K) * 0.5
            # Exponential relaxation: r -> r_inf
            dr = (r_inf - r) * (1 - math.exp(-rate * dt))
            r = max(1e-6, min(1.0 - 1e-6, r + dr))
            trajectory.append(r)
        return trajectory

    def predict_chi(self, r_trajectory: list[float], N: int = 8192) -> float:
        """Estimate chi from predicted r variance."""
        if len(r_trajectory) < 3:
            return 0.0
        arr = np.array(r_trajectory)
        var_r = float(np.var(arr))
        return min(1.0, N * var_r)

    def predict(self, state: CerebellumState, action: np.ndarray,
                horizon: int = 5) -> Prediction:
        """Full fast prediction from state + action."""
        # Determine which K to vary based on action type
        K_aa = state.K_aa + action[N_ACTION_TYPES]
        K_cc = state.K_cc + action[N_ACTION_TYPES + 1]

        # Predict analytical and creative populations separately
        traj_a = self.predict_r(state.r_a, K_aa, omega_std=0.03, n_ticks=horizon)
        traj_c = self.predict_r(state.r_c, K_cc, omega_std=0.30, n_ticks=horizon)
        # Mean r from both populations
        traj_mean = [(a + c) / 2 for a, c in zip(traj_a, traj_c)]

        chi_pred = self.predict_chi(traj_mean)
        delta_r = traj_mean[-1] - state.r

        return Prediction(
            delta_r=delta_r,
            delta_chi=chi_pred - state.chi,
            r_trajectory=traj_mean,
            confidence=1.0,  # physics model is always confident
        )


# ── Layer 2: Slow Learned Predictor (MLP) ────────────────────────────────

class LearnedPredictor:
    """Small MLP trained on experience — 5K params in NumPy."""

    def __init__(self, lr: float = 0.001):
        self.lr = lr
        # Xavier init
        scale1 = math.sqrt(2.0 / N_INPUT_DIMS)
        scale2 = math.sqrt(2.0 / 64)
        scale3 = math.sqrt(2.0 / 64)
        self.W1 = np.random.randn(N_INPUT_DIMS, 64).astype(np.float32) * scale1
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = np.random.randn(64, 64).astype(np.float32) * scale2
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = np.random.randn(64, N_OUTPUT_DIMS).astype(np.float32) * scale3
        self.b3 = np.zeros(N_OUTPUT_DIMS, dtype=np.float32)
        self.n_training_samples = 0

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return x * 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning intermediates for backprop."""
        z1 = x @ self.W1 + self.b1
        h1 = self._gelu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = self._gelu(z2)
        out = h2 @ self.W3 + self.b3
        return out, h2, h1, x

    def predict(self, state: CerebellumState, action: np.ndarray) -> Prediction:
        """Forward pass through MLP."""
        x = np.concatenate([state.to_array(), action])
        out, _, _, _ = self._forward(x)
        return Prediction(
            delta_r=float(out[0]),
            delta_chi=float(out[1]),
            delta_hunger=float(out[2]),
            delta_curiosity=float(out[3]),
            delta_satiation=float(out[4]),
            avalanche_prob=float(1.0 / (1.0 + math.exp(-float(out[5])))),  # sigmoid
        )

    def train_batch(self, states: np.ndarray, actions: np.ndarray,
                    targets: np.ndarray, l2_reg: float = 1e-4) -> float:
        """Train on a batch with MSE loss + L2 regularization.

        Args:
            states: (B, N_STATE_DIMS)
            actions: (B, ACTION_DIM)
            targets: (B, N_OUTPUT_DIMS)
            l2_reg: L2 regularization coefficient

        Returns:
            Mean loss for the batch.
        """
        B = states.shape[0]
        x = np.concatenate([states, actions], axis=1)  # (B, N_INPUT_DIMS)

        # Forward
        z1 = x @ self.W1 + self.b1
        h1 = self._gelu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = self._gelu(z2)
        out = h2 @ self.W3 + self.b3

        # MSE loss
        diff = out - targets  # (B, N_OUTPUT_DIMS)
        loss = float(np.mean(diff ** 2))

        # Backward (simplified — GELU approx gradient)
        d_out = 2.0 * diff / (B * N_OUTPUT_DIMS)

        # Layer 3
        dW3 = h2.T @ d_out + 2 * l2_reg * self.W3
        db3 = np.sum(d_out, axis=0)
        d_h2 = d_out @ self.W3.T

        # GELU derivative approximation: gelu'(x) ≈ sigmoid(1.702*x)
        sig2 = 1.0 / (1.0 + np.exp(-1.702 * z2))
        d_z2 = d_h2 * sig2

        # Layer 2
        dW2 = h1.T @ d_z2 + 2 * l2_reg * self.W2
        db2 = np.sum(d_z2, axis=0)
        d_h1 = d_z2 @ self.W2.T

        sig1 = 1.0 / (1.0 + np.exp(-1.702 * z1))
        d_z1 = d_h1 * sig1

        # Layer 1
        dW1 = x.T @ d_z1 + 2 * l2_reg * self.W1
        db1 = np.sum(d_z1, axis=0)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.n_training_samples += B
        return loss

    def save(self, path: str) -> None:
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 n_training_samples=np.array(self.n_training_samples))

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W3 = data["W3"]
        self.b3 = data["b3"]
        self.n_training_samples = int(data["n_training_samples"])
        log.info(f"Cerebellum MLP loaded from {path} ({self.n_training_samples} samples)")
        return True


# ── Experience Buffer ────────────────────────────────────────────────────

class ExperienceBuffer:
    """Ring buffer of (state, action, outcome) transitions."""

    def __init__(self, capacity: int = 2000, horizon: int = 5):
        self._capacity = capacity
        self._horizon = horizon
        self._buffer: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=capacity)
        self._pending: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._tick = 0

    def record(self, state: CerebellumState, action: np.ndarray) -> None:
        """Record current state + action. Completed transitions added when outcome arrives."""
        state_arr = state.to_array()
        self._pending.append((state_arr, action, self._tick))
        self._tick += 1

        # Complete any pending transitions whose horizon has elapsed
        completed = []
        for i, (s, a, t) in enumerate(self._pending):
            if self._tick - t >= self._horizon:
                # Use current state as the outcome
                outcome = state.to_array()
                target = self._compute_target(s, outcome)
                self._buffer.append((s, a, target))
                completed.append(i)

        for i in reversed(completed):
            self._pending.pop(i)

    def _compute_target(self, state_before: np.ndarray, state_after: np.ndarray) -> np.ndarray:
        """Compute target deltas from state transition."""
        delta_r = state_after[0] - state_before[0]
        delta_chi = state_after[3] - state_before[3]
        delta_hunger = state_after[8] - state_before[8]
        delta_curiosity = state_after[9] - state_before[9]
        delta_satiation = state_after[10] - state_before[10]
        # Avalanche proxy: did r drop significantly?
        avalanche = 1.0 if (state_after[0] - state_before[0]) < -0.15 else 0.0
        return np.array([delta_r, delta_chi, delta_hunger, delta_curiosity,
                         delta_satiation, avalanche], dtype=np.float32)

    def sample_batch(self, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Random batch for training. Returns None if insufficient data."""
        if len(self._buffer) < batch_size:
            return None
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states = np.stack([self._buffer[i][0] for i in indices])
        actions = np.stack([self._buffer[i][1] for i in indices])
        targets = np.stack([self._buffer[i][2] for i in indices])
        return states, actions, targets

    def __len__(self) -> int:
        return len(self._buffer)


# ── Cerebellum (facade) ──────────────────────────────────────────────────

class Cerebellum:
    """Forward model blending fast analytical + slow learned predictions."""

    def __init__(self, cfg: Halo3Config):
        self._cfg = cfg
        self.fast = FastPredictor()
        self.learned = LearnedPredictor(lr=cfg.cerebellum_lr)
        self.buffer = ExperienceBuffer(
            capacity=cfg.cerebellum_buffer_size,
            horizon=cfg.cerebellum_horizon,
        )
        self._checkpoint_path = "data/checkpoints/cerebellum_mlp.npz"
        self.learned.load(self._checkpoint_path)

    @property
    def confidence(self) -> float:
        """How much to trust the learned predictor [0, 1].

        sigmoid((n - scale) / (scale/4)) — near 0 when n=0, 0.5 at n=scale.
        """
        n = self.learned.n_training_samples
        scale = self._cfg.cerebellum_confidence_scale
        return 1.0 / (1.0 + math.exp(-(n - scale) / (scale / 4 + 1e-8)))

    def predict(self, state: CerebellumState, action: np.ndarray) -> Prediction:
        """Blend fast + learned predictions based on confidence."""
        fast_pred = self.fast.predict(state, action, horizon=self._cfg.cerebellum_horizon)
        c = self.confidence

        if c < 0.01:
            # Learned model untrained — return fast only
            fast_pred.confidence = 0.0
            return fast_pred

        learned_pred = self.learned.predict(state, action)

        return Prediction(
            delta_r=(1 - c) * fast_pred.delta_r + c * learned_pred.delta_r,
            delta_chi=(1 - c) * fast_pred.delta_chi + c * learned_pred.delta_chi,
            delta_hunger=c * learned_pred.delta_hunger,
            delta_curiosity=c * learned_pred.delta_curiosity,
            delta_satiation=c * learned_pred.delta_satiation,
            avalanche_prob=c * learned_pred.avalanche_prob,
            r_trajectory=fast_pred.r_trajectory,
            confidence=c,
        )

    def record(self, state: CerebellumState, action: np.ndarray) -> None:
        """Record a state-action pair for future training."""
        self.buffer.record(state, action)

    def train(self, n_steps: int | None = None, batch_size: int = 32) -> float | None:
        """Train the learned predictor on buffered experience. Returns mean loss or None."""
        if len(self.buffer) < batch_size:
            log.debug(f"Cerebellum: insufficient data ({len(self.buffer)}/{batch_size})")
            return None

        steps = n_steps or self._cfg.cerebellum_train_steps
        total_loss = 0.0
        actual_steps = 0

        for _ in range(steps):
            batch = self.buffer.sample_batch(batch_size)
            if batch is None:
                break
            states, actions, targets = batch
            loss = self.learned.train_batch(states, actions, targets)
            total_loss += loss
            actual_steps += 1

        if actual_steps > 0:
            mean_loss = total_loss / actual_steps
            log.info(f"Cerebellum trained {actual_steps} steps, loss={mean_loss:.6f}, "
                     f"confidence={self.confidence:.3f}")
            self.learned.save(self._checkpoint_path)
            return mean_loss
        return None

    def predict_soc_trajectory(self, r_now: float, K_proposed: float,
                               omega_std: float, horizon: int = 5) -> list[float]:
        """Quick r trajectory prediction for SOC controller preview."""
        return self.fast.predict_r(r_now, K_proposed, omega_std, n_ticks=horizon)
