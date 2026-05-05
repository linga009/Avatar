# FEP-Swarm: Free Energy Principle Multi-Agent System
## Design Specification

**Author:** Design session — May 5, 2026
**Status:** Approved for implementation
**Stack:** JAX + equinox + diffrax + optax

---

## 1. Goal

Build a physics-principled, 4-layer Active Inference system where every architectural decision derives directly from Karl Friston's Free Energy Principle (FEP). The system proves — empirically and mathematically — that N individual agents minimizing variational free energy spontaneously generate:

1. Adaptive belief-driven behavior without reward functions (Layer 2)
2. Generalized synchrony and emergent tribalism (Layer 3)
3. A macroscopic Markov Blanket with time-scale separation (Layer 4)

This is a HALO-style research prototype: no production deployment, no external APIs, pure physics-to-code mapping validated by four proof plots.

---

## 2. Theoretical Basis

### 2.1 The Four-Layer Physics

| Layer | Physics | Proof |
|---|---|---|
| 1 — Generative Model | P(s\|η), P(η'\|η,a), prior prefs C | A/B matrices valid probability distributions |
| 2 — Active Inference Agent | μ̇ = −∇_μ F(μ,s), G(π) = pragmatic + epistemic | F decreases monotonically; T-maze solved without reward |
| 3 — Swarm Coupling | s_B ∝ κ·a_A; generalized synchrony μ̇_A ≈ μ̇_B | S(t) → 0; emergent clustering by prior C |
| 4 — Macro Blanket | F_macro ≤ ΣF_i − I(sync); Jacobian time-scale separation | Eigenvalue gap > 10×; bound holds ≥95% of steps |

### 2.2 Markov Blanket Definition

The universe is partitioned into four sets:
- **η** — External states (hidden world reality)
- **μ** — Internal states (agent beliefs)
- **s** — Sensory states (observations)
- **a** — Active states (actions)

The Markov Blanket b = (s, a) enforces:

```
P(μ, η | s, a) = P(μ | s, a) · P(η | s, a)
```

μ is permanently sealed from η — it can only interact with reality through the blanket.

### 2.3 Variational Free Energy

```
F(μ, s) = KL[Q(η; μ) || P(η)] − E_Q[ln P(s | η)]
         = E_Q[ln Q(η) − ln P(s, η)]
```

Minimizing F = maximizing model evidence = Bayesian belief updating.

Continuous-time gradient flow:
```
μ̇ = −∇_μ F(μ, s) + ω_μ
ȧ = −∇_a F(μ, s, a) + ω_a
```

### 2.4 Expected Free Energy

Policy selection via Expected Free Energy over planning horizon τ:

```
G(π) = −E_Q[ DKL(Q(s_τ|π) || P(s_τ)) ]   ← Pragmatic value (goal-seeking)
      + E_Q[ H(s_τ | η_τ, π) ]              ← Epistemic value (curiosity)
```

π* = softmax(−β · G(π)) over all candidate policies.

### 2.5 Multi-Scale Emergence

Coupling equations:
```
s_B(t+Δt) ∝ κ · a_A(t)
s_A(t+Δt) ∝ κ · a_B(t)
```

At synchrony: μ̇_A ≈ μ̇_B (generalized synchrony).

Macro free energy bound (Renormalization Group result):
```
F_macro(M, S, A) ≤ Σᵢ Fᵢ(μᵢ, sᵢ) − I(synchrony)
```

Jacobian time-scale separation: the eigenvalue spectrum of J = ∂μ̇/∂μ partitions into fast micro modes (|λ_micro| large) and slow macro modes (|λ_macro| small), with gap ratio > eig_gap.

---

## 3. Architecture

### 3.1 File Structure

```
fep_swarm/
├── config.py                          # FEPConfig dataclass — all hyperparameters
├── generative_model/
│   ├── __init__.py
│   ├── discrete_gm.py                 # A, B, C, D matrices as equinox params
│   └── continuous_likelihood.py       # diffrax NeuralODE: dx/dt = f_θ(x, η, t)
├── agent/
│   ├── __init__.py
│   ├── markov_blanket.py              # Blanket topology + conditional independence check
│   ├── belief_update.py               # μ̇ = −∇_μ F via jax.grad + fori_loop
│   └── action_selection.py            # G(π) decomposition + softmax policy
├── swarm/
│   ├── __init__.py
│   ├── environment.py                 # Shared world state η, noisy local sampling
│   ├── coupling.py                    # Coupling matrix W, obs_new = (1-κ)s + κ(W@a)
│   └── synchrony.py                   # S(t) Frobenius norm + I(synchrony) MI estimator
├── macro/
│   ├── __init__.py
│   ├── renormalization.py             # Coarse-graining operator R, group → M, S, A
│   ├── macro_blanket.py               # F_macro bound computation + violation logging
│   └── eigenanalysis.py               # jax.jacobian → jnp.linalg.eig → gap ratio
├── training/
│   ├── __init__.py
│   └── trainer.py                     # optax Adam, jax.lax.scan episode loop
├── data/
│   ├── __init__.py
│   └── synthetic_world.py             # Configurable grid/graph environment
├── viz/
│   ├── __init__.py
│   └── proof_dashboard.py             # 4-panel matplotlib proof figure
└── tests/
    ├── __init__.py
    ├── test_generative_model.py
    ├── test_agent.py
    ├── test_swarm.py
    ├── test_macro.py
    └── test_integration.py
```

### 3.2 FEPConfig

```python
@dataclass
class FEPConfig:
    # Generative model
    n_hidden: int = 8          # |η| discrete hidden state dimension
    n_obs: int = 4             # |s| discrete observation dimension
    n_actions: int = 4         # |a| action space size
    n_policies: int = 8        # number of candidate policies π to evaluate
    tau: int = 3               # planning horizon (timesteps ahead for EFE)

    # Continuous likelihood NeuralODE
    obs_dim: int = 16          # continuous observation embedding dimension
    ode_width: int = 64        # equinox MLP hidden width for vector field f_θ
    ode_depth: int = 2         # MLP depth

    # Agent belief inference
    inf_steps: int = 16        # gradient descent steps per belief update
    inf_lr: float = 0.1        # belief update step size
    beta: float = 1.0          # policy temperature for softmax(−β·G)

    # Swarm
    n_agents: int = 256        # N agents (all vmapped, zero Python loops)
    kappa: float = 0.3         # coupling strength κ ∈ [0, 1]
    topology: str = "all2all"  # "all2all" | "sparse" | "grid"
    sparse_p: float = 0.1      # edge probability for sparse topology

    # Macro
    coarse_k: int = 16         # agents per coarse-grain group (N must divide by k)
    eig_gap: float = 10.0      # |λ_micro|/|λ_macro| threshold for time-scale proof

    # Training
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42
```

### 3.3 Layer 1 — Discrete Generative Model (`discrete_gm.py`)

Four equinox parameters stored as log-softmax values:

| Parameter | Shape | Physics |
|---|---|---|
| `log_A` | `[n_obs, n_hidden]` | log P(s\|η) — sensory likelihood |
| `log_B` | `[n_hidden, n_hidden, n_actions]` | log P(η'\|η,a) — transition model |
| `log_C` | `[n_obs]` | log prior preferences over observations |
| `log_D` | `[n_hidden]` | log P(η) — initial hidden state prior |

All accessed via `jax.nn.softmax` to recover proper distributions.

### 3.4 Layer 1 — Continuous Likelihood (`continuous_likelihood.py`)

diffrax NeuralODE maps discrete hidden state η → continuous observation embedding:

```
dx/dt = f_θ(x, η, t)     where f_θ is an equinox MLP
x(0) ~ N(0, I)
x(1) = diffrax.diffeqsolve(f_θ, t0=0, t1=1, y0=x0)
```

Solver: `diffrax.Euler` (training), `diffrax.Dopri5` (validation). Output `x(1) ∈ R^obs_dim` is the continuous sensory embedding fed to the agent alongside discrete `s`.

### 3.5 Layer 2 — Markov Blanket (`markov_blanket.py`)

Enforces the conditional independence constraint:
```
P(μ, η | s, a) = P(μ | s, a) · P(η | s, a)
```

Implemented as a structural assertion: `μ` updates depend only on `(μ, s)`, never directly on `η`. Active states `a` depend only on `(μ, s)`. Verified in tests via mutual information between `μ` and `η` conditioned on blanket.

### 3.6 Layer 2 — Belief Update (`belief_update.py`)

Variational free energy F and its gradient descent, fully JIT-compilable:

```python
def free_energy(mu, s, gm_params):
    q_eta = jax.nn.softmax(mu)                         # Q(η; μ)
    p_eta = jax.nn.softmax(gm_params.log_D)            # P(η)
    kl = jnp.sum(q_eta * (jnp.log(q_eta) - jnp.log(p_eta)))
    A = jax.nn.softmax(gm_params.log_A, axis=0)        # P(s|η)
    log_lik = jnp.log(A[s] @ q_eta + 1e-8)
    return kl - log_lik

def belief_update(mu_init, s, gm_params, cfg):
    grad_F = jax.grad(free_energy)
    return jax.lax.fori_loop(
        0, cfg.inf_steps,
        lambda i, mu: mu - cfg.inf_lr * grad_F(mu, s, gm_params),
        mu_init
    )
```

### 3.7 Layer 2 — Action Selection (`action_selection.py`)

Expected Free Energy G(π) computed for all policies in parallel:

```
G(π) = Pragmatic(π) + Epistemic(π)

Pragmatic(π) = −DKL[ Q(s_τ|π) || P(s_τ) ]       # seek preferred obs
Epistemic(π) = E_Q[ H(s_τ | η_τ, π) ]            # minimize expected ambiguity

π* = softmax(−β · G(π))     over n_policies policies
```

### 3.8 Layer 3 — Swarm vmap Pattern (`swarm/`)

The architectural spine. All N agents are a single batched pytree:

```python
AgentState = NamedTuple(
    mu:     Array[N, n_hidden],
    action: Array[N, n_actions],
    obs:    Array[N, n_obs],
)

# Single-agent step
def agent_step(state, env_obs, gm_params, cfg): ...

# Vectorized over N — zero Python loops
swarm_step = jax.vmap(agent_step, in_axes=(0, 0, None, None))
```

Coupling matrix `W ∈ R^[N,N]` (configurable topology):
```
obs_new[i] = (1 − κ) · obs_self[i]  +  κ · Σ_j W[i,j] · action[j]
```

### 3.9 Layer 3 — Synchrony Metrics (`synchrony.py`)

Two measurements logged each step:

**Synchrony decay:** `S(t) = ||μ̇_A − μ̇_B||_F / N²`
- Computed as pairwise Frobenius norm of belief rate differences
- Should decrease toward 0 as κ increases

**Mutual information:** `I(t)` estimated over sliding window of μ history
- Should increase as agents couple

Phase transition scan: sweep κ ∈ [0, 1], plot S(T) vs κ — critical κ* is the coupling threshold for synchrony.

### 3.10 Layer 4 — Coarse-Graining (`renormalization.py`)

```python
# mu: Array[N, n_hidden]
mu_grouped = mu.reshape(N // k, k, n_hidden)
M = mu_grouped.mean(axis=1)          # [N//k, n_hidden]  — macro internal states
S_macro = boundary_mean(obs, W)      # [N//k, n_obs]     — macro sensory states
A_macro = boundary_mean(actions, W)  # [N//k, n_actions] — macro active states
```

### 3.11 Layer 4 — Macro Free Energy Bound (`macro_blanket.py`)

```
F_macro(M, S_macro, A_macro) ≤ Σᵢ Fᵢ(μᵢ, sᵢ) − I(synchrony)
```

Computed every step. Violation rate logged. At equilibrium: ≤5% violation rate.

### 3.12 Layer 4 — Jacobian Eigenanalysis (`eigenanalysis.py`)

```python
def compute_jacobian(mu_flat, gm_params, env_state):
    # mu_flat: Array[N * n_hidden]
    mu_dot = lambda m: swarm_belief_rates(m, gm_params, env_state)
    J = jax.jacobian(mu_dot)(mu_flat)          # [N*d, N*d]
    eigenvalues, _ = jnp.linalg.eig(J)
    magnitudes = jnp.abs(eigenvalues)
    gap = magnitudes.max() / (magnitudes.min() + 1e-8)
    return eigenvalues, gap
```

**Proof:** `gap > cfg.eig_gap` (default 10×) demonstrates that the swarm operates at two distinct time scales — individual agents react fast (large λ), macro structure evolves slow (small λ).

---

## 4. Training Loop (`training/trainer.py`)

Uses `jax.lax.scan` for the episode loop (JIT-compilable, no Python overhead):

```python
@jax.jit
def episode_step(carry, _):
    state, env, gm_params = carry
    obs = env.observe(state)
    state = swarm_step(state, obs, gm_params, cfg)
    env = env.step(state.action)
    metrics = compute_metrics(state, env, gm_params)
    return (state, env, gm_params), metrics

final_carry, metrics_history = jax.lax.scan(
    episode_step, init_carry, xs=None, length=cfg.n_steps
)
```

Generative model parameters updated via `optax.adam(cfg.lr)` on episode free energy.

---

## 5. Data (`data/synthetic_world.py`)

Configurable synthetic environment:
- **Grid world:** N×N grid, agents navigate to goal states. Hidden state η = agent position.
- **Graph world:** random graph, η = node occupancy. Tests non-uniform topology effects on synchrony.
- **T-maze:** classic active inference benchmark. Two arms, one rewarded. Validates Layer 2 epistemic drive.

All environments implemented as pure JAX functions (no Python state) for JIT compatibility.

---

## 6. Proof Dashboard (`viz/proof_dashboard.py`)

Four-panel matplotlib figure — the single deliverable proving all 4 layers work:

| Panel | X-axis | Y-axis | Pass condition |
|---|---|---|---|
| 1 | inference step | F(μ, s) | strictly decreasing |
| 2 | simulation step | S(t) synchrony | converges toward 0 |
| 3 | simulation step | F_macro vs ΣFᵢ−I | blue line stays below red |
| 4 | eigenvalue index | \|λ\| magnitude | visible gap between fast and slow |

---

## 7. Test Plan

### test_generative_model.py
- `A = softmax(log_A, axis=0)` sums to 1 per column
- `B[:, :, a]` is column-stochastic for each action `a` (sums to 1 over η', i.e. `softmax(log_B, axis=0)`)
- NeuralODE output shape: `[obs_dim]`, no NaN/Inf
- diffrax solve terminates without step-size underflow

### test_agent.py
- `free_energy` decreases monotonically over `inf_steps` gradient steps
- EFE epistemic term > pragmatic term when agent is in a novel (unseen) state
- EFE pragmatic term > epistemic term when agent is in a familiar (seen) state
- Markov blanket: `μ` update depends only on `(μ, s)`, not directly on `η`
- T-maze: agent reaches goal arm ≥80% of 100 episodes (no reward function)

### test_swarm.py
- `swarm_step` runs N=256 agents without Python loop (verified via `jax.make_jaxpr`)
- Coupling: `obs_new.shape == (N, n_obs)`, values ∈ [0, 1]
- S(t) at step 1000 < S(t) at step 0 for κ=0.5
- S(t) remains flat (< 5% change) for κ=0 (uncoupled baseline)
- Tribalism: k-means on final μ states, ARI > 0.7 when agents have distinct prior C

### test_macro.py
- Coarse-graining: `M.shape == (N // coarse_k, n_hidden)`
- F_macro bound holds ≥95% of steps at quasi-equilibrium
- Jacobian: `J.shape == (N * n_hidden, N * n_hidden)`, no NaN
- Eigenvalue gap > `cfg.eig_gap` (10×) after convergence
- Macro temporal horizon (1/|λ_macro|) > micro temporal horizon (1/|λ_micro|)

### test_integration.py
- Full 4-layer pipeline runs 1000 steps without error or NaN
- All 4 proof plots generate cleanly (no matplotlib errors, no Inf values)
- `jax.jit(swarm_step)` compiles without error
- N=256, 1000 steps completes in < 60 seconds on CPU

---

## 8. Success Criteria

1. All 5 test files pass: `pytest fep_swarm/tests/ -v` → 100% pass rate
2. Proof Plot 1: F(t) is strictly decreasing over inference steps
3. Proof Plot 2: S(t) synchrony metric converges toward 0 for κ ≥ 0.3
4. Proof Plot 3: F_macro bound holds visually (blue ≤ red line)
5. Proof Plot 4: Eigenvalue spectrum shows clear fast/slow gap ≥ 10×
6. JIT compilation: `jax.jit(swarm_step)` succeeds, N=256 step < 100ms CPU

---

## 9. What is NOT in Scope

- Real-world data integration (live feeds, Bloomberg, Twitter firehose)
- MiroFish codebase integration — this is an independent prototype
- Production deployment or API exposure
- GPU optimization (single CPU/GPU prototype)
- Continuous action spaces (discrete actions only in v1)
- Multi-GPU or distributed JAX (single device)
- Hierarchical generative models deeper than 2 levels
