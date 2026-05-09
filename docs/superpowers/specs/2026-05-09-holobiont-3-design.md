# HoloBiont 3.0 "Physics Engine" — Design Spec

> A physics simulation with learned parameters, not a neural network
> with physics-inspired components. 12M params, ~227 MB training VRAM.

---

## Goal

Replace all neural approximations with their exact physical counterparts:
MERA tensor networks for the bulk geometry, Hamiltonian ODEs for dynamics,
Kuramoto oscillators for collective inference. The result is a 12M-parameter
model with the representational capacity of 1.2B dense parameters, training
in ~227 MB on a 6 GB GPU.

## Architecture Overview

```
Perception (web + embed)
    │
    ▼
LorentzEmbedding → q₀ ∈ Lorentz^64
    │
    ├──→ Initialize momenta p₀ ~ N(0, σ²)
    │
    ▼
Halo3Backbone: [S,S,S,S,S,H] × 4 = 24 layers
  ├─ 20× SelectiveSSM S7 (from v2.1)
  ├─ 4× SharedHoloAttention Zamba2 + LoRA (from v2.1)
  ├─ 24× MERA-FFN (tensor train, replaces SwiGLU)
  └─ Reversible coupling (from v2.1)
    │
    ▼
Hamiltonian ODE (replaces flow matching)
  ├─ H(q,p) = ½||p||² + V_learned(q) + V_ads(q)
  ├─ Leapfrog symplectic integrator
  └─ Adjoint backprop (O(1) memory)
    │
    ▼
Kuramoto Oscillators on n-Torus (replaces FEP swarm)
  ├─ θ_k ∈ [0,2π)^{n_hidden} per cluster
  ├─ dθ/dt = ω + K·sin(θ_mean - θ) + η·obs
  ├─ Actions from phase velocity
  └─ Order parameter r replaces free energy
    │
    ▼
MetaLayer (modifies coupling K and frequencies ω)
HomeostaticRegulator (order parameter r replaces novelty)
```

---

## 1. MERA-FFN (Tensor Train FFN Replacement)

### 1.1 Concept

Replace dense SwiGLU weight matrices with tensor train decomposition.
A matrix W of shape (m, n) is decomposed into k core tensors:

```
W[i₁...iₖ, j₁...jₖ] = G₁[i₁,j₁] × G₂[i₂,j₂] × ... × Gₖ[iₖ,jₖ]
```

where each Gₗ has shape (χ, mₗ, nₗ, χ) with bond dimension χ.

### 1.2 MERA-FFN Module

```python
class MERAFFN(eqx.Module):
    """Tensor train FFN — replaces SwiGLU with 518× fewer params."""
    cores_gate: list   # k tensor cores for gate path
    cores_up: list     # k tensor cores for up path
    cores_down: list   # k tensor cores for down path

    def __call__(self, x):
        gate = tt_contract(self.cores_gate, x)
        up = tt_contract(self.cores_up, x)
        down = tt_contract(self.cores_down, jax.nn.silu(gate) * up)
        return down
```

### 1.3 Tensor Train Contraction

```python
def tt_contract(cores, x):
    """Contract input x through tensor train cores."""
    # Reshape x into tensor indices
    # Contract through cores left-to-right
    # Reshape output back to vector
```

### 1.4 Configuration

- Bond dimension χ = 64
- Number of cores k = 4
- Input/output reshape: d_model=2048 → (8, 8, 8, 4) or similar factorization
- Params per FFN: 3 paths × k × χ² × (mₗ × nₗ) ≈ 65K (vs 34.6M for SwiGLU)

---

## 2. Hamiltonian Neural ODE

### 2.1 Phase Space

```
q ∈ R^{n_tokens × d_boundary}  — positions (Lorentz boundary coords)
p ∈ R^{n_tokens × d_boundary}  — conjugate momenta (learned initialization)
```

### 2.2 Hamiltonian

```
H(q, p) = T(p) + V(q)

T(p) = ½ Σᵢ ||pᵢ||²                              — kinetic (fixed)

V(q) = V_ads(q) + V_learned(q)                     — potential (structured + learned)

V_ads(q) = Σᵢ<ⱼ log(cosh(d_geo(qᵢ, qⱼ)))         — AdS gravity wells

V_learned(q) = MLP(mean_pool(q))                   — data-specific correction
  MLP: d_boundary → 64 → 64 → 1  (scalar potential, ~8K params)
```

### 2.3 Leapfrog Integrator

```
p_{1/2} = p₀ - (ε/2)·∇_q V(q₀)        — half-step momentum
q₁     = q₀ + ε·p_{1/2}                — full-step position
p₁     = p_{1/2} - (ε/2)·∇_q V(q₁)    — half-step momentum

Repeat for n_leapfrog_steps (default: 3)
```

Symplectic: energy is conserved exactly (up to floating point), trajectories
never diverge, large step sizes are stable.

### 2.4 Momentum Initialization

```python
p₀ = MLP_init(backbone_output)  # small MLP: d_model → d_boundary
```

The backbone output conditions the initial momenta, connecting the SSM/attention
representation to the Hamiltonian dynamics.

### 2.5 Loss

```
L = L_recon + λ_energy · L_energy

L_recon  = MSE(q_T, q_data)                      — final position matches data
L_energy = (H(q_T, p_T) - H(q₀, p₀))²           — energy conservation
```

### 2.6 Backpropagation

`diffrax.diffeqsolve` with `RecursiveCheckpointAdjoint` — O(1) memory.

---

## 3. Kuramoto Oscillators on n-Torus

### 3.1 State

```python
class KuramotoState(NamedTuple):
    theta: jnp.ndarray       # (K, n_hidden) — phases on n-torus
    omega: jnp.ndarray       # (K, n_hidden) — natural frequencies (learned)
    coupling: float           # scalar K — global coupling strength
    key: jnp.ndarray
```

K=32 clusters, n_hidden=16 phases each.

### 3.2 Dynamics

```
dθ_k/dt = ω_k + (K/N) Σ_j sin(θ_j - θ_k) + η · obs_k

Mean-field approximation:
dθ_k/dt = ω_k + K · sin(θ_mean - θ_k) + η · obs_k
```

Integration: single Euler step per tick (dt=0.1).

### 3.3 Action Selection

Actions derived from phase velocity — no EFE computation needed:

```python
def kuramoto_action(state, n_actions):
    phase_velocity = jnp.sin(
        jnp.mean(state.theta, axis=0)[None,:] - state.theta
    )
    return jax.nn.softmax(phase_velocity[:, :n_actions], axis=-1)
```

### 3.4 Order Parameter

```python
def order_parameter(theta):
    """r ∈ [0,1] per dimension. r=1 = full sync = low surprise."""
    return jnp.abs(jnp.mean(jnp.exp(1j * theta), axis=0))
```

Replaces free energy as the system health metric:
- High r → synchronized → confident → exploit
- Low r → desynchronized → uncertain → explore

### 3.5 Bridge Interface

- **ObsBridge**: backbone → (K, n_obs) → phase kicks η·obs for Kuramoto
- **ActionBridge**: phase velocities → (n_tokens, d_boundary) boundary bias
- **BeliefBridge**: sin(θ)/cos(θ) encoding → (n_tokens, d_model) flow conditioning

### 3.6 MetaLayer Integration

MetaLayer fires every K=10 ticks and modifies:
- Coupling strength K (was: log_C goal preferences)
- Natural frequencies ω (was: meta-belief updates)

HomeostaticRegulator uses order parameter r instead of novelty EMA:
- r < threshold → explore mode (reduce K → desynchronize)
- r ≥ threshold → exploit mode (increase K → synchronize)

---

## 4. Training Pipeline

### 4.1 Optimizer

Adafactor (unchanged from v2.1). With 12M params, optimizer state is negligible.

### 4.2 Memory Techniques

- **Reversible backbone**: zero activation storage
- **GaLore**: rank-64 projection on MERA cores (already small, but helps)
- **LISA**: 2/24 active layers per step
- **Adjoint backprop**: O(1) through Hamiltonian ODE
- **FP16 mixed precision**: halves param/gradient memory

### 4.3 Loss Function

```
L_total = L_recon + λ_energy · L_energy + λ_sync · L_sync

L_recon  = MSE(q_T, q_data)           — Hamiltonian reconstruction
L_energy = (H_T - H_0)²               — energy conservation
L_sync   = -mean(r)                    — encourage synchronization
```

### 4.4 VRAM Budget (d_model=2048)

| Component | Memory |
|-----------|--------|
| Params (12M × 2B FP16) | 24 MB |
| Gradients (LISA 2/24) | 2 MB |
| Adafactor state | 0.1 MB |
| Activations (reversible) | 0 MB |
| Hamiltonian ODE (adjoint) | 1 MB |
| Kuramoto state | 2 KB |
| JIT overhead | 200 MB |
| **Total** | **~227 MB** |

---

## 5. Parameter Count

| Component | Params |
|-----------|--------|
| 20× S7 SSM | 5.3M |
| 2× SharedHoloAttention | 4.2M |
| 4× LoRA adapters | 262K |
| 24× MERA-FFN | 1.6M |
| 24× LayerNorm ×2 | 98K |
| LorentzEmbedding | 66K |
| Hamiltonian V_learned MLP | 8K |
| Momentum init MLP | 66K |
| Kuramoto ω (32×16) | 512 |
| Bridges ×3 | 93K |
| MetaLayer | 50K |
| **Total** | **~12M** |

---

## 6. Config

```python
@dataclass(frozen=True)
class Halo3Config:
    # Backbone
    d_model: int = 2048
    d_boundary: int = 64
    n_heads: int = 16
    d_head: int = 128
    n_layers: int = 24
    d_state: int = 64
    layer_pattern: str = "SSSSSH"
    n_shared_attn: int = 2
    lora_rank: int = 16
    reversible: bool = True

    # MERA-FFN
    mera_bond_dim: int = 64
    mera_n_cores: int = 4

    # Lorentz
    init_curvature: float = 1.0

    # Hamiltonian
    n_leapfrog_steps: int = 3
    leapfrog_step_size: float = 0.1
    lambda_energy: float = 0.1

    # Kuramoto
    n_clusters: int = 32
    n_hidden: int = 16
    kuramoto_dt: float = 0.1
    init_coupling: float = 1.0
    lambda_sync: float = 0.01

    # Page memory
    max_cache: int = 128
    island_size: int = 32

    # Training
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42
    galore_rank: int = 64
    lisa_active_layers: int = 2

    # Bridges
    n_tokens: int = 32
    n_obs: int = 8
    n_actions: int = 8

    # Meta-layer
    meta_n_hidden: int = 8
    meta_n_actions: int = 4
    meta_k: int = 10

    # Homeostatic
    homeo_sync_threshold: float = 0.6
    homeo_blend_clip: float = 1.0

    # Heartbeat
    tick_interval: int = 60
```

---

## 7. File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `halo3/__init__.py` | Create | Package init |
| `halo3/config.py` | Create | Halo3Config |
| `halo3/lorentz_ops.py` | Copy from halo2 | Unchanged |
| `halo3/lorentz_embedding.py` | Copy from halo2 | Unchanged |
| `halo3/ssm_s7.py` | Copy from halo2 | Unchanged |
| `halo3/holo_attention_shared.py` | Copy from halo2 | Unchanged |
| `halo3/mera_ffn.py` | Create | MERA tensor train FFN |
| `halo3/backbone.py` | Create | Reversible backbone with MERA-FFN |
| `halo3/hamiltonian.py` | Create | H(q,p), V_ads, V_learned, leapfrog, momentum init |
| `halo3/kuramoto.py` | Create | KuramotoState, kuramoto_step, kuramoto_action, order_parameter |
| `halo3/bridge/obs_bridge.py` | Create | Phase kick interface |
| `halo3/bridge/action_bridge.py` | Create | Phase velocity → boundary bias |
| `halo3/bridge/belief_bridge.py` | Create | sin/cos(θ) → d_model conditioning |
| `halo3/model.py` | Create | Halo3Model, Halo3Carry, halo3_step |
| `halo3/loss.py` | Create | L_recon + L_energy + L_sync |
| `halo3/page_memory.py` | Copy from halo2 | Unchanged |
| `halo3/intellect/meta_layer.py` | Create | Modifies K and ω |
| `halo3/intellect/homeostatic_regulator.py` | Create | Order parameter r |
| `halo3/intellect/goal_updater.py` | Create | Coupling decay |
| `halo3/training/trainer.py` | Create | Adafactor + GaLore + LISA |
| `halo3/training/galore.py` | Copy from halo2 | Unchanged |
| `halo3/training/bootstrap.py` | Create | Phase 0 pre-training |
| `halo3/main.py` | Create | Heartbeat loop |
| `halo3/tests/test_config.py` | Create | Config tests |
| `halo3/tests/test_mera.py` | Create | MERA-FFN tests |
| `halo3/tests/test_hamiltonian.py` | Create | Hamiltonian ODE tests |
| `halo3/tests/test_kuramoto.py` | Create | Kuramoto oscillator tests |
| `halo3/tests/test_backbone.py` | Create | Backbone tests |
| `halo3/tests/test_model.py` | Create | End-to-end model tests |
| `halo3/tests/test_integration.py` | Create | Integration tests |

---

## 8. Key References

- [MERA as neural network layers](https://openreview.net/forum?id=rkGZuJb0b) — 3,900× param compression
- [CompactifAI: tensor network compression of LLMs](https://arxiv.org/abs/2401.14109) — 93% memory reduction
- [Holographic codes from hyperinvariant tensor networks](https://www.nature.com/articles/s41467-023-42743-z) — MERA = AdS
- [Symplectic Generative Networks](https://arxiv.org/abs/2505.22527) — reversible symplectic layers
- [SPINI: structure-preserving neural integrator](https://www.nature.com/articles/s41598-025-28710-2) — Yoshida leapfrog
- [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563) — learned Hamiltonians
- [Neural ODE adjoint method](https://docs.kidger.site/diffrax/api/adjoints/) — O(1) memory backprop
- [Kuramoto model](https://en.wikipedia.org/wiki/Kuramoto_model) — coupled oscillator synchronization
- [Convergence of Swarm Intelligence and Active Inference](https://www.alphanome.ai/post/the-convergence-of-swarm-intelligence-antetic-ai-cellular-automata-active-inference-reshaping-m)
