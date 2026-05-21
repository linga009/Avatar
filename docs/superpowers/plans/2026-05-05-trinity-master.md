# Master Trinity Implementation Plan: FEP + HALO + Gemma

**Goal:** Build the "Trinity" — a unified synthetic organism where **FEP-Swarm** (Active Inference) provides the biological drive, **HALO** (Holographic Perception) provides high-dimensional sensory processing, and **Hermes** (Intellect) provides high-level reasoning, web-learning, and self-evolution locally.

**Architecture:** A closed-loop JAX system where a local **Llama-3-Hermes** model acts as the "Prefrontal Cortex," setting strategic goals that are executed by a swarm of holographic agents. The system autonomously browses the web to research improvements to its own code and evolves via self-modification.

---

## 🗺️ File Map

| Component | Files | Responsibility |
|---|---|---|
| **Core** | `halo_fep/config.py` | Unified hyperparameters for all layers |
| **Perception** | `halo_fep/halo_jax/` | JAX port of HALO (Attention, SSM, PageMemory) |
| **Drive** | `fep_swarm/` | FEP agents (Belief updates, Action selection) |
| **Intellect** | `halo_fep/intellect/` | Hermes Local Bridge, Web Explorer, Evolution Engine |
| **Bridge** | `halo_fep/bridge/` | ObsBridge, ActionBridge, BeliefBridge |
| **Evolution** | `halo_fep/main.py` | The autonomous "Heartbeat" loop |

---

## 🛠️ Phase 1: JAX Foundations (HALO-FEP Port)

- [ ] **Task 1: Scaffold & Unified Config**
    - Create `halo_fep/config.py` with support for both 6GB VRAM and A100 profiles.
    - Setup all package stubs (`__init__.py`).

- [ ] **Task 2: HALO JAX Port**
    - Port `HoloEmbedding`, `HoloAttention`, `SimpleSSM`, and `PageCurveMemory` to JAX/Equinox.
    - Implement the `HALOBackbone` stack ([S,S,S,H,S,S,S,H]).

- [ ] **Task 3: FEP-Swarm Coupling**
    - Implement `ObsBridge` (HALO → FEP observations).
    - Implement `ActionBridge` (FEP actions → HALO boundary bias).
    - Implement `BeliefBridge` (FEP beliefs → HALO flow conditioning).

---

## 🧠 Phase 2: The Intellect (Hermes Local Layer)

- [ ] **Task 4: Hermes Local Integration**
    - Implement `halo_fep/intellect/hermes_bridge.py` using `llama-cpp-python` or `transformers`.
    - Load **Llama-3-Hermes-8B** (4-bit quantization) to fit within 6GB VRAM limits.
    - Create the "Strategic Goal" parser: Hermes output → FEP Prior ($C$ matrix).

- [ ] **Task 5: Semantic Projection**
    - Map HALO token embeddings directly to the Hermes embedding space.
    - Allow Hermes to "read" the holographic state of the swarm via direct neural injection.

---

## 🌐 Phase 3: Cyber-Evolution (Browse & Evolve)

- [ ] **Task 6: Autonomous Web Explorer**
    - Implement `halo_fep/intellect/web_researcher.py`.
    - Trigger: High Free Energy for >100 steps.
    - Logic: Hermes searches the web (via Search API) -> HALO filters content -> Hermes learns.

- [ ] **Task 7: Self-Modification Engine**
    - Implement `halo_fep/intellect/evolution_engine.py`.
    - Logic: Hermes proposes a code "Mutation" -> Sandbox test via `pytest` -> If pass, overwrite source and **Self-Restart**.

---

## 🚀 Phase 4: Deployment & Benchmark

- [ ] **Task 8: Multimodal Goal-Inference Benchmark**
    - Create `MultimodalWorld` with complex, shifting text/image goals.
    - Verify that the "Trinity" (FEP+HALO+Gemma) solves the task faster than FEP alone.

- [ ] **Task 9: The "Always-On" Heartbeat**
    - Launch `main.py`. The organism begins its first "Life Cycle."

---

## 🧪 Verification Plan

### Automated
1. **The "Surprise" Test:** Artificially spike uncertainty and verify Hermes triggers a web search.
2. **The "Safe Mutation" Test:** Ensure the system rejects a code change that breaks the `pytest` suite.

### Manual
1. **Evolution Log:** Monitor `evolution.log` to track how the organism's hyperparameters change over 24 hours of autonomous learning.
2. **Holographic Dashboard:** Use `viz/proof_dashboard.py` to watch the swarm sync up under Gemma's strategic guidance.
