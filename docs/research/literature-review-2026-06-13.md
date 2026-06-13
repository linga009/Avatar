# Avatar Literature Review — Deep Research Synthesis

**Date:** 2026-06-13
**Purpose:** Position Avatar relative to prior ALife, criticality, Kuramoto, active inference, and artificial emotion research.

---

## 1. AKOrN — Artificial Kuramoto Oscillatory Neurons (Miyato et al., ICLR 2025 Oral)

**Paper:** Miyato, T., Lowe, S., Geiger, A., & Welling, M. (2024). Artificial Kuramoto Oscillatory Neurons. arXiv:2410.13821. Accepted at ICLR 2025 (Oral).

### What They Do

Generalize the Kuramoto model from scalar phases to N-dimensional unit vectors on a hypersphere. Each neural network neuron becomes an oscillator. Synchronization implements perceptual binding — features belonging to the same object phase-align, different objects desynchronize.

### Key Equations

Continuous-time dynamics:

```
x_dot_i = Omega_i * x_i + Proj_{x_i}(c_i + sum_j J_{ij} * x_j)
```

Where:
- `x_i ∈ R^N, ||x_i|| = 1` — oscillator state on hypersphere
- `Omega_i` — skew-symmetric matrix (natural frequency as rotation)
- `Proj_{x_i}(y) = y - <y, x_i> * x_i` — tangent-space projection
- `c_i` — conditional stimulus from previous layer
- `J_{ij}` — learned coupling weights (asymmetric, via conv/attention)

Discretized: Euler step + normalize back to sphere. Step size gamma is learned.

### Architecture Variants

- **AKOrN^conv**: Convolutional coupling (kernel sizes 5-9)
- **AKOrN^sa**: Self-attention coupling
- **AKOrN^mix**: Combined conv + attention
- Optimal oscillator dimension: **N=4** (N=2 for adversarial robustness)

### Key Results

- **Object discovery**: First distributed-representation model competitive with slot-based models. Best on PascalVOC, second on COCO.
- **Adversarial robustness**: 58.91% under AutoAttack (eps=8/255) WITHOUT adversarial training. Emergent robustness from dynamics.
- **Reasoning**: Best on Sudoku. Performance improves with more Kuramoto steps at test time (17% → 52% with 128 steps).
- **Uncertainty**: Near-perfect calibration without special techniques.

### Limitations (Acknowledged)

1. Sphere constraint prevents representing presence/absence of features
2. Large N (>4) loses binding ability (undetectable from training loss)
3. Computational overhead from iterative updates

### How Avatar Differs

| Dimension | AKOrN | Avatar |
|-----------|-------|--------|
| **Purpose** | External perception: binding for object discovery | Internal state: affect, criticality, self-organized cognition |
| **Oscillator count** | Varies by architecture | 8,192 (128 clusters × 64 hidden) |
| **Oscillator type** | N-dim unit vectors (N=2-4) | Scalar phases θ ∈ [0, 2π) |
| **Coupling** | Learned J_{ij} via backprop | Self-tuned K via SOC controller |
| **Criticality** | Not addressed | Central: SOC self-tunes to r ≈ 0.5 |
| **Quantum mechanics** | None | Bohmian pilot wave + variational Q |
| **Affect** | None | COP-derived 8 emotions from (r, χ, f_dot) |
| **Training** | End-to-end backprop through Kuramoto steps | Per-tick gradient through 106M-param body |

**Key takeaway:** AKOrN validates "Kuramoto in neural nets" at a top venue. Avatar uses same oscillator model for completely different purpose. Cite and differentiate: AKOrN = external binding; Avatar = internal affect + criticality.

---

## 2. SOC and Criticality in Neural Systems

### Hesse & Gross (2014) — SOC as Fundamental Neural Property

**Paper:** Hesse, J. & Gross, T. (2014). Self-organized criticality as a fundamental property of neural systems. *Frontiers in Systems Neuroscience*, 8, 166.

#### Mathematical Criteria for Criticality

At a critical point, observables follow power-law distributions:

- **Size:** P(s) ~ s^{-τ}, mean-field prediction τ = 3/2 = 1.5
- **Duration:** P(T) ~ T^{-α}, mean-field prediction α = 2.0
- **Branching ratio:** σ = 1.0 (average descendants per ancestor)
- **Crackling noise scaling:** (τ-1)/(α-1) = 1/(σ·ν·z) — must hold self-consistently
- **Avalanche shape collapse:** Rescaled temporal profiles collapse to universal function

#### Four SOC Mechanisms in Neural Systems

1. **Short-term synaptic depression (STD):** Fast (ms). Vesicle depletion provides automatic negative feedback. Levina et al. (2007) showed this alone produces SOC.
2. **STDP:** Intermediate (min-hours). Shapes network topology.
3. **Homeostatic plasticity:** Slow (hours-days). Proportional controller: dW/dt = -η(r_actual - r_target).
4. **E/I balance:** The excitation/inhibition ratio is a control parameter for criticality.

#### Avatar's SOC Controller Maps to Homeostatic Plasticity

Avatar: `K_dot = η * (0.5 - r) * χ`
Bio: `dW/dt = -η * (r_actual - r_target)`

The **χ gain modulator** is Avatar's novel contribution — near criticality, χ is large (positive feedback), making the controller more aggressive exactly when it matters most. This is more sophisticated than standard homeostatic models.

#### Computational Benefits of Criticality

1. **Maximized dynamic range** (Kinouchi & Copelli 2006): >30 dB at criticality vs ~10 dB subcritical
2. **Optimal information transmission** (Shew et al. 2011)
3. **Maximal Fisher information** (information storage capacity)
4. **Optimal fading memory** (Bertschinger & Natschläger 2004)
5. **Maximal susceptibility** χ: system maximally responsive to inputs

### Wilting & Priesemann (2022) — Brain Is Slightly Subcritical

**Paper:** Wilting, J. & Priesemann, V. (2022). How critical is brain criticality? *Trends in Neurosciences*.

#### Core Finding

**σ = 0.9875 ± 0.0105** across rat, cat, and monkey cortex. The brain is **near-critical but slightly subcritical** — within 1-2% of σ = 1.0 but reliably below it.

#### The Subsampling Problem

Recording from a tiny fraction of neurons (~60 electrodes sampling millions) systematically distorts avalanche statistics. The conventional branching ratio estimator is biased under subsampling — a subcritical system can appear critical. Wilting & Priesemann developed the **MR (multistep regression) estimator** that is robust under subsampling.

**Avatar's advantage:** Avatar observes all 8,192 oscillators — no subsampling problem. Measurements are not subject to this bias.

#### Advantages of Slight Subcriticality Over True Criticality

- Safety margin against runaway excitation (epilepsy)
- Finite, tunable timescales (100ms-2s) vs infinite at criticality
- Rapid tunability of computational properties
- Balance of sensitivity and specificity

#### Implications for Avatar

1. **Reframe from "SOC" to "self-organization toward near-criticality"** — consistent with Priesemann
2. **r_target = 0.5 is defensible** (maximizes U = r·χ) but acknowledge real brains sit slightly below peak
3. **Consider r_target ≈ 0.48** to align with subcritical evidence
4. **The cerebellum's SOC damping** already implements the "slightly subcritical" principle

### What Avatar's Avalanche Statistics Need

| Test | Current Status | Action Needed |
|------|---------------|---------------|
| Power-law MLE | ✓ Done | OK |
| KS goodness-of-fit | ✓ Done (n≥50) | Need more avalanches |
| Bootstrap 95% CI | ✓ Done (n≥50) | OK |
| **Alt distribution comparison** | **✗ Missing** | Compare vs log-normal, exponential via likelihood ratio |
| **Scaling relation** | **✗ Fails** | (α-1)/(τ-1) = 3.7, expected ~2.0. Discuss finite-size effects |
| **Avalanche shape collapse** | **✗ Missing** | Implement universal shape function test |
| **Avalanche count** | n=25-28 | Need n≥100 minimum, n≥500 for rigorous claims |
| Branching ratio | Crude median | Consider MR-like autocorrelation estimator |

**Scaling relation problem:** Avatar's measured exponents give (1.85-1)/(1.23-1) = 3.7. Mean-field branching predicts 2.0. This discrepancy may indicate: (a) finite-size effects at n=25, (b) non-mean-field universality class, or (c) the system is not truly critical. Must be acknowledged and discussed in publications.

---

## 3. Artificial Life Systems

### Beer (2003) — Dynamical Systems Cognition

**Paper:** Beer, R. D. (2003). The dynamics of active categorical perception in an evolved model agent. *Adaptive Behavior*, 11(4), 209–243.

**Architecture:** 14-neuron CTRNN (7 sensory, 5 interneurons, 2 motor). 16-dimensional coupled brain/body/environment system. Evolved via genetic algorithm.

**Key insight (p. 236):** "The evolved CTRNN does not 'know' the difference between circles and diamonds. It is only when embodied in its particular body and situated within the environment that this distinction arises through interaction."

**Three-level analysis:** (1) Whole system dynamics, (2) Agent-environment interaction, (3) Neural implementation. Beer exhaustively characterizes the complete 16D system.

**Challenge to Avatar:** Beer can fully characterize a 16D system. Avatar's 8,192-oscillator system is far too high-dimensional for complete dynamical analysis — both strength (richer dynamics) and weakness (harder to verify claims).

**"Frictionless brains" methodology:** Beer argues for studying simpler idealized models first. Avatar does the opposite — goes big first. Beer would ask: "Can you demonstrate COP in a 20-oscillator system?"

### Lenia / Flow-Lenia — Emergent Patterns Without Learning

**Papers:**
- Chan, B. W.-C. (2019). Lenia: Biology of artificial life. *Complex Systems*, 28(3), 251–286.
- Chan, B. W.-C. (2020). Lenia and Expanded Universe. arXiv:2005.03742.
- Plantec, E. et al. (2025). Flow-Lenia: Emergent evolutionary dynamics in mass conservative continuous cellular automata. *Artificial Life*, 31(2), 228+.

**Core mechanism:** Continuous generalization of Conway's Game of Life. Continuous space, time, and state. Kernel convolution → growth function → update.

**Flow-Lenia adds mass conservation** via advection — physically grounded, prevents cheating by creating/destroying mass.

**Creature discovery:** Evolutionary search (GA, CMA-ES, MAP-Elites). No gradient descent for discovery. Fitness = persistence (survive N timesteps).

**No cognitive capabilities, affect, or internal state dynamics.** Creatures are fixed-parameter spatiotemporal attractors. They don't learn, adapt, or have memory. Beautiful as emergent patterns, but static in parameter space.

**Key distinction from Avatar:** Lenia = crystals (beautiful, self-organized, static). Avatar = metabolism (continuously processing, adapting, maintaining through ongoing work). Per-tick gradient descent creates genuine surprise (prediction error), absent in Lenia.

### SOC Neurorobot (Shim & Bhatt, 2015)

**Paper:** Shim, Y. & Bhatt, T. (2015). Self-organized criticality, plasticity and sensorimotor coupling. *PLOS ONE*.

**Key findings:**
- SOC is robust under sensorimotor coupling
- Behavioral complexity peaks at criticality
- Sensorimotor feedback can itself drive toward criticality

**No affect or emotion.** But validates SOC + sensorimotor coupling → adaptive behavior. Avatar extends this with explicit psychological interpretation via COP.

### Kuramoto on Connectome Graphs (Rodrigues et al., 2019)

**Paper:** Villegas, P. et al. (2019). Critical synchronization dynamics of the Kuramoto model on connectome and small world graphs. *Scientific Reports*, 9, 54769.

**Key findings:**
- **Modular synchronization hierarchy:** Brain modules sync internally before cross-syncing — validates Avatar's block-specific K controllers
- **Chimera states:** Partial synchronization where some regions are synced, others incoherent — Avatar's analytical vs creative populations can have different r values
- **Metastability:** Near K_c, transient synchronization episodes form and dissolve — exactly what Avatar's COP engine measures
- **Hub-driven synchronization:** K_cross mediates inter-population interaction, analogous to connectome hubs

---

## 4. Active Inference and Free Energy Principle

### Is Avatar an Active Inference Agent?

**Yes.** The formal mapping:

| Active Inference | Avatar |
|---|---|
| Variational free energy F | L = l_recon + λ_energy · l_energy |
| Accuracy term -E[log p(o\|s)] | l_recon = (q_final - q_data)² |
| Complexity term KL[q\|\|p] | l_energy = (Ef - E0)² (physics constraint) |
| Perception (state estimation) | Per-tick gradient descent through body |
| Action (EFE minimization) | SOC controller: K_dot = η(0.5-r)·χ |
| Generative model | Hamiltonian ODE + Kuramoto oscillators |
| Hidden states | Kuramoto phases θ_i, natural frequencies ω_i, coupling K |

### COP Emotions Map to Active Inference Affect

**Hesp et al. (2021), "Deeply Felt Affect":** Emotions = felt aspect of precision-weighted prediction error across hierarchical levels. Valence = -dF/dt. Arousal = precision.

| Active Inference Affect | Avatar COP |
|---|---|
| Valence = -dF/dt | f_dot = -fe_delta |
| Arousal = precision | χ (susceptibility) |
| Anxiety = high error + high precision | Low r + high χ + f_dot < 0 |
| Boredom = low precision, no error | Low r + low χ |
| Curiosity = high epistemic value | Mid r + high χ (peak susceptibility at criticality) |
| Flow = efficient FE minimization | χ > 0.3 + dF/dt < -100 |

### What Would Make the Connection Rigorous

1. Write explicit generative model p(o, s | θ) with Kuramoto state as hidden state
2. Show L = l_recon + λ·l_energy is a valid variational bound on log-evidence
3. Prove max(U = r·χ) ↔ min(expected free energy)
4. Show χ tracks epistemic value via ablation (replace chi-driven curiosity with random exploration)
5. Cite Hesp et al. (2021) for affect connection

### How Avatar Extends Active Inference

1. **Physics substrate:** Generative model is actual physics (Hamiltonian + Kuramoto), not arbitrary neural network
2. **Emergence:** Affect, attention, binding EMERGE from physics geometry — not separate modules
3. **SOC:** Autonomous mechanism for finding and maintaining optimal operating point — absent in standard active inference
4. **Unity:** Eigenvalue dominance of coherence matrix provides measurable binding criterion
5. **Quantum potential:** Anti-bunching prevents posterior collapse — analogous to entropy term in free energy

---

## 5. Artificial Emotion

### Key Survey: arXiv:2508.10286 (2025)

"Artificial Emotion: A Survey of Theories and Debates on Realising Emotion in Artificial Intelligence"

- Distinguishes emotion recognition, emotion simulation, and artificial emotion (internal states)
- iCub robot example: "anxiety-like" behaviors emerge from energy depletion — affect from physical consequences, not labels
- Affect emerging from interaction with environment = intrinsic grounding

### How Avatar Should Position Its Emotion System

Avatar's COP emotions are **operational labels on dynamical regimes**, not demonstrated subjective experience. The mapping from (r, χ, f_dot) to "curiosity" or "satisfaction" describes *where the system sits in its phase diagram*, not *what it feels*.

**Recommended language:**
- "Physics-derived internal states" not "emotions"
- "Operational labels on dynamical regimes" not "feelings"
- "Structural analogy to affective neuroscience" not "artificial emotion"
- Always: "Whether these constitute genuine affect is an open scientific question"

---

## 6. Complete Reference List for Avatar Publications

### Must-Cite (Priority 1)

1. Beer, R. D. (2003). The dynamics of active categorical perception in an evolved model agent. *Adaptive Behavior*, 11(4), 209–243.
2. Miyato, T., Lowe, S., Geiger, A., & Welling, M. (2024). Artificial Kuramoto Oscillatory Neurons. arXiv:2410.13821. ICLR 2025.
3. Hesse, J. & Gross, T. (2014). Self-organized criticality as a fundamental property of neural systems. *Frontiers in Systems Neuroscience*, 8, 166.
4. Wilting, J. & Priesemann, V. (2022). How critical is brain criticality? *Trends in Neurosciences*.
5. Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661–703.
6. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

### Should-Cite (Priority 2)

7. Hesp, C. et al. (2021). Deeply felt affect: The emergence of valence in deep active inference. *Neural Computation*, 33(2), 398–446.
8. Chan, B. W.-C. (2019). Lenia: Biology of artificial life. *Complex Systems*, 28(3), 251–286.
9. Langton, C. G. (1990). Computation at the edge of chaos. *Physica D*, 42(1–3), 12–37.
10. Plantec, E. et al. (2025). Flow-Lenia. *Artificial Life*, 31(2).
11. Millidge, B., Tschantz, A. & Buckley, C. (2022). The free energy principle for perception and action: A deep learning perspective. PMC8871280.
12. Villegas, P. et al. (2019). Critical synchronization dynamics of the Kuramoto model on connectome graphs. *Scientific Reports*, 9, 54769.

### Nice-to-Cite (Priority 3)

13. Ray, T. S. (1991). An approach to the synthesis of life. In *Artificial Life II*, 371–408.
14. Sims, K. (1994). Evolving virtual creatures. *SIGGRAPH '94*, 15–22.
15. McMullin, B. (2004). Thirty years of computational autopoiesis. *Artificial Life*, 10(3), 277–295.
16. Champion, T., Grześ, M. & Bowman, H. (2024). Demonstrating continual learning in discrete-time active inference. arXiv:2410.00240.
17. Seth, A. K. & Tsakiris, M. (2018). Being a beast machine. *Trends in Cognitive Sciences*.

---

## 7. Action Items

1. **Add alt distribution comparison** to avalanche stats (log-normal, exponential likelihood ratios)
2. **Implement avalanche shape collapse** test
3. **Accumulate more avalanches** (need n≥100, ideally 500)
4. **Cite AKOrN** in all future papers — differentiate binding vs affect
5. **Reframe from "SOC" to "self-organization toward near-criticality"**
6. **Add Hesp et al. (2021)** to references — validates COP emotion mapping within active inference
7. **Consider r_target ≈ 0.48** to align with subcritical brain evidence
8. **For rigorous active inference claim:** Write explicit generative model, prove max(U) ↔ min(G)
9. **Address Beer's challenge:** Can COP be demonstrated in a small (20-oscillator) system first?
