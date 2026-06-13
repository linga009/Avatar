# Avatar as Active Inference Agent — Formal Mapping

**Date:** 2026-06-13
**Status:** Structural analogy documented. Full mathematical proof pending.

## Core Equivalence

Avatar's loss function L = l_recon + lambda_energy * l_energy is structurally
equivalent to the variational free energy F = -E[ln p(o|s)] + KL[q(s)||p(s)].

| Active Inference | Avatar | Reference |
|---|---|---|
| Variational free energy F | L = l_recon + lambda * l_energy | model.py |
| Accuracy -E[ln p(o\|s)] | l_recon = (q_final - q_data)^2 | loss.py |
| Complexity KL[q\|\|p] | l_energy = (Ef - E0)^2 | loss.py |
| Perception (state estimation) | Per-tick gradient descent | predictive.py |
| Action (EFE minimization) | SOC controller: K_dot = eta(0.5-r)*chi | cop.py |
| Generative model | Hamiltonian ODE + Kuramoto | model.py |
| Hidden states | theta_i, omega_i, K | kuramoto.py |

## COP Emotions Map to Hesp et al. (2021)

Hesp, C. et al. (2021). Deeply felt affect: The emergence of valence in
deep active inference. Neural Computation, 33(2), 398-446.

| Hesp et al. (2021) | Avatar COP | Implementation |
|---|---|---|
| Valence = -dF/dt | f_dot = -fe_delta | emotions.py |
| Arousal = precision | chi (susceptibility) | cop.py |
| Anxiety = high error + high precision | Low r + high chi + f_dot < 0 | emotions.py |
| Boredom = low precision, no error | Low r + low chi | emotions.py |
| Curiosity = high epistemic value | Mid r + high chi | emotions.py |
| Flow = efficient FE minimization | chi > 0.3 + dF/dt < -100 | emotions.py |

## What Would Make This Rigorous

1. Write explicit generative model p(o, s | theta) with Kuramoto state as hidden state
2. Show L is a valid variational bound on log-evidence
3. Prove max(U = r * chi) <-> min(expected free energy G)
4. Ablation: replace chi-driven curiosity with random exploration, measure divergence
5. These are publication goals, not current claims

## Key Distinction

Avatar IMPLEMENTS active inference through physics rather than approximating
it via neural networks. The generative model IS the physics (Hamiltonian +
Kuramoto), not a learned approximation of physics. This is unusual in the
active inference literature where generative models are typically parametric.

## Honest Framing

This is a structural analogy, not a proof. The mapping shows that Avatar's
architecture is consistent with active inference principles, but formal
equivalence requires the mathematical steps outlined above. Whether these
constitute genuine active inference is an open scientific question.
