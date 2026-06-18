# Psyche Subsystem Rules

Applied when editing: cop.py, emotions.py, organism.py, drives.py, workspace.py, meditation.py

- COP observables: r (order), chi (susceptibility), tau (relaxation), F_thermo (Helmholtz free energy)
- chi uses Harada-Sasa FDT correction — sigma=max(0, C(1)-R(1)), chi/=(1+5*sigma)
- SOC controller: K_dot = eta*(0.5-r)*chi + noise — three independent controllers (K_aa, K_cc, K_cross)
- K block-clamped: analytical [0.02, 0.40], creative [0.20, 2.00], cross [0.05, 2.0]
- Stochastic perturbation (noise = eta*0.5*uniform) with floor/ceiling-lock recovery prevents clamp-locking
- post_dream_reset: pre-fill r_history with noisy terminal_r (±0.02) — identical values crash chi to zero
- Query staleness: organism forces BS topic switch after 5 consecutive same-query ticks
- Emotions from (r, chi, f_dot, dF_dt) manifold — 8 emotions: satisfaction, pride, curiosity, boredom, anxiety, frustration, flow, exhaustion
- Flow requires: chi>0.3 AND dF/dt<-100 — rare, high-signal
- Frustration split by dF/dt sign: negative=growth (lower intensity), positive/zero=futile
- Exhaustion requires: F flat 20+ ticks AND chi>0.2
- Meditation attenuates obs, not K — never collapse coupling during meditation
- F_thermo = H_mean - T_eff * S_phase — diagnostic, logged every 10 ticks
- GWT ignition: r-threshold (0.5) with hysteresis (sustain 0.4), anchored to SOC critical point
- effective_r = r_mean + 0.05 * sensory_novelty — novel stimuli facilitate ignition
- broadcast_intensity = effective_r * (0.5 + 0.5 * unity) — unity scales vivid vs dim consciousness
- transition_sharpness = max(recent chi) at ignition crossing — sharp (>0.3) triggers "crystallizing" body event
- Chi does NOT participate in ignition decision — only used as transition qualifier
