"""Experiment configurations for ablation study."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Flags controlling which components are active."""
    name: str
    n_ticks: int = 200
    disable_cop: bool = False
    disable_senses: bool = False
    disable_dreams: bool = False
    disable_quantum_potential: bool = False
    is_transformer_baseline: bool = False
    description: str = ""


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "full_avatar": ExperimentConfig(
        name="full_avatar",
        description="Full Avatar v4.0 — control condition",
    ),
    "no_cop": ExperimentConfig(
        name="no_cop",
        disable_cop=True,
        description="COP disabled — fixed emotions (v3.10 if/elif), no SOC, fixed K=0.3",
    ),
    "no_senses": ExperimentConfig(
        name="no_senses",
        disable_senses=True,
        description="Senses disabled — zero sensory injection, FNO/VQ-VAE inactive",
    ),
    "no_dreams": ExperimentConfig(
        name="no_dreams",
        disable_dreams=True,
        description="Dreams disabled — continuous waking, no consolidation",
    ),
    "no_bohmian_q": ExperimentConfig(
        name="no_bohmian_q",
        disable_quantum_potential=True,
        description="Quantum potential disabled — pure Kuramoto without anti-bunching",
    ),
    "transformer_baseline": ExperimentConfig(
        name="transformer_baseline",
        is_transformer_baseline=True,
        description="Standard 12-layer transformer (~100M params), same data and schedule",
    ),
}


def get_experiment_config(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Options: {list(EXPERIMENTS)}")
    return EXPERIMENTS[name]
