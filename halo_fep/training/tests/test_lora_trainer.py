# halo_fep/training/tests/test_lora_trainer.py
import jax
import numpy as np
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.memory.schema import Episode
from halo_fep.training.lora_trainer import LoRATrainer


def make_episodes(cfg, n=3):
    return [
        Episode(
            query=f"ep{i}",
            tokens=np.random.randn(cfg.n_tokens, cfg.d_model).astype(np.float32),
            swarm_mu=np.random.randn(cfg.n_agents, cfg.n_hidden).astype(np.float32),
            free_energy=1.0,
            free_energy_delta=-0.2,
        )
        for i in range(n)
    ]


def test_run_returns_model():
    cfg = HaloFEPConfig(n_tokens=32)  # tiny config for speed
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    episodes = make_episodes(cfg, n=2)
    new_model, log = trainer.run(model, episodes)
    assert new_model is not None
    assert "loss_before" in log
    assert "loss_after" in log
    assert "n_episodes" in log


def test_run_logs_episode_count():
    cfg = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    episodes = make_episodes(cfg, n=4)
    _, log = trainer.run(model, episodes)
    assert log["n_episodes"] == 4


def test_run_empty_episodes_returns_same_model():
    cfg = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    new_model, log = trainer.run(model, [])
    # Should return original model unchanged
    assert log["n_episodes"] == 0
