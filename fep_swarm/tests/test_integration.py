import jax
import jax.numpy as jnp
import pytest
import time
from fep_swarm.config import FEPConfig
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.training.trainer import run_episode
from fep_swarm.viz.proof_dashboard import plot_proof_dashboard


@pytest.fixture
def small_cfg():
    """Small config for fast integration tests."""
    return FEPConfig(n_agents=16, coarse_k=4, n_steps=100)


@pytest.fixture
def small_gm(small_cfg):
    return DiscreteGenerativeModel(small_cfg, jax.random.PRNGKey(0))


def test_full_pipeline_no_error(small_cfg, small_gm):
    result = run_episode(small_cfg, small_gm, jax.random.PRNGKey(0), n_steps=50)
    assert len(result["F_history"]) == 50
    assert len(result["S_history"]) == 50
    assert not any(jnp.isnan(jnp.array(result["F_history"])))
    assert not any(jnp.isnan(jnp.array(result["S_history"])))


def test_proof_plots_generate(small_cfg, small_gm, tmp_path):
    result = run_episode(small_cfg, small_gm, jax.random.PRNGKey(1), n_steps=20)
    fig = plot_proof_dashboard(
        F_history=result["F_history"],
        S_history=result["S_history"],
        F_macro_history=result["F_macro_history"],
        F_micro_sum_history=result["F_micro_sum_history"],
        I_sync_history=result["I_sync_history"],
        eigenvalue_magnitudes=result["eigenvalue_magnitudes"],
        save_path=str(tmp_path / "proof.png"),
    )
    assert fig is not None
    assert (tmp_path / "proof.png").exists()


def test_eigenvalue_gap_positive(small_cfg, small_gm):
    result = run_episode(small_cfg, small_gm, jax.random.PRNGKey(2), n_steps=20)
    assert result["eig_gap"] > 0.0


def test_macro_horizon_gte_micro(small_cfg, small_gm):
    result = run_episode(small_cfg, small_gm, jax.random.PRNGKey(3), n_steps=20)
    assert result["macro_horizon"] >= result["micro_horizon"]


def test_step_time_acceptable():
    """N=256 agents, 100 steps should complete in < 120s on CPU."""
    cfg = FEPConfig(n_agents=256, coarse_k=16)
    gm = DiscreteGenerativeModel(cfg, jax.random.PRNGKey(0))
    start = time.time()
    run_episode(cfg, gm, jax.random.PRNGKey(0), n_steps=100,
                compute_jacobian_at_end=False)
    elapsed = time.time() - start
    assert elapsed < 120.0, f"Took {elapsed:.1f}s, expected < 120s"
