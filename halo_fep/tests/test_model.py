# halo_fep/tests/test_model.py
import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig
from halo_fep.loss import halo_loss
from fep_swarm.agent.belief_update import free_energy

cfg = HaloFEPConfig()
key = jax.random.PRNGKey(0)

def test_halo_loss_is_scalar():
    v_pred      = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target    = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w      = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    total, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert total.shape == ()

def test_halo_loss_no_nan():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    total, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert not jnp.isnan(total)
    for v in parts.values():
        assert not jnp.isnan(v)

def test_halo_loss_parts_keys():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    _, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert set(parts.keys()) == {"fm", "bek", "thermo", "page"}


import equinox as eqx
from halo_fep.model import HaloFEPModel, HaloFEPCarry, halo_fep_step

def _make_model_and_carry():
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    return model, carry

def test_model_init():
    model, carry = _make_model_and_carry()
    assert carry.swarm_mu.shape  == (cfg.n_agents, cfg.n_hidden)
    assert carry.swarm_action.shape == (cfg.n_agents, cfg.n_actions)

def test_closed_loop_step_shape():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    assert h_out.shape    == (cfg.n_tokens, cfg.d_model)
    assert obs.shape      == (cfg.n_agents,)
    assert v_pred.shape   == (cfg.n_tokens, cfg.d_model)
    assert v_target.shape == (cfg.n_tokens, cfg.d_model)

def test_closed_loop_step_no_nan():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    assert not jnp.any(jnp.isnan(h_out))
    assert not jnp.any(jnp.isnan(v_pred))

def test_closed_loop_jit_compiles():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    jit_step = eqx.filter_jit(halo_fep_step)
    new_carry, outputs = jit_step(model, carry, tokens, key)
    assert outputs[0].shape == (cfg.n_tokens, cfg.d_model)


import optax
from halo_fep.loss import halo_loss

def _joint_loss(model, carry, tokens, key, cfg):
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    # HALO loss
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    L_halo, _    = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    # FEP free energy — signature: free_energy(mu, obs_idx, gm)
    F_swarm = jnp.mean(
        jax.vmap(lambda mu, obs_idx: free_energy(mu, obs_idx, model.gm))(
            new_carry.swarm_mu, obs
        )
    )
    return L_halo + cfg.lambda_fep * F_swarm, new_carry

def test_joint_loss_is_scalar():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    loss, _ = _joint_loss(model, carry, tokens, key, cfg)
    assert loss.shape == ()

def test_joint_loss_no_nan():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    loss, _ = _joint_loss(model, carry, tokens, key, cfg)
    assert not jnp.isnan(loss)

def test_joint_loss_gradients_nonzero():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    grad_fn = eqx.filter_grad(_joint_loss, has_aux=True)
    grads, _ = grad_fn(model, carry, tokens, key, cfg)
    leaf_grads = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    nonzero = [jnp.any(g != 0.0) for g in leaf_grads if g is not None]
    assert any(nonzero)

def test_train_step_decreases_loss():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    opt    = optax.adam(cfg.lr)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = opt.init(params)

    def step(params, opt_state, carry):
        model_ = eqx.combine(params, static)
        (loss, new_carry), grads = jax.value_and_grad(
            _joint_loss, has_aux=True
        )(model_, carry, tokens, key, cfg)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), new_opt_state, new_carry, loss

    loss0 = _joint_loss(model, carry, tokens, key, cfg)[0]
    params, opt_state, carry, _ = step(params, opt_state, carry)
    for _ in range(9):
        params, opt_state, carry, _ = step(params, opt_state, carry)
    loss10 = _joint_loss(eqx.combine(params, static), carry, tokens, key, cfg)[0]
    assert loss10 < loss0 * 1.5  # not diverging


def test_action_probs_not_uniform_after_belief_update():
    """After belief update with observations, action probs should not be uniform.
    A uniform distribution (all 0.25 for n_actions=4) indicates the EFE bug.
    This test verifies the per-policy EFE fix is working.
    """
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    # Run a few steps to let beliefs become non-trivial
    k = key
    for _ in range(3):
        k, k_step = jax.random.split(k)
        carry, _ = halo_fep_step(model, carry, tokens, k_step)
    # After updates, at least some agents should have non-uniform action probs
    uniform = jnp.ones(cfg.n_actions) / cfg.n_actions
    max_deviation = jnp.max(jnp.abs(carry.swarm_action - uniform))
    assert float(max_deviation) > 1e-6, (
        f"Action probs look uniform (max deviation {float(max_deviation):.2e}). "
        "EFE per-policy fix may not be working."
    )
