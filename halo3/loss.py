"""Halo3 loss — reconstruction + energy conservation + sync."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from halo3.kuramoto import order_parameter


def halo3_loss(model, carry, tokens, key):
    from halo3.model import halo3_step
    from halo3.hamiltonian import leapfrog_integrate

    new_carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, key)

    # Reconstruction: leapfrog should bring q back near original embedding
    l_recon = jnp.mean((q_final - q_data) ** 2)

    # Energy conservation: H(q0, p0) ≈ H(q_final, p_final)
    p0 = model.momentum_init(h_out)
    E0 = model.hamiltonian(q_data, p0)
    q_check, p_check = leapfrog_integrate(
        model.hamiltonian, q_data, p0,
        model.cfg.n_leapfrog_steps, model.cfg.leapfrog_step_size
    )
    Ef = model.hamiltonian(q_check, p_check)
    l_energy = (Ef - E0) ** 2

    # Synchrony: maximise order parameter
    r = order_parameter(new_carry.kuramoto.theta)
    l_sync = -jnp.mean(r)

    total = l_recon + model.cfg.lambda_energy * l_energy + model.cfg.lambda_sync * l_sync
    return total, {"l_recon": l_recon, "l_energy": l_energy, "l_sync": l_sync}
