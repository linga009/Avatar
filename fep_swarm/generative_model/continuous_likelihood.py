import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from fep_swarm.config import FEPConfig


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    n_hidden: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(self, cfg: FEPConfig, key: jax.random.PRNGKey):
        self.n_hidden = cfg.n_hidden
        self.obs_dim = cfg.obs_dim
        # input: concat(x [obs_dim], eta_onehot [n_hidden], t [1])
        self.mlp = eqx.nn.MLP(
            in_size=cfg.obs_dim + cfg.n_hidden + 1,
            out_size=cfg.obs_dim,
            width_size=cfg.ode_width,
            depth=cfg.ode_depth,
            key=key,
        )

    def __call__(self, t: float, x: jnp.ndarray, eta_onehot: jnp.ndarray) -> jnp.ndarray:
        inp = jnp.concatenate([x, eta_onehot, jnp.array([t])])
        return self.mlp(inp)


class ContinuousLikelihood(eqx.Module):
    vf: VectorField
    n_hidden: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(self, cfg: FEPConfig, key: jax.random.PRNGKey):
        self.vf = VectorField(cfg, key)
        self.n_hidden = cfg.n_hidden
        self.obs_dim = cfg.obs_dim

    def __call__(
        self,
        eta_idx: int,
        x0: jnp.ndarray,
        use_dopri5: bool = False,
    ) -> jnp.ndarray:
        """Map discrete hidden state index -> continuous obs embedding x(1)."""
        eta_onehot = jax.nn.one_hot(eta_idx, self.n_hidden)
        term = diffrax.ODETerm(lambda t, x, args: self.vf(t, x, args))
        if use_dopri5:
            solver = diffrax.Dopri5()
            controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
            dt0 = None
        else:
            solver = diffrax.Euler()
            controller = diffrax.ConstantStepSize()
            dt0 = 0.1
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=dt0,
            y0=x0, args=eta_onehot,
            stepsize_controller=controller,
            max_steps=100,
        )
        return sol.ys[-1]
