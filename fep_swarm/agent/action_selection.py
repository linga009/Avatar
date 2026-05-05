import jax
import jax.numpy as jnp
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.config import FEPConfig


def expected_free_energy(
    mu: jnp.ndarray,       # [n_hidden]
    policy: jnp.ndarray,   # [tau, n_actions] one-hot actions
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> tuple:
    """
    G(π) = Pragmatic + Epistemic
    Pragmatic = −DKL[Q(s_τ|π) || P(s_τ)]   seek preferred observations
    Epistemic = E_Q[H(s_τ|η_τ,π)]           reduce ambiguity about world
    Returns (G, pragmatic, epistemic).
    """
    q_eta = jax.nn.softmax(mu)  # [n_hidden]

    def rollout_step(q, action_onehot):
        a = jnp.argmax(action_onehot)
        q_next = gm.B[:, :, a] @ q  # [n_hidden]
        return q_next, q_next

    _, q_traj = jax.lax.scan(rollout_step, q_eta, policy)
    q_eta_tau = q_traj[-1]                       # [n_hidden]
    q_obs_tau = gm.A @ q_eta_tau                 # [n_obs]
    p_obs = jax.nn.softmax(gm.log_C)             # [n_obs]

    pragmatic = -jnp.sum(
        q_obs_tau * (jnp.log(q_obs_tau + 1e-8) - jnp.log(p_obs + 1e-8))
    )

    # H(s|η) = −Σ_s A[s,η] ln A[s,η]  per hidden state
    H_s_given_eta = -jnp.sum(gm.A * jnp.log(gm.A + 1e-8), axis=0)  # [n_hidden]
    epistemic = jnp.sum(q_eta_tau * H_s_given_eta)

    return pragmatic + epistemic, pragmatic, epistemic


def build_policies(cfg: FEPConfig, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Build [n_policies, tau, n_actions] random one-hot action sequences."""
    n = cfg.n_policies * cfg.tau
    keys = jax.random.split(key, n)
    actions = jax.vmap(
        lambda k: jax.nn.one_hot(
            jax.random.randint(k, (), 0, cfg.n_actions), cfg.n_actions
        )
    )(keys)
    return actions.reshape(cfg.n_policies, cfg.tau, cfg.n_actions)


def select_action(
    mu: jnp.ndarray,
    policies: jnp.ndarray,  # [n_policies, tau, n_actions]
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
    key: jax.random.PRNGKey,
) -> tuple:
    """
    Compute G(π) for all policies, sample π* ~ softmax(−β·G).
    Returns (action [n_actions], G_all [n_policies]).
    """
    G_all, _, _ = jax.vmap(
        lambda p: expected_free_energy(mu, p, gm, cfg)
    )(policies)

    pi_dist = jax.nn.softmax(-cfg.beta * G_all)
    pi_idx = jax.random.choice(key, cfg.n_policies, p=pi_dist)
    action = policies[pi_idx, 0]  # first action of selected policy
    return action, G_all
