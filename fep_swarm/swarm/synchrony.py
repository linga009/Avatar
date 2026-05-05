import jax
import jax.numpy as jnp


def synchrony_metric(
    mu: jnp.ndarray,       # [N, n_hidden]
    mu_prev: jnp.ndarray,  # [N, n_hidden]
) -> jnp.ndarray:
    """
    S(t) = ||μ̇_A − μ̇_B||_F / N²
    Pairwise Frobenius norm of belief-rate differences, normalized.
    Lower = more synchronized.
    """
    mu_dot = mu - mu_prev                             # [N, n_hidden]
    diff = mu_dot[:, None, :] - mu_dot[None, :, :]   # [N, N, n_hidden]
    frob = jnp.sqrt(jnp.sum(diff ** 2))
    N = mu.shape[0]
    return frob / (N ** 2)


def mutual_information_estimate(
    mu_history: jnp.ndarray,  # [T, N, n_hidden]
    n_bins: int = 10,
) -> jnp.ndarray:
    """
    Estimate average pairwise MI between agents via binned 1D projections.
    Uses mean over hidden dim as the 1D projection.
    """
    mu_1d = mu_history.mean(axis=-1)  # [T, N]
    ref = mu_1d[:, 0]                  # [T] reference agent

    def pairwise_mi(other: jnp.ndarray) -> jnp.ndarray:
        def to_bins(x):
            lo, hi = x.min(), x.max()
            b = jnp.floor((x - lo) / (hi - lo + 1e-8) * n_bins).astype(int)
            return jnp.clip(b, 0, n_bins - 1)

        rb = to_bins(ref)
        ob = to_bins(other)
        joint = jnp.zeros((n_bins, n_bins)).at[rb, ob].add(1.0)
        joint = joint / (joint.sum() + 1e-10)
        p_r = joint.sum(axis=1)
        p_o = joint.sum(axis=0)
        outer = p_r[:, None] * p_o[None, :]
        mi = jnp.where(
            joint > 1e-10,
            joint * jnp.log(joint / (outer + 1e-10)),
            0.0,
        ).sum()
        return mi

    N = mu_1d.shape[1]
    mi_vals = jax.vmap(pairwise_mi)(mu_1d[:, 1:].T)  # [N-1]
    return mi_vals.mean()
