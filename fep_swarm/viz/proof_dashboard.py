from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless for tests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import jax.numpy as jnp


def plot_proof_dashboard(
    F_history: list,
    S_history: list,
    F_macro_history: list,
    F_micro_sum_history: list,
    I_sync_history: list,
    eigenvalue_magnitudes: jnp.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel proof figure validating all 4 FEP-Swarm layers:
    1. F(t) decreasing        - Layer 2 belief convergence
    2. S(t) -> 0              - Layer 3 generalized synchrony
    3. F_macro <= sum(Fi) - I  - Layer 4 macro bound
    4. Eigenvalue gap          - Layer 4 time-scale separation
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("FEP-Swarm: 4-Layer Emergence Proof", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    steps = list(range(len(F_history)))

    # Panel 1: Free energy convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, F_history, color="#4ad9a0", linewidth=2)
    ax1.set_xlabel("Simulation step")
    ax1.set_ylabel("Mean F(mu, s)")
    ax1.set_title("Layer 2: Free Energy Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, "should decrease", transform=ax1.transAxes,
             color="gray", fontsize=9, va="top")

    # Panel 2: Synchrony
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, S_history, color="#f77e7e", linewidth=2)
    ax2.set_xlabel("Simulation step")
    ax2.set_ylabel("S(t)")
    ax2.set_title("Layer 3: Generalized Synchrony")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, "should converge to 0", transform=ax2.transAxes,
             color="gray", fontsize=9, va="top")

    # Panel 3: Macro bound
    ax3 = fig.add_subplot(gs[1, 0])
    rhs = [f - i for f, i in zip(F_micro_sum_history, I_sync_history)]
    ax3.plot(steps, F_macro_history, color="#7eb8f7", linewidth=2, label="F_macro")
    ax3.plot(steps, rhs, color="#f7e87e", linewidth=2, linestyle="--", label="sum(Fi) - I")
    ax3.set_xlabel("Simulation step")
    ax3.set_ylabel("Free Energy")
    ax3.set_title("Layer 4: Bound F_macro <= sum(Fi) - I")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.05, 0.95, "blue <= yellow = proof holds",
             transform=ax3.transAxes, color="gray", fontsize=9, va="top")

    # Panel 4: Eigenvalue spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    mags = np.array(eigenvalue_magnitudes)
    mags_sorted = np.sort(mags)[::-1]
    n = len(mags_sorted)
    colors = ["#d94a4a"] * (n // 2) + ["#4a90d9"] * (n - n // 2)
    ax4.bar(range(n), mags_sorted, color=colors, width=1.0, alpha=0.8)
    ax4.set_xlabel("Eigenvalue index (sorted)")
    ax4.set_ylabel("|lambda|")
    ax4.set_title("Layer 4: Time-Scale Separation")
    ax4.grid(True, alpha=0.3, axis="y")
    gap = mags_sorted[0] / (mags_sorted[-1] + 1e-8)
    ax4.text(0.05, 0.95, f"Gap: {gap:.1f}x (need >=10x)",
             transform=ax4.transAxes, color="gray", fontsize=9, va="top")
    ax4.legend(handles=[
        Patch(facecolor="#d94a4a", label="fast (micro)"),
        Patch(facecolor="#4a90d9", label="slow (macro)"),
    ], fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
