"""
Evaluation utilities: RMSE scoring and diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt


def rmse(estimate, truth):
    """Root-mean-square error between an estimate and the true trajectory.
    Averages over both time and state dimension."""
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def rmse_timeseries(estimate, truth):
    """RMSE at each time step, averaged over the state dimension only.
    Useful for plotting error growth/decay over time."""
    return np.sqrt(np.mean((estimate - truth) ** 2, axis=1))


def plot_state_reconstruction(t, true_traj, analysis_mean, ensemble_spread,
                               obs_times=None, obs_values=None, obs_indices=None,
                               var_idx=0, title=None, save_path=None):
    """
    Plot true state, EnKF estimate (+/- uncertainty band), and observations
    for a single state variable over time. This is the core Stage-A figure:
    "can I recover the true state from imperfect observations?"
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, true_traj[:, var_idx], color="black", lw=2, label="True state")
    ax.plot(t, analysis_mean[:, var_idx], color="tab:blue", lw=1.8, ls="--",
            label="EnKF estimate")
    ax.fill_between(
        t,
        analysis_mean[:, var_idx] - ensemble_spread[:, var_idx],
        analysis_mean[:, var_idx] + ensemble_spread[:, var_idx],
        color="tab:blue", alpha=0.2, label="Ensemble spread (\u00b11 std)"
    )

    if obs_times is not None and obs_values is not None and obs_indices is not None:
        if var_idx in obs_indices:
            col = list(obs_indices).index(var_idx)
            obs_t = [t[i] for i in obs_times]
            obs_v = [ov[col] for ov in obs_values]
            ax.scatter(obs_t, obs_v, color="tab:red", s=18, zorder=5, label="Observations")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"$x_{{{var_idx}}}$")
    ax.set_title(title or f"State reconstruction (variable {var_idx})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rmse_over_time(t, rmse_series_dict, title="RMSE over time", save_path=None):
    """
    Plot RMSE(t) for one or more methods on the same axes.

    rmse_series_dict : dict[str, np.ndarray]
        Maps method name -> RMSE time series (shape (T,)).
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, series in rmse_series_dict.items():
        ax.plot(t, series, lw=1.8, label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_hovmoller(t, true_traj, analysis_mean, save_path=None):
    """
    Space-time ("Hovmoller") view of the full state: true vs EnKF estimate
    vs absolute error, side by side. Good for showing recovery across ALL
    variables at once, not just one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    vmin, vmax = true_traj.min(), true_traj.max()

    im0 = axes[0].imshow(true_traj.T, aspect="auto", origin="lower",
                          extent=[t[0], t[-1], 0, true_traj.shape[1]],
                          cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("True state")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Variable index")

    axes[1].imshow(analysis_mean.T, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], 0, true_traj.shape[1]],
                   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("EnKF estimate")
    axes[1].set_xlabel("Time")

    err = np.abs(analysis_mean - true_traj)
    im2 = axes[2].imshow(err.T, aspect="auto", origin="lower",
                          extent=[t[0], t[-1], 0, true_traj.shape[1]],
                          cmap="viridis")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("Time")

    fig.colorbar(im0, ax=axes[:2].tolist(), shrink=0.85, label="State value")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="|error|")

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def print_summary_table(results_dict):
    """
    results_dict : dict[str, float]
        Maps method name -> scalar RMSE.
    Prints a simple aligned table to stdout and returns it as a string.
    """
    lines = ["Method".ljust(20) + "RMSE", "-" * 30]
    for name, val in results_dict.items():
        lines.append(f"{name:<20}{val:.4f}")
    table = "\n".join(lines)
    print(table)
    return table
