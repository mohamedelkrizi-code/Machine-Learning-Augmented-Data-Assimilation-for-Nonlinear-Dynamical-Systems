"""
Sensor-placement optimization: given the 28 real GHCN candidate station
sites used in `real_observation_network.py`, which subset should actually
be instrumented to minimize reconstruction RMSE across the full 40-point
ring, at a given sensor budget K?

This directly follows up on the real-network finding that 28 real
(clustered) stations reconstructed the ring WORSE than 20 evenly-spaced
synthetic ones -- the natural question that leaves open is whether
picking the RIGHT real subset (rather than "whichever real candidates
happen to exist nearby") can close or beat that gap.

Two phases:
  1. SEARCH (cheap proxy settings: short trajectory, small ensemble) --
     greedy forward selection builds the full budget-vs-RMSE curve for
     k=1..28, and a random-subset baseline is computed for comparison.
  2. VALIDATE (full settings, matching real_network_experiment.py exactly)
     -- the greedy-selected subsets at a few interesting budgets are
     re-scored at full accuracy, directly comparable to the earlier
     0.195 (synthetic 20/40) / 0.285 (real, all 28/40) numbers.

Run with:
    python experiments/sensor_placement_experiment.py
Figures are written to ../figures/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lorenz96 import generate_true_trajectory, step
from src.observations import make_observation_operator
from src.enkf import run_enkf
from src.evaluation import rmse, print_summary_table
from src.real_observation_network import load_real_station_network
from src.sensor_placement import greedy_sensor_placement, random_subset_baseline, _evaluate_subset

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    N = 40
    F = 8.0
    dt = 0.05
    inflation = 1.05
    localization_radius = 10

    print("=" * 60)
    print("SENSOR-PLACEMENT OPTIMIZATION: which real stations to instrument?")
    print("=" * 60)

    H_real, R_real, real_idx, stations = load_real_station_network(N=N)
    R_diag = np.diag(R_real)
    n_candidates = H_real.shape[0]
    print(f"{n_candidates} real candidate stations available (same pool as real_observation_network.py).")

    # ---- PHASE 1: search, cheap proxy settings ---------------------------------
    t_proxy, traj_proxy = generate_true_trajectory(N=N, F=F, dt=dt, n_steps=80, seed=0)
    model_step_fn = lambda x: step(x, dt, F=F)
    proxy_kwargs = dict(ensemble_size=15, init_spread_std=2.0, inflation=inflation,
                         localization_radius=localization_radius, seed=42)

    print("\nPhase 1: greedy forward selection (proxy settings: 80 steps, ensemble=15)...")
    history = greedy_sensor_placement(traj_proxy, H_real, R_diag, model_step_fn,
                                       max_k=n_candidates, **proxy_kwargs)

    H_synth20, synth_idx20 = make_observation_operator(N, obs_every=2)
    R_synth20_diag = np.full(H_synth20.shape[0], 0.5 ** 2)
    synth_rmse_proxy = _evaluate_subset(traj_proxy, H_synth20, R_synth20_diag,
                                         list(range(H_synth20.shape[0])), model_step_fn,
                                         **proxy_kwargs)

    rand_mean, rand_std = random_subset_baseline(traj_proxy, H_real, R_diag, model_step_fn,
                                                  k=20, n_trials=10, **proxy_kwargs)
    greedy_k20_proxy = next(h["rmse"] for h in history if h["k"] == 20)
    print(f"At K=20 (proxy scale): greedy={greedy_k20_proxy:.3f}, "
          f"random subset avg={rand_mean:.3f} (+/-{rand_std:.3f}), "
          f"synthetic 20/40 evenly-spaced={synth_rmse_proxy:.3f}")

    # ---- PHASE 2: validate the winning subsets at FULL settings ----------------
    print("\nPhase 2: validating greedy-selected subsets at full settings "
          "(400 steps, ensemble=30) -- comparable to real_network_experiment.py...")
    t_full, traj_full = generate_true_trajectory(N=N, F=F, dt=dt, n_steps=400, seed=0)
    full_kwargs = dict(ensemble_size=30, init_spread_std=2.0, inflation=inflation,
                        localization_radius=localization_radius, seed=42)

    validate_budgets = [10, 14, 20, 28]
    final_results = {}
    for k in validate_budgets:
        subset = next(h["selected"] for h in history if h["k"] == k)
        r = _evaluate_subset(traj_full, H_real, R_diag, subset, model_step_fn, **full_kwargs)
        final_results[f"Greedy-selected real stations, K={k}"] = r

    final_results["Synthetic (20/40, evenly spaced)"] = _evaluate_subset(
        traj_full, H_synth20, R_synth20_diag, list(range(H_synth20.shape[0])), model_step_fn, **full_kwargs
    )
    final_results["All 28 real candidates (no selection)"] = _evaluate_subset(
        traj_full, H_real, R_diag, list(range(n_candidates)), model_step_fn, **full_kwargs
    )

    print()
    print_summary_table(final_results)
    print(f"\nFor reference, climatological std of the state: {traj_full.std():.3f}")

    best_stations_k20 = [stations[i]["name"] for i in next(h["selected"] for h in history if h["k"] == 20)]
    print(f"\nGreedy-selected 20 real stations (in order added): {best_stations_k20}")

    # ---- Plots -------------------------------------------------------------------
    ks = [h["k"] for h in history]
    rmses = [h["rmse"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, rmses, marker="o", ms=3, color="tab:green", label="Greedy-selected real subset")
    ax.axhline(synth_rmse_proxy, color="tab:blue", ls="--", label="Synthetic 20/40, evenly spaced")
    ax.errorbar([20], [rand_mean], yerr=[rand_std], fmt="s", color="tab:red",
                label="Random 20-of-28 subset (avg +/- std, 10 trials)")
    ax.set_xlabel("Sensor budget K (of 28 real candidates)")
    ax.set_ylabel("RMSE (proxy scale: 80 steps, ensemble=15)")
    ax.set_title("Sensor-placement optimization: budget vs. reconstruction accuracy")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sensor_placement_budget_curve.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    names = list(final_results.keys())
    vals = list(final_results.values())
    colors = ["tab:green"] * len(validate_budgets) + ["tab:blue", "tab:gray"]
    ax2.bar(range(len(names)), vals, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("RMSE (full settings)")
    ax2.set_title("Sensor placement: validated RMSE at full settings")
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "sensor_placement_validated_comparison.png"), dpi=150)

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")

    return {"history": history, "final_results": final_results, "stations": stations}


if __name__ == "__main__":
    main()
