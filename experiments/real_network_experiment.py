"""
Real-observation-network experiment: does the EnKF (+ localization) still
work when the observation network is a REAL, non-uniform, gap-riddled
one instead of the synthetic "every k-th variable" pattern used
everywhere else in this project?

The Lorenz-96 dynamics are still synthetic (so we still know the exact
truth and can score RMSE honestly -- see README, "A note on the data").
What changes here is only the observation operator H and noise R: instead
of picking them by hand, they come from `src/real_observation_network.py`,
which is built from real NOAA GHCN-Daily weather-station locations in a
real 49-51N latitude band, with real gaps over open ocean and real,
heterogeneous per-station noise levels (reference-network vs. ordinary
co-op stations).

Two EnKF runs are compared on the SAME true trajectory:
    1. Synthetic network  -- observations.make_observation_operator (every 2nd variable, 50%)
    2. Real network        -- real_observation_network (28/40 = 70%, but patchy/clustered, not evenly spaced)

Run with:
    python experiments/real_network_experiment.py
Figures are written to ../figures/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lorenz96 import generate_true_trajectory, step
from src.observations import make_observation_operator, generate_observations
from src.enkf import run_enkf
from src.evaluation import rmse, rmse_timeseries, plot_rmse_over_time, plot_state_reconstruction, print_summary_table
from src.real_observation_network import load_real_station_network, generate_real_network_observations

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    N = 40
    F = 8.0
    dt = 0.05
    n_steps = 400

    ensemble_size = 30
    init_spread_std = 2.0
    inflation = 1.05
    localization_radius = 10

    print("=" * 60)
    print("REAL OBSERVATION NETWORK: EnKF with real GHCN-Daily station geometry")
    print("=" * 60)

    # ---- 1. True trajectory (same synthetic L96 dynamics as the rest of the project)
    t, true_traj = generate_true_trajectory(N=N, F=F, dt=dt, n_steps=n_steps, seed=0)
    model_step_fn = lambda x: step(x, dt, F=F)

    # ---- 2. Real network -------------------------------------------------------
    H_real, R_real, real_indices, stations = load_real_station_network(N=N)
    obs_times_real, obs_values_real = generate_real_network_observations(true_traj, H_real, R_real, seed=1)
    print(f"Real network: {H_real.shape[0]}/{N} grid sectors have a real station "
          f"(49-51N band, real GHCN-Daily data). Gaps at sectors: "
          f"{sorted(set(range(N)) - set(real_indices.tolist()))}")

    # ---- 3. Synthetic network for comparison (Stage A's usual pattern) --------
    H_synth, synth_indices = make_observation_operator(N, obs_every=2)
    obs_times_synth, obs_values_synth, R_synth = generate_observations(
        true_traj, H_synth, obs_std=0.5, obs_every_steps=1, seed=1
    )
    print(f"Synthetic network (comparison): {H_synth.shape[0]}/{N}, evenly spaced.")

    # ---- 4. Run EnKF (with localization) on both -------------------------------
    mean_real, spread_real = run_enkf(
        true_traj, obs_times_real, obs_values_real, H_real, R_real, model_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=42,
    )
    mean_synth, spread_synth = run_enkf(
        true_traj, obs_times_synth, obs_values_synth, H_synth, R_synth, model_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=42,
    )

    # ---- 5. Score ----------------------------------------------------------------
    results = {
        f"Synthetic network ({H_synth.shape[0]}/{N}, even spacing)": rmse(mean_synth, true_traj),
        f"Real network ({H_real.shape[0]}/{N}, real GHCN stations)": rmse(mean_real, true_traj),
    }
    print()
    print_summary_table(results)
    print(f"\nFor reference, climatological std of the state: {true_traj.std():.3f}")

    # ---- 6. Plots ------------------------------------------------------------------
    rmse_dict = {
        "Synthetic network": rmse_timeseries(mean_synth, true_traj),
        "Real network": rmse_timeseries(mean_real, true_traj),
    }
    plot_rmse_over_time(t, rmse_dict, title="Real vs. synthetic observation network: RMSE over time",
                         save_path=os.path.join(FIG_DIR, "real_network_rmse_over_time.png"))

    # coverage map: which of the 40 sectors are observed, real vs synthetic
    fig, ax = plt.subplots(figsize=(10, 2.8))
    lons = np.array([-180 + (i + 0.5) * (360 / N) for i in range(N)])
    ax.scatter(lons, np.ones(N) * 1, color="lightgray", s=40, label="unobserved sector")
    ax.scatter(lons[synth_indices], np.ones(len(synth_indices)) * 1.15, color="tab:blue",
               marker="^", s=60, label="synthetic network (even spacing)")
    ax.scatter(lons[real_indices], np.ones(len(real_indices)) * 0.85, color="tab:red",
               marker="o", s=60, label="real network (GHCN-Daily, 49-51N)")
    ax.set_yticks([])
    ax.set_xlabel("Longitude (deg) -- Lorenz-96 ring mapped onto a real 49-51N latitude band")
    ax.set_title("Real vs. synthetic observation coverage around the ring")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "real_network_coverage_map.png"), dpi=150)

    plot_state_reconstruction(
        t, true_traj, mean_real, spread_real,
        obs_times=obs_times_real, obs_values=obs_values_real, obs_indices=real_indices,
        var_idx=int(real_indices[0]), title="Real network: EnKF reconstruction at an observed real station",
        save_path=os.path.join(FIG_DIR, "real_network_reconstruction.png"),
    )

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")

    return {"t": t, "true_traj": true_traj, "results": results,
            "mean_real": mean_real, "mean_synth": mean_synth, "stations": stations}


if __name__ == "__main__":
    main()
