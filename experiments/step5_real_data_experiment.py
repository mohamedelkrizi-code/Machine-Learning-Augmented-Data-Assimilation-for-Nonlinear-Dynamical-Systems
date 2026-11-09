"""
Step 5: fully real data -- real ground truth, real observations, and a
model learned directly from real data (no Lorenz-96 ODE at all).

This is a genuinely different regime from every earlier experiment: Stage
A / Step 3 / Step 4 all score against a synthetic (but exactly known)
Lorenz-96 truth; the "real observation network" addition (Step 4.5) kept
that synthetic truth and only borrowed real station geometry/noise. Here,
the truth itself is real recorded (ERA5 reanalysis) hourly wind speed at
28 real GHCN-Daily station locations, and the forecast model M is a small
neural network trained directly on that real historical series -- there's
no equation to fall back on for real near-surface wind.

Of the 28 real stations, 21 are "observed" (fed to the EnKF, with
realistic instrument noise added) and 7 are held out completely -- never
observed by any method -- so their real recorded values can be used as an
honest generalization test, not just a fit-to-training-data check.

Four things are compared on a real, held-out TIME period (never used for
training the surrogate model):
    1. Persistence            -- x_hat(t) = x_hat(t-1) (freeze the last known state; a famously hard-to-beat real-world baseline)
    2. Learned-M rollout alone -- the trained surrogate model, no observations ever (open-loop, like Step 3's "physical model alone")
    3. EnKF, no localization   -- learned M + real observed stations
    4. EnKF, with localization -- same, + covariance localization

Run with:
    python experiments/step5_real_data_experiment.py
Figures are written to ../figures/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.enkf import run_enkf
from src.evaluation import rmse, rmse_timeseries, plot_rmse_over_time, print_summary_table
from src.real_data_pipeline import (
    load_real_wind_data, split_observed_held_out, build_real_H_R,
    train_surrogate_model, make_surrogate_step_fn, ANEMOMETER_NOISE_STD_KMH,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    ensemble_size = 30
    init_spread_std = 3.0
    inflation = 1.05
    localization_radius = 7  # scaled down from Step 4's 10, since N=28 here (not 40)

    print("=" * 60)
    print("STEP 5: Fully real data (real GT + real obs + learned model)")
    print("=" * 60)

    # ---- 1. Load real data, split train/test in TIME --------------------------
    times, X, grid_indices = load_real_wind_data()
    n_stations = X.shape[1]
    n_train = 1500
    train_data = X[:n_train]
    test_data = X[n_train:]
    test_times = times[n_train:]
    print(f"Real wind-speed data: {X.shape[0]} real hourly values at {n_stations} real stations "
          f"({times[0]} to {times[-1]}). Train: {n_train} hours, test: {len(test_data)} hours.")

    # ---- 2. Observed vs. held-out real stations --------------------------------
    observed_pos, held_out_pos = split_observed_held_out(n_stations=n_stations)
    print(f"{len(observed_pos)}/{n_stations} real stations observed, "
          f"{len(held_out_pos)} held out completely (never given to any method).")

    # ---- 3. Train the surrogate forecast model on REAL training data ----------
    model, scaler_X, scaler_y = train_surrogate_model(train_data)
    step_fn = make_surrogate_step_fn(model, scaler_X, scaler_y)
    print("Trained MLP surrogate model x(t) -> x(t+1) on real training data "
          "(no Lorenz-96 physics involved).")

    # ---- 4. Real, noisy observations of the TEST period (observed stations only)
    H, R = build_real_H_R(n_stations, observed_pos, noise_std=ANEMOMETER_NOISE_STD_KMH)
    rng = np.random.default_rng(1)
    obs_times = list(range(len(test_data)))
    obs_values = [H @ test_data[t] + rng.normal(scale=ANEMOMETER_NOISE_STD_KMH, size=len(observed_pos))
                  for t in obs_times]

    # ---- 5. Method 1: persistence (freeze last known real state) --------------
    persistence = np.zeros_like(test_data)
    persistence[0] = test_data[0]
    for t in range(1, len(test_data)):
        persistence[t] = persistence[t - 1]

    # ---- 6. Method 2: learned-M rollout alone (open-loop, no observations) ----
    ml_only = np.zeros_like(test_data)
    ml_only[0] = test_data[0]
    for t in range(1, len(test_data)):
        ml_only[t] = step_fn(ml_only[t - 1])

    # ---- 7. Method 3/4: EnKF with the learned model, without/with localization
    enkf_nolocal, _ = run_enkf(
        test_data, obs_times, obs_values, H, R, step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=None, seed=42,
    )
    enkf_local, _ = run_enkf(
        test_data, obs_times, obs_values, H, R, step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=42,
    )

    # ---- 8. Score: overall (all 28) AND held-out-only (the honest test) -------
    def rmse_held_out(estimate):
        return rmse(estimate[:, held_out_pos], test_data[:, held_out_pos])

    results_overall = {
        "Persistence": rmse(persistence, test_data),
        "Learned-M rollout alone": rmse(ml_only, test_data),
        "EnKF, no localization": rmse(enkf_nolocal, test_data),
        "EnKF, with localization": rmse(enkf_local, test_data),
    }
    results_held_out = {
        "Persistence": rmse_held_out(persistence),
        "Learned-M rollout alone": rmse_held_out(ml_only),
        "EnKF, no localization": rmse_held_out(enkf_nolocal),
        "EnKF, with localization": rmse_held_out(enkf_local),
    }

    print("\nRMSE over ALL 28 real stations (21 observed + 7 held-out):")
    print_summary_table(results_overall)
    print("\nRMSE over the 7 HELD-OUT real stations only (never observed -- the honest test):")
    print_summary_table(results_held_out)
    print(f"\nFor reference, real test-period wind-speed std: {test_data.std():.3f} km/h")

    # ---- 9. Plots ------------------------------------------------------------------
    rmse_dict = {
        "Persistence": rmse_timeseries(persistence[:, held_out_pos], test_data[:, held_out_pos]),
        "Learned-M alone": rmse_timeseries(ml_only[:, held_out_pos], test_data[:, held_out_pos]),
        "EnKF, no localization": rmse_timeseries(enkf_nolocal[:, held_out_pos], test_data[:, held_out_pos]),
        "EnKF, with localization": rmse_timeseries(enkf_local[:, held_out_pos], test_data[:, held_out_pos]),
    }
    t_hours = np.arange(len(test_data))
    plot_rmse_over_time(t_hours, rmse_dict,
                         title="Step 5: RMSE over time, held-out real stations only",
                         save_path=os.path.join(FIG_DIR, "step5_rmse_over_time_held_out.png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(results_held_out.keys())
    vals = list(results_held_out.values())
    ax.bar(names, vals, color=["tab:gray", "tab:orange", "tab:red", "tab:green"])
    ax.set_ylabel("RMSE (km/h), held-out real stations")
    ax.set_title("Step 5: overall RMSE, held-out real stations only")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step5_rmse_bar_held_out.png"), dpi=150)

    held_out_var = held_out_pos[0]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(t_hours, test_data[:, held_out_var], color="black", lw=2, label="Real recorded truth (held out)")
    ax2.plot(t_hours, persistence[:, held_out_var], color="tab:gray", lw=1.2, ls=":", label="Persistence")
    ax2.plot(t_hours, ml_only[:, held_out_var], color="tab:orange", lw=1.2, ls="--", label="Learned-M alone")
    ax2.plot(t_hours, enkf_local[:, held_out_var], color="tab:green", lw=1.8, label="EnKF + localization")
    ax2.set_xlabel("Hour into test period")
    ax2.set_ylabel("Wind speed (km/h)")
    ax2.set_title(f"Step 5: real held-out station (grid_{grid_indices[held_out_var]}) reconstruction")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "step5_held_out_reconstruction.png"), dpi=150)

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")

    return {
        "test_data": test_data, "results_overall": results_overall,
        "results_held_out": results_held_out, "observed_pos": observed_pos,
        "held_out_pos": held_out_pos,
    }


if __name__ == "__main__":
    main()
