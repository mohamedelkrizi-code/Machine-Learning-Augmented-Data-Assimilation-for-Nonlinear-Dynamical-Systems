"""
Step 4 experiment: CNN-augmented EnKF for high-resolution observations.

Inspired by Howard, Subramanian & Hoteit (2023), "A Machine Learning
Augmented Data Assimilation Method for High-Resolution Observation".

Question: dense/high-resolution observations are often too expensive to
assimilate operationally and get discarded. Can a small CNN, trained
offline to reproduce what a full EnKF analysis would look like, be used
online to cheaply assimilate that otherwise-wasted high-resolution data
alongside a standard EnKF handling routine low-resolution observations?

Three methods compared on a held-out test period:
    1. EnKF AllObs     -- upper bound: full obs assimilated by EnKF every step
    2. EnKF SparseObs  -- realistic baseline: high-res obs discarded
    3. Augmented        -- EnKF (low-res) + CNN (high-res), alternating steps

Run with:
    python experiments/step4_cnn_augmented_enkf.py
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
from src.evaluation import (
    rmse, rmse_timeseries, plot_rmse_over_time, plot_state_reconstruction,
    print_summary_table,
)
from src.cnn_augmented import (
    generate_cnn_training_data, train_analysis_cnn,
    run_augmented_enkf, run_sparse_enkf_baseline,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # ---- Configuration -----------------------------------------------------
    N = 40
    F = 8.0
    dt = 0.05
    n_train_steps = 3000
    n_test_steps = 400

    obs_std = 0.5
    ensemble_size = 30  # same as Stage A/Step 3 -- covariance localization below (not a
                        # bigger ensemble) is what keeps the filter stable under sparse obs
    init_spread_std = 2.0
    inflation = 1.05
    localization_radius = 10  # Gaspari-Cohn taper reaches 0 at 2*radius=20 grid points away;
                               # without this, ensemble_size=30 diverges under this sparse
                               # (25%-space, half-frequency) observation network (see README)

    print("=" * 60)
    print("STEP 4: CNN-augmented EnKF for high-resolution observations")
    print("=" * 60)

    # ---- 1. Nature run: spin-up -> training period -> test period ---------
    t_full, full_traj = generate_true_trajectory(
        N=N, F=F, dt=dt, n_steps=n_train_steps + n_test_steps, spinup_steps=300, seed=5
    )
    train_traj = full_traj[: n_train_steps + 1]
    test_traj = full_traj[n_train_steps:]
    t_test = t_full[n_train_steps:]
    print(f"Nature run: {n_train_steps} training steps + {n_test_steps} test steps generated.")

    # ---- 2. Observation operators -------------------------------------------
    H_low, low_indices = make_observation_operator(N, obs_every=4)  # 25% of variables ("low-res")
    H_full = np.eye(N)                                              # 100% of variables ("high-res")
    print(f"Low-resolution obs: {H_low.shape[0]}/{N} variables. "
          f"High-resolution obs: {N}/{N} variables (discarded by the baseline).")

    model_step_fn = lambda x: step(x, dt, F=F)

    # ---- 3. Build CNN training data ON-POLICY -------------------------------
    # Train on the SAME alternating cadence the CNN sees online (EnKF on
    # low-res odd steps, real EnKF-on-full-obs as the target on even steps),
    # not on forecasts from an EnKF that always sees full obs. Training on
    # the latter looks fine offline but generalizes poorly online: those
    # forecasts have much smaller errors than what the CNN actually has to
    # correct once observations are this sparse (the exact distribution-
    # mismatch limitation Howard et al. 2023 flag in their Discussion, Sec 4).
    obs_times_low_tr, obs_values_low_tr, R_low = generate_observations(
        train_traj, H_low, obs_std=obs_std, obs_every_steps=1, seed=888
    )
    obs_times_full_tr, obs_values_full_tr, R_full = generate_observations(
        train_traj, H_full, obs_std=obs_std, obs_every_steps=1, seed=777
    )
    X, y = generate_cnn_training_data(
        train_traj, H_low, R_low, obs_values_low_tr, H_full, R_full, obs_values_full_tr,
        model_step_fn, ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=42,
    )
    print(f"Generated {X.shape[0]} on-policy (forecast, innovation) -> analysis training pairs.")

    # ---- 4. Train the CNN ----------------------------------------------------
    cnn_model, history = train_analysis_cnn(X, y, val_frac=0.2, epochs=600, lr=5e-3, seed=0)
    n_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"Trained AnalysisCNN ({n_params} trainable weights). "
          f"Final train RMSE={history['train_loss'][-1]:.4f}, val RMSE={history['val_loss'][-1]:.4f}")

    # ---- 5. Pre-generate shared test-period observations (fair comparison) ---
    obs_times_low, obs_values_low, R_low_test = generate_observations(
        test_traj, H_low, obs_std=obs_std, obs_every_steps=1, seed=88
    )
    obs_times_full, obs_values_full, R_full_test = generate_observations(
        test_traj, H_full, obs_std=obs_std, obs_every_steps=1, seed=77
    )

    # ---- 6. Method 1: EnKF AllObs (upper bound) -------------------------------
    enkf_allobs_mean, _ = run_enkf(
        test_traj, obs_times_full, obs_values_full, H_full, R_full_test, model_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=123,
    )

    # ---- 7. Method 2: EnKF SparseObs (realistic baseline, high-res discarded) -
    sparse_mean = run_sparse_enkf_baseline(
        test_traj, H_low, R_low_test, obs_values_low, model_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=123,
    )

    # ---- 8. Method 3: Augmented (EnKF low-res + CNN high-res) -----------------
    augmented_mean, augmented_spread = run_augmented_enkf(
        test_traj, H_low, R_low_test, obs_values_low, H_full, obs_values_full,
        cnn_model, model_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, localization_radius=localization_radius, seed=123,
    )

    # ---- 9. Score --------------------------------------------------------------
    results = {
        "EnKF AllObs (upper bound)": rmse(enkf_allobs_mean, test_traj),
        "EnKF SparseObs (baseline)": rmse(sparse_mean, test_traj),
        "Augmented (EnKF+CNN)": rmse(augmented_mean, test_traj),
    }
    print()
    print_summary_table(results)
    improvement = 100 * (
        (results["EnKF SparseObs (baseline)"] - results["Augmented (EnKF+CNN)"])
        / results["EnKF SparseObs (baseline)"]
    )
    print(f"\nAugmented method reduces RMSE vs. EnKF SparseObs by {improvement:.1f}%")
    print(f"For reference, climatological std of the state: {test_traj.std():.3f}")

    # ---- 10. Plots ----------------------------------------------------------------
    rmse_dict = {
        "EnKF AllObs": rmse_timeseries(enkf_allobs_mean, test_traj),
        "EnKF SparseObs": rmse_timeseries(sparse_mean, test_traj),
        "Augmented (EnKF+CNN)": rmse_timeseries(augmented_mean, test_traj),
    }
    plot_rmse_over_time(t_test, rmse_dict, title="Step 4: RMSE over time (3 methods)",
                         save_path=os.path.join(FIG_DIR, "step4_rmse_over_time.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(results.keys())
    vals = list(results.values())
    colors = ["tab:gray", "tab:red", "tab:green"]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("RMSE (whole test period)")
    ax.set_title("Step 4: overall RMSE comparison")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step4_rmse_bar_comparison.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs, history["train_loss"], label="Training (vs. EnKF-AllObs analysis)")
    ax2.plot(epochs, history["val_loss"], label="Validation (vs. EnKF-AllObs analysis)")
    ax2.set_xlabel("Training epoch")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Step 4: CNN training curve")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "step4_cnn_training_curve.png"), dpi=150)

    plot_state_reconstruction(
        t_test, test_traj, augmented_mean, augmented_spread,
        obs_times=obs_times_low, obs_values=obs_values_low, obs_indices=low_indices,
        var_idx=0, title="Step 4: Augmented (EnKF+CNN) reconstruction (variable 0)",
        save_path=os.path.join(FIG_DIR, "step4_augmented_reconstruction_var0.png"),
    )

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")

    return {
        "t_test": t_test, "test_traj": test_traj, "results": results,
        "enkf_allobs_mean": enkf_allobs_mean, "sparse_mean": sparse_mean,
        "augmented_mean": augmented_mean, "history": history,
    }


if __name__ == "__main__":
    main()
