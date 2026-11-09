"""
Stage A experiment: "Can I recover the true state from imperfect observations?"

Pipeline:
  1. Generate the true Lorenz-96 trajectory (SciPy integration).
  2. Generate sparse, noisy observations of it.
  3. Run an Ensemble Kalman Filter to reconstruct the hidden state.
  4. Score with RMSE and produce diagnostic plots.

Run with:
    python experiments/lorenz96_experiment.py
Figures are written to ../figures/.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lorenz96 import generate_true_trajectory, step
from src.observations import make_observation_operator, generate_observations
from src.enkf import run_enkf
from src.evaluation import (
    rmse, rmse_timeseries, plot_state_reconstruction,
    plot_rmse_over_time, plot_hovmoller, print_summary_table
)
from src.ml_correction import (
    make_delta_fn, generate_nature_run, build_training_dataset,
    train_correction_model, make_corrected_step_fn
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # ---- Configuration -----------------------------------------------
    N = 40                  # number of Lorenz-96 variables
    F = 8.0                 # forcing (F=8 -> chaotic regime)
    dt = 0.05                # integration time step
    n_steps = 300             # number of assimilation steps

    obs_every_space = 2       # observe every 2nd variable (20 of 40 observed)
    obs_every_steps = 1       # observe every time step
    obs_std = 0.5             # observation noise std

    ensemble_size = 30
    init_spread_std = 2.0
    inflation = 1.05          # multiplicative covariance inflation

    print("=" * 60)
    print("STAGE A: Lorenz-96 + Ensemble Kalman Filter")
    print("=" * 60)

    # ---- 1. True trajectory -------------------------------------------
    t, true_traj = generate_true_trajectory(N=N, F=F, dt=dt, n_steps=n_steps, seed=0)
    print(f"True trajectory generated: shape={true_traj.shape}")

    # ---- 2. Sparse noisy observations ----------------------------------
    H, obs_indices = make_observation_operator(N, obs_every=obs_every_space)
    obs_times, obs_values, R = generate_observations(
        true_traj, H, obs_std=obs_std, obs_every_steps=obs_every_steps, seed=1
    )
    print(f"Observing {H.shape[0]}/{N} variables, every {obs_every_steps} step(s), "
          f"noise std={obs_std}")

    # ---- 3. EnKF reconstruction -----------------------------------------
    model_step_fn = lambda x: step(x, dt, F=F)
    analysis_mean, ensemble_spread = run_enkf(
        true_traj, obs_times, obs_values, H, R, model_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, seed=42
    )

    # ---- 4. Baseline: raw model, no assimilation at all -----------------
    rng = np.random.default_rng(7)
    naive = np.zeros_like(true_traj)
    naive[0] = true_traj[0] + rng.normal(scale=init_spread_std, size=N)
    for i in range(1, true_traj.shape[0]):
        naive[i] = step(naive[i - 1], dt, F=F)

    # ---- 5. Scoring -------------------------------------------------------
    results = {
        "Physical model (no assim.)": rmse(naive, true_traj),
        "EnKF": rmse(analysis_mean, true_traj),
    }
    print()
    print_summary_table(results)
    print(f"\nFor reference, climatological std of the state: {true_traj.std():.3f}")

    # ---- 6. Plots -----------------------------------------------------------
    plot_state_reconstruction(
        t, true_traj, analysis_mean, ensemble_spread,
        obs_times=obs_times, obs_values=obs_values, obs_indices=obs_indices,
        var_idx=0, title="Stage A: EnKF reconstruction (variable 0)",
        save_path=os.path.join(FIG_DIR, "stageA_reconstruction_var0.png"),
    )

    plot_state_reconstruction(
        t, true_traj, analysis_mean, ensemble_spread,
        obs_times=obs_times, obs_values=obs_values, obs_indices=obs_indices,
        var_idx=1, title="Stage A: EnKF reconstruction (variable 1, unobserved)",
        save_path=os.path.join(FIG_DIR, "stageA_reconstruction_var1.png"),
    )

    rmse_dict = {
        "Physical model (no assim.)": rmse_timeseries(naive, true_traj),
        "EnKF": rmse_timeseries(analysis_mean, true_traj),
    }
    plot_rmse_over_time(t, rmse_dict, title="Stage A: RMSE over time",
                         save_path=os.path.join(FIG_DIR, "stageA_rmse_over_time.png"))

    plot_hovmoller(t, true_traj, analysis_mean,
                   save_path=os.path.join(FIG_DIR, "stageA_hovmoller.png"))

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")

    return {
        "t": t, "true_traj": true_traj, "analysis_mean": analysis_mean,
        "ensemble_spread": ensemble_spread, "naive": naive,
        "obs_times": obs_times, "obs_values": obs_values,
        "H": H, "R": R, "obs_indices": obs_indices, "results": results,
    }


def run_step3_ml_comparison():
    """
    Step 3: "the heart of the project".

    Deliberately corrupt the physics with an unknown model error delta(x),
    then compare 4 ways of dealing with it:
        1. Physical model alone         (wrong, no correction at all)
        2. EnKF                         (wrong model, corrected online by obs)
        3. ML correction alone          (model + learned delta, no live obs)
        4. ML + EnKF                    (model + learned delta, AND corrected by obs)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Machine-learning model-error correction")
    print("=" * 60)

    # ---- Configuration --------------------------------------------------
    N = 40
    F = 8.0
    dt = 0.05
    n_train_steps = 600     # length of the "historical data" used to train the NN
    n_test_steps = 300      # length of the held-out evaluation period

    obs_every_space = 2
    obs_every_steps = 1
    obs_std = 0.5
    ensemble_size = 30
    init_spread_std = 2.0
    inflation = 1.05

    delta_fn = make_delta_fn(a=-0.5, b=0.3)  # the "unknown" hidden physics

    # ---- 1. Generate nature run: spin-up -> training period -> test period ----
    rng = np.random.default_rng(10)
    x0 = F * np.ones(N) + rng.normal(scale=0.5, size=N)
    _, spinup = generate_nature_run(x0, dt, 300, F=F, delta_fn=delta_fn)
    x0_train = spinup[-1]

    _, train_traj = generate_nature_run(x0_train, dt, n_train_steps, F=F, delta_fn=delta_fn)
    x0_test = train_traj[-1]

    t_test, test_traj = generate_nature_run(x0_test, dt, n_test_steps, F=F, delta_fn=delta_fn)
    print(f"Nature run: {n_train_steps} training steps + {n_test_steps} test steps generated.")

    # ---- 2. Train the ML correction on the training period ONLY ---------------
    X, y = build_training_dataset(train_traj, dt, F=F)
    ml_model, scaler_X, scaler_y = train_correction_model(X, y, hidden_layer_sizes=(32, 32))
    print(f"Trained correction network on {X.shape[0]} (state, residual) pairs.")

    base_step_fn = lambda x: step(x, dt, F=F)                       # wrong model M
    corrected_step_fn = make_corrected_step_fn(base_step_fn, ml_model, scaler_X, scaler_y)

    # ---- 3. Observations of the TEST period (from the true/nature trajectory) --
    H, obs_indices = make_observation_operator(N, obs_every=obs_every_space)
    obs_times, obs_values, R = generate_observations(
        test_traj, H, obs_std=obs_std, obs_every_steps=obs_every_steps, seed=99
    )

    # ---- 4. Method 1: physical model alone (wrong, no correction) --------------
    naive = np.zeros_like(test_traj)
    naive[0] = test_traj[0] + rng.normal(scale=init_spread_std, size=N)
    for i in range(1, test_traj.shape[0]):
        naive[i] = base_step_fn(naive[i - 1])

    # ---- 5. Method 2: EnKF (wrong model, corrected online by observations) -----
    # NOTE: model_noise_std > 0 here is important. The model now carries a
    # persistent STRUCTURAL bias (the hidden delta), not just chaotic
    # sensitivity to initial conditions. Without injecting some forecast
    # model-error noise, the ensemble becomes falsely overconfident (the
    # deterministic wrong model collapses the spread), the Kalman gain
    # shrinks, and the filter barely trusts the observations -- a classic
    # "confidently wrong" filter divergence. Injecting model_noise_std
    # tells the filter "I know my model can't be fully trusted", which
    # widens the ensemble and lets it actually use the incoming data.
    enkf_mean, enkf_spread = run_enkf(
        test_traj, obs_times, obs_values, H, R, base_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, model_noise_std=0.3, seed=123
    )

    # ---- 6. Method 3: ML correction alone (no live observations) ---------------
    ml_only = np.zeros_like(test_traj)
    ml_only[0] = test_traj[0] + rng.normal(scale=init_spread_std, size=N)
    for i in range(1, test_traj.shape[0]):
        ml_only[i] = corrected_step_fn(ml_only[i - 1])

    # ---- 7. Method 4: ML + EnKF (corrected model AND observations) -------------
    # The corrected model still isn't perfect (R^2 < 1 on the residual fit),
    # so a small model_noise_std is kept for consistency/fairness, though
    # less structural bias remains than in the raw model above.
    ml_enkf_mean, ml_enkf_spread = run_enkf(
        test_traj, obs_times, obs_values, H, R, corrected_step_fn,
        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
        inflation=inflation, model_noise_std=0.3, seed=123
    )

    # ---- 8. Score everything -----------------------------------------------------
    results = {
        "Physical model": rmse(naive, test_traj),
        "EnKF": rmse(enkf_mean, test_traj),
        "ML correction": rmse(ml_only, test_traj),
        "ML + EnKF": rmse(ml_enkf_mean, test_traj),
    }
    print()
    print_summary_table(results)
    print(f"\nFor reference, climatological std of the state: {test_traj.std():.3f}")

    # ---- 9. Plots -------------------------------------------------------------------
    rmse_dict = {
        "Physical model": rmse_timeseries(naive, test_traj),
        "EnKF": rmse_timeseries(enkf_mean, test_traj),
        "ML correction": rmse_timeseries(ml_only, test_traj),
        "ML + EnKF": rmse_timeseries(ml_enkf_mean, test_traj),
    }
    plot_rmse_over_time(t_test, rmse_dict, title="Step 3: RMSE over time (4 methods)",
                         save_path=os.path.join(FIG_DIR, "step3_rmse_over_time.png"))

    # bar chart of overall RMSE
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(results.keys())
    vals = list(results.values())
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("RMSE (whole test period)")
    ax.set_title("Step 3: overall RMSE comparison")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step3_rmse_bar_comparison.png"), dpi=150)

    plot_state_reconstruction(
        t_test, test_traj, ml_enkf_mean, ml_enkf_spread,
        obs_times=obs_times, obs_values=obs_values, obs_indices=obs_indices,
        var_idx=0, title="Step 3: ML + EnKF reconstruction (variable 0)",
        save_path=os.path.join(FIG_DIR, "step3_ml_enkf_reconstruction_var0.png"),
    )

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")

    return {
        "t_test": t_test, "test_traj": test_traj, "naive": naive,
        "enkf_mean": enkf_mean, "ml_only": ml_only, "ml_enkf_mean": ml_enkf_mean,
        "results": results,
    }


if __name__ == "__main__":
    main()
    run_step3_ml_comparison()
