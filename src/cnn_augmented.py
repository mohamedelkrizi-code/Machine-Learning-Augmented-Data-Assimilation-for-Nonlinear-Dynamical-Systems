"""
Step 4: CNN-augmented EnKF for high-resolution observations.

Inspired by Howard, Subramanian & Hoteit (2023), "A Machine Learning
Augmented Data Assimilation Method for High-Resolution Observation"
(ESSOAr preprint essoar.168201784, submitted to JAMES).

Setup
-----
In an operational setting, some observations are available at a higher
spatial resolution than a data assimilation system can afford to
assimilate with the EnKF (e.g. dense satellite swaths), so they are
simply discarded. This module trains a small convolutional neural
network OFFLINE to reproduce what an EnKF assimilating full-resolution
observations would have produced, then uses that trained network
ONLINE to cheaply assimilate the otherwise-discarded high-resolution
observations, alternating with a standard EnKF that handles routine
low-resolution observations.

The CNN takes two co-located spatial channels (forecast ensemble mean,
and the innovation = observation - forecast) and outputs the analysis
mean. Cyclic (circular) padding is used throughout so the network
respects the periodic boundary of the Lorenz-96 ring, matching the
architecture in Howard et al. (2023): 2 hidden conv layers, filter
size 3, 5 feature maps each, ReLU, linear output layer -- 131
trainable weights total, same as the paper.
"""

import numpy as np
import torch
import torch.nn as nn

from .enkf import init_ensemble, forecast_step, analysis_step, build_localization_matrix


class AnalysisCNN(nn.Module):
    """
    2-channel-in / 1-channel-out cyclic CNN that emulates one EnKF
    analysis step. Input: (batch, 2, N) = [forecast_mean, innovation].
    Output: (batch, N) analysis mean.
    """

    def __init__(self, n_features=5, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(2, n_features, kernel_size, padding=pad, padding_mode="circular")
        self.conv2 = nn.Conv1d(n_features, n_features, kernel_size, padding=pad, padding_mode="circular")
        self.conv_out = nn.Conv1d(n_features, 1, kernel_size, padding=pad, padding_mode="circular")
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        out = self.conv_out(h)  # linear output, as in Howard et al. Sec 2.3
        return out.squeeze(1)


def generate_cnn_training_data(train_traj, H_low, R_low, obs_values_low,
                                H_full, R_full, obs_values_full, model_step_fn,
                                ensemble_size=30, init_spread_std=2.0,
                                inflation=1.03, localization_radius=None, seed=42):
    """
    Generate CNN training data ON-POLICY: run the exact alternating
    cadence the augmented method uses at test time (EnKF analysis with
    low-resolution obs on odd steps, real EnKF analysis with
    full-resolution obs on even steps), and record the even-step
    (forecast mean, innovation, EnKF-full analysis) triples.

    This matters because training instead on forecasts from an EnKF
    that assimilates full observations at EVERY step (as in Howard et
    al. 2023's original offline procedure) produces forecasts with much
    smaller errors than what the CNN actually has to correct online,
    once observations are this sparse -- exactly the train/test
    distribution mismatch the paper's Discussion (Sec 4) flags as a
    limitation. Generating on-policy data (forecasts that already come
    from the same half-frequency, low-resolution-only correction
    cadence) closes that gap.

    obs_values_low / obs_values_full must be indexable by time step t,
    covering every t in [0, T) (as produced by
    src.observations.generate_observations with obs_every_steps=1).

    Returns
    -------
    X : np.ndarray, shape (n_even_steps, 2, N)
    y : np.ndarray, shape (n_even_steps, N)
    """
    rng = np.random.default_rng(seed)
    T, N = train_traj.shape
    loc = build_localization_matrix(N, localization_radius) if localization_radius is not None else None

    x0_guess = train_traj[0] + rng.normal(scale=init_spread_std, size=N)
    ensemble = init_ensemble(x0_guess, ensemble_size, spread_std=init_spread_std, seed=seed)

    X_list, y_list = [], []
    for t in range(1, T):
        ensemble = forecast_step(ensemble, model_step_fn, inflation=inflation, seed=seed + t)

        if t % 2 == 0:
            forecast_mean = ensemble.mean(axis=0)
            y_obs = obs_values_full[t]
            innovation = y_obs - H_full @ forecast_mean
            ensemble, _ = analysis_step(ensemble, y_obs, H_full, R_full, localization_matrix=loc, seed=seed + t)
            X_list.append(np.stack([forecast_mean, innovation]))
            y_list.append(ensemble.mean(axis=0))
        else:
            y_obs = obs_values_low[t]
            ensemble, _ = analysis_step(ensemble, y_obs, H_low, R_low, localization_matrix=loc, seed=seed + t)

    return np.array(X_list), np.array(y_list)


def train_analysis_cnn(X, y, val_frac=0.2, epochs=300, lr=5e-3, seed=0):
    """
    Train the AnalysisCNN by supervised regression against EnKF
    (full-obs) analyses, holding out the last `val_frac` of the
    training trajectory as a validation set -- checked for overfitting,
    as in Howard et al. Fig 3 (training vs. validation RMSE curves).

    Returns
    -------
    model : trained AnalysisCNN
    history : dict with "train_loss" and "val_loss" (RMSE) per epoch
    """
    torch.manual_seed(seed)
    n = X.shape[0]
    n_val = int(n * val_frac)
    n_train = n - n_val

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    X_train, y_train = X_t[:n_train], y_t[:n_train]
    X_val, y_val = X_t[n_train:], y_t[n_train:]

    model = AnalysisCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)

        history["train_loss"].append(loss.item() ** 0.5)
        history["val_loss"].append(val_loss.item() ** 0.5)

    return model, history


def apply_cnn_analysis(model, forecast_mean, innovation):
    """Run one CNN analysis step: (forecast_mean, innovation) -> analysis_mean."""
    x = np.stack([forecast_mean, innovation])[None, :, :]  # (1, 2, N)
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        out = model(x_t).numpy()[0]
    return out


def recreate_ensemble(forecast_ensemble, cnn_analysis_mean):
    """
    Rebuild an analysis ensemble consistent with a CNN point analysis:
    shift every forecast member by the same vector so the new ensemble
    mean equals the CNN analysis, preserving the forecast spread
    (Howard et al. 2023, Sec 2.4).
    """
    forecast_mean = forecast_ensemble.mean(axis=0)
    delta = forecast_mean - cnn_analysis_mean
    return forecast_ensemble - delta[None, :]


def run_augmented_enkf(true_traj, H_low, R_low, obs_values_low, H_full, obs_values_full,
                        cnn_model, model_step_fn, ensemble_size=30, init_spread_std=2.0,
                        inflation=1.03, localization_radius=None, seed=123):
    """
    The augmented method (Howard et al. 2023, Fig 2): alternate time
    steps between a standard EnKF analysis using low-resolution
    observations and a CNN analysis using high-resolution observations
    that would otherwise be discarded.

        even t: "high-resolution" step -> CNN analysis
        odd t:  "low-resolution" step  -> EnKF analysis

    obs_values_low / obs_values_full must be indexable by time step t
    (as produced by src.observations.generate_observations with
    obs_every_steps=1), so both methods being compared see the same
    noisy observation realizations at each shared time step.

    Returns
    -------
    analysis_mean : np.ndarray, shape (T, N)
    ensemble_spread : np.ndarray, shape (T, N)
    """
    rng = np.random.default_rng(seed)
    T, N = true_traj.shape
    loc = build_localization_matrix(N, localization_radius) if localization_radius is not None else None

    x0_guess = true_traj[0] + rng.normal(scale=init_spread_std, size=N)
    ensemble = init_ensemble(x0_guess, ensemble_size, spread_std=init_spread_std, seed=seed)

    analysis_mean = np.zeros((T, N))
    ensemble_spread = np.zeros((T, N))
    analysis_mean[0] = ensemble.mean(axis=0)
    ensemble_spread[0] = ensemble.std(axis=0)

    for t in range(1, T):
        ensemble = forecast_step(ensemble, model_step_fn, inflation=inflation, seed=seed + t)

        if t % 2 == 0:
            forecast_mean = ensemble.mean(axis=0)
            y_obs = obs_values_full[t]
            innovation = y_obs - H_full @ forecast_mean
            cnn_analysis = apply_cnn_analysis(cnn_model, forecast_mean, innovation)
            ensemble = recreate_ensemble(ensemble, cnn_analysis)
        else:
            y_obs = obs_values_low[t]
            ensemble, _ = analysis_step(ensemble, y_obs, H_low, R_low, localization_matrix=loc, seed=seed + t)

        analysis_mean[t] = ensemble.mean(axis=0)
        ensemble_spread[t] = ensemble.std(axis=0)

    return analysis_mean, ensemble_spread


def run_sparse_enkf_baseline(true_traj, H_low, R_low, obs_values_low, model_step_fn,
                              ensemble_size=30, init_spread_std=2.0, inflation=1.03,
                              localization_radius=None, seed=123):
    """
    Baseline "EnKF SparseObs": identical time-stepping to the augmented
    method, but the high-resolution observations on even steps are
    simply discarded (no assimilation happens on those steps) -- the
    real-world default this method is trying to improve on.
    """
    rng = np.random.default_rng(seed)
    T, N = true_traj.shape
    loc = build_localization_matrix(N, localization_radius) if localization_radius is not None else None

    x0_guess = true_traj[0] + rng.normal(scale=init_spread_std, size=N)
    ensemble = init_ensemble(x0_guess, ensemble_size, spread_std=init_spread_std, seed=seed)

    analysis_mean = np.zeros((T, N))
    analysis_mean[0] = ensemble.mean(axis=0)

    for t in range(1, T):
        ensemble = forecast_step(ensemble, model_step_fn, inflation=inflation, seed=seed + t)

        if t % 2 != 0:
            y_obs = obs_values_low[t]
            ensemble, _ = analysis_step(ensemble, y_obs, H_low, R_low, localization_matrix=loc, seed=seed + t)
        # else: high-resolution observation available but discarded (unused)

        analysis_mean[t] = ensemble.mean(axis=0)

    return analysis_mean
