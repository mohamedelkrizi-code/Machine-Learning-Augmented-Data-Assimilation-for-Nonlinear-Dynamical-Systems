"""
Ensemble Kalman Filter (EnKF).

Model:
    x_{t+1} = M(x_t) + eta_t          (forecast / dynamics)
    y_t     = H(x_t) + eps_t          (observation)

Two stages, repeated every time an observation becomes available:

1. FORECAST
   Push every ensemble member through the (possibly imperfect) model M:
       x_t^f_(e) = M(x_{t-1}^a_(e)) + eta_(e),   e = 1..E
   The ensemble mean is the point forecast; the ensemble covariance P^f
   (estimated empirically from the spread of the E members) is the filter's
   uncertainty estimate — no need to propagate a full NxN covariance matrix
   analytically, which is what makes EnKF tractable for large nonlinear
   systems.

2. ANALYSIS
   Blend the forecast with the incoming observation y_t via the Kalman gain:
       K = P^f H^T (H P^f H^T + R)^-1
       x_t^a_(e) = x_t^f_(e) + K (y_t + eps_(e) - H x_t^f_(e))
   (perturbed-observation EnKF: each member is updated with an independently
   perturbed copy of the observation, which keeps the analysis ensemble
   spread statistically consistent.)
"""

import numpy as np


def init_ensemble(x0_mean, ensemble_size, spread_std=1.0, seed=2):
    """
    Initialize an ensemble of state vectors around a mean guess.

    Returns
    -------
    ensemble : np.ndarray, shape (E, N)
    """
    rng = np.random.default_rng(seed)
    N = len(x0_mean)
    ensemble = x0_mean[None, :] + rng.normal(scale=spread_std, size=(ensemble_size, N))
    return ensemble


def forecast_step(ensemble, model_step_fn, model_noise_std=0.0, inflation=1.0, seed=None):
    """
    Forecast stage: x_t^f = M(x_{t-1}^a) + eta_t

    Parameters
    ----------
    ensemble : np.ndarray, shape (E, N)
        Analysis ensemble from the previous time step.
    model_step_fn : callable
        Function taking an (E, N) array (or (N,) for a single member) and
        returning the state advanced by one time step: x -> M(x).
    model_noise_std : float
        Std of additive Gaussian model-error noise eta_t (represents
        stochastic/unresolved physics). 0.0 disables it.
    inflation : float
        Multiplicative covariance inflation factor (>= 1.0). In a chaotic
        system the ensemble tends to under-estimate its own uncertainty
        after repeated analysis steps ("filter collapse" / "ensemble
        collapse"): members get pulled so close together that the filter
        becomes overconfident and stops trusting new observations, which
        lets the estimate drift away from truth unchecked. Inflation
        counteracts this by stretching the forecast ensemble's spread
        around its mean by a small factor (e.g. 1.03 = +3%) every step,
        which is standard practice in operational EnKF systems.

    Returns
    -------
    forecast_ensemble : np.ndarray, shape (E, N)
    """
    forecast = model_step_fn(ensemble)
    if model_noise_std > 0:
        rng = np.random.default_rng(seed)
        forecast = forecast + rng.normal(scale=model_noise_std, size=forecast.shape)

    if inflation != 1.0:
        mean = forecast.mean(axis=0, keepdims=True)
        forecast = mean + inflation * (forecast - mean)

    return forecast


def analysis_step(forecast_ensemble, y_obs, H, R, seed=None):
    """
    Analysis stage: correct the forecast ensemble using an observation.

        K = P^f H^T (H P^f H^T + R)^-1
        x^a_(e) = x^f_(e) + K (y_obs + eps_(e) - H x^f_(e))

    Parameters
    ----------
    forecast_ensemble : np.ndarray, shape (E, N)
    y_obs : np.ndarray, shape (M,)
        Observation vector at this time step.
    H : np.ndarray, shape (M, N)
        Observation operator.
    R : np.ndarray, shape (M, M)
        Observation error covariance.

    Returns
    -------
    analysis_ensemble : np.ndarray, shape (E, N)
    K : np.ndarray, shape (N, M)
        Kalman gain actually used (handy for diagnostics/teaching).
    """
    rng = np.random.default_rng(seed)
    E, N = forecast_ensemble.shape
    M = H.shape[0]

    # Ensemble mean and anomalies (deviations from the mean)
    x_bar = forecast_ensemble.mean(axis=0)                      # (N,)
    X_prime = forecast_ensemble - x_bar[None, :]                 # (E, N)

    # Empirical forecast covariance P^f = (1/(E-1)) X'^T X'
    Pf = (X_prime.T @ X_prime) / (E - 1)                          # (N, N)

    # Innovation covariance and Kalman gain
    S = H @ Pf @ H.T + R                                          # (M, M)
    K = Pf @ H.T @ np.linalg.inv(S)                               # (N, M)

    # Perturbed observations (one noisy copy of y_obs per ensemble member)
    obs_perturbations = rng.multivariate_normal(np.zeros(M), R, size=E)  # (E, M)
    y_perturbed = y_obs[None, :] + obs_perturbations                      # (E, M)

    # Innovation for each member: y_perturbed - H x^f_(e)
    innovation = y_perturbed - forecast_ensemble @ H.T            # (E, M)

    analysis_ensemble = forecast_ensemble + innovation @ K.T      # (E, N)
    return analysis_ensemble, K


def run_enkf(true_traj, obs_times, obs_values, H, R, model_step_fn,
             ensemble_size=40, init_spread_std=2.0, model_noise_std=0.0,
             inflation=1.03, seed=42):
    """
    Run the full EnKF over the length of true_traj, cycling
    forecast -> (analysis if observation available) at every time step.

    Returns
    -------
    analysis_mean : np.ndarray, shape (T, N)
        EnKF state estimate (ensemble mean) at every time step.
    ensemble_spread : np.ndarray, shape (T, N)
        Per-variable ensemble standard deviation at every time step
        (this is the "uncertainty" band to plot).
    """
    rng = np.random.default_rng(seed)
    T, N = true_traj.shape
    obs_dict = dict(zip(obs_times, obs_values))

    # Initialize ensemble around a deliberately imperfect first guess
    x0_guess = true_traj[0] + rng.normal(scale=init_spread_std, size=N)
    ensemble = init_ensemble(x0_guess, ensemble_size, spread_std=init_spread_std, seed=seed)

    analysis_mean = np.zeros((T, N))
    ensemble_spread = np.zeros((T, N))

    # t=0: treat initial ensemble as the analysis at time 0 (assimilate obs if present)
    if 0 in obs_dict:
        ensemble, _ = analysis_step(ensemble, obs_dict[0], H, R, seed=seed)
    analysis_mean[0] = ensemble.mean(axis=0)
    ensemble_spread[0] = ensemble.std(axis=0)

    for t in range(1, T):
        # 1. FORECAST: push every member through the model (+ covariance inflation)
        ensemble = forecast_step(ensemble, model_step_fn, model_noise_std=model_noise_std,
                                  inflation=inflation, seed=seed + t)

        # 2. ANALYSIS: correct with observation if one is available at this step
        if t in obs_dict:
            ensemble, _ = analysis_step(ensemble, obs_dict[t], H, R, seed=seed + t)

        analysis_mean[t] = ensemble.mean(axis=0)
        ensemble_spread[t] = ensemble.std(axis=0)

    return analysis_mean, ensemble_spread
