"""
Step 3 of the project: learn an unknown model error with a neural network.

Setup
-----
We deliberately make the physics model WRONG by hiding an extra term from
it:

    "Nature" (true system):   dx_i/dt = (x_{i+1}-x_{i-2})x_{i-1} - x_i + F + delta(x_i)
    "Our model" M (wrong):    dx_i/dt = (x_{i+1}-x_{i-2})x_{i-1} - x_i + F

delta(x_i) here is a pointwise (elementwise) function of each individual
variable -- e.g. an unresolved damping/forcing effect, analogous to a
missing sub-grid-scale parameterization in a real climate/ocean model
(this elementwise-with-parameter-sharing setup is the same idea used in
classic Lorenz-96 "two-scale" subgrid-parameterization-learning studies).

We NEVER give the neural network the analytic formula for delta. Instead,
it only ever sees data: pairs of (x_t, one-step residual), where the
residual is the gap between what really happened (from the nature run)
and what the wrong model M predicted. This is the realistic, data-driven
setting: in the real world you don't know the missing physics, you can only
see that your model doesn't quite match observations/reanalysis.

    residual_t = x_{t+1}^{true} - M(x_t^{true})   (one-step-ahead)

The network learns  delta_hat_theta(x) ~ residual(x),  which is then added
back onto the model:

    x_{t+1} = M(x_t) + delta_hat_theta(x_t)
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .lorenz96 import integrate, step


def make_delta_fn(a=-0.5, b=0.3):
    """
    Build an artificial, "unknown" elementwise model-error function.

        delta(x_i) = a * sin(x_i) + b * x_i

    This is applied identically (with parameter sharing) to every grid
    point of the Lorenz-96 state -- i.e. it depends only on the local
    value x_i, not on neighboring variables. This makes it something a
    small pointwise neural network can realistically learn from data.

    Returns
    -------
    delta_fn : callable
        Takes a state vector x (N,) and returns delta(x) (N,), suitable to
        pass as `delta_fn` to lorenz96.lorenz96_rhs / integrate.
    """
    def delta_fn(x):
        return a * np.sin(x) + b * x
    return delta_fn


def generate_nature_run(x0, dt, n_steps, F=8.0, delta_fn=None, method="RK45"):
    """
    Integrate the TRUE system (physics + hidden delta) forward -- this
    plays the role of "nature"/reality, which we only get to see through
    noisy sparse observations in the full pipeline, but which we DO get to
    use directly here to build a supervised training set for the ML model
    (exactly as one would use historical reanalysis data in practice).
    """
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    traj = integrate(x0, (0, n_steps * dt), t, F=F, delta_fn=delta_fn, method=method)
    return t, traj


def build_training_dataset(true_traj, dt, F=8.0):
    """
    Build a supervised dataset of (x_t_i, residual_t_i) pairs, flattened
    across ALL time steps and ALL state variables (parameter sharing: one
    scalar function learned for every grid point at once).

    residual_t_i = x_{t+1,i}^{true} - [M(x_t^{true})]_i

    Parameters
    ----------
    true_traj : np.ndarray, shape (T, N)
        Nature-run trajectory (from generate_nature_run).
    dt : float
    F : float

    Returns
    -------
    X : np.ndarray, shape (T-1)*N, 1)
        Input feature: the scalar state value x_{t,i}.
    y : np.ndarray, shape ((T-1)*N,)
        Target: the residual (model error) at that point.
    """
    T, N = true_traj.shape
    X_list = []
    y_list = []
    for t_idx in range(T - 1):
        x_t = true_traj[t_idx]
        x_true_next = true_traj[t_idx + 1]
        x_model_next = step(x_t, dt, F=F, delta_fn=None)  # wrong model, no delta
        residual = x_true_next - x_model_next
        X_list.append(x_t.reshape(-1, 1))
        y_list.append(residual)
    X = np.vstack(X_list)          # ((T-1)*N, 1)
    y = np.concatenate(y_list)     # ((T-1)*N,)
    return X, y


def train_correction_model(X, y, hidden_layer_sizes=(32, 32), max_iter=2000, seed=0):
    """
    Train a small MLP to approximate the elementwise model-error function
    delta_hat_theta(x_i) ~ residual(x_i).

    A scikit-learn MLPRegressor is used for portability (no heavy deep
    learning framework required); swapping in a PyTorch/Keras network with
    the same input/output shape is a drop-in replacement.

    Returns
    -------
    model : trained MLPRegressor
    scaler_X, scaler_y : StandardScaler
        Feature/target scalers (MLPs train much better on standardized
        inputs); must be applied consistently at prediction time, which
        `apply_correction` below takes care of.
    """
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    Xs = scaler_X.transform(X)
    ys = scaler_y.transform(y.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="tanh",
        solver="adam",
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=25,
    )
    model.fit(Xs, ys)
    return model, scaler_X, scaler_y


def apply_correction(x, model, scaler_X, scaler_y):
    """
    Apply the trained elementwise correction network to a state vector or
    batch of ensemble members.

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (E, N)

    Returns
    -------
    delta_hat : np.ndarray, same shape as x
    """
    x = np.asarray(x)
    orig_shape = x.shape
    flat = x.reshape(-1, 1)
    flat_scaled = scaler_X.transform(flat)
    pred_scaled = model.predict(flat_scaled).reshape(-1, 1)
    pred = scaler_y.inverse_transform(pred_scaled).ravel()
    return pred.reshape(orig_shape)


def make_corrected_step_fn(base_step_fn, model, scaler_X, scaler_y):
    """
    Wrap a base model-step function M(x) into a corrected one:
        x -> M(x) + delta_hat_theta(x)

    This corrected function can be used as a drop-in replacement anywhere
    the raw model_step_fn was used before -- including as the forecast
    model inside the EnKF (Step 3's "ML + EnKF" row).
    """
    def corrected_step(x):
        base = base_step_fn(x)
        correction = apply_correction(x, model, scaler_X, scaler_y)
        return base + correction
    return corrected_step
