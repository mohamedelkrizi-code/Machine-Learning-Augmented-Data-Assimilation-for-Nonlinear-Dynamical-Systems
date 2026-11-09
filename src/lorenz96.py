"""
Lorenz-96 dynamical system.

The model:
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F     for i = 0..N-1 (cyclic)

This is the "true" physics M(x). Optionally, an artificial unknown model
error delta(x) can be added to simulate a *wrong* physical model, which is
what the ML-correction stage (Step 3 of the project) will try to learn back.

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F + delta(x)
"""

import numpy as np
from scipy.integrate import solve_ivp


def lorenz96_rhs(t, x, F=8.0, delta_fn=None):
    """
    Right-hand side of the Lorenz-96 ODE system.

    Parameters
    ----------
    t : float
        Time (unused, system is autonomous, but required by solve_ivp signature).
    x : np.ndarray, shape (N,)
        State vector.
    F : float
        Forcing constant. F=8 is the classic chaotic regime.
    delta_fn : callable or None
        Optional function delta(x) -> np.ndarray, shape (N,), representing an
        artificial/unknown model error added on top of the "true" physics.
        Leave as None for the clean/true model.

    Returns
    -------
    dxdt : np.ndarray, shape (N,)
    """
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F

    if delta_fn is not None:
        dxdt = dxdt + delta_fn(x)

    return dxdt


def integrate(x0, t_span, t_eval, F=8.0, delta_fn=None, method="RK45", rtol=1e-6, atol=1e-9):
    """
    Integrate the Lorenz-96 system forward in time using SciPy.

    Parameters
    ----------
    x0 : np.ndarray, shape (N,)
        Initial condition.
    t_span : tuple (t0, tf)
    t_eval : np.ndarray
        Time points at which to store the solution.
    F : float
        Forcing constant.
    delta_fn : callable or None
        Optional artificial model error (see lorenz96_rhs).

    Returns
    -------
    traj : np.ndarray, shape (len(t_eval), N)
        State trajectory, one row per time point.
    """
    sol = solve_ivp(
        lorenz96_rhs,
        t_span,
        x0,
        t_eval=t_eval,
        args=(F, delta_fn),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.y.T  # shape (T, N)


def step(x, dt, F=8.0, delta_fn=None, method="RK45"):
    """
    Advance a single state vector x by one time step dt.
    This is the function M(x_t) -> x_{t+1} used inside the EnKF forecast step
    (and, with delta_fn, the "wrong physics + artificial error" model used in
    Step 3 of the project).

    Works on a single state vector (N,) OR a batch of ensemble members (E, N).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        traj = integrate(x, (0.0, dt), np.array([dt]), F=F, delta_fn=delta_fn, method=method)
        return traj[-1]
    elif x.ndim == 2:
        # batch of ensemble members, integrate each independently
        out = np.zeros_like(x)
        for e in range(x.shape[0]):
            out[e] = step(x[e], dt, F=F, delta_fn=delta_fn, method=method)
        return out
    else:
        raise ValueError("x must be 1D (single state) or 2D (ensemble, N)")


def generate_true_trajectory(N=40, F=8.0, dt=0.05, n_steps=400, spinup_steps=200, seed=0):
    """
    Generate the reference ("true") Lorenz-96 trajectory.

    A short spin-up is run first and discarded so the trajectory starts on
    the system's chaotic attractor rather than on an arbitrary initial
    condition.

    Returns
    -------
    t : np.ndarray, shape (n_steps+1,)
    traj : np.ndarray, shape (n_steps+1, N)
    """
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(N) + rng.normal(scale=0.5, size=N)

    # spin-up (discarded) so we start on the attractor
    if spinup_steps > 0:
        t_spinup = np.linspace(0, spinup_steps * dt, spinup_steps + 1)
        spinup_traj = integrate(x0, (0, spinup_steps * dt), t_spinup, F=F)
        x0 = spinup_traj[-1]

    t = np.linspace(0, n_steps * dt, n_steps + 1)
    traj = integrate(x0, (0, n_steps * dt), t, F=F)
    return t, traj
