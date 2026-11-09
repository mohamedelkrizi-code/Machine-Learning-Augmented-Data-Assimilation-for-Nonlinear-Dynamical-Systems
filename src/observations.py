"""
Observation model:

    y_t = H(x_t) + eps_t,      eps_t ~ N(0, R)

H picks out a subset of the state (sparse observations: e.g. only every
k-th variable is observed, mimicking real sensor networks that never cover
the full state). eps_t is Gaussian observation noise.
"""

import numpy as np


def make_observation_operator(N, obs_every=4):
    """
    Build a sparse linear observation operator H as a matrix, plus the
    indices of the observed state components.

    Parameters
    ----------
    N : int
        Full state dimension.
    obs_every : int
        Observe one out of every `obs_every` state variables.
        obs_every=1 means fully observed.

    Returns
    -------
    H : np.ndarray, shape (M, N)
        Linear observation matrix such that y = H @ x.
    obs_indices : np.ndarray, shape (M,)
        Indices of the observed components.
    """
    obs_indices = np.arange(0, N, obs_every)
    M = len(obs_indices)
    H = np.zeros((M, N))
    for row, idx in enumerate(obs_indices):
        H[row, idx] = 1.0
    return H, obs_indices


def generate_observations(true_traj, H, obs_std=1.0, obs_every_steps=1, seed=1):
    """
    Generate noisy observations from a true trajectory.

    Parameters
    ----------
    true_traj : np.ndarray, shape (T, N)
        True state trajectory (as produced by lorenz96.generate_true_trajectory).
    H : np.ndarray, shape (M, N)
        Observation operator (spatial sparsity).
    obs_std : float
        Standard deviation of Gaussian observation noise.
    obs_every_steps : int
        Observe only every k-th time step (temporal sparsity). Steps in
        between have no observation available (represented as None).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    obs_times : list[int]
        Time-step indices at which an observation is available.
    obs_values : list[np.ndarray]
        Corresponding noisy observation vectors, each shape (M,).
    R : np.ndarray, shape (M, M)
        Observation error covariance matrix (diagonal, iid noise assumed).
    """
    rng = np.random.default_rng(seed)
    T = true_traj.shape[0]
    M = H.shape[0]
    R = (obs_std ** 2) * np.eye(M)

    obs_times = []
    obs_values = []
    for t in range(0, T, obs_every_steps):
        clean = H @ true_traj[t]
        noisy = clean + rng.normal(scale=obs_std, size=M)
        obs_times.append(t)
        obs_values.append(noisy)

    return obs_times, obs_values, R
