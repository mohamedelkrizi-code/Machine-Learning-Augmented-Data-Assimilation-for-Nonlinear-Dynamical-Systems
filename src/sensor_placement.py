"""
Sensor-placement optimization: given a pool of real candidate station
locations, which K of them should actually be instrumented to minimize
reconstruction RMSE across the whole ring?

This ties together the two earlier real-data additions: the real
observation network showed that a "smart" evenly-spaced synthetic H can
beat a "dumb" real-but-clustered H even with fewer observed points --
which raises the natural operational question this module answers: given
a real, physically-realizable candidate pool (you can only put a sensor
where a real site exists), can SELECTING the right subset recover -- or
beat -- what the evenly-spaced synthetic network achieves, instead of
just taking whatever real candidates happen to already exist (as
`real_observation_network.py` did)?

Algorithm: greedy forward selection, the standard, well-established
approach for this exact class of problem (sensor placement under a
reconstruction-error objective is well known to be approximately
submodular -- greedy is provably near-optimal and, unlike exhaustive
search, tractable: choosing the best K of 28 candidates exactly would
require checking C(28,K) subsets, e.g. ~40 million for K=14; greedy
instead evaluates at most 28+27+...+1 subsets total, one candidate added
at a time).
"""

import numpy as np

from .enkf import run_enkf
from .evaluation import rmse


def _evaluate_subset(true_traj, H_candidates, R_diag_candidates, subset, model_step_fn,
                      ensemble_size, init_spread_std, inflation, localization_radius, seed):
    H = H_candidates[subset]
    R = np.diag(R_diag_candidates[subset])
    rng = np.random.default_rng(seed)
    T = true_traj.shape[0]
    noise_std = np.sqrt(R_diag_candidates[subset])
    obs_times = list(range(T))
    obs_values = [H @ true_traj[t] + rng.normal(scale=noise_std) for t in obs_times]
    mean, _ = run_enkf(true_traj, obs_times, obs_values, H, R, model_step_fn,
                        ensemble_size=ensemble_size, init_spread_std=init_spread_std,
                        inflation=inflation, localization_radius=localization_radius, seed=seed)
    return rmse(mean, true_traj)


def greedy_sensor_placement(true_traj, H_candidates, R_diag_candidates, model_step_fn,
                             max_k, ensemble_size=15, init_spread_std=2.0, inflation=1.05,
                             localization_radius=10, seed=42):
    """
    Greedily build up a sensor subset, one candidate at a time, always
    adding whichever remaining candidate most reduces reconstruction RMSE.

    Parameters
    ----------
    true_traj : np.ndarray, shape (T, N)
    H_candidates : np.ndarray, shape (n_candidates, N)
        Each row is one candidate sensor's observation operator row.
    R_diag_candidates : np.ndarray, shape (n_candidates,)
        Observation noise VARIANCE for each candidate.
    max_k : int
        Build the curve from k=1 up to this budget.

    Returns
    -------
    history : list of dict, one per k, each with:
        {"k": int, "selected": list[int] (indices into H_candidates, in the
         order added), "rmse": float, "added_this_step": int}
    """
    n_candidates = H_candidates.shape[0]
    remaining = list(range(n_candidates))
    selected = []
    history = []

    for k in range(1, max_k + 1):
        best_candidate, best_rmse = None, np.inf
        for c in remaining:
            trial = selected + [c]
            r = _evaluate_subset(true_traj, H_candidates, R_diag_candidates, trial, model_step_fn,
                                  ensemble_size, init_spread_std, inflation, localization_radius, seed)
            if r < best_rmse:
                best_rmse, best_candidate = r, c

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        history.append({"k": k, "selected": list(selected), "rmse": best_rmse,
                         "added_this_step": best_candidate})

    return history


def random_subset_baseline(true_traj, H_candidates, R_diag_candidates, model_step_fn, k,
                            n_trials=10, ensemble_size=15, init_spread_std=2.0, inflation=1.05,
                            localization_radius=10, seed=42):
    """
    Average RMSE of `n_trials` random size-k subsets -- the "no smart
    selection at all" baseline that greedy selection should beat.
    """
    n_candidates = H_candidates.shape[0]
    rng = np.random.default_rng(seed + 999)
    rmses = []
    for trial in range(n_trials):
        subset = list(rng.choice(n_candidates, size=k, replace=False))
        r = _evaluate_subset(true_traj, H_candidates, R_diag_candidates, subset, model_step_fn,
                              ensemble_size, init_spread_std, inflation, localization_radius,
                              seed + trial)
        rmses.append(r)
    return float(np.mean(rmses)), float(np.std(rmses))
