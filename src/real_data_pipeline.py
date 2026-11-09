"""
Step 5: fully real data -- real ground truth AND real observations.

Every earlier experiment keeps the ground truth synthetic (the Lorenz-96
ODE), even when the observation *network* (Step 4.5 /
`real_observation_network.py`) was built from real station geometry. This
module goes one step further: the ground truth itself is now a REAL
recorded physical field, not a synthetic ODE solution.

Data
----
Real hourly 10m wind speed (a different field from Step 4.5's
temperature-network exercise, on purpose), from the free, no-API-key
Open-Meteo historical archive (ERA5 reanalysis, https://open-meteo.com/),
at the SAME 28 real GHCN-Daily station locations used in
`real_observation_network.py` (continuity with that earlier work -- same
real geometry, now with real values behind it too). 2160 real hourly
values (2026-01-01 to 2026-03-31) per station, saved locally at
`data/real_wind_speed_49_51N.csv` so this reproduces fully offline.

Why this needs a different forecast model
------------------------------------------
Lorenz-96's ODE has no relationship to how real wind actually evolves --
using it here would be physically meaningless. There IS no simple
closed-form equation for real near-surface wind at this granularity, so
this is exactly the situation Howard et al.-style "model surrogate"
approaches (and Step 3's model-error-correction idea) are for: a small
neural network is trained directly on the real historical series to learn
x(t) -> x(t+1), and plays the role of M in the same EnKF machinery used
everywhere else in this project (`src/enkf.py` is completely agnostic to
where M comes from).

Real ground truth, held out honestly
--------------------------------------
Of the 28 real stations, most are treated as "observed" (fed into the
EnKF, with realistic instrument noise added on top of their real
recorded values), and a held-out subset are NEVER given to any method --
their real recorded values are used only for scoring, exactly mirroring
how Stage A scores against a truth the algorithm never sees, except this
truth is real recorded data, not synthetic.
"""

import csv
import os

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_wind_speed_49_51N.csv")

# Cup-anemometer instrument accuracy per WMO Guide to Instruments and
# Methods of Observation is roughly +/-0.5 m/s; the data here is in km/h.
ANEMOMETER_NOISE_STD_KMH = 0.5 * 3.6  # ~1.8 km/h

# Fixed, reproducible held-out subset: every ~3rd of the 28 real stations
# (by real longitude order) never gets observed -- reconstructed purely
# from its real neighbors' observations + the learned model.
HELD_OUT_POSITIONS = [2, 6, 10, 14, 18, 22, 26]  # positions within the 28, not grid_index values


def load_real_wind_data(csv_path=DEFAULT_CSV_PATH):
    """
    Returns
    -------
    times : list[str]  (ISO timestamps, length T)
    X : np.ndarray, shape (T, 28)  -- real recorded wind speed (km/h)
    grid_indices : np.ndarray, shape (28,)  -- original Lorenz-96-ring grid index of each column
    """
    times = []
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        grid_indices = np.array([int(h.split("_")[1]) for h in header[1:]])
        for row in reader:
            times.append(row[0])
            rows.append([float(v) for v in row[1:]])
    X = np.array(rows)
    return times, X, grid_indices


def split_observed_held_out(n_stations=28, held_out_positions=HELD_OUT_POSITIONS):
    """
    Returns
    -------
    observed_positions : np.ndarray -- indices (into the 28-station array) that are "observed"
    held_out_positions : np.ndarray -- indices never observed, used only for scoring
    """
    held_out = np.array(held_out_positions)
    observed = np.array([i for i in range(n_stations) if i not in held_out])
    return observed, held_out


def build_real_H_R(n_stations, observed_positions, noise_std=ANEMOMETER_NOISE_STD_KMH):
    """Simple selection operator H (observed subset of the 28 real stations) + diagonal R."""
    n_obs = len(observed_positions)
    H = np.zeros((n_obs, n_stations))
    for row, idx in enumerate(observed_positions):
        H[row, idx] = 1.0
    R = (noise_std ** 2) * np.eye(n_obs)
    return H, R


def train_surrogate_model(train_data, hidden_layer_sizes=(64, 64), max_iter=2000, seed=0):
    """
    Train a small MLP surrogate forecast model directly on REAL data:
    x(t) -> x(t+1), across all 28 stations at once (vector-to-vector
    regression -- sklearn's MLPRegressor supports multi-output natively).
    This plays the role of M for the EnKF -- there is no analytic
    equation to fall back on, unlike Lorenz-96.

    Parameters
    ----------
    train_data : np.ndarray, shape (T_train, 28)

    Returns
    -------
    model : trained MLPRegressor
    scaler_X, scaler_y : StandardScaler
    """
    X = train_data[:-1]
    y = train_data[1:]

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="tanh",
        solver="adam",
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=25,
    )
    model.fit(scaler_X.transform(X), scaler_y.transform(y))
    return model, scaler_X, scaler_y


def make_surrogate_step_fn(model, scaler_X, scaler_y):
    """
    Wrap the trained surrogate into a step function x -> M(x), usable as
    `model_step_fn` anywhere in `src/enkf.py` -- works on a single state
    (28,) or a batch/ensemble (E, 28), matching `lorenz96.step`'s contract.
    """
    def step_fn(x):
        x = np.asarray(x)
        orig_ndim = x.ndim
        x2d = x.reshape(1, -1) if orig_ndim == 1 else x
        pred = scaler_y.inverse_transform(model.predict(scaler_X.transform(x2d)))
        return pred[0] if orig_ndim == 1 else pred
    return step_fn
