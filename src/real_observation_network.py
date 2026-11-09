"""
A REAL observation network for the Lorenz-96 ring.

Stage A / Step 3 / Step 4 all use a synthetic observation operator H
(`observations.make_observation_operator`): "observe every k-th grid
point," evenly spaced, all with identical noise. Real sensor networks
look nothing like that -- station placement is driven by where people
and infrastructure are, not by a tidy spacing rule, and instrument
quality varies station to station.

This module builds H (and a heterogeneous R) from REAL weather-station
data instead: NOAA's GHCN-Daily station list
(https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt,
~132,500 real stations worldwide). Lorenz-96 is explicitly described (in
Lorenz, 2005) as a 1-D analog of a variable at N evenly-spaced points
around a latitude circle -- so we take that literally: real stations in
a real ~2-degree latitude band (49-51N, chosen for its mix of dense
coverage over Canada/Europe/Russia and genuine gaps over open ocean) are
binned into N=40 longitude sectors (9 degrees each) around the globe,
and the closest real station to each sector's center (if any exist in
that sector) becomes that grid point's real "sensor." The result:
28 of the 40 sectors have a real observation, 12 don't (mostly Pacific/
Atlantic ocean gaps) -- an authentically patchy, real-world-derived H,
not an evenly-spaced synthetic one.

The extracted station subset is saved at `data/real_stations_49_51N.csv`
so this runs fully offline (no network access needed to reproduce).

Observation noise is also made heterogeneous based on each station's real
GHCN quality flag (GSN / HCN-CRN "reference network" vs. an ordinary
co-op station), rather than one arbitrary constant for every sensor --
reference-network stations carry documented tighter instrument tolerances
(WMO Guide to Instruments and Methods of Observation; NOAA's Climate
Reference Network is specified to roughly 0.1 degC vs. ~0.3-0.5 degC for
an ordinary co-op station). REFERENCE_NOISE_STD / ORDINARY_NOISE_STD below
mirror that real ratio, expressed in this project's Lorenz-96 units.
"""

import csv
import os

import numpy as np

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_stations_49_51N.csv")

REFERENCE_NOISE_STD = 0.2  # GSN / HCN-CRN "reference network" station
ORDINARY_NOISE_STD = 0.5   # ordinary co-op station (same default as the rest of the project)


def load_real_station_network(N=40, csv_path=DEFAULT_CSV_PATH):
    """
    Load the real GHCN-Daily-derived station network.

    Returns
    -------
    H : np.ndarray, shape (n_observed, N)
        Real, non-uniform observation operator -- one row per real station.
    R : np.ndarray, shape (n_observed, n_observed)
        Diagonal observation-error covariance, heterogeneous per station
        (tighter for reference-network stations, looser for ordinary ones).
    obs_indices : np.ndarray, shape (n_observed,)
        Which of the N grid points (longitude sectors) have a real station.
    stations : list[dict]
        Metadata (station id, lat, lon, name, quality tier) per observed
        grid point, same order as obs_indices/H's rows -- for plotting/reporting.
    """
    stations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["grid_index"] = int(row["grid_index"])
            row["lat"] = float(row["lat"])
            row["lon"] = float(row["lon"])
            stations.append(row)

    stations.sort(key=lambda s: s["grid_index"])
    obs_indices = np.array([s["grid_index"] for s in stations])
    n_observed = len(obs_indices)

    H = np.zeros((n_observed, N))
    for row, idx in enumerate(obs_indices):
        H[row, idx] = 1.0

    noise_std = np.array([
        REFERENCE_NOISE_STD if s["quality_tier"] == "reference" else ORDINARY_NOISE_STD
        for s in stations
    ])
    R = np.diag(noise_std ** 2)

    return H, R, obs_indices, stations


def generate_real_network_observations(true_traj, H, R, seed=1):
    """
    Generate noisy observations from the real station network, with each
    station's own (heterogeneous) noise level from R -- unlike
    `observations.generate_observations`, which assumes one shared noise
    std for every sensor.

    Returns
    -------
    obs_times : list[int]  (every time step -- real station networks report continuously)
    obs_values : list[np.ndarray], each shape (n_observed,)
    """
    rng = np.random.default_rng(seed)
    T = true_traj.shape[0]
    noise_std = np.sqrt(np.diag(R))

    obs_times, obs_values = [], []
    for t in range(T):
        clean = H @ true_traj[t]
        noisy = clean + rng.normal(scale=noise_std)
        obs_times.append(t)
        obs_values.append(noisy)

    return obs_times, obs_values
