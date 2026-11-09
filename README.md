# Machine-Learning-Augmented Data Assimilation for Nonlinear Dynamical Systems

Progression demonstrated by this project:

**Physics + mathematical modeling → dynamical systems → state estimation → data assimilation → ML**

## Problem statement

Given a nonlinear dynamical system

$$x_{t+1} = M(x_t) + \eta_t \qquad y_t = H(x_t) + \epsilon_t$$

where $x_t$ is the true (hidden) physical state, $M$ is a (possibly imperfect)
dynamical model, $y_t$ is a sparse/noisy observation, $H$ is the observation
operator, and $\eta_t, \epsilon_t$ are model and observation noise: **can we
recover the true hidden state from imperfect observations, and can machine
learning improve the underlying model itself?**

## Stage A — Lorenz-96 + Ensemble Kalman Filter

- `src/lorenz96.py` — Lorenz-96 equations, integrated with `scipy.integrate.solve_ivp`.
- `src/observations.py` — sparse spatial observation operator `H`, Gaussian observation noise.
- `src/enkf.py` — Ensemble Kalman Filter: forecast step (propagate an ensemble
  through `M`) + analysis step (Kalman-gain correction using observations),
  with multiplicative covariance inflation to prevent filter/ensemble collapse.
- `src/evaluation.py` — RMSE scoring and diagnostic plots (state reconstruction,
  RMSE-over-time, Hovmöller true/estimate/error panels).

**Result:** EnKF RMSE ≈ **0.41**, vs. a naive (no-assimilation) rollout RMSE
of ≈ **5.04** (worse than the climatological std of ≈3.67 — i.e. the raw
model is *useless* without assimilation, chaos amplifies any initial error).
EnKF also accurately reconstructs *unobserved* variables via the ensemble's
empirical spatial covariance.

Run: `python experiments/lorenz96_experiment.py`

## Step 2 — The mathematics (see report / notebooks)

- Forecast: $x_t^f = M(x_{t-1}^a)$, with ensemble spread standing in for the
  forecast error covariance $P^f$ (no need to propagate an explicit NxN
  covariance matrix — this is what makes EnKF tractable for nonlinear, high-
  dimensional systems).
- Analysis: $x_t^a = x_t^f + K(y_t - Hx_t^f)$, $K = P^f H^\top (HP^fH^\top+R)^{-1}$.
- Covariance inflation and (for real systems) localization are the two
  practical tricks that keep EnKF from collapsing/diverging in chaotic
  systems — both are implemented and discussed in `src/enkf.py`.

## Step 3 — Machine learning model-error correction (the heart of the project)

An artificial, unknown model error is injected:

$$x_{t+1} = M(x_t) + \delta(x_t), \qquad \delta(x_i) = a\sin(x_i) + b x_i$$

`M` (what we assume) never sees $\delta$. A small neural network
(`src/ml_correction.py`, `sklearn.neural_network.MLPRegressor`) is trained
**only on data** — one-step residuals between a nature run and the (wrong)
model's prediction — to learn $\hat\delta_\theta(x) \approx \delta(x)$.

Four methods are compared on a held-out test period:

| Method | RMSE | What it uses |
|---|---|---|
| Physical model alone | 5.75 | wrong physics, no correction |
| EnKF | 1.31 | wrong physics + live observations |
| ML correction alone | 6.24 | learned correction, no live observations |
| **ML + EnKF** | **0.91** | learned correction + live observations |

Key finding: **ML correction alone actually performs worse than doing
nothing**, because it's an open-loop correction — in a chaotic system, any
residual imperfection in $\hat\delta_\theta$ still grows exponentially with
no observational anchor to keep it in check. EnKF alone helps a lot (it has
real-time data), but ML+EnKF is best: the corrected model starts each
forecast step much closer to the truth, so the observations only need to
do fine, cheap corrections rather than fight a badly biased model.

Run: `python experiments/lorenz96_experiment.py` (runs Stage A, then Step 3)

## Step 4 — 2D Shallow-Water model (in progress)

Extends the same framework ($M$, $H$, EnKF, ML correction) to a 2D
shallow-water system (height $h$, velocities $u,v$) as a geophysically
realistic stand-in for ocean/atmosphere data assimilation.

## Repository layout

```
ml-data-assimilation/
├── README.md
├── requirements.txt
├── src/
│   ├── lorenz96.py        # Stage A: physics + integration
│   ├── observations.py    # H operator + noisy/sparse obs generation
│   ├── enkf.py             # Ensemble Kalman Filter (forecast + analysis)
│   ├── ml_correction.py   # Step 3: neural-network model-error correction
│   └── evaluation.py      # RMSE + plotting
├── experiments/
│   ├── lorenz96_experiment.py       # Stage A + Step 3, end to end
│   └── shallow_water_experiment.py  # Step 4 (geophysical extension)
├── notebooks/              # exploratory / teaching notebooks
├── figures/                # generated plots
└── report/                 # write-up
```

## Setup

```bash
pip install -r requirements.txt
python experiments/lorenz96_experiment.py
```
