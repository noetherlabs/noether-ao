"""Short-horizon image-shift prediction and range forecasting (protocol v1).

A tip-tilt benchmark for a small solar aperture: frozen-flow von Karman
turbulence crosses a 38 cm pupil sampled at an assumed 1 kHz. The target is
the least-squares (Z) tilt along the wind, in arcsec. Inputs are the measured
global tilt and a 4 x 4 grid of subaperture tilts.

Trusted benchmark models use ``PastOnly`` to request lagged inputs relative
to evaluator-supplied anchors. This indexing contract is not a sandbox for
untrusted models. NumPy only: the benchmark does not use the JAX optics.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

RAD2AS = 180 / np.pi * 3600
LAM = 500e-9  # wavelength at which r0 is quoted (m)
D = 0.38  # aperture diameter (m)
DX = 0.01  # screen pixel (m); one pixel per frame is 10 m/s at 1 kHz
NPIX = int(round(D / DX))
N, NTRAIN, HIST = 40000, 20000, 1000
HORIZONS = [1, 2, 5, 10, 20, 30, 50, 100, 1000, 10000]
SUB, SUBPIX = 4, 9  # 4 x 4 subapertures of 9 cm
GLOBAL_LAGS = np.r_[np.arange(20), np.arange(20, 101, 10), np.arange(150, HIST, 50)]
SPATIAL_LAGS = np.arange(20)
EVAL_SEEDS = [0, 1, 2, 3, 4]
TWIN_SEED = 99
UPPER_SEEDS = list(range(100, 110))
T95 = 2.776  # two-sided 95% t quantile, 4 degrees of freedom
LAMS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

REGIMES = {
    # layers: (fraction of turbulence, pixels per frame); noise: global tilt, arcsec
    "one layer, 10 m/s": {"layers": [(1.0, 1)], "noise": 0.03},
    "two layers, 10 + 20 m/s": {"layers": [(0.6, 1), (0.4, 2)], "noise": 0.03},
    "low contrast (noisy sensor)": {"layers": [(1.0, 1)], "noise": 0.15},
}


# ----------------------------------------------------------------- the twin

def screen_strip(nx, r0, L0, rng, width=512):
    """Central NPIX-wide strip of a von Karman FFT phase screen (rad at LAM)."""
    fx = np.fft.fftfreq(nx, DX)
    fy = np.fft.fftfreq(width, DX)
    f2 = fx[:, None] ** 2 + fy[None, :] ** 2
    psd = 0.023 * r0 ** (-5 / 3) * (f2 + (1 / L0) ** 2) ** (-11 / 6)
    psd[0, 0] = 0.0
    amp = np.sqrt(psd / (nx * DX * width * DX))
    del f2, psd
    coef = (rng.standard_normal((nx, width)) + 1j * rng.standard_normal((nx, width))) * amp
    del amp
    screen = np.fft.ifft2(coef) * nx * width
    y0 = (width - NPIX) // 2
    return np.ascontiguousarray(screen.real[:, y0:y0 + NPIX])


def tilt_kernel():
    """Least-squares tilt along the first axis over the circular pupil."""
    c = (np.arange(NPIX) - (NPIX - 1) / 2) * DX
    X, Y = np.meshgrid(c, c, indexing="ij")
    mask = X**2 + Y**2 <= (D / 2) ** 2
    return np.where(mask, X, 0.0) / np.sum(X[mask] ** 2)


def kernels():
    """Global tilt kernel, then 16 subaperture kernels; row index runs along the wind."""
    ks = [tilt_kernel()]
    c = (np.arange(SUBPIX) - (SUBPIX - 1) / 2) * DX
    X, _ = np.meshgrid(c, c, indexing="ij")
    local = X / np.sum(X**2)
    offsets = [1 + SUBPIX * i for i in range(SUB)]
    for i in offsets:
        for j in offsets:
            k = np.zeros((NPIX, NPIX))
            k[i:i + SUBPIX, j:j + SUBPIX] = local
            ks.append(k)
    return np.stack(ks)


def sensor_series(strip, step, n, K, chunk=4000):
    windows = sliding_window_view(strip, K.shape[1:])[:, 0]
    out = np.empty((n, K.shape[0]))
    for s in range(0, n, chunk):
        pos = np.arange(s, min(s + chunk, n)) * step
        out[s:s + len(pos)] = np.einsum("tij,kij->tk", windows[pos], K)
    return out * LAM / (2 * np.pi) * RAD2AS


def drifting_r0(n, r0, rng, corr_frames=5000, spread=0.25):
    """Seeing that wanders slowly: log r0 is smoothed noise (roughly 3-7 cm)."""
    k = np.exp(-0.5 * (np.arange(-3 * corr_frames, 3 * corr_frames + 1) / corr_frames) ** 2)
    z = np.convolve(rng.standard_normal(n + len(k)), k, mode="same")[:n]
    z = (z - z.mean()) / z.std()
    return r0 * np.exp(spread * z)


def simulate(regime, seed, n=N, r0=0.04, L0=20.0, drift=False):
    """True global tilt (target) and measured inputs: global + 16 subapertures."""
    rng = np.random.default_rng(seed)
    K = kernels()
    truth = np.zeros((n, K.shape[0]))
    for frac, step in regime["layers"]:
        nx = 1 << int(np.ceil(np.log2(n * step + NPIX)))
        strip = screen_strip(nx, r0 * frac ** (-3 / 5), L0, rng)
        truth += sensor_series(strip, step, n, K)
    if drift:
        truth *= ((drifting_r0(n, r0, rng) / r0) ** (-5 / 6))[:, None]
    # a subaperture collects 1/16 of the light, so its noise is 4x the global noise
    sigma = np.r_[regime["noise"], np.full(K.shape[0] - 1, regime["noise"] * SUB)]
    measured = truth + sigma * rng.standard_normal(truth.shape)
    return truth[:, 0], measured


# ---------------------------------------------------------- history access

class PastOnly:
    """Bounds-checked lagged inputs for trusted, evaluator-anchored models.

    Lags include zero (the anchor frame). The evaluator controls the anchors;
    Python callers can still access the underlying array or choose other anchors.
    """

    def __init__(self, measured):
        self._m = np.asarray(measured)
        if self._m.ndim != 2 or 0 in self._m.shape:
            raise ValueError("measured must be a nonempty frame-by-channel array")

    def window(self, idx, lags, channels):
        def indices(value, name, limit):
            a = np.asarray(value)
            if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
            if np.any(a < 0) or np.any(a >= limit):
                raise ValueError(f"{name} must lie in [0, {limit})")
            return a.astype(np.intp, copy=False)

        idx = indices(idx, "anchors", self._m.shape[0])
        lags = indices(lags, "lags", HIST)
        channels = indices(channels, "channels", self._m.shape[1])
        if np.any(idx < lags.max()):
            raise ValueError("requested history precedes the start of the recording")
        rows = idx[:, None] - lags[None, :]
        return self._m[rows[:, :, None], channels[None, None, :]].reshape(len(idx), -1)


def _ridge(X, y, lam):
    mu = X.mean(0)
    ym = y.mean()
    Xc = X - mu
    G = Xc.T @ Xc
    w = np.linalg.solve(G + lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), Xc.T @ (y - ym))
    return w, mu, ym


def _predict(model, X):
    w, mu, ym = model
    return (X - mu) @ w + ym


def ridge(X, y):
    """Ridge regression with the penalty chosen on the last 25% of the rows."""
    cut = int(0.75 * len(y))
    best = None
    for lam in LAMS:
        m = _ridge(X[:cut], y[:cut], lam)
        err = np.mean((_predict(m, X[cut:]) - y[cut:]) ** 2)
        if best is None or err < best[0]:
            best = (err, lam)
    return _ridge(X, y, best[1])


# --------------------------------------------------------------- the models

class NoChange:
    name = "no change"

    def fit(self, view, idx, y):
        return self

    def predict(self, view, idx):
        return view.window(idx, [0], [0])[:, 0]


class GlobalLinear:
    name = "global linear"

    def features(self, view, idx):
        return view.window(idx, GLOBAL_LAGS, [0])

    def fit(self, view, idx, y):
        self.m = ridge(self.features(view, idx), y)
        return self

    def predict(self, view, idx):
        return _predict(self.m, self.features(view, idx))


class SpatialLinear(GlobalLinear):
    """Global tilt and every subaperture; upwind subapertures see turbulence first."""
    name = "full-sensor linear"

    def features(self, view, idx):
        return view.window(idx, SPATIAL_LAGS, range(1 + SUB * SUB))


class SubaperturesOnly(GlobalLinear):
    """Subapertures only, for the post-hoc check of the shared-noise artefact."""
    name = "subapertures only"

    def features(self, view, idx):
        return view.window(idx, SPATIAL_LAGS, range(1, 1 + SUB * SUB))


class RandomFeatures(GlobalLinear):
    """Nonlinear challenger: linear terms plus 500 random Fourier features."""
    name = "nonlinear (random features)"

    def fit(self, view, idx, y):
        X = view.window(idx, GLOBAL_LAGS, [0])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        rng = np.random.default_rng(0)
        best = None
        cut = int(0.75 * len(y))
        for scale in (0.5, 1.0, 2.0):
            W = rng.standard_normal((Z.shape[1], 500)) / (scale * np.sqrt(Z.shape[1]))
            b = rng.uniform(0, 2 * np.pi, 500)
            F = np.c_[Z, np.sqrt(2 / 500) * np.cos(Z @ W + b)]
            m = ridge(F[:cut], y[:cut])
            err = np.mean((_predict(m, F[cut:]) - y[cut:]) ** 2)
            if best is None or err < best[0]:
                best = (err, W, b)
        self.W, self.b = best[1], best[2]
        self.m = ridge(self._feat(Z), y)
        return self

    def _feat(self, Z):
        return np.c_[Z, np.sqrt(2 / 500) * np.cos(Z @ self.W + self.b)]

    def predict(self, view, idx):
        Z = (view.window(idx, GLOBAL_LAGS, [0]) - self.mu) / self.sd
        return _predict(self.m, self._feat(Z))


MODELS = [NoChange, GlobalLinear, SpatialLinear, RandomFeatures]
# Historical result key; this is a more-data reference, not a mathematical bound.
UPPER = "upper bound (global linear, 400k frames)"
SHUFFLED = "shuffled-target check"


# ------------------------------------------------------------ the evaluator

def skill(pred, target):
    """Fraction of target variance explained: 1 is perfect, 0 is the mean."""
    return float(1 - np.mean((pred - target) ** 2) / np.var(target))


def split(h):
    if (isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
            or not 0 < h < min(NTRAIN - HIST, N - NTRAIN - HIST)):
        raise ValueError("horizon must be a positive integer leaving nonempty training and test sets")
    h = int(h)
    tr = np.arange(HIST, NTRAIN - h)
    te = np.arange(NTRAIN + HIST, N - h)
    # leakage guard: no training target, and no test input window, crosses NTRAIN
    if tr.max() + h >= NTRAIN or te.min() - (HIST - 1) < NTRAIN:
        raise ValueError("training targets and test histories must be disjoint")
    return tr, te


def upper_bound(runs, h):
    """More-data global-linear reference; the legacy name is not a bound claim."""
    X, y = [], []
    for target, measured in runs:
        idx = np.arange(HIST, N - h)
        X.append(PastOnly(measured).window(idx, GLOBAL_LAGS, [0]))
        y.append(target[idx + h])
    return _ridge(np.concatenate(X), np.concatenate(y), 1e-4)


def range_forecast(target, measured, q_twin):
    """Marginal coverage over overlapping forecast-window/time pairs.

    The legacy ``conformal`` key denotes an empirical training quantile. With
    dependent windows and drifting seeing, no conformal guarantee is claimed.
    """
    x = measured[:, 0]

    def windows(lo, hi):
        rows = []
        for t0 in range(lo, hi - 10000 + 1, 1000):
            past = x[t0 - HIST:t0]
            rows.append((past.mean(), past.std(), target[t0:t0 + 10000]))
        return rows

    calib = windows(HIST, NTRAIN)
    test = windows(NTRAIN + HIST, N)
    scores = np.concatenate([np.abs(f - m) / s for m, s, f in calib])
    q_conf = float(np.quantile(scores, 0.954))
    out = {}
    for name, q in (("naive", 2.0), ("twin", q_twin), ("conformal", q_conf)):
        cover = np.mean(np.concatenate([np.abs(f - m) <= q * s for m, s, f in test]))
        width = np.mean([q * s for m, s, f in test])
        out[name] = {"coverage": float(cover), "half_width_arcsec": float(width), "q": float(q)}
    return out


def twin_q(regime):
    """Band multiplier from an independent drifting twin run."""
    target, measured = simulate(regime, TWIN_SEED, n=60000, drift=True)
    x = measured[:, 0]
    num, den = [], []
    for t0 in range(HIST, len(x) - 10000, 1000):
        past = x[t0 - HIST:t0]
        num.append(np.mean((target[t0:t0 + 10000] - past.mean()) ** 2))
        den.append(past.var())
    return float(2 * np.sqrt(np.mean(num) / np.mean(den)))


def wind_estimate(measured):
    """Speed from the delay between the upwind and downwind subaperture rows."""
    col = [measured[NTRAIN:, 1 + i * SUB:1 + (i + 1) * SUB].mean(1) for i in range(SUB)]
    a, b = col[SUB - 1] - col[SUB - 1].mean(), col[0] - col[0].mean()
    lags = np.arange(0, 61)
    corr = [np.corrcoef(a[:len(a) - L], b[L:])[0, 1] for L in lags]
    lag = int(lags[int(np.argmax(corr))])
    sep = SUBPIX * (SUB - 1) * DX
    return {"delay_frames": lag, "speed_m_s": float(sep / (lag * 1e-3)) if lag else None}


def importance_map(model, view, idx):
    """Summed |weight x feature sd| per channel: the global tilt, then a 4 x 4 map."""
    X = model.features(view, idx)
    w = model.m[0] * X.std(0)
    imp = np.abs(w).reshape(len(SPATIAL_LAGS), -1).sum(0)
    return float(imp[0]), imp[1:].reshape(SUB, SUB)


def paired(a, b):
    """Mean paired difference and its 95% t-interval over seeds."""
    d = np.asarray(a) - np.asarray(b)
    half = T95 * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def run_benchmark(log=print):
    report = {}
    for rname, regime in REGIMES.items():
        data = {s: simulate(regime, s) for s in EVAL_SEEDS}
        upper_runs = [simulate(regime, s) for s in UPPER_SEEDS]
        res = {m.name: {} for m in MODELS}
        res[UPPER], res[SHUFFLED] = {}, {}
        imp = None
        for h in HORIZONS:
            tr, te = split(h)
            ub = upper_bound(upper_runs, h)
            for key in res:
                res[key][h] = []
            for s, (target, measured) in data.items():
                view = PastOnly(measured)
                for M in MODELS:
                    model = M().fit(view, tr, target[tr + h])
                    res[M.name][h].append(skill(model.predict(view, te), target[te + h]))
                    if M is SpatialLinear and h == 20 and s == 0:
                        imp = importance_map(model, view, te)
                res[UPPER][h].append(skill(_predict(ub, GlobalLinear().features(view, te)), target[te + h]))
                if h == 10:
                    shuffled = np.random.default_rng(s).permutation(target[tr + h])
                    res[SHUFFLED][h].append(
                        skill(GlobalLinear().fit(view, tr, shuffled).predict(view, te), target[te + h]))
        q = twin_q(regime)
        report[rname] = {
            "skill": {k: {str(h): v for h, v in d.items() if v} for k, d in res.items()},
            "range": [range_forecast(*simulate(regime, s, drift=True), q) for s in EVAL_SEEDS],
            "q_twin": q,
            "wind": [wind_estimate(data[s][1]) for s in EVAL_SEEDS],
            "true_speeds_m_s": [10 * step for _, step in regime["layers"]],
            "importance_h20_seed0": {"global": imp[0], "subapertures": imp[1].tolist()},
        }
        log(f"{rname}: benchmark done")
    return report


def run_posthoc(horizons=(1, 5, 10, 20, 30), log=print):
    """Post-hoc, not pre-registered: global tilt derived from the measured subapertures."""
    out = {}
    for rname, regime in REGIMES.items():
        rows = {h: {"global from subapertures": [], "subapertures only": []} for h in horizons}
        for s in EVAL_SEEDS:
            target, measured = simulate(regime, s)
            measured = measured.copy()
            measured[:, 0] = measured[:, 1:].mean(1)
            view = PastOnly(measured)
            for h in horizons:
                tr, te = split(h)
                for M, key in ((GlobalLinear, "global from subapertures"), (SubaperturesOnly, "subapertures only")):
                    rows[h][key].append(skill(M().fit(view, tr, target[tr + h]).predict(view, te), target[te + h]))
        out[rname] = {str(h): v for h, v in rows.items()}
        log(f"{rname}: post-hoc check done")
    return out
