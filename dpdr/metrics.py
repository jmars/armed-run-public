"""Metrics: regime classification, stuck/rescue/relapse measures, gate checks.

Regime taxonomy (plan §4 Phase 3):
  baseline       G never collapsed (never below 0.1) and healthy at the end
  depersonalized stuck at the end (G < 0.1, E > Theta_eff, c > 0.5 sustained
                 >= 5*tau_G) and never made a sustained recovery before it
  recovering     collapsed at some point but NOT stuck at the end (escaped)
  relapsed       made a sustained recovery (G > 0.5 for >= 2*tau_G) and then
                 re-entered a stuck state

Gate checks (plan §4c, G1-G2c) reproduce the sim0 gate logic and are used as
regression tests in tests/test_model.py.
"""
from __future__ import annotations

import numpy as np

from .events import baseline_schedule, failure_schedule, rescue_schedule
from .integrate import simulate
from .model import Params

REGIME_CODES = {"baseline": 0, "depersonalized": 1, "recovering": 2,
                "relapsed": 3}

EPISODE = (100.0, 200.0)
RESCUE = (800.0, 1000.0)


# ------------------------------------------------------------- primitives
def first_below(G, t, thresh=0.1):
    """First time G crosses below thresh (None if never)."""
    idx = np.where(G < thresh)[0]
    return None if idx.size == 0 else float(t[idx[0]])


def first_above_after(G, t, thresh=0.5, t_after=0.0):
    """First time G rises above thresh after t_after (None if never)."""
    idx = np.where((G > thresh) & (t > t_after))[0]
    return None if idx.size == 0 else float(t[idx[0]])


def is_stuck(sol, p: Params, k_tau: float = 5.0) -> bool:
    """Stuck at the end: G < 0.1, E > Theta_eff, c > 0.5 over the final
    k_tau*tau_G window (plan's stuck criterion, sustained > 5*tau_G)."""
    t, G, E, c, Teff = (sol["t"], sol["G"], sol["E"], sol["c"],
                        sol["Theta_eff"])
    w = t >= t[-1] - k_tau * p.tau_G
    if not w.any():
        w = slice(None)
    return bool(G[w].mean() < 0.1 and E[w].mean() > Teff[w].mean()
                and c[w].mean() > 0.5)


def stuck_duration(sol, p: Params) -> float:
    """Duration of the final stuck epoch: T minus the last time G >= 0.1."""
    t, G = sol["t"], sol["G"]
    idx = np.where(G >= 0.1)[0]
    if idx.size == 0:
        return float(t[-1] - t[0])
    return float(t[-1] - t[idx[-1]])


def detect_relapse(sol, p: Params):
    """Sustained recovery (G > 0.5 for >= 2*tau_G) starting AFTER the first
    collapse, followed by re-collapse (G < 0.1).  The initial healthy period
    is not a recovery epoch.  Returns (relapsed, t_relapse)."""
    t, G = sol["t"], sol["G"]
    t_coll = first_below(G, t)
    if t_coll is None:
        return False, None
    hi = (G > 0.5).astype(int)
    d = np.diff(hi)
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if hi[0]:
        starts = [0] + starts
    if hi[-1]:
        ends = ends + [len(hi)]
    for s, e in zip(starts, ends):
        if t[s] <= t_coll:
            continue                   # pre-collapse healthy period
        if t[e - 1] - t[s] >= 2 * p.tau_G:      # sustained recovery epoch
            below = np.where(G[e:] < 0.1)[0]
            if below.size:
                return True, float(t[e + below[0]])
    return False, None


def classify(sol, p: Params) -> str:
    """Classify a trajectory into one of the four regime labels."""
    relapsed, _ = detect_relapse(sol, p)
    if relapsed:
        return "relapsed"
    if is_stuck(sol, p):
        return "depersonalized"
    if sol["G"].min() < 0.1:
        return "recovering"
    return "baseline"


def rescue_success(sol, p: Params) -> bool:
    """Full re-couple (plan §4 Phase 3): G(t_end) > 0.5, E(t_end) < 0.8*Theta,
    no relapse within 2*tau_G of the end."""
    if not (sol["G"][-1] > 0.5 and sol["E"][-1] < 0.8 * p.Theta):
        return False
    relapsed, t_rel = detect_relapse(sol, p)
    return not (relapsed and t_rel is not None
                and t_rel > sol["t"][-1] - 2 * p.tau_G)


def summarize(sol, p: Params) -> dict:
    """Flat numeric summary of one run (sweep output row)."""
    reg = classify(sol, p)
    relapsed, t_rel = detect_relapse(sol, p)
    return dict(
        G_end=float(sol["G"][-1]), D_end=float(sol["D"][-1]),
        S_end=float(sol["S"][-1]), g_end=float(sol["g"][-1]),
        E_end=float(sol["E"][-1]), c_end=float(sol["c"][-1]),
        Theta_eff_end=float(sol["Theta_eff"][-1]),
        u_loop_end=float(sol["u_loop"][-1]),
        G_min=float(sol["G"].min()), c_max=float(sol["c"].max()),
        t_collapse=first_below(sol["G"], sol["t"]),
        t_recover=(np.nan if (r := first_above_after(sol["G"], sol["t"], 0.5,
                                                     EPISODE[1])) is None
                   else float(r)),
        T_stuck=stuck_duration(sol, p),
        relapsed=float(relapsed),
        t_relapse=(np.nan if t_rel is None else float(t_rel)),
        regime=REGIME_CODES[reg],
        collapsed_at=(np.nan if sol["collapsed_at"] is None
                      else float(sol["collapsed_at"])),
        nfev=int(sol["nfev"]),
    )


# ------------------------------------------------------------- gate checks
def check_gates(p: Params | None = None, T_base: float = 1000.0,
                T_fail: float = 600.0, T_resc: float = 1400.0) -> dict:
    """Evaluate plan §4c gates G1-G2c on the three standard scenarios.

    Horizons: baseline only needs settling (tau_S = 100), failure needs
    10*tau_G = 200 past the episode end plus margin, rescue needs the
    post-rescue stay window.  The full-horizon (T=2000) values are printed by
    sim0.py; these trimmed horizons test the same conditions faster.
    """
    p = p if p is not None else Params()
    out: dict[str, bool] = {}

    s = simulate(p, baseline_schedule(), T_base)          # G1
    out["G1"] = bool(0.0 < s["G"][-1] < 1.0 and s["E"][-1] < p.Theta
                     and s["c"][-1] < 0.05)

    s = simulate(p, failure_schedule(), T_fail)           # G2, G2b
    w = s["t"] >= EPISODE[1] + 10 * p.tau_G
    out["G2"] = bool(s["G"][-1] < 0.1 and s["E"][-1] > p.Theta
                     and s["c"][-1] > 0.5)
    out["G2b"] = bool(s["G"][w].max() < 0.1 and stuck_duration(s, p)
                      >= 10 * p.tau_G)

    s = simulate(p, rescue_schedule(), T_resc)            # G2c
    w = s["t"] >= RESCUE[1]
    out["G2c"] = bool(s["G"][-1] > 0.5 and s["G"][w].min() > 0.5)
    return out
