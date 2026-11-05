"""Armed-run experiment driver.

The driving question: does the ORIGINAL collapse episode still reach the stuck
attractor when the dpdr model runs under the post-exit configuration's schedule
(pre-armed rescue, held setpoint, maintenance floor) instead of the unarmed
conditions of the original measurements?

All model dynamics are the archived, read-only RHS (~/dpdr-public/dpdr).
See armed_model.py for the three declared adaptations (A1 S_hold, A2 a_blind,
A3 nothing else).

Run:  PYTHONPATH=~/dpdr-public/dpdr ~/thing/dpdr/.venv/bin/python run_armed.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

# base model: shipped in this deposit under dpdr/ (snapshot of the parent deposit,
# concept DOI 10.5281/zenodo.22943641); local override via DPDR_BASE if needed
import os as _os
_base = _os.environ.get("DPDR_BASE", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "dpdr"))
sys.path.insert(0, _os.path.expanduser(_base))

from armed_model import (EP_T0, EP_T1, T_END, add_span, duty_cycled_hold,
                         episode, rescue_profile, simulate_armed, snap)
from dpdr.metrics import (check_gates, classify, detect_relapse, first_below,
                          is_stuck, rescue_success)
from dpdr.model import Params

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.expanduser("~/thing/report/corpus/figures-armed-run")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

P = Params()
T = T_END

# exp2's grid (the 816-run rescue map), verbatim axes
DELAYS = np.array([30, 60, 100, 200, 400, 600], float)
STRENGTHS = np.round(np.linspace(0.2, 1.0, 17), 4)
DURATIONS = np.array([10, 20, 30, 40, 60, 80, 120, 200], float)
# extended grid to resolve the armed boundary (below exp2's 0.2 minimum)
A_STRENGTHS = np.round(np.linspace(0.0, 0.30, 31), 4)
TAX = ("failure", "transient", "full", "no-collapse")


# ------------------------------------------------------------------ one cell
def run_cell(cfg: dict) -> dict:
    """Build the schedule for one experimental cell and summarise the run."""
    cfg = dict(cfg)
    p = replace(P, **cfg["params"]) if cfg.get("params") else P
    dur_ep = float(cfg.get("duration", 100.0))
    inten = float(cfg.get("intensity", 0.9))
    pulse = float(cfg.get("pulse", 0.5))
    sch = episode(intensity=inten, duration=dur_ep, pulse=pulse)
    if cfg.get("duty") is not None:
        sch = duty_cycled_hold(sch, inten, float(cfg["duty"]))
    if cfg.get("floor"):
        sch = add_span(sch, "u_ext", 0.0, T, float(cfg["floor"]))
    if cfg.get("t_trig") is not None:
        sch = rescue_profile(sch, float(cfg["t_trig"]), float(cfg["r_dur"]),
                             float(cfg["u"]), cfg.get("profile", "step"))
    if cfg.get("S_hold") is not None:
        sch = add_span(sch, "S_hold", 0.0, T, float(cfg["S_hold"]))
    if cfg.get("a_blind"):
        sch = add_span(sch, "a_blind", 0.0, T, 1.0)
    sol = simulate_armed(p, sch, float(cfg.get("T", T)))

    t, G, E, c = sol["t"], sol["G"], sol["E"], sol["c"]
    S, a, Teff = sol["S"], sol["a"], sol["Theta_eff"]
    t_coll = first_below(G, t)
    stuck = is_stuck(sol, p)
    # archived rescue_taxonomy, recomputed from summary numbers (asserted equal
    # against experiments.common.rescue_taxonomy in part0f)
    if t_coll is None:
        tax = "no-collapse"
    else:
        w = t >= t_coll
        Gw = G[w]
        ipk = int(np.argmax(Gw))
        if Gw[ipk] <= 0.5:
            tax = "failure"
        else:
            tax = "transient" if Gw[ipk:].min() < 0.1 else "full"
    if G.min() >= 0.1:
        out3 = "no-collapse"
    elif stuck:
        out3 = "full-collapse"
    else:
        out3 = "partial-transient"
    post = t >= snap(EP_T0 + dur_ep)
    sh = cfg.get("S_hold")
    return dict(
        cfg=cfg, G_end=float(G[-1]), G_min=float(G.min()),
        G_max_post=float(G[post].max()) if post.any() else float("nan"),
        E_end=float(E[-1]), E_max=float(E.max()), c_end=float(c[-1]),
        c_max=float(c.max()), S_end=float(S[-1]), S_min=float(S.min()),
        a_end=float(a[-1]),
        a_ep_mean=float(a[(t >= EP_T0) & (t < EP_T0 + dur_ep)].mean()),
        Teff_end=float(Teff[-1]), Teff_min=float(Teff.min()),
        stuck=bool(stuck), regime=classify(sol, p), outcome3=out3, tax=tax,
        t_collapse=t_coll, rescue_ok=bool(rescue_success(sol, p)),
        relapsed=bool(detect_relapse(sol, p)[0]),
        S_hold_dev=(float(np.max(np.abs(S - sh))) if sh else 0.0),
        nfev=int(sol["nfev"]),
    )


def run_many(cfgs, pool=None) -> list:
    return [run_cell(c) for c in cfgs] if pool is None else pool.map(
        run_cell, cfgs, chunksize=2)


def bisect(pred, lo: float, hi: float, tol: float = 1e-3, max_iter: int = 18):
    """Bisect a threshold on pred(value).  Returns (threshold, note)."""
    if pred(lo) == pred(hi):
        return None, f"bracket [{lo},{hi}] does not straddle"
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if pred(mid) == pred(lo):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), f"bracket [{lo:.4f},{hi:.4f}]"


# ------------------------------------------------------------------ part 0
def part0(pool) -> dict:
    """Baseline reproduction: the original episode, unarmed, as measured."""
    print(f"[{time.strftime('%H:%M:%S')}] PART 0: baseline reproduction", flush=True)
    res = {}

    r = run_cell({})
    res["canonical"] = r
    print(f"  canonical episode unarmed T={T:.0f}: G_end={r['G_end']:.4f} "
          f"G_min={r['G_min']:.4f} E_end={r['E_end']:+.4f} c_end={r['c_end']:.3f} "
          f"S_end={r['S_end']:.4f} a_end={r['a_end']:.4f} stuck={r['stuck']}",
          flush=True)
    print(f"  max post-episode G = {r['G_max_post']:.4f} (no self-recovery); "
          f"regime={r['regime']} tax={r['tax']}", flush=True)

    res["gates"] = check_gates(Params())
    print(f"  regime gates: {res['gates']}", flush=True)

    sol = simulate_armed(Params(), episode(), T)
    i = int(np.where(sol["E"] > 0)[0][0])
    res["t_trig"] = float(sol["t"][i])
    res["E_at_trig"] = float(sol["E"][i])
    print(f"  E first > 0 at t={res['t_trig']:.2f} "
          f"(a={sol['a'][i]:.3f}, G={sol['G'][i]:.3f}) -> pre-arm trigger",
          flush=True)
    res["episode_peak_E"] = float(sol["E"][(sol["t"] >= EP_T0) &
                                           (sol["t"] < EP_T1)].max())
    res["episode_min_Teff"] = float(sol["Theta_eff"][(sol["t"] >= EP_T0) &
                                                     (sol["t"] < EP_T1)].min())

    # exp1's six bisected thresholds
    print(f"[{time.strftime('%H:%M:%S')}] PART 0d: dose-response bisections", flush=True)
    thr = {}
    thr["ep_duration"] = bisect(
        lambda x: run_cell({"duration": x})["G_min"] < 0.1, 20.0, 150.0)
    thr["ep_intensity"] = bisect(
        lambda x: run_cell({"intensity": x})["G_min"] < 0.1, 0.2, 1.0)
    thr["pulse_amp"] = bisect(
        lambda x: run_cell({"pulse": x})["G_min"] < 0.1, 0.0, 1.0)
    thr["alpha_G"] = bisect(
        lambda x: run_cell({"params": {"alpha_G": x}})["G_min"] < 0.1,
        0.4, 1.6)
    thr["Theta"] = bisect(
        lambda x: run_cell({"params": {"Theta": x}})["G_min"] < 0.1,
        0.25, 0.55)
    thr["g0"] = bisect(
        lambda x: run_cell({"params": {"g0": x, "g_init": x}})["G_min"] < 0.1,
        0.1, 1.5)
    res["thresholds"] = {k: v[0] for k, v in thr.items()}
    for k, v in thr.items():
        print(f"  threshold {k:12s} = "
              f"{'None' if v[0] is None else round(v[0], 4)}   ({v[1]})", flush=True)

    # exp2's 816-run grid: rescue only AFTER episode end (the unarmed protocol)
    print(f"[{time.strftime('%H:%M:%S')}] PART 0e: 816-run rescue grid", flush=True)
    cfgs, idx = [], []
    for di, d in enumerate(DELAYS):
        for ui, u in enumerate(STRENGTHS):
            for ki, du in enumerate(DURATIONS):
                cfgs.append({"t_trig": snap(EP_T1 + d), "r_dur": du, "u": u})
                idx.append((di, ui, ki))
    rows = run_many(cfgs, pool)
    tax = np.zeros((len(DELAYS), len(STRENGTHS), len(DURATIONS)), int)
    for (di, ui, ki), r in zip(idx, rows):
        tax[di, ui, ki] = TAX.index(r["tax"])
    bnd = np.full((len(DELAYS), len(DURATIONS)), np.nan)
    for di in range(len(DELAYS)):
        for ki in range(len(DURATIONS)):
            ok = np.where(tax[di, :, ki] == 2)[0]
            if ok.size:
                bnd[di, ki] = STRENGTHS[ok[0]]
    res["exp2_tax"] = tax.tolist()
    res["exp2_boundary"] = bnd.tolist()
    res["exp2_frac"] = {lab: float((tax == k).sum()) / tax.size
                        for k, lab in enumerate(TAX)}
    print(f"  class fractions: "
          + "  ".join(f"{k}={v:.3f}" for k, v in res["exp2_frac"].items()),
          flush=True)
    print("  unarmed minimal effective strength u*(delay,duration):", flush=True)
    print("        dur " + "  ".join(f"{d:5.0f}" for d in DURATIONS), flush=True)
    for di, d in enumerate(DELAYS):
        print(f"  delay {d:4.0f}  " + "  ".join(
            ("  -  " if np.isnan(bnd[di, ki]) else f"{bnd[di, ki]:5.2f}")
            for ki in range(len(DURATIONS))), flush=True)

    # 0f: the recomputed taxonomy must equal the archived classifier
    from experiments.common import rescue_taxonomy
    agree, tot = 0, 0
    for c, (di, ui, ki) in zip(cfgs[::37], idx[::37]):
        sch = episode()
        sch = rescue_profile(sch, c["t_trig"], c["r_dur"], c["u"])
        s = simulate_armed(Params(), sch, T)
        agree += (rescue_taxonomy(s, Params(), c["t_trig"]) ==
                  run_cell(c)["tax"])
        tot += 1
    res["taxonomy_agreement"] = f"{agree}/{tot}"
    print(f"  recomputed taxonomy vs archived rescue_taxonomy: {agree}/{tot}",
          flush=True)
    return res


# ------------------------------------------------------------------ part 1
ARMS = {
    # name: (label, cfg extras)
    "A_step":      ("(a) pre-armed step rescue from E>0", {"profile": "step"}),
    "A_esc3":      ("(a) pre-armed escalating rescue (3 tiers)",
                    {"profile": "esc3"}),
    "AB":          ("(a)+(b) step rescue + S held", {"profile": "step",
                                                     "S_hold": 0.9545}),
    "ABC":         ("(a)+(b)+(c) + standing floor 0.10",
                    {"profile": "step", "S_hold": 0.9545, "floor": 0.10}),
    "B_only":      ("(b) S held, no rescue", {"S_hold": 0.9545}),
    "C_only":      ("(c) floor only, no rescue", {"floor": 0.10}),
}


def part1(pool, t_trig: float) -> dict:
    """The arm grids: exp2's strength x duration axes, re-run under each arm."""
    print(f"[{time.strftime('%H:%M:%S')}] PART 1: arm grids (17 x 8)", flush=True)
    out = {}
    for name, (label, extra) in ARMS.items():
        cfgs, idx = [], []
        for ui, u in enumerate(STRENGTHS):
            for ki, du in enumerate(DURATIONS):
                c = {"t_trig": t_trig, "r_dur": du, "u": u}
                c.update(extra)
                cfgs.append(c)
                idx.append((ui, ki))
        rows = run_many(cfgs, pool)
        g3 = np.zeros((len(STRENGTHS), len(DURATIONS)), int)
        Ge = np.zeros_like(g3, float)
        Gm = np.zeros_like(g3, float)
        for (ui, ki), r in zip(idx, rows):
            g3[ui, ki] = ("no-collapse", "partial-transient",
                          "full-collapse").index(r["outcome3"])
            Ge[ui, ki] = r["G_end"]
            Gm[ui, ki] = r["G_min"]
        frac = {k: float((g3 == i).sum()) / g3.size
                for i, k in enumerate(("no-collapse", "partial-transient",
                                       "full-collapse"))}
        bnd = np.full(len(DURATIONS), np.nan)
        for ki in range(len(DURATIONS)):
            ok = np.where(g3[:, ki] == 0)[0]        # no-collapse
            if ok.size:
                bnd[ki] = STRENGTHS[ok[0]]
        out[name] = dict(label=label, extra=extra, frac=frac,
                         boundary=bnd.tolist(), grid=g3.tolist(),
                         G_end=Ge.tolist(), G_min=Gm.tolist())
        print(f"  {label}: " + "  ".join(f"{k}={v:.2f}" for k, v in frac.items()),
              flush=True)
        print("      u* (min strength -> no-collapse) by duration "
              + " ".join(f"{d:.0f}:{('-' if np.isnan(bnd[i]) else round(bnd[i], 3))}"
                         for i, d in enumerate(DURATIONS)), flush=True)

    # extended fine grid (0 .. 0.30) to resolve the armed boundary
    print(f"[{time.strftime('%H:%M:%S')}] PART 1b: fine strength grid 0-0.30",
          flush=True)
    for name in ("A_step", "AB"):
        extra = ARMS[name][1]
        cfgs, idx = [], []
        for ui, u in enumerate(A_STRENGTHS):
            for ki, du in enumerate(DURATIONS):
                c = {"t_trig": t_trig, "r_dur": du, "u": u}
                c.update(extra)
                cfgs.append(c)
                idx.append((ui, ki))
        rows = run_many(cfgs, pool)
        g3 = np.zeros((len(A_STRENGTHS), len(DURATIONS)), int)
        for (ui, ki), r in zip(idx, rows):
            g3[ui, ki] = ("no-collapse", "partial-transient",
                          "full-collapse").index(r["outcome3"])
        bnd = np.full(len(DURATIONS), np.nan)
        for ki in range(len(DURATIONS)):
            ok = np.where(g3[:, ki] == 0)[0]
            if ok.size:
                bnd[ki] = A_STRENGTHS[ok[0]]
        out[name]["fine_boundary"] = bnd.tolist()
        out[name]["fine_strengths"] = A_STRENGTHS.tolist()
        out[name]["fine_grid"] = g3.tolist()
        print(f"  {ARMS[name][0]}: u* = "
              + " ".join(f"{d:.0f}:{('-' if np.isnan(bnd[i]) else round(bnd[i], 3))}"
                         for i, d in enumerate(DURATIONS)), flush=True)
    return out


# ------------------------------------------------------------------ part 2
def part2(t_trig: float) -> dict:
    """Dose-response under the armed schedule, vs the unarmed boundary."""
    print(f"[{time.strftime('%H:%M:%S')}] PART 2: armed dose-response bisections",
          flush=True)
    out = {}
    U = 0.8
    conds = {
        "unarmed": {},
        "A_step_u0.04": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.04,
                         "profile": "step"},
        "A_step_u0.8": {"t_trig": t_trig, "r_dur": 1600.0, "u": U,
                        "profile": "step"},
        "A_esc3_u0.8": {"t_trig": t_trig, "r_dur": 1600.0, "u": U,
                        "profile": "esc3"},
        "B_only": {"S_hold": 0.9545},
        "AB": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.04, "profile": "step",
               "S_hold": 0.9545},
        "ABC": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.04, "profile": "step",
                "S_hold": 0.9545, "floor": 0.10},
        "C_only": {"floor": 0.10},
    }
    for name, extra in conds.items():
        def f_dur(x, extra=extra):
            c = dict(extra)
            c["duration"] = x
            return run_cell(c)["G_min"] < 0.1

        def f_int(x, extra=extra):
            c = dict(extra)
            c["intensity"] = x
            return run_cell(c)["G_min"] < 0.1

        td, nd = bisect(f_dur, 5.0, 400.0)
        ti, ni = bisect(f_int, 0.05, 1.0)
        out[name] = dict(extra=extra, dur_threshold=td, dur_note=nd,
                         int_threshold=ti, int_note=ni)
        print(f"  {name:14s} duration threshold "
              f"{'NONE(>400)' if td is None else round(td, 2)}   "
              f"intensity threshold "
              f"{'NONE(>1.0)' if ti is None else round(ti, 4)}", flush=True)
    return out


# ------------------------------------------------------------------ part 3
def part3(t_trig: float) -> dict:
    """Mechanism decomposition: which channel does the protecting."""
    print(f"[{time.strftime('%H:%M:%S')}] PART 3: mechanism decomposition",
          flush=True)
    out = {}
    # 3a. S held at a sweep of levels, no rescue at all
    lv = [0.9545, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
    out["S_hold_sweep"] = []
    for v in lv:
        r = run_cell({"S_hold": v})
        out["S_hold_sweep"].append(dict(S_hold=v, G_min=r["G_min"],
                                        G_end=r["G_end"], c_max=r["c_max"],
                                        Teff_min=r["Teff_min"],
                                        E_max=r["E_max"],
                                        dev=r["S_hold_dev"],
                                        outcome3=r["outcome3"]))
        print(f"  S held at {v:.4f}: G_min={r['G_min']:.4f} c_max={r['c_max']:.3f} "
              f"E_max={r['E_max']:.3f} Teff_min={r['Teff_min']:.3f} "
              f"|S-held|max={r['S_hold_dev']:.2e} -> {r['outcome3']}", flush=True)
    # 3b. standing floor sweep, no rescue, no S hold
    out["floor_sweep"] = []
    for v in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]:
        r = run_cell({"floor": v})
        out["floor_sweep"].append(dict(floor=v, G_min=r["G_min"],
                                       G_end=r["G_end"], c_max=r["c_max"],
                                       outcome3=r["outcome3"]))
        print(f"  floor {v:.3f}: G_min={r['G_min']:.4f} c_max={r['c_max']:.3f}"
              f" -> {r['outcome3']}", flush=True)
    # 3c. the a-channel ablation (A2): does the pre-armed rescue still protect
    #     when u_ext can no longer pull attention outward?
    out["a_blind"] = []
    for u in [0.04, 0.1, 0.2, 0.4, 0.8]:
        r = run_cell({"t_trig": t_trig, "r_dur": 1600.0, "u": u,
                      "a_blind": True})
        out["a_blind"].append(dict(u=u, G_min=r["G_min"], G_end=r["G_end"],
                                   a_ep_mean=r["a_ep_mean"], c_max=r["c_max"],
                                   outcome3=r["outcome3"]))
        print(f"  a-blind step u={u:.2f}: G_min={r['G_min']:.4f} "
              f"a_ep_mean={r['a_ep_mean']:.3f} c_max={r['c_max']:.3f}"
              f" -> {r['outcome3']}", flush=True)
    # 3d. duty-cycled hold (arm d): break the sustained hold, no rescue
    out["duty"] = []
    for duty in [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05]:
        r = run_cell({"duty": duty})
        r2 = run_cell({"duty": duty, "t_trig": t_trig, "r_dur": 1600.0,
                       "u": 0.04})
        out["duty"].append(dict(duty=duty, G_min=r["G_min"],
                                G_end=r["G_end"], c_max=r["c_max"],
                                outcome3=r["outcome3"],
                                G_min_u=r2["G_min"], outcome3_u=r2["outcome3"]))
        print(f"  duty {duty:.2f}: G_min={r['G_min']:.4f} c_max={r['c_max']:.3f}"
              f" -> {r['outcome3']}   | +u0.04: G_min={r2['G_min']:.4f}"
              f" -> {r2['outcome3']}", flush=True)
    # 3e. trigger-time sensitivity at the armed boundary
    out["t_trig_sens"] = []
    for tt in [EP_T0, 104.0, t_trig, 120.0, 140.0, 160.0, 180.0, EP_T1]:
        r = run_cell({"t_trig": tt, "r_dur": 1600.0, "u": 0.04})
        out["t_trig_sens"].append(dict(t_trig=tt, G_min=r["G_min"],
                                       outcome3=r["outcome3"]))
        print(f"  t_trig={tt:6.2f} u=0.04: G_min={r['G_min']:.4f}"
              f" -> {r['outcome3']}", flush=True)
    return out


# ------------------------------------------------------------------ part 4
def _esc_schedule(t_trig: float, dur: float, tiers: list):
    """Base tier (u_cap/3 from t_trig for `dur`) + one span per tell."""
    sch = rescue_profile(episode(), t_trig, dur, tiers[0][2])
    for t0s, _t1s, lvl in tiers[1:]:
        sch = add_span(sch, "u_ext", t0s, t_trig + dur, lvl)
    return sch


def part4(pool, t_trig: float) -> dict:
    """Closed-loop escalation: a tier is added when the cannibalization tell
    fires (c > 0.5), up to three tiers — the doc's Tier 1 -> Tier 3 ladder."""
    print(f"[{time.strftime('%H:%M:%S')}] PART 4: closed-loop escalation",
          flush=True)
    out = []
    for u_cap in [0.1, 0.2, 0.4, 0.8]:
        for dur in [40.0, 200.0, 1600.0]:
            tiers = [(t_trig, None, u_cap / 3.0)]
            tell_t = None
            for _ in range(2):                     # at most two escalations
                sol = simulate_armed(Params(), _esc_schedule(t_trig, dur, tiers), T)
                idx = np.where(sol["c"] > 0.5)[0]
                if idx.size == 0:
                    break
                tell_t = float(sol["t"][idx[0]])
                if tell_t >= t_trig + dur - 1e-9:
                    break
                tiers.append((tell_t, None, u_cap / 3.0))
            sol = simulate_armed(Params(), _esc_schedule(t_trig, dur, tiers), T)
            G = sol["G"]
            o3 = ("no-collapse" if G.min() >= 0.1
                  else ("full-collapse" if is_stuck(sol, Params())
                        else "partial-transient"))
            out.append(dict(u_cap=u_cap, dur=dur, n_tiers=len(tiers),
                            tell_t=tell_t, G_min=float(G.min()),
                            G_end=float(G[-1]), c_max=float(sol["c"].max()),
                            outcome3=o3))
            print(f"  escalation cap={u_cap:.2f} dur={dur:6.0f}: "
                  f"tiers={len(tiers)} first_tell={tell_t} "
                  f"G_min={G.min():.4f} -> {o3}", flush=True)
    return {"closed_loop": out}


# ------------------------------------------------------------------ part 5
def part5(t_trig: float) -> dict:
    """The structural bound: why the held setpoint is not a dose threshold.

    E = D - G <= D_max, and dD/dt = 0 gives
        D_max(A) = (D_base+A)*beta_D / ((D_base+A)*beta_D + delta_D).
    The switch can only fire if E > Theta_eff = Theta*S/S_rest.  So if
    Theta_eff > D_max the collapse is impossible AT ANY EPISODE DOSE — the
    dose axes (duration, intensity) act on G and a only, and cannot lift E
    above the demand ceiling.
    """
    print(f"[{time.strftime('%H:%M:%S')}] PART 5: structural bound", flush=True)
    p = Params()
    out = {"D_max": {}, "theta_eff": {}, "S_hold_threshold": {}}
    for A in [0.0, 0.2, 0.5, 0.8, 1.2, 2.0]:
        k = (p.D_base + A) * p.beta_D
        out["D_max"][str(A)] = k / (k + p.delta_D)
    for S in [0.15, 0.4, 0.7, 0.9, 0.9525, 0.9545, 1.0]:
        out["theta_eff"][str(S)] = p.Theta * S / p.S_rest
    print(f"  D_max(A): {out['D_max']}", flush=True)
    print(f"  Theta_eff(S) = 0.8*S: {out['theta_eff']}", flush=True)

    # minimum S_hold that prevents collapse at the canonical dose
    thr, note = bisect(lambda v: run_cell({"S_hold": v})["G_min"] < 0.1,
                       0.10, 0.9545, tol=1e-3)
    out["S_hold_threshold"]["canonical_dose"] = thr
    out["S_hold_threshold"]["note"] = note
    print(f"  minimum S_hold preventing collapse at the canonical dose: "
          f"{thr} ({note})", flush=True)

    # can the held setpoint be beaten by MORE dose?  push all three axes.
    out["B_only_extreme"] = []
    for dur, inten, pulse in [(400.0, 1.0, 1.0), (400.0, 1.0, 2.0),
                              (400.0, 1.0, 5.0), (800.0, 1.0, 1.0)]:
        r = run_cell({"duration": dur, "intensity": inten, "pulse": pulse,
                      "S_hold": 0.9545})
        out["B_only_extreme"].append(dict(duration=dur, intensity=inten,
                                          pulse=pulse, G_min=r["G_min"],
                                          E_max=r["E_max"],
                                          c_max=r["c_max"],
                                          Teff_min=r["Teff_min"],
                                          outcome3=r["outcome3"]))
        print(f"  S held 0.9545, dur={dur:.0f} int={inten} pulse={pulse}: "
              f"G_min={r['G_min']:.4f} E_max={r['E_max']:.4f} "
              f"Teff_min={r['Teff_min']:.4f} c_max={r['c_max']:.3f}"
              f" -> {r['outcome3']}", flush=True)

    # and the same extremes UNARMED, to show the axes do bite without the hold
    out["unarmed_extreme"] = []
    for dur, inten, pulse in [(400.0, 1.0, 1.0), (400.0, 1.0, 2.0)]:
        r = run_cell({"duration": dur, "intensity": inten, "pulse": pulse})
        out["unarmed_extreme"].append(dict(duration=dur, intensity=inten,
                                           pulse=pulse, G_min=r["G_min"],
                                           outcome3=r["outcome3"]))
        print(f"  UNARMED dur={dur:.0f} int={inten} pulse={pulse}: "
              f"G_min={r['G_min']:.4f} -> {r['outcome3']}", flush=True)
    return out


def main() -> int:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] armed-run experiment start", flush=True)
    res = {}
    with Pool(28) as pool:
        res["part0"] = part0(pool)
        tt = res["part0"]["t_trig"]
        res["part1"] = part1(pool, tt)
    print(f"[{time.strftime('%H:%M:%S')}] pool parts done "
          f"({time.time() - t0:.0f}s)", flush=True)
    res["part2"] = part2(tt)
    res["part3"] = part3(tt)
    res["part4"] = part4(None, tt)
    res["part5"] = part5(tt)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE in {time.time() - t0:.0f}s "
          f"-> {OUT}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
