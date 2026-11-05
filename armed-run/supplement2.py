"""Supplement 2: the P4 claim under the armed configuration.

paper2 §4.3 / P4: 'the recovered generator's head start buys no protection:
collapse onset timing is identical to the first episode', with a second-episode
duration threshold of 66.0 t.u. (with affect load) vs the first episode's 66.3.
That was measured for the UNARMED schedule.

Here the recovered state at t=800 is held FIXED (identical first episode and
identical post-episode rescue, exactly exp2's canonical geometry), and only the
schedule the SECOND episode runs under is changed:
  unarmed  : nothing (reproduces P4)
  armed    : S held at the healthy value, standing floor 0.10, pre-armed
             external demand from the moment E first goes positive again.

Everything before t=800 is bit-identical between the arms, so the difference at
t>=800 is attributable to the schedule alone.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dpdr-public/dpdr"))

from armed_model import T_END, add_span, episode, rescue_profile, simulate_armed, snap
from dpdr.metrics import first_below, is_stuck
from dpdr.model import Params, Schedule
from run_armed import OUT, run_cell

P = Params()
EP2_T0, EP2_PULSE_LEN = 800.0, 60.0


def build(arm: str, dur2: float, t_trig2: float | None):
    """First episode + canonical post-episode rescue (unarmed, identical in
    both arms) + the second episode; the armed schedule applies from t=800."""
    sch = episode()                                  # [100,200) a_hold .9, A .5
    sch = rescue_profile(sch, 400.0, 60.0, 0.8)      # exp2 canonical rescue
    sch = add_span(sch, "a_hold", EP2_T0, EP2_T0 + dur2, 0.9)
    sch = add_span(sch, "A", EP2_T0, EP2_T0 + EP2_PULSE_LEN, 0.5)
    if arm == "armed":
        sch = add_span(sch, "S_hold", EP2_T0, T_END, 0.9545)
        sch = add_span(sch, "u_ext", EP2_T0, T_END, 0.10)      # floor
        if t_trig2 is not None:
            sch = rescue_profile(sch, t_trig2, T_END - t_trig2, 0.04)
    return sch


def run(arm: str, dur2: float, t_trig2: float | None = None) -> dict:
    sol = simulate_armed(P, build(arm, dur2, t_trig2), T_END)
    t, G = sol["t"], sol["G"]
    w = t >= EP2_T0                      # every metric below is the SECOND episode
    t_coll2 = first_below(G[w], t[w])
    G_pre = float(G[np.searchsorted(t, EP2_T0)])
    stuck = bool(is_stuck(sol, P))
    return dict(arm=arm, dur2=dur2, G_pre=G_pre,
                G_min=float(G[w].min()), G_end=float(G[-1]),
                c_max=float(sol["c"][w].max()), c_end=float(sol["c"][-1]),
                S_min=float(sol["S"][w].min()), S_end=float(sol["S"][-1]),
                a_post=float(sol["a"][w].mean()),
                Teff_min=float(sol["Theta_eff"][w].min()),
                t_coll2=t_coll2,
                onset=(None if t_coll2 is None else t_coll2 - EP2_T0),
                stuck=stuck, E_end=float(sol["E"][-1]))


def bisect_dur(arm: str, t_trig2: float | None, lo=20.0, hi=300.0,
               tol=0.05, it=22):
    def collapsed(d):
        return run(arm, d, t_trig2)["stuck"]
    if collapsed(lo) == collapsed(hi):
        return None, "bracket does not straddle"
    for _ in range(it):
        if hi - lo < tol:
            break
        m = 0.5 * (lo + hi)
        if collapsed(m) == collapsed(lo):
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi), f"[{lo:.2f},{hi:.2f}]"


def main() -> int:
    out = {}
    # the trigger for the armed arm: E first > 0 in the unarmed second episode
    s = simulate_armed(P, build("unarmed", 100.0, None), T_END)
    w = s["t"] >= EP2_T0
    i = np.where(s["E"][w] > 0)[0]
    t_trig2 = float(s["t"][w][i[0]]) if i.size else None
    out["t_trig2"] = t_trig2
    print(f"second-episode pre-arm trigger: E first > 0 at t={t_trig2}", flush=True)

    for arm in ("unarmed", "armed"):
        r = run(arm, 100.0, t_trig2)
        print(f"  {arm:8s} dur2=100: G_pre={r['G_pre']:.4f} G_min={r['G_min']:.4f} "
              f"G_end={r['G_end']:.4f} c_max={r['c_max']:.3f} "
              f"S_min={r['S_min']:.3f} stuck={r['stuck']}", flush=True)

    out["threshold"] = {}
    for arm in ("unarmed", "armed"):
        thr, note = bisect_dur(arm, t_trig2)
        out["threshold"][arm] = (thr, note)
        print(f"  {arm:8s} second-episode duration threshold: "
              f"{'NONE(>300)' if thr is None else round(thr, 2)} ({note})",
              flush=True)

    out["sweep"] = []
    for arm in ("unarmed", "armed"):
        for d in (40.0, 60.0, 66.0, 70.0, 100.0, 150.0, 200.0, 300.0):
            r = run(arm, d, t_trig2)
            out["sweep"].append(r)
            print(f"  {arm:8s} dur2={d:6.0f}: G_min={r['G_min']:.4f} "
                  f"onset={r['onset']} c_max={r['c_max']:.3f} "
                  f"stuck={r['stuck']} -> "
                  f"{'no-collapse' if r['G_min'] >= 0.1 else ('full-collapse' if r['stuck'] else 'partial-transient')}",
                  flush=True)

    with open(os.path.join(OUT, "supplement2.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
