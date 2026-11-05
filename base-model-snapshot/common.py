"""Shared schedule/taxonomy helpers for the experiment drivers.

The three drivers sweep SCHEDULE axes (episode duration, rescue timing, ...)
alongside Params axes, but dpdr.sweep sweeps Params overrides against ONE
fixed schedule.  Rather than fork the sweep machinery, each driver builds the
per-cell schedule directly and writes its own fixed-name .npz into cache/.
"""
from __future__ import annotations

import numpy as np

from dpdr.events import Schedule, affect_pulse, external_demand, inward_episode
from dpdr.integrate import simulate
from dpdr.metrics import first_above_after, first_below
from dpdr.model import Params

# Canonical scenario geometry (README / metrics): episode [100, 200), pulse
# [100, 160), rescue AFTER episode end to avoid the relapse confound.
EP_T0, EP_T1 = 100.0, 200.0
PULSE_END = 160.0


def episode_schedule(t0: float = EP_T0, t1: float = EP_T1,
                     intensity: float = 0.9, pulse: float = 0.5,
                     pulse_end: float | None = None) -> Schedule:
    """Inward episode + affect pulse (the trigger), parameterized.

    Times are snapped to the 0.05 integration grid; the pulse is clamped to
    end within the episode (its default end 160 < t1=200 keeps the canonical
    scenario unchanged)."""
    t0, t1 = snap(t0), snap(t1)
    pe = snap(min(PULSE_END if pulse_end is None else pulse_end, t1))
    s = inward_episode(t0, t1, intensity)
    if pe > t0 and pulse != 0.0:
        s = affect_pulse(s, t0, pe - t0, pulse)
    return s


def rescue_schedule(ep: Schedule, t_rescue: float, dur: float,
                    u: float) -> Schedule:
    return external_demand(ep, snap(t_rescue), snap(dur), u)


def snap(x: float, dt: float = 0.05) -> float:
    """Snap a schedule time onto the integration grid (integrate() requires
    breakpoints on the dt grid)."""
    return round(round(x / dt) * dt, 10)


# --------------------------------------------------------------- taxonomy
def rescue_taxonomy(sol, p: Params, t_rescue: float):
    """Plan §4 Phase-3 three-class taxonomy, judged at the horizon.

    full      final rescue_success-style re-couple: G(T) > 0.5, E(T) < 0.8*Theta
              and G(T) stays (no re-collapse to < 0.1 after its post-rescue peak)
    transient G rose past 0.5 at some point after the first collapse, then
              re-crossed below 0.1 (lift + relapse)
    failure   G never exceeded 0.5 after collapsing
    """
    t, G = sol["t"], sol["G"]
    t_coll = first_below(G, t)
    if t_coll is None:
        return "no-collapse"
    w = t >= t_coll
    Gw, tw = G[w], t[w]
    ipk = int(np.argmax(Gw))
    if Gw[ipk] > 0.5:
        return "transient" if Gw[ipk:].min() < 0.1 else "full"
    return "failure"


TAXONOMY_CODES = {"failure": 0, "transient": 1, "full": 2, "no-collapse": 3}


def one_run(p: Params, sch: Schedule, T: float) -> dict:
    """simulate + summary for drivers that need per-cell detail."""
    from dpdr.metrics import summarize
    sol = simulate(p, sch, T)
    d = summarize(sol, p)
    d["sol"] = sol
    return d
