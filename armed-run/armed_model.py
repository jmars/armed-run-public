"""Armed-run scratch layer: the archived dpdr RHS, plus three declared adaptations.

NOTHING here reimplements the model.  The five-state RHS is `dpdr.model.deriv`
imported verbatim from the read-only archive (~/dpdr-public/dpdr) and called
inside `deriv_armed`; the integrator is a verbatim copy of
`dpdr.integrate.simulate` with the single token `deriv` -> `deriv_armed`.

ADAPTATIONS (all declared in the deliverable; A1-A3 are the only deviations):
  A1  'S_hold' schedule channel (arm b, held setpoint).  When active with
      value v, dS is replaced by  k_hold*(v - S)/tau_S.  This is a fast
      restoring spring toward v, NOT an instantaneous clamp: it is smooth
      (C0) so the RK45 driver needs no special handling, and its time
      constant is tau_S/k_hold = 1.0 t.u. against a 100 t.u. episode.  The
      residual |S(t) - v| is reported so "held" is checkable, not asserted.
  A2  'a_blind' schedule channel (mechanism decomposition only).  When
      active, the -k_ext*u_ext*a term is removed from da.  This is a
      DIAGNOSTIC ablation that isolates the G-side effect of u_ext from its
      attention-side effect; it is never used for a headline number.
  A3  Nothing else.  a_hold / A / u_ext keep the archived semantics
      (Schedule.value SUMS overlapping spans, so duty-cycled holds and
      multi-tier escalation ramps are built as multiple spans).

Everything downstream (Params, Schedule, cannibalization_vec, metrics) is
imported from the archive unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

import dpdr.model as M
from dpdr.model import Params, Schedule, cannibalization_vec

K_HOLD = 100.0          # A1 restoring rate; time constant tau_S/k_hold = 1 t.u.
DT = 0.05


# ----------------------------------------------------------------- RHS (A1, A2)
def deriv_armed(t, y, p: Params, sch: Schedule):
    """Archived dpdr.model.deriv, with adaptations A1 (S_hold) and A2 (a_blind)."""
    d = M.deriv(t, y, p, sch)              # <- the archived RHS, verbatim
    if sch.value("S_hold", t) > 0.0:       # A1
        d[3] = (K_HOLD * (sch.value("S_hold", t) - y[3])) / p.tau_S
    if sch.value("a_blind", t) > 0.0:      # A2: undo the -k_ext*u_ext*a term
        a = y[0]
        d[0] += (p.k_ext * sch.value("u_ext", t) * a) / p.tau_a
    return d


# --------------------------------------------------- integrator (verbatim copy)
def collapse_event(t, y, p, sch):
    return y[1] - p.G_floor


collapse_event.terminal = True
collapse_event.direction = -1.0


def simulate_armed(p: Params, sch: Schedule, T: float, dt: float = DT) -> dict:
    """Verbatim copy of dpdr.integrate.simulate, with deriv -> deriv_armed."""
    grid = np.arange(0.0, T + dt / 2, dt)
    bps = [b for b in sch.breakpoints() if 0.0 < b < T]
    for b in bps:
        if abs(b / dt - round(b / dt)) > 1e-9:
            raise ValueError(f"breakpoint {b} not on the dt={dt} grid")
    edges = [0.0] + bps + [T]
    y = np.array([p.a0, p.G0, p.D0, p.S0, p.g_init], float)
    ts, ys, methods = [], [], []
    nfev_total = 0
    collapsed_at = None
    for t0, t1 in zip(edges[:-1], edges[1:]):
        tev = grid[(grid >= t0 - 1e-9) & (grid < t1 - 1e-9)] if t1 < T \
            else grid[(grid >= t0 - 1e-9) & (grid <= T + 1e-9)]
        kw = dict(args=(p, sch), rtol=1e-6, atol=1e-8, max_step=0.5,
                  t_eval=tev, events=collapse_event)
        sol = solve_ivp(deriv_armed, (t0, t1), y, method="RK45", **kw)
        method = "RK45"
        if sol.status == -1 or sol.nfev > max(5000.0, 60.0 * (t1 - t0)):
            sol = solve_ivp(deriv_armed, (t0, t1), y, method="LSODA", **kw)
            method = "LSODA"
        methods.append(f"[{t0:g},{t1:g}]:{method}({sol.nfev})")
        nfev_total += sol.nfev
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
        if sol.status == 1:
            collapsed_at = float(sol.t_events[0][0])
            rest = grid[grid > sol.t[-1] + 1e-9]
            if rest.size:
                ts.append(rest)
                ys.append(np.repeat(y[:, None], rest.size, axis=1))
            break
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    a, G, D, S, g = Y
    E = D - G
    c, Theta_eff = cannibalization_vec(E, S, p)
    return dict(t=t, a=a, G=G, D=D, S=S, g=g, E=E, c=c,
                Theta_eff=Theta_eff, u_loop=G / (D + 1e-9),
                collapsed_at=collapsed_at, methods=methods, nfev=nfev_total)


# -------------------------------------------------------------- geometry helpers
EP_T0, EP_T1, PULSE_END = 100.0, 200.0, 160.0
T_END = 1800.0


def snap(x: float, dt: float = DT) -> float:
    return round(round(x / dt) * dt, 10)


def episode(intensity: float = 0.9, duration: float = 100.0,
            pulse: float = 0.5, t0: float = EP_T0) -> Schedule:
    """The archived canonical geometry (experiments.common.episode_schedule)."""
    t0s, t1s = snap(t0), snap(t0 + duration)
    ch = {"a_hold": [(t0s, t1s, intensity)]}
    pe = snap(min(PULSE_END, t1s))
    if pe > t0s and pulse != 0.0:
        ch["A"] = [(t0s, pe, pulse)]
    return Schedule(ch)


def add_span(sch: Schedule, ch: str, t0: float, t1: float, v: float) -> Schedule:
    out = dict(sch.channels)
    out[ch] = out.get(ch, []) + [(snap(t0), snap(t1), v)]
    return Schedule(out)


def rescue_profile(sch: Schedule, t0: float, dur: float, u: float,
                   profile: str = "step") -> Schedule:
    """Pre-armed rescue profiles on u_ext, all starting at t0."""
    if profile == "step":
        return add_span(sch, "u_ext", t0, t0 + dur, u)
    if profile == "esc3":
        d3 = dur / 3.0
        s = add_span(sch, "u_ext", t0, t0 + d3, u / 3.0)
        s = add_span(s, "u_ext", t0 + d3, t0 + 2 * d3, 2 * u / 3.0)
        return add_span(s, "u_ext", t0 + 2 * d3, t0 + dur, u)
    raise ValueError(profile)


def duty_cycled_hold(sch: Schedule, intensity: float, duty: float,
                     period: float = 10.0) -> Schedule:
    """Replace the sustained a_hold with an on/off hold at `duty` (arm d)."""
    out = {k: v for k, v in sch.channels.items() if k != "a_hold"}
    on = duty * period
    t = EP_T0
    while t < EP_T1 - 1e-9:
        t1 = min(t + on, EP_T1)
        if t1 > t:
            out.setdefault("a_hold", []).append((snap(t), snap(t1), intensity))
        t += period
    return Schedule(out)
