"""Integrate: segmented solve_ivp driver (plan §3).

RK45, rtol=1e-6, atol=1e-8, max_step=0.5, t_eval spacing 0.05; the horizon is
split at every schedule breakpoint and the integrator restarted there.  A
segment is retried with LSODA when RK45 struggles (status -1, or nfev above
max(5000, 60*segment length) — a bare nfev>5000 threshold fires on every long
segment by construction under max_step=0.5).  Terminal event at G < G_floor
(deep collapse); the trajectory is held flat afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .model import Params, Schedule, cannibalization_vec, deriv

DT = 0.05


def collapse_event(t, y, p, sch):
    return y[1] - p.G_floor


collapse_event.terminal = True
collapse_event.direction = -1.0


def simulate(p: Params, sch: Schedule, T: float, dt: float = DT) -> dict:
    """Integrate the five-state system over [0, T]; return a Solution dict
    {t, a, G, D, S, g, c, E, Theta_eff, u_loop, collapsed_at, methods, nfev}."""
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
        sol = solve_ivp(deriv, (t0, t1), y, method="RK45", **kw)
        method = "RK45"
        if sol.status == -1 or sol.nfev > max(5000.0, 60.0 * (t1 - t0)):
            sol = solve_ivp(deriv, (t0, t1), y, method="LSODA", **kw)
            method = "LSODA"
        methods.append(f"[{t0:g},{t1:g}]:{method}({sol.nfev})")
        nfev_total += sol.nfev
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
        if sol.status == 1:            # terminal collapse event fired
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
