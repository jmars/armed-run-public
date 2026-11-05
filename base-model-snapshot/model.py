"""Model: parameters, exogenous schedules, and the five-state RHS (plan §1).

States: a (attention), G (generator), D (reducer demand), S (setpoint),
g (adaptive gain).  Algebraic: E = D - G; c = sigma_c*max(0,tanh((E-Theta_eff)/w)),
Theta_eff = Theta*S/S_rest.

Deviations from the plan's literal equations (each forced by a gate conflict,
tuned in Step A per plan risk #6; full discussion in sim0.py / README):
  D1. Generator turnover -gam_G*G (healthy attractor strictly inside (0,1)).
  D2. Control correction gated (G+eps0)(1-G) (the stuck attractor must exist).
  D3. Setpoint tracks externality (1-a), not a (matches the plan's own prose
      "chronic inward context drags S down").
  D4. |dE/dt|_filt approximated by instantaneous |dE/dt| (tau_g is the filter).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Params:
    # time constants (four-timescale separation, plan §3)
    tau_a: float = 1.0     # attention (fastest)
    tau_G: float = 20.0    # generator
    tau_D: float = 50.0    # reducer demand
    tau_S: float = 100.0   # allostatic setpoint
    tau_g: float = 200.0   # adaptive gain (slowest)
    # generator
    beta_G: float = 3.0    # growth under external attention  (tuned; §3: 1.0)
    alpha_G: float = 1.2   # allostatic weakening under inward attention (0.6)
    eta: float = 1.2       # cannibalization cost
    gam_G: float = 0.30    # turnover / maintenance cost (deviation D1)
    eps0: float = 0.25     # control gating at low G (deviation D2)
    # reducer demand
    beta_D: float = 0.8
    delta_D: float = 0.2
    D_base: float = 0.3
    # allostatic setpoint
    k_s: float = 0.5       # context tracking (tuned; §3: 0.3)
    lam_S: float = 0.05
    S_rest: float = 0.5
    # cannibalization switch
    Theta: float = 0.4
    w: float = 0.05
    sigma_c: float = 3.0
    # attention
    k_in: float = 1.5
    chi: float = 1.0
    k_ext: float = 4.0     # external pull (tuned; §3: 2.0)
    rho_a: float = 0.2     # attention decay (tuned; §3: 1.0)
    # control loop / adaptive gain
    g0: float = 0.5
    Es: float = 0.5
    pi: float = 1.5
    mu: float = 0.3
    dEdt_ref: float = 0.002  # deadband on |dE/dt| upregulation
    # numerics
    G_floor: float = 1e-4  # terminal-event deep-collapse level
    # initial state
    a0: float = 0.05
    G0: float = 0.7
    D0: float = 0.5
    S0: float = 0.8
    g_init: float = 0.5


@dataclass
class Schedule:
    """Piecewise-constant exogenous channels: name -> [(t0, t1, value), ...].

    Channels used by the model RHS: 'a_hold' (inward-attention hold),
    'A' (affective input), 'u_ext' (external demand / rescue).
    """
    channels: dict = field(default_factory=dict)

    def value(self, ch: str, t: float) -> float:
        return sum(v for (t0, t1, v) in self.channels.get(ch, [])
                   if t0 <= t < t1)

    def breakpoints(self) -> list:
        pts = {0.0}
        for spans in self.channels.values():
            for (t0, t1, _v) in spans:
                pts.add(float(t0))
                pts.add(float(t1))
        return sorted(pts)


def cannibalization(E: float, S: float, p: Params):
    """Return (c, Theta_eff); c clipped to [0, 1] (max rate sigma_c)."""
    Theta_eff = p.Theta * S / p.S_rest
    c = p.sigma_c * max(0.0, math.tanh((E - Theta_eff) / p.w))
    return min(c, 1.0), Theta_eff


def cannibalization_vec(E, S, p: Params):
    """Vectorized version for trajectory post-processing."""
    import numpy as np
    Theta_eff = p.Theta * S / p.S_rest
    c = p.sigma_c * np.maximum(0.0, np.tanh((E - Theta_eff) / p.w))
    return np.minimum(c, 1.0), Theta_eff


def deriv(t, y, p: Params, sch: Schedule):
    """RHS of the five-state system; y = [a, G, D, S, g]."""
    a, G, D, S, g = y
    E = D - G
    c, _te = cannibalization(E, S, p)
    a_hold = sch.value("a_hold", t)
    A = sch.value("A", t)
    u_ext = sch.value("u_ext", t)
    dG = (p.beta_G * (1 - a) * G * (1 - G)
          - p.alpha_G * a * G
          - p.eta * c * G
          - p.gam_G * G
          + g * math.tanh(E / p.Es) * (G + p.eps0) * (1 - G)) / p.tau_G
    dD = ((p.D_base + A) * p.beta_D * (1 - D) - p.delta_D * D) / p.tau_D
    dS = (p.k_s * ((1 - a) - S) - p.lam_S * (S - p.S_rest)) / p.tau_S
    da = (p.k_in * (a_hold + p.chi * c) * (1 - a)
          - p.k_ext * u_ext * a - p.rho_a * a) / p.tau_a
    dE = dD - dG                       # deviation D4: instantaneous |dE/dt|
    dg = (p.pi * max(0.0, abs(dE) - p.dEdt_ref) - p.mu * (g - p.g0)) / p.tau_g
    return [da, dG, dD, dS, dg]
