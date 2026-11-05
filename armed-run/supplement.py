"""Armed-run supplement: like-for-like controls, trajectory dumps, figures.

  S1  exp2 Phase 3b reproduced — rescue OVERLAPPING the episode (the design
      exp2's main grid excludes as the confound).  Establishes what the
      archive already knew about in-episode rescue.
  S2  like-for-like: same (u, duration) applied pre-armed vs post-episode.
  S3  the S_hold settling diagnostic (is "held" actually held?).
  S4  G_end for the partial-transient cells.
  S5  figures -> ~/thing/report/corpus/figures-armed-run/
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dpdr-public/dpdr"))

from armed_model import (EP_T0, EP_T1, T_END, add_span, episode,
                         rescue_profile, simulate_armed, snap)
from dpdr.metrics import is_stuck
from dpdr.model import Params
from run_armed import (DELAYS, DURATIONS, STRENGTHS, TAX, OUT, FIG, run_cell,
                       run_many)

CONF_ONSET = np.array([120.0, 140.0, 160.0, 180.0])
CONF_STRENGTHS = np.round(np.linspace(0.4, 1.0, 7), 4)
CONF_DURATIONS = np.array([20.0, 60.0, 100.0, 200.0])


def s1_confound(pool) -> dict:
    """exp2 Phase 3b: rescue starting DURING the episode (t = 120..180)."""
    print("S1: exp2 Phase 3b confound grid (rescue DURING the episode)", flush=True)
    cfgs, idx = [], []
    for oi, t0 in enumerate(CONF_ONSET):
        for ui, u in enumerate(CONF_STRENGTHS):
            for ki, du in enumerate(CONF_DURATIONS):
                cfgs.append({"t_trig": snap(t0), "r_dur": du, "u": u})
                idx.append((oi, ui, ki))
    rows = run_many(cfgs, pool)
    tax = np.zeros((len(CONF_ONSET), len(CONF_STRENGTHS), len(CONF_DURATIONS)), int)
    for (oi, ui, ki), r in zip(idx, rows):
        tax[oi, ui, ki] = TAX.index(r["tax"])
    frac = {lab: float((tax == k).sum()) / tax.size for k, lab in enumerate(TAX)}
    print(f"  class fractions: " + "  ".join(f"{k}={v:.3f}" for k, v in frac.items()),
          flush=True)
    for oi, t0 in enumerate(CONF_ONSET):
        print(f"  onset {t0:5.0f}: " + " ".join(
            f"{CONF_DURATIONS[ki]:.0f}:" + ("F" if tax[oi, -1, ki] == 0 else
                                           "T" if tax[oi, -1, ki] == 1 else
                                           "R" if tax[oi, -1, ki] == 2 else "n")
            for ki in range(len(CONF_DURATIONS))) + "   (u=1.0 row)", flush=True)
    return dict(tax=tax.tolist(), frac=frac, onsets=CONF_ONSET.tolist(),
                strengths=CONF_STRENGTHS.tolist(),
                durations=CONF_DURATIONS.tolist())


def s2_like_for_like(pool, t_trig: float) -> dict:
    """Same (u, engagement duration), armed (starts at E>0, t=108) vs unarmed
    (starts after episode end, delay 30 -> t=230)."""
    print("S2: like-for-like, pre-armed vs post-episode", flush=True)
    out = []
    cfgs = []
    for u in (0.2, 0.4, 0.6, 0.8, 1.0):
        for dur in (60.0, 200.0):
            cfgs.append({"t_trig": t_trig, "r_dur": dur, "u": u, "arm": "armed"})
            cfgs.append({"t_trig": snap(EP_T1 + 30.0), "r_dur": dur, "u": u,
                         "arm": "post"})
    rows = run_many(cfgs, pool)
    for c, r in zip(cfgs, rows):
        out.append(dict(arm=c["arm"], u=c["u"], dur=c["r_dur"],
                        G_min=r["G_min"], G_end=r["G_end"],
                        outcome3=r["outcome3"], tax=r["tax"]))
        print(f"  {c['arm']:6s} u={c['u']:.2f} dur={c['r_dur']:5.0f}: "
              f"G_min={r['G_min']:.4f} G_end={r['G_end']:.4f} -> {r['outcome3']}",
              flush=True)
    return dict(cells=out)


def s3_settling() -> dict:
    """Is the S hold actually held?  Deviation over the post-settling window."""
    print("S3: S_hold settling diagnostic", flush=True)
    out = []
    for v in (0.9545, 0.8, 0.657, 0.6, 0.4):
        sch = add_span(episode(), "S_hold", 0.0, T_END, v)
        sol = simulate_armed(Params(), sch, T_END)
        t, S = sol["t"], sol["S"]
        m = t >= 20.0
        out.append(dict(S_hold=v, dev_all=float(np.max(np.abs(S - v))),
                        dev_after20=float(np.max(np.abs(S[m] - v))),
                        S_end=float(S[-1]),
                        S_min=float(S.min()),
                        G_end=float(sol["G"][-1]),
                        G_min=float(sol["G"].min()),
                        c_max=float(sol["c"].max()),
                        stuck=bool(is_stuck(sol, Params()))))
        print(f"  S_hold={v:.4f}: |S-v| max all={out[-1]['dev_all']:.2e} "
              f"after t=20: {out[-1]['dev_after20']:.2e}  S_end={S[-1]:.4f}  "
              f"G_end={sol['G'][-1]:.4f} G_min={sol['G'].min():.4f} "
              f"c_max={sol['c'].max():.3f} stuck={out[-1]['stuck']}", flush=True)
    return dict(rows=out)


def s4_trajectories() -> dict:
    """Trajectory dump for the figure: the canonical dose under each arm."""
    print("S4: trajectory dump", flush=True)
    t_trig = 107.95
    runs = {
        "unarmed": {},
        "A_u0.04": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.04},
        "A_u0.20": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.20},
        "A_u0.80": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.80},
        "A_esc3_u0.8": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.80,
                        "profile": "esc3"},
        "B_only": {"S_hold": 0.9545},
        "C_only": {"floor": 0.10},
        "ABC": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.04,
                "S_hold": 0.9545, "floor": 0.10},
        "duty0.25": {"duty": 0.25},
        "a_blind_u0.8": {"t_trig": t_trig, "r_dur": 1600.0, "u": 0.80,
                         "a_blind": True},
    }
    store = {}
    for name, cfg in runs.items():
        sch = episode()
        if cfg.get("duty") is not None:
            from armed_model import duty_cycled_hold
            sch = duty_cycled_hold(sch, 0.9, cfg["duty"])
        if cfg.get("floor"):
            sch = add_span(sch, "u_ext", 0.0, T_END, cfg["floor"])
        if cfg.get("t_trig") is not None:
            sch = rescue_profile(sch, cfg["t_trig"], cfg["r_dur"], cfg["u"],
                                 cfg.get("profile", "step"))
        if cfg.get("S_hold") is not None:
            sch = add_span(sch, "S_hold", 0.0, T_END, cfg["S_hold"])
        if cfg.get("a_blind"):
            sch = add_span(sch, "a_blind", 0.0, T_END, 1.0)
        sol = simulate_armed(Params(), sch, T_END)
        keep = sol["t"] <= 600.0
        store[name] = {k: sol[k][keep] for k in ("t", "a", "G", "E", "c", "S",
                                                "Theta_eff")}
        print(f"  {name}: G_min={sol['G'].min():.4f} G_end={sol['G'][-1]:.4f} "
              f"c_max={sol['c'].max():.3f}", flush=True)
    np.savez(os.path.join(OUT, "trajectories.npz"),
             **{f"{n}_{k}": v for n, d in store.items() for k, v in d.items()},
             names=np.array(list(store.keys())))
    return {"names": list(store.keys())}


def s5_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    res = json.load(open(os.path.join(OUT, "results.json")))
    supp = json.load(open(os.path.join(OUT, "supplement.json")))
    cmap = ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])   # no-collapse/partial/full

    # ---- F1: outcome maps
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    panels = [
        ("exp2 unarmed: rescue after episode end (delay 30)", None),
        ("(a) pre-armed step from E>0", "A_step"),
        ("(a) pre-armed escalating (3 tiers)", "A_esc3"),
        ("(a)+(b) pre-armed + S held", "AB"),
    ]
    for ax, (title, key) in zip(axs, panels):
        if key is None:
            tax = np.array(res["part0"]["exp2_tax"])[0]     # delay 30
            g = np.zeros_like(tax)
            g[tax == 0] = 2          # failure -> full collapse
            g[tax == 1] = 1
            g[tax == 2] = 0          # full rescue -> no collapse
            g[tax == 3] = 0
            uu, dd = STRENGTHS, DURATIONS
        else:
            g = np.array(res["part1"][key]["grid"])
            uu, dd = STRENGTHS, DURATIONS
        ax.imshow(g.T, origin="lower", aspect="auto", vmin=0, vmax=2, cmap=cmap,
                  interpolation="nearest",
                  extent=[uu.min(), uu.max(), dd.min(), dd.max()])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("u_ext strength")
        ax.axvline(0.2, color="w", ls=":", lw=0.8)
    axs[0].set_ylabel("engagement duration (t.u.)")
    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color=cmap(i), label=lab)
                        for i, lab in enumerate(("no collapse", "partial transient",
                                                 "full collapse"))],
               loc="upper center", ncol=3)
    fig.suptitle("Fig 1 — three-outcome maps: the same episode dose, armed vs "
                 "unarmed (grid strengths 0.2-1.0, exp2's axes)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    f1 = os.path.join(FIG, "f1_outcome_maps.png")
    fig.savefig(f1, dpi=110)
    plt.close(fig)
    print(f"  -> {f1}")

    # ---- F2: trajectories
    z = np.load(os.path.join(OUT, "trajectories.npz"))
    names = [str(n) for n in z["names"]]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for n in names:
        axs[0].plot(z[f"{n}_t"], z[f"{n}_G"], lw=1.1, label=n)
        axs[1].plot(z[f"{n}_t"], z[f"{n}_c"], lw=1.1)
    axs[0].axvspan(100, 200, color="tab:red", alpha=0.08)
    axs[0].axhline(0.1, color="gray", ls=":", lw=0.8)
    axs[0].set_xlabel("t"); axs[0].set_ylabel("G (generator)")
    axs[0].set_title("generator G(t), canonical episode")
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].axvspan(100, 200, color="tab:red", alpha=0.08)
    axs[1].set_xlabel("t"); axs[1].set_ylabel("c (cannibalization switch)")
    axs[1].set_title("switch c(t): does it ever fire?")
    for n in ("unarmed", "B_only", "C_only", "a_blind_u0.8"):
        axs[2].plot(z[f"{n}_t"], z[f"{n}_S"], lw=1.1, label=f"{n}: S")
    axs[2].axvspan(100, 200, color="tab:red", alpha=0.08)
    axs[2].axhline(0.9545, color="gray", ls=":", lw=0.8, label="healthy S")
    axs[2].set_xlabel("t"); axs[2].set_ylabel("S (setpoint)")
    axs[2].set_title("the setpoint drain is the collapse mechanism")
    axs[2].legend(fontsize=6)
    fig.suptitle("Fig 2 — trajectories: the armed configuration does not reach "
                 "the stuck attractor", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f2 = os.path.join(FIG, "f2_trajectories.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)
    print(f"  -> {f2}")

    # ---- F3: the dose-response boundary, unarmed vs armed
    p2 = res["part2"]
    order = ["unarmed", "A_step_u0.04", "A_step_u0.8", "A_esc3_u0.8",
             "C_only", "B_only", "AB", "ABC"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.arange(len(order))
    dur = [p2[k]["dur_threshold"] if p2[k]["dur_threshold"] else 420.0 for k in order]
    inten = [p2[k]["int_threshold"] if p2[k]["int_threshold"] else 1.05 for k in order]
    axs[0].bar(xs, dur, color=["#d62728"] + ["#1f77b4"] * (len(order) - 1))
    axs[0].axhline(66.32, color="k", ls="--", lw=1, label="unarmed 66.32 t.u.")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("episode-duration collapse threshold (t.u.)")
    axs[0].set_title("dose threshold: duration\n(bar at 420 = NONE up to 400 tested)")
    axs[1].bar(xs, inten, color=["#d62728"] + ["#1f77b4"] * (len(order) - 1))
    axs[1].axhline(0.5997, color="k", ls="--", lw=1, label="unarmed 0.600")
    axs[1].axhline(1.0, color="gray", ls=":", lw=0.8,
                   label="a_hold ceiling = 1.0")
    axs[1].set_ylabel("episode-intensity collapse threshold")
    axs[1].set_title("dose threshold: intensity\n(bar at 1.05 = NONE up to 1.0)")
    for ax in axs:
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Fig 3 — the dose-response boundary moves by an order of "
                 "magnitude under the armed schedule", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    f3 = os.path.join(FIG, "f3_dose_boundary.png")
    fig.savefig(f3, dpi=110)
    plt.close(fig)
    print(f"  -> {f3}")

    # ---- F4: the structural bound
    p5 = res["part5"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ss = np.linspace(0.05, 1.0, 200)
    axs[0].plot(ss, 0.8 * ss, color="tab:purple", lw=1.5,
                label=r"$\Theta_{eff}(S)=0.8S$ (switch fires if $E>\Theta_{eff}$)")
    for A, dmax in p5["D_max"].items():
        if float(A) in (0.5, 1.0, 2.0):
            axs[0].axhline(dmax, ls="--", lw=1,
                           label=f"$D_{{max}}(A={A})$ = {dmax:.3f}")
    sh = np.array([r["S_hold"] for r in supp["s3"]["rows"]])
    axs[0].plot(sh, np.array([r["S_min"] for r in supp["s3"]["rows"]]), "o",
                color="tab:blue", label="measured $S_{min}$ (held setpoint)")
    axs[0].set_xlabel("held setpoint S"); axs[0].set_ylabel("E / threshold")
    axs[0].set_title("the structural bound: $E \\leq D_{max}$, and the switch "
                     "needs $E>\\Theta_{eff}$")
    axs[0].legend(fontsize=7)
    x = np.array([r["S_hold"] for r in res["part3"]["S_hold_sweep"]])
    y = np.array([r["c_max"] for r in res["part3"]["S_hold_sweep"]])
    axs[1].semilogx(np.maximum(x, 1e-2), y, "o-")
    axs[1].set_xlabel("held setpoint S (log)")
    axs[1].set_ylabel("max cannibalization c")
    axs[1].set_title("switch fires only once the setpoint is dragged low\n"
                     "(unarmed S falls to 0.153)")
    axs[1].axhline(0.5, color="gray", ls=":", lw=0.8)
    fig.suptitle("Fig 4 — why the held setpoint is not a dose threshold: the "
                 "collapse requires the setpoint drain", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    f4 = os.path.join(FIG, "f4_structural_bound.png")
    fig.savefig(f4, dpi=110)
    plt.close(fig)
    print(f"  -> {f4}")


def main() -> int:
    res = {}
    with Pool(28) as pool:
        res["s1"] = s1_confound(pool)
        res["s2"] = s2_like_for_like(pool, 107.95)
    res["s3"] = s3_settling()
    res["s4"] = s4_trajectories()
    with open(os.path.join(OUT, "supplement.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    s5_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
