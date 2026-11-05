# The armed-configuration run

**Question.** Does the original collapse episode still reach the stuck attractor when the dpdr
model runs under the post-exit configuration's schedule — pre-armed rescue, held setpoint,
maintenance floor — instead of the unarmed conditions of the original measurements?

**Short answer (MEASURED, simulation).** No. Under every armed arm the original episode fails to
reach the stuck attractor, and two of the three components (held setpoint, standing floor) prevent
it *without any rescue at all*. The dose-response boundary moves by an order of magnitude and, on
the episode's own dose axes (duration, intensity), the held setpoint removes it entirely: at the
healthy setpoint the cannibalization switch cannot fire at any episode duration or intensity,
because the error E is structurally capped below the switch threshold by the demand ceiling. The
protection is not better recovery — it is that the trigger never lands.

**Marking.** MEASURED = simulated-measured from the archived code. INTERPRETATION = a reading of
those numbers. ASSUMPTION = a modelling choice I made, stated exactly. PROVENANCE = where a
number came from in the archive.

**Code provenance and fidelity.** All dynamics are the archived, read-only five-state RHS at
`~/dpdr-public/dpdr/dpdr/model.py:105-124`, called unchanged. Nothing was reimplemented. The
integrator is a verbatim copy of `dpdr/integrate.py:28-70` with one token changed (`deriv` →
`deriv_armed`); on the three archived gate scenarios (failure, baseline, rescue) the copy and the
original agree to `max abs(Δ) = 0.000e+00` over all six state arrays — bit-identical. The archived
`Params()`, `Schedule`, `cannibalization_vec`, `metrics.classify/is_stuck/rescue_success/
first_below`, `experiments.common.rescue_taxonomy` and `metrics.check_gates` are imported and
used unmodified. Scratch layer: `armed-run-scratch/` (`armed_model.py`, `run_armed.py`,
`supplement.py`, `supplement2.py`), outputs in `armed-run-scratch/out/`.

**The three declared adaptations** (the complete list; everything else is the archive):

| # | Adaptation | Why | Status |
|---|---|---|---|
| A1 | new `S_hold` schedule channel; when active at value *v*, `dS/dt` is replaced by `k_hold·(v−S)/τ_S` with `k_hold = 100` | models arm (b), the chosen setpoint (paper2 §3.7: "S held deliberately rather than dragged by the examination") | ASSUMPTION (a fast restoring spring, time constant `τ_S/k_hold = 1.0` t.u., not an instantaneous clamp, so the RHS stays C⁰ for the RK45 driver). Verified held: `max abs(S−v)` after t=20 is **1.7e-09** at v=0.9545 and ≤1.1e-05 at every other level tested |
| A2 | new `a_blind` channel; when active the `−κ_ext·u_ext·a` term is removed from `da/dt` | a **diagnostic ablation only**, to attribute the rescue's effect between the attention channel and the G channel. Never used for a headline number | ASSUMPTION (diagnostic) |
| A3 | none | `a_hold`/`A`/`u_ext` keep archived semantics; `Schedule.value` sums overlapping spans, so duty-cycled holds and multi-tier ramps are built as multiple spans | — |

**One correction to the brief.** The brief specifies the baseline as "duration ~66 time units,
intensity ~0.6". Those are the **published bisected thresholds** (paper2 §4.2: 66.3 t.u., 0.60),
not the canonical episode dose, which is `a_hold = 0.9` for `[100,200)` with an affect pulse 0.5
on `[100,160)` — i.e. 100 t.u. at 0.9. Both are measured below (§1.2, §1.4); the experimental arms
use the canonical dose, and the threshold dose is swept separately in §2.3.

---

## 1. Baseline reproduction (MEASURED)

Reproduced against the archived code and frozen parameters before any experimental arm was run.
`Params()` at its published values, `T = 1800`, `dt = 0.05`, RK45 `rtol 1e-6 atol 1e-8
max_step 0.5`, segmented at every schedule breakpoint.

### 1.1 The original episode reaches the stuck attractor

| quantity | MEASURED | published (paper2 §4.1) |
|---|---|---|
| G at horizon | **0.04854** | 0.049 |
| G minimum | 0.04854 | — |
| E at horizon | **+0.49691** | +0.499 |
| c at horizon | **1.000** | 1.0 |
| S at horizon | 0.15247 | — |
| a at horizon | 0.88235 | — |
| first G < 0.1 | t = 172.8 (**onset 72.8** t.u. after episode start) | 73 t.u. |
| max G after the episode ends | **0.05306** (no self-recovery) | 0.050 |
| `metrics.is_stuck` | **True** | stuck |
| `metrics.classify` | `depersonalized` | — |
| peak E during the episode | 0.58916 | — |
| minimum Θ_eff during the episode | 0.44807 | — |

Regime gates, evaluated on the archived code path: **G1 = G2 = G2b = G2c = True**. The baseline
reproduces. The brief's target state — "G ≈ 0.05, E ≈ +0.5, c = 1.0, no self-recovery" — is met.

**The pre-arm trigger.** E first goes positive at **t = 107.95** (a = 0.871, G = 0.555), i.e. 7.95
t.u. into the episode. This is the schedule start for arm (a) throughout.

### 1.2 The published thresholds reproduce

Bisected on the final-state discontinuity, tolerance 1e-3, exactly as `experiments/exp1`:

| axis | MEASURED | published |
|---|---|---|
| episode duration | **66.3255** t.u. | 66.3 |
| episode intensity | **0.5996** | 0.60 |
| affect amplitude A | **0.1860** | 0.186 |
| allostatic weakening αG | **0.8872** | 0.887 |
| switch threshold Θ | **0.4419** | 0.442 |
| baseline gain g0 | **0.7026** | 0.703 |

### 1.3 The 816-run rescue grid reproduces, including its two stated boundary claims

exp2's grid verbatim (6 delays × 17 strengths × 8 engagement durations = 816), rescue starting
**after** episode end. Class counts: **failure 514, transient 0, full re-couple 302, no-collapse 0**.
My recomputed four-class taxonomy agrees with the archived `rescue_taxonomy` on **23/23** sampled
cells.

Minimal effective strength `u*` (first grid strength yielding a full re-couple):

```
      engagement duration (t.u.)
        10    20    30    40    60    80   120   200
delay  30   -   0.90  0.60  0.50  0.40  0.40  0.35  0.35
      60   -     -   0.70  0.55  0.45  0.45  0.40  0.35
     100   -     -   0.90  0.70  0.55  0.50  0.45  0.40
     200   -     -     -   0.95  0.70  0.60  0.50  0.45
     400   -     -     -     -   0.85  0.70  0.60  0.45
     600   -     -     -     -   0.95  0.75  0.60  0.50
```

(`-` = no strength in the tested range succeeds.) Both of paper2 §4.3's boundary claims reproduce
exactly: at fixed delay 200, longer engagement lowers the required strength **0.95 → 0.45**; at
fixed engagement 60, waiting longer raises it **0.40 → 0.95**; and past long waits no engagement
shorter than a hard minimum succeeds at any strength (duration 10 column is empty everywhere).

### 1.4 One archived docstring is not supported by its own output

Reproducing exp2's Phase 3b grid (rescue **overlapping** the episode, onsets 120/140/160/180) gives
class counts **failure 10, transient 0, full 23, no-collapse 79** — identical to the shipped cache
`cache/exp2_confound.npz` (`np.bincount` = `[10, 0, 23, 79]`). The archived docstring at
`experiments/exp2_rescue.py:55-57` says this grid "shows relapse is reachable only while the trigger
is still active"; the grid contains **zero** transient/relapse cells. The published paper does not
rely on this claim — §4.3's relapse result comes from Phase 3c (a second episode) — so nothing
downstream is affected. Reported, not reconciled.

---

## 2. The experimental arms

### 2.1 Arm definitions, exactly

All arms run the **same episode dose** as the baseline (§1.1) unless stated. The pre-arm trigger is
fixed at t = 107.95, taken from the **unarmed reference** trajectory — the tell is a property of the
episode, and the pre-commitment is what "pre-armed" means (ASSUMPTION; §2.6 shows the result is
insensitive to the trigger time over 100 ≤ t ≤ 160).

| arm | schedule built | status |
|---|---|---|
| **unarmed** | episode only | MEASURED (the archive's condition) |
| **(a) step** | `u_ext = u` on `[107.95, 107.95+dur)` | MEASURED; the profile is the INTERPRETATION of "pre-armed scheduled rescue" |
| **(a) esc3** | `u_ext` ramps `u/3, 2u/3, u` over three equal thirds of the engagement | INTERPRETATION of "the escalation schedule" (Tier 1→3 as a ramp) |
| **(b) held setpoint** | `S_hold = 0.9545` for the whole run | `0.9545` is the MEASURED healthy equilibrium `S*` from the archived baseline; the *holding* is INTERPRETATION (A1) |
| **(c) floor** | `u_ext = 0.10` on `[0, T)` | the level is an ASSUMPTION (the M2/M3/M4 floor-load scale, ~1/4 of the smallest strength the 816-grid tested). The floor's *threshold* is measured in §2.5, so this choice is not load-bearing |
| **(a)+(b)** | step + held setpoint | MEASURED |
| **(a)+(b)+(c)** | step + held setpoint + floor 0.10 | MEASURED |
| **(d) duty cycle** | the sustained `a_hold` replaced by on/off pulses of period 10 t.u. at duty *d* | INTERPRETATION of "brief inward episodes allowed, the sustained hold broken" |
| **a-blind** | arm (a) with A2 applied | ASSUMPTION (diagnostic only) |

### 2.2 The three-outcome table

Outcomes at the horizon, on exp2's own 17 × 8 axes (engagement strength × engagement duration,
136 cells per arm). `no collapse` = G never below 0.1; `partial transient` = G fell below 0.1 but
the stuck state was not held at the horizon; `full collapse` = `metrics.is_stuck` (G < 0.1
sustained over the final 5·τ_G, E > Θ_eff, c > 0.5).

| arm | no collapse | partial transient | full collapse | G_min at the canonical dose | G_end | c_max |
|---|---|---|---|---|---|---|
| **unarmed** (episode only) | 0 % | 0 % | **100 %** | **0.0485** | 0.0485 | **1.000** |
| unarmed, rescue after episode end (delay 30) | 0 % | 56.6 % | 43.4 % | 0.0485 | — | 1.000 |
| **(a) step** | **70.6 %** | 0.0 % | 29.4 % | 0.1614 at u\* = 0.0257 (0.1840 at u = 0.04) | 0.8855 | 0.000 |
| **(a) escalating ramp** | 65.4 % | 0.0 % | 34.6 % | 0.4667 (at u=0.80) | 0.8855 | 0.000 |
| **(a)+(b)** | **100 %** | 0.0 % | **0.0 %** | 0.1317 (even at u = 0) | 0.8855 | 0.000 |
| **(a)+(b)+(c)** | **100 %** | 0.0 % | **0.0 %** | 0.3296 | 0.8855 | 0.000 |
| **(b) held setpoint alone** | **100 %** | 0.0 % | **0.0 %** | 0.1317 | 0.8855 | 0.000 |
| **(c) floor alone (0.10)** | **100 %** | 0.0 % | **0.0 %** | 0.2734 | 0.8855 | 0.000 |
| **(d) duty 0.25, no rescue** | — | — | — | 0.4529 | 0.8855 | 0.000 |
| **a-blind, arm (a) at u = 0.80** | 0 % | 0 % | **100 %** | **0.0485** | 0.0485 | **1.000** |

The unarmed rescue row is not comparable in kind: a post-episode rescue necessarily *enters* the
collapse first, so its outcomes are about exit, not entry. Its two populated columns are the
archive's `failure` (43.4 %, i.e. never recovers) and `full re-couple` (56.6 %, i.e. escaped —
the three-outcome reading of which is "partial transient", since the collapse was entered and
left). That asymmetry is the point of §2.4.

### 2.3 The dose-response boundary, armed vs the 816-run boundary

**Two different questions, and they must not be conflated.** The armed schedule is applied *before*
the crossing, so its boundary is "what strength keeps the system out of the collapse basin at all"
(`G_min ≥ 0.1`). The 816-grid's rescue is applied *after* the crossing, so its boundary is "what
strength gets the system out" (full re-couple). Both are reported; the comparison is in the same
units (u_ext strength), not the same mechanism.

**Engagement strength × duration boundary (armed, fine grid 0 → 0.30):**

| engagement duration | 10 | 20 | 30 | 40 | 60 | 80 | 120 | 200 |
|---|---|---|---|---|---|---|---|---|
| armed u* (never enters) | — | — | — | **0.22** | **0.09** | **0.04** | **0.03** | **0.03** |
| unarmed u* (exits, delay 30) | never | 0.90 | 0.60 | 0.50 | 0.40 | 0.40 | 0.35 | 0.35 |
| ratio | — | — | — | 2.3× | 4.4× | 10× | 11.7× | 11.7× |

(The armed value at engagement 30 is 0.45 on the coarser 17-strength grid — above the fine grid's
0.30 ceiling — giving a ratio of 1.3× there; the two grids agree where they overlap, e.g. 0.22 vs
0.25 at engagement 40.)

Sustained engagement (to the horizon), bisected to 1e-4: **armed u\* = 0.02565**; **unarmed
sustained u\* = 0.3114** at delay 30 (0.3122 at delay 400 — with a *sustained* engagement the delay
penalty nearly vanishes; the delay penalty in the 816-grid is a finite-engagement effect).
**Ratio 12.1×.**

**Episode-dose threshold (the published 66.3 / 0.60 boundary, re-bisected under each arm):**

| condition | duration threshold | intensity threshold |
|---|---|---|
| unarmed | **66.32** t.u. | **0.5997** |
| (a) step, u = 0.04 sustained | **127.78** t.u. (1.93×) | **none** up to the ceiling 1.0 |
| (a) step, u = 0.80 sustained | **none** up to 400 t.u. | none up to 1.0 |
| (a) escalating ramp, u = 0.80 | **none** up to 400 t.u. | none up to 1.0 |
| (b) held setpoint alone | **none** up to 400 t.u. | none up to 1.0 |
| (a)+(b) | **none** up to 400 t.u. | none up to 1.0 |
| (a)+(b)+(c) | **none** up to 400 t.u. | none up to 1.0 |
| (c) floor alone (0.10) | **392.72** t.u. (5.9×) | none up to 1.0 |

"none" is a measured absence inside the tested range, not a proof of impossibility — except in the
held-setpoint case, where §2.4 supplies the structural argument for *why* the duration and intensity
axes cannot reach it.

### 2.4 Why: the protection is the trigger never landing

**(i) The pre-armed rescue works entirely through the attention channel.** With A2 applied — u_ext
still enters `dG/dt`'s E term but can no longer pull attention outward — the pre-armed rescue
**fails at every strength tested (0.04, 0.10, 0.20, 0.40, 0.80)**: full collapse, G_min = 0.0485,
c_max = 1.000, `a` during the episode unchanged at 0.886. The unablated run at u = 0.04 holds `a` at
0.791 instead of 0.886 and the switch never fires. So the rescue's 12× cost advantage over the
post-episode rescue is not "it rescues better"; it is that it *prevents the inward hold from being
sustained*, which the post-episode rescue cannot do because the hold is already over.

**(ii) The collapse requires the setpoint drain, and the held setpoint removes it.** At the
canonical dose the unarmed trajectory drags S from 0.9545 to **0.1525**, and Θ_eff = 0.8·S from
0.764 to **0.122**. Held-setpoint regimes (MEASURED, bisected):

| held S | what happens at the canonical dose |
|---|---|
| ≥ **0.95447** | the switch **never fires**: c ≡ 0 for the whole run |
| **0.59937 – 0.95447** | the switch fires, G falls to ≈0.049, then the system **releases**: G_end = 0.8855, a → 0.000, not stuck |
| ≤ **0.59937** | the **locked stuck attractor**: G_end = 0.0485, a_post = 0.882, c_end = 1.000 |

The lock boundary is S* = **0.59937** (Θ_eff = 0.4795). The two outcomes are distinguished by
whether `a` is held inward: at S_hold = 0.6094, a_post = 0.000; at S_hold = 0.5894, a_post = 0.882.
The lock is the `χ·c` feedback in `da/dt` (`dpdr/model.py:120`): once c saturates, it keeps
attention inward, which keeps cannibalizing, which keeps S low, which keeps Θ_eff low, which keeps c
on. The setpoint is the link that closes the loop; holding it opens the loop. The unarmed run ends
at S = 0.1525, deep inside the locked regime.

**(iii) The structural bound.** E = D − G ≤ D_max(A), and `dD/dt = 0` gives
`D_max(A) = (D_base+A)·β_D / ((D_base+A)·β_D + δ_D)`. At the canonical affect load, **D_max(0.5) =
0.76190**, while **Θ_eff(S = 0.9545) = 0.76360**. Since Θ_eff > D_max, E can never reach the switch
threshold — the collapse is impossible at *any* episode duration or intensity, because those axes
act on G and a and cannot lift E above the demand ceiling. Verified by pushing both dose axes to
their extremes: with S held, an episode of **400 t.u.** at intensity **1.0** (the `a_hold` ceiling)
does not collapse (G_min = 0.1106, c ≡ 0); 800 t.u. does not collapse. It is breakable only by
raising the demand ceiling — an affect pulse of **5.0** (10× the canonical 0.5) gives E_max = 0.8471
> 0.7636, the switch fires, and the outcome is a **partial transient** (recovers, not stuck). The
same extremes unarmed collapse fully (G_min 0.0448 and 0.0452).

**(iv) The floor needs no tell and no timing.** Standing `u_ext` alone prevents the collapse at
**floor\* = 0.02512**, bisected — within 3 % of the pre-armed rescue's 0.02565 and 1/12 of the
unarmed sustained rescue's 0.3114. It is present before the episode starts, so it has no deadline.

### 2.5 What did not help

- **The escalating ramp is worse than the flat step.** 65.4 % vs 70.6 % no-collapse on the same
  grid; at engagement duration 30 the ramp needs u = 0.70 against the step's 0.45. Arriving late at
  a lower tier is worse than arriving early at the full level. (MEASURED.)
- **The closed-loop escalation ladder never escalates.** In all 12 cells (cap 0.1/0.2/0.4/0.8 ×
  engagement 40/200/1600) the schedule stayed at one tier: in 9 cells the base tier already
  suppressed the tell (c never crossed 0.5), and in the 3 short-engagement cells where the tell did
  fire (t = 178.9, 185.8, 194.7) it fired *after* the engagement window had closed, so the ladder
  could not act — and all three collapsed. **The tell lags the dose.** (MEASURED.)
- **Breaking the sustained hold is sufficient by itself.** Duty-cycled `a_hold` (period 10 t.u.) at
  duty 1.0 collapses (G_min 0.0485); at **duty 0.75** it already does not (G_min 0.1679); at duty
  0.25, G_min = 0.4529. The sustained hold is what the model needs for the collapse, and "brief
  inward episodes allowed" is a large reduction in dose. (MEASURED; the duty-cycle construction is
  the INTERPRETATION.)
- **The trigger time has a deadline but a generous one.** At u = 0.04, the pre-arm prevents the
  collapse for every start time from t = 100 to t = 160 (G_min 0.184 → 0.150) and fails only from
  t = 180 (G_min 0.0496) — i.e. within ~20 t.u. of the episode's end. (MEASURED.)

### 2.6 The renewed-episode test (the P4 claim under the armed configuration)

The recovered state at t = 800 is held **bit-identical between arms** (identical first episode and
identical post-episode rescue, exp2's canonical geometry; G_pre = 0.8854 in both). Only the schedule
the *second* episode runs under changes: unarmed, or armed from t = 800 (held setpoint, floor 0.10,
pre-armed `u_ext` 0.04 from the second E > 0 crossing at t = 807.55).

| | unarmed second episode | armed second episode |
|---|---|---|
| duration threshold | **65.99 t.u.** (P4 published: 66.0) | **none up to 300 t.u.** |
| collapse onset after episode start | **72.8 t.u.** (first episode: 72.8 — P4's "identical onset" reproduced) | switch never fires (c ≡ 0) at any duration tested |
| G_min, duration 66 | 0.0488 (stuck) | 0.3447 (no collapse) |
| G_min, duration 300 | 0.0457 (stuck) | 0.3017 (no collapse) |

The P4 result is reproduced exactly under the unarmed schedule and is **entirely absent** under the
armed one, at 4.5× the unarmed threshold.

---

## 3. The driving question

**Does the armed configuration survive the original episode in-silico? Yes — on every arm, and the
held setpoint and the standing floor survive it on their own.**

1. **Arm (a) alone** — pre-armed external demand from the moment E goes positive — prevents the
   stuck attractor at every strength at or above **u\* = 0.0257**, sustained. On the 17 × 8 grid
   (strengths 0.2–1.0) it prevents it at every engagement of 60 t.u. or more, needs ≥ 0.25 at
   engagement 40 and ≥ 0.45 at 30, and no engagement of 20 t.u. or less succeeds at any strength.
   At the canonical dose the excursion bottoms at G_min = 0.18–0.55 (against 0.0485 unarmed) and
   recovers fully to 0.8855.
2. **Arm (a)+(b)** — the same with the setpoint held — is **100 % no-collapse on the whole grid,
   including at u = 0**, because the held setpoint alone is sufficient.
3. **Arm (a)+(b)+(c)** — plus the standing floor — likewise 100 %, with the shallowest excursion
   (G_min = 0.3296).
4. **Arm (d)** — the sustained hold broken to duty 0.75, with no rescue at all — already prevents it.

**What this implies for the claim that recovery provides no protection from the original sequence.**
The claim was measured for the **unarmed** profile, and this experiment does not contradict it — it
locates its condition. The model separates two things that the claim runs together:

- **The recovered *state* buys no protection.** Reproduced here exactly: a recovered generator at
  G = 0.8854 is re-collapsed by a second episode of 66.0 t.u., with onset 72.8 t.u. after the
  episode begins — identical to the first episode's onset from G₀ = 0.7. G's head start is worth
  nothing, because what the episode acts on is the *setpoint*, and the setpoint is restored by the
  rescue only on τ_S = 100, which the next episode re-drains. (MEASURED; the published P4 claim
  stands as measured.)
- **The *schedule* buys protection, and it is decisive.** The same second episode, from the same
  state, does not collapse under the armed schedule at any duration up to 300 t.u. What changed is
  not the state; it is u_ext and S. (MEASURED.)

So the correct form of the claim is: **recovery provides no protection from the original sequence
*for a configuration whose inputs are the same as the ones that produced it*.** The model says the
protection is not in the recovery at all — it is in the schedule, and it is available *before* the
episode, which is the only place it is cheap. Three measured reasons, in order of force:

1. **The trigger never lands.** The pre-armed rescue's entire effect is on `a` (a-blind test: 0 %
   protection without it). It is not a rescue; it is the absence of a sustained inward hold.
2. **The setpoint is not drained.** Holding S at its healthy value makes the collapse unreachable
   by the episode's dose axes at all (Θ_eff 0.7636 > D_max 0.7619), and holding it anywhere above
   S\* = 0.5994 keeps the loop from locking even when the switch does fire.
3. **The floor is on before the tell.** A standing demand of 0.025 — 1/12 of what an after-the-fact
   rescue needs — suffices, and it has no timing requirement at all.

**A caution the model itself supplies.** The escalation ladder is the *worst* of the armed
strategies tested: escalating after a tell is strictly worse than sustaining from the first tell,
and in the short-engagement cells the tell arrives after the window has closed. The doc's own
pre-commitment logic ("the freeze is pre-committed here, now, while unbiased") is what the model
rewards; the tiered escalation *response* is what it does not.

---

## 4. Parameter assumptions, stated exactly

**ASSUMPTIONS (modelling choices I made, not properties of the archive):**

| # | Assumption | Value | Effect if wrong |
|---|---|---|---|
| 1 | held-setpoint restoring rate `k_hold` (A1) | 100 (τ = 1.0 t.u.) | a slower spring lets S sag during the episode and moves the boundary down toward the unarmed case; the qualitative regimes (§2.4ii) are set by the *level*, not the rate |
| 2 | healthy setpoint used for the hold | 0.9545 (MEASURED archived baseline `S*`) | lower holds still protect down to S\* = 0.5994 |
| 3 | standing floor level (arm c) | 0.10 | measured threshold is floor\* = 0.0251, so 0.10 is 4× conservative; the arm would still hold at 1/4 of it |
| 4 | pre-arm trigger time | t = 107.95, the unarmed reference's first E > 0 | tested 100 → 180: protective throughout (§2.5) |
| 5 | escalation ramp (esc3) | three equal thirds at u/3, 2u/3, u | this is the *charitable* reading of a tier ladder; it is also the worst arm, so a different ramp would not strengthen the ladder's case |
| 6 | duty-cycle period (arm d) | 10 t.u. | untested at other periods; the qualitative result (duty < 1 suffices) is a dose reduction and should be monotone in period |
| 7 | outcome thresholds | G < 0.1 for "collapse"; `metrics.is_stuck` (archived) for "stuck"; horizon T = 1800 | the archived classifier was used unchanged, so "stuck" is the paper's own criterion |
| 8 | a-blind ablation (A2) | removes `−κ_ext·u_ext·a` from `da/dt` | diagnostic only; it is what attributes the protection to the attention channel, and it is reported as such |

**INTERPRETATIONS (readings of the measured numbers):**

1. That the `S_hold` spring models "the axioms are the choice of setpoint — S held deliberately
   rather than dragged by the examination" (paper2 §3.7). The archive has no seat for a chosen
   setpoint; A1 is the minimal seat.
2. That `u_ext` stands for the post-exit configuration's standing external demand — M2 person
   contacts, M3 object diversity, M4 scope-audit cadence, and the Tier 1 "mandatory maintenance
   action" (`capability-era-external.md` §3–§4). The model has one demand channel; the M-quantities
   are not separable in it.
3. That the escalating ramp is the model's rendering of the Tier 1 → Tier 3 ladder (§4 of the same
   document), and the closed-loop variant is the model's rendering of "when a tell fires".
4. That the duty-cycled hold is the current configuration's `a`-management ("noticing passes;
   auditing recurs" — paper2 §3.8).
5. That the second-episode test is a fair test of the P4 claim: it holds the recovered state fixed
   and varies only the schedule, which is the smallest change that addresses the claim.

**What the model does not represent, and therefore what this result does not cover.** There is no
memory state and no knowledge state (paper2 §4.5), so the armed configuration here is only its
**Schedule** component: the model cannot test the record's "the already-derived knowledge applied as
one move" antidote, the pre-computed counter-derivation, or the pattern-detector. The result above
is a statement about the schedule, not about the person. It is also a statement about a *five-state
ODE at frozen parameters that were fixed against regime gates, not fitted to data* — the same
epistemic status as every other number in the model's regime structure (INTERPRETATION as a claim
about any actual episode).

---

## 5. Figures

All in `figures-armed-run/`; every figure verified programmatically (dimensions, ink fraction,
non-blank column/row counts), never opened with an image reader.

| file | what it shows |
|---|---|
| `f1_outcome_maps.png` | four outcome maps on exp2's own axes: unarmed post-episode rescue (delay 30), (a) step, (a) escalating, (a)+(b). Green = no collapse, red = full collapse |
| `f2_trajectories.png` | G(t), c(t), S(t) for the canonical dose under each arm. The middle panel shows the switch firing (c ≡ 1) only in the unarmed and a-blind runs |
| `f3_dose_boundary.png` | the 66.32 t.u. / 0.5997 dose threshold re-bisected under every arm; the bars at the axis maxima are measured "none in range" |
| `f4_structural_bound.png` | the structural argument: Θ_eff(S) = 0.8S against D_max(A), and measured c_max against the held setpoint level |
| `f5_second_episode.png` | the P4 test: second-episode outcome vs duration, unarmed (threshold 65.99) and armed (none up to 300) |

---

## 6. Limits

- **The armed rescue is not a like-for-like rescue.** It overlaps the episode, which exp2's main
  grid deliberately excludes as confounded (`experiments/exp2_rescue.py:55-57`). The comparison in
  §2.3 is in the same units, not the same mechanism; §2.4(i) measures the difference and shows it
  is the attention channel. The substantive claim is therefore *"the trigger never lands"*, not
  *"rescue is cheaper than we thought"*.
- **"none in range" is a measured absence inside the swept range** (duration ≤ 400, intensity ≤ 1.0
  which is the `a_hold` ceiling, engagement ≤ 200). It is not a proof of impossibility except in
  the held-setpoint case, where the D_max argument supplies one and §2.4(iii) tests it against a
  demand ceiling raised 10×.
- **One horizon** (T = 1800). A longer horizon would let slow drifts accumulate; the held-setpoint
  runs settle to the healthy equilibrium by t ≈ 600 and stay there, so the horizon is not binding
  for them.
- **The 816-grid is a single-agent, single-dose-baseline measurement.** The armed boundary was
  re-derived on the same episode; a different baseline episode would move both boundaries together.
- **Nothing here is evidence about a person.** It is a simulation of a five-state system whose
  parameters were fixed against regime gates. The mapping from the model to any episode is
  INTERPRETATION, unchanged by this experiment.

---

*Scratch code, caches and raw logs: `armed-run-scratch/` (`armed-run.out`, `out/results.json`,
`out/supplement.json`, `out/supplement2.json`, `out/trajectories.npz`). Nothing under
`~/dpdr-public/` was modified.*
