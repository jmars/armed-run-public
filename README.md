# The armed-configuration run

In-silico experiment: does the original collapse episode of the dpdr self-regulation
model still reach the stuck attractor when the model runs under a post-recovery
configuration — pre-armed rescue, held setpoint, standing maintenance floor — instead of
the unarmed conditions of the original measurements?

**Headline result (MEASURED, simulation).** No. Under every armed arm the stuck attractor
is not reached. The held setpoint alone prevents collapse at *any* episode dose
(structurally: the effective switch threshold sits above the demand ceiling, so the error
cannot reach it). The standing floor alone prevents it at 1/12 of the strength the
after-the-fact rescue needs, with no timing requirement. The pre-armed response prevents
it entirely through the attention channel (ablation-confirmed). The escalating response
ladder is the worst armed strategy tested.

Full write-up: `armed-run-results.md` (question, baseline reproduction, arms,
dose-response boundaries, mechanism, honest limits).

## Contents

- `armed-run/` — the experimental code. `armed_model.py` is the archived dpdr RHS called
  verbatim inside a wrapper with exactly three declared adaptations (held-setpoint
  channel, an attention-ablation diagnostic, duty-cycled holds); `run_armed.py` and the
  two supplements drive the grids and figures.
- `dpdr/` — the base-model snapshot this run executed against (copied verbatim from the
  parent deposit), so the deposit is self-contained.
- `results/` — raw outputs: results.json, supplement.json, supplement2.json,
  trajectories.npz, and the run log.
- `figures/` — five figures, verified programmatically.
- `armed-run-results.md` — the full write-up with marking discipline
  (MEASURED / INTERPRETATION / ASSUMPTION) stated per claim.
- `base-model-snapshot/` — duplicate of `dpdr/` kept flat for readers who prefer
  un-packaged files; identical content.

## Provenance and dependency

All dynamics are the archived five-state RHS called unchanged; the wrapper's adaptations
are declared in `armed-run/armed_model.py` (A1–A3) and mirrored in
`armed-run-results.md` §"declared adaptations". The baseline reproduction (original
episode, thresholds, 816-run rescue grid) was run against this snapshot and matches the
published values (see `armed-run-results.md` §1). `run_armed.py` resolves the base
package from `dpdr/` in this deposit; a local override via the `DPDR_BASE` environment
variable is supported.

## Relation to the other artifacts

- Base model and model paper: dpdr-public deposit, concept DOI
  10.5281/zenodo.22943641 (the model paper cites it; this deposit extends it with the
  armed-configuration experiment).
- Single-subject manuscript (in preparation): the record and stage-to-variable mapping;
  its transition-threshold and rescue-boundary sections report the unarmed numbers this
  experiment re-derives and extends.
- Self-regulation protocol: the standing schedule simulated here (external floor, held
  setpoint, escalation limits) is the operational form described in the author's
  external protocol document.

## License

MIT (code), consistent with the parent deposit.
