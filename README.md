# GTA VI Trophy Rarity Forecaster

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate — the percent of
Exophase-tracked players who have earned it — at a case's target instant
(week one: 2026-11-26T00:00:00Z for the final case). Entry for make.fun
bounty `gta6-trophy-rarity-forecast-01`. MIT licensed. Python 3 standard
library only; deterministic; runs offline in well under a second per case.

Run it the way the grader does:

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

## Why a ratio needs two models

An earn rate is `earners / players_tracked`, and the two sides move for
different reasons:

- **Earners** only grow. A tracked player earns a trophy when their play
  reaches it, so earner counts follow each trophy's attainment curve.
- **Players tracked** also only grows, and fast in week one. Every new
  tracked player arrives with almost no trophies and *dilutes* every rate —
  the house's own launch-night record shows the first Exophase refresh
  cutting nearly every rate at once (638 → 3,868 tracked players) even
  though nobody un-earned anything.

A forecaster that only extrapolates the rate confuses dilution with
regression. This one models both sides and the rate itself, then blends.

## Method

1. **Refresh states.** Exophase refreshes in steps; the recorded history is
   long runs of identical reads. Consecutive duplicates are collapsed to a
   single state before any fitting, so a stale page repeated for hours
   cannot outweigh a fresh one.
2. **Attainment curve (direct view).** For a fixed cohort, the share that
   has earned a trophy by time *t* is the CDF of a completion-time
   distribution. Completion times for story content are roughly lognormal,
   so that CDF is close to linear in `logit(rate)` against `ln(1 + hours)`.
   A weighted least-squares fit over the states (weights √players, so
   better-measured states count more) yields a slope; the forecast is
   **anchored at the last observed logit** and walks forward with a shrunk
   slope (`n_eff / (n_eff + 3)`) clipped to [−1.5, 2.5]. Anchoring keeps a
   bad early state from moving the level; shrinkage keeps a slope fitted on
   three states from being trusted like one fitted on fifty.
3. **Earners / players decomposition.** `players_tracked` is extrapolated
   by the median of three fits (log–log, log-linear on the recent half of
   states, linear-in-log-time), clipped to a sane band. Earners
   `E_i = rate_i × P / 100` are extrapolated linearly in `ln(1 + hours)`,
   anchored, never falling, never exceeding the tracked base. The implied
   rate `E_i(target) / P(target)` is the decomposition view.
4. **Blend.** Median logit = 0.7 × attainment view + 0.3 × decomposition
   view (whichever exists if only one does; persistence with fewer than two
   usable states).
5. **Archetype priors.** Each trophy is classified from its tier and
   description — story completion, collectible grind, skill/challenge,
   exploration, platinum, other. Base rates from comparable open-world
   launches (GTA V, Red Dead Redemption 2 and similar) are consistent:
   early story trophies reach tens of percent within days, mid-story and
   exploration trophies trail by roughly a chapter's worth of playtime,
   collectible/completion grinds sit in the low single digits in week one,
   and a platinum is a fraction of a percent. The model does not hard-code
   those numbers; it applies the archetype gently — grinds and the platinum
   get damped slopes (×0.80 / ×0.55) and wider bands (×1.20 / ×1.35), story
   trophies keep their fitted slope — because grind attainment is later and
   more uneven than story attainment at the same observed level.
6. **Honest uncertainty.** σ (logit units) combines a random-walk term
   `0.42·√(horizon days)`, the fit's residual scale inflated by the
   extrapolation leverage, and the observed jump size between consecutive
   refresh states — an unseen refresh is the dominant risk, as the practice
   cases prove (every pre-refresh read identical, the answer on the other
   side of a refresh). With fewer than three states the model has never
   seen a refresh, so σ is floored at 1.0. Boundary rates (≥99.9% or
   ≤0.1%) are small-sample artifacts more often than facts: their median
   logit is capped at the 99%/1% points and σ is floored at 1.5. σ is
   clipped to [0.30, 2.40]; quantiles are read from a normal in logit
   space, clamped to [0.01, 99.99] and rounded to 4 decimals.

## Validation

- Practice cases `p00`/`p01` (kit input): valid, deterministic output;
  byte-identical across runs.
- A synthetic final-shaped case (55 trophies, lognormal attainment curves,
  tracked players growing 638 → ~62k over 62 hours of 5-hourly refreshes,
  4.5-day horizon): mean pinball loss on the logit scale **0.370** for this
  model vs **0.744** for persistence (the house baseline's median), i.e.
  the trend + decomposition roughly halves the loss when a real trend
  exists to find. The simulation is a sanity check, not the contest data.

## Layout

- `main.py` — the whole forecaster (the entry point the grader runs).
- `README.md` — this methodology.
- `LICENSE` — MIT.
