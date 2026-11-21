"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case file (Exophase reads of the GTA VI PS5 trophy page, oldest
first) and writes, for every trophy, five quantiles (5/25/50/75/95%) of its
earn rate at the case's target instant. Standard library only, deterministic,
no network.

Model (see README.md for the full methodology)
----------------------------------------------
An earn rate is a ratio, earners / players_tracked, and the two sides move
for different reasons, so the forecaster models both and the rate directly,
then blends the views in logit space:

1.  Refresh states. Exophase refreshes its counts in steps, so the recorded
    history is long runs of identical reads. Consecutive duplicates are
    collapsed to one *state* each; all fitting happens on states, so a stale
    page repeated for hours cannot dominate a fit.

2.  Attainment curve (direct view). For a fixed cohort of players, the share
    that has earned a trophy by time t is the CDF of a completion-time
    distribution. Completion times are roughly lognormal, so that CDF is
    close to linear in logit(rate) against ln(1 + hours). A weighted least
    squares fit of logit(rate) on ln(1 + hours) over the states (weights
    sqrt(players), so better-measured later states count more) gives a
    slope; the forecast is *anchored* at the last observed logit and walks
    forward with a shrunk, clipped slope. Anchoring keeps one bad early
    state from moving the level; shrinkage (n/(n+3)) keeps a slope fitted
    on two or three states from being trusted like one fitted on fifty.

3.  Earners / players decomposition. Players tracked P(t) is extrapolated
    by the median of three simple fits (log-log, log-linear on the recent
    half, linear-in-log-time). Earners E_i(t) = rate_i * P / 100 are
    extrapolated linearly in ln(1 + hours), anchored at the last state and
    never allowed to fall or to exceed the tracked base. The implied rate
    E_i(target) / P(target) is the decomposition view: it separates real
    progress (earners rising) from dilution (new tracked players arriving
    with almost no trophies, which pushes every rate down).

4.  Blend. Median logit = 0.7 * attainment view + 0.3 * decomposition view
    when both exist, else whichever exists; with fewer than two usable
    states it is persistence (the last rate).

5.  Archetype adjustment. Trophies are classified from their description
    (story completion, collectible grind, skill/challenge, exploration,
    platinum, other). Grinds and the platinum earn slowly and late, so
    their slopes are damped and their uncertainty widened; story trophies
    keep their slope. This is a prior, applied gently (a multiplier on the
    slope and on sigma), never a hard-coded rate.

6.  Uncertainty. Sigma (logit units) combines a random-walk term
    0.42 * sqrt(horizon days), the fit's residual scale inflated by the
    extrapolation leverage, and the observed jump size between consecutive
    refresh states (a refresh the model has not seen is the main risk, as
    the practice cases show). When fewer than three states exist the model
    has never seen a refresh, so sigma is floored at 1.0. Sigma is clipped
    to [0.30, 2.40] and quantiles are read off a normal in logit space,
    clamped to [0.01, 99.99] percent and rounded to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs python3 -P -s: keep this file's directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

EPS_LO = 0.01
EPS_HI = 99.99


def clamp_rate(x):
    return min(EPS_HI, max(EPS_LO, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------- archetypes

def archetype(trophy):
    """Classify a trophy from its tier and description keywords."""
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = desc + " " + name
    if trophy.get("tier") == "platinum":
        return "platinum"
    if any(k in text for k in ("every", "all of", "collect", "find all", "complete all")):
        return "grind"
    if any(k in text for k in ("win", "beat", "race", "challenge", "without", "under ")):
        return "skill"
    if any(k in text for k in ("complete", "finish", "story", "mission")):
        return "story"
    if any(k in text for k in ("visit", "explore", "discover", "travel", "reach")):
        return "explore"
    return "other"


# Slope damping and sigma widening per archetype: grinds and the platinum
# are earned late and unevenly; story trophies track playtime most cleanly.
ARCH = {
    "platinum": (0.55, 1.35),
    "grind": (0.80, 1.20),
    "skill": (0.90, 1.10),
    "story": (1.00, 1.00),
    "explore": (0.95, 1.05),
    "other": (0.95, 1.05),
}


# ------------------------------------------------------------------- fitting

def wls_slope(xs, ys, ws):
    """Weighted least squares; returns (slope, intercept, resid_sd, n_eff)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return 0.0, (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return 0.0, my, 0.0, float(len(xs))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1.0, len(xs) - 2.0)
    resid_sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / max(sw, 1e-9) * len(xs) / dof) if len(xs) > 2 else 0.0
    n_eff = sw * sw / sum(w * w for w in ws)
    return slope, intercept, resid_sd, n_eff


def ols_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def forecast_players(states, x_last, x_target):
    """Extrapolate players_tracked to the target; median of three fits."""
    xs = [s[0] for s in states]
    ps = [float(s[1]) for s in states]
    p_last = ps[-1]
    if len(states) < 3:
        return p_last
    preds = []
    # log-log: ln P against ln(1+t)
    lx = [math.log1p(x) for x in xs]
    ly = [math.log(max(1.0, p)) for p in ps]
    b = ols_slope(lx, ly)
    a = ly[-1] - b * lx[-1]
    preds.append(math.exp(a + b * math.log1p(x_target)))
    # log-linear on the recent half of states
    half = max(3, len(states) // 2)
    b2 = ols_slope(xs[-half:], ly[-half:])
    a2 = ly[-1] - b2 * xs[-1]
    preds.append(math.exp(a2 + b2 * x_target))
    # linear in log time
    b3 = ols_slope(lx, ps)
    a3 = ps[-1] - b3 * lx[-1]
    preds.append(a3 + b3 * math.log1p(x_target))
    preds.sort()
    p_med = preds[len(preds) // 2]
    return min(max(p_med, p_last), p_last * 12.0)


# -------------------------------------------------------------------- driver

def unique_states(history):
    """Collapse consecutive identical reads into refresh states."""
    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    states = unique_states(history)
    t0 = parse_time(states[0]["slot"])
    xs = [(parse_time(s["slot"]) - t0).total_seconds() / 3600.0 for s in states]
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = (target - t0).total_seconds() / 3600.0
    x_last = xs[-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    lx = [math.log1p(x) for x in xs]
    lx_last = lx[-1]
    lx_target = math.log1p(x_target)
    dlx = max(0.0, lx_target - lx_last)

    players = [float(s.get("players_tracked") or 0) for s in states]
    p_target = forecast_players(list(zip(xs, players)), x_last, x_target)
    p_last = players[-1] if players else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        arch = archetype(trophy)
        damp, widen = ARCH[arch]
        # Series over states, skipping reads that lacked this trophy.
        sx, sy, sw, se = [], [], [], []
        last_rate = None
        for j, s in enumerate(states):
            rates = s.get("rates") or []
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            last_rate = r
            sx.append(lx[j])
            sy.append(logit(r))
            sw.append(math.sqrt(max(1.0, players[j])))
            se.append(r / 100.0 * players[j])
        if last_rate is None:
            last_rate = 0.0
        last_logit = logit(last_rate)

        if len(sx) >= 2 and dlx > 0:
            slope, _intercept, resid_sd, n_eff = wls_slope(sx, sy, sw)
            shrink = n_eff / (n_eff + 3.0)
            slope = max(-1.5, min(2.5, slope)) * shrink * damp
            view_direct = last_logit + slope * dlx
            # Decomposition view: earners linear in log time, anchored.
            if p_last > 0 and p_target > 0 and len(se) >= 2:
                e_slope = ols_slope(sx, se) * shrink * damp
                e_pred = se[-1] + max(0.0, e_slope) * dlx
                e_pred = min(e_pred, 0.999 * p_target)
                e_pred = max(e_pred, se[-1])
                view_decomp = logit(100.0 * e_pred / p_target)
                median_logit = 0.7 * view_direct + 0.3 * view_decomp
            else:
                median_logit = view_direct
                resid_sd = resid_sd
            # Jump scale between consecutive states (refresh risk).
            diffs = [abs(sy[k] - sy[k - 1]) for k in range(1, len(sy))]
            jump = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
            leverage = 1.0 + dlx / max(0.5, (sx[-1] - sx[0]) if len(sx) > 1 else 1.0)
            sigma = math.sqrt(
                (0.42 * math.sqrt(horizon_days)) ** 2
                + (resid_sd * math.sqrt(leverage)) ** 2
                + (0.8 * jump) ** 2
            )
        elif len(sx) == 1 or dlx == 0:
            median_logit = last_logit
            sigma = math.sqrt((0.42 * math.sqrt(horizon_days)) ** 2 + 1.0 ** 2)
        else:
            median_logit = last_logit
            sigma = 1.0
        if len(sx) < 3:
            sigma = max(sigma, 1.0)
        # Boundary rates carry less information than they appear to: a rate
        # printed as 100.00 (or 0.00) comes from a clamped logit, and on a
        # small tracked base it is usually a pre-refresh artifact that the
        # next refresh dilutes hard. Cap the median's logit at the 99% /
        # 1% points and widen the band, so the quantiles stay honest.
        if last_rate >= 99.9 or last_rate <= 0.1:
            median_logit = max(logit(1.0), min(logit(99.0), median_logit))
            sigma = max(sigma, 1.5)
        elif last_rate >= 99.0 or last_rate <= 0.5:
            median_logit = max(logit(0.5), min(logit(99.5), median_logit))
            sigma = max(sigma, 0.9)
        sigma = min(2.40, max(0.30, sigma * widen))
        qs = []
        for z in Z:
            q = inv_logit(median_logit + z * sigma)
            qs.append(round4(min(EPS_HI, max(EPS_LO, q))))
        qs.sort()
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
