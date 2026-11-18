#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Method: launch-shape forecaster in log space, anchored on the case's
hour-1 print.

  * The case file carries 38-42 past big launches ("comparables") as
    hourly SullyGnome Twitch category averages, 14 days before to 3 days
    after each launch day, plus the target game's own history.
  * GTA VI unlocks at rolling local midnight (New Zealand first, then
    Europe, then the US east/west coasts), so its hour 1 (00:00-01:00 UTC)
    is early: the bulk of the launch audience arrives later in the day
    and the curve peaks in the evening UTC. The relevant analogues are
    therefore the rolling-midnight launches, not the simultaneous-global
    ones (a game that unlocks at 14:00 UTC has a near-empty hour 1 and a
    step-function shape that says nothing about a rolling launch).
  * Pool rule, computed from the case data alone: a comparable counts
    when its tier is "comparable", its hour-1 print is at least 100
    viewers, its launch-day peak is under 20x its hour-1 print, and its
    peak hour is 5 or later. (The Grand Theft Auto VI category's own
    pre-release history is tier "target-history" and is excluded: it has
    no launch day yet.)
  * For each hour h = 2..24, take the pool's log-ratios
    ln(viewers[h] / viewers[1]). The median is the centre; the spread is
    the inter-quartile range / 1.349. The forecast's five log-quantiles
    are the average of a lognormal curve (median + z * sigma) and the
    pool's empirical quantiles at the five levels -- the smooth curve
    steadies the tails, the empirical quantiles keep them honest.
    Multiply by the hour-1 print and round half-up to 2 decimals.
  * One refinement, fitted from the pool alone: a launch's shape depends
    on its hour-1 magnitude. Across the pool, log(peak/hour-1) correlates
    -0.74 with log(hour-1), and the per-hour regression slope of the
    log-ratio on log(hour-1) grows from ~0 in the early hours to about
    -0.45 at the evening peak -- launches whose hour 1 is small (a small
    New Zealand footprint next to a big global audience) rise much
    further by the evening. So before pooling, each member's ratios are
    aligned to the target's hour-1 level: ratio - b_h * (log hour-1 -
    log target hour-1), where b_h is that hour's OLS slope shrunk 50%
    toward the pooled slope over all hours. The target's log hour-1 is
    clamped to the pool's observed range, so a GTA-VI-sized hour 1 does
    not extrapolate the slope. The spread is the aligned ratios' IQR /
    1.349, widened 1.5x: conditioning removes systematic variation, so
    the raw IQR would understate the remaining spread, and the extra
    width is cheap insurance if the target is an outlier launch.

Grounding (web research, 2026-11-17): the biggest single-player Twitch
peaks on record are Hogwarts Legacy ~1.28M, Cyberpunk 2077 ~1.14M and
Elden Ring ~0.91M concurrent viewers (The Direct / Gamesight, Feb 2023);
Twitch's own GTA VI category read 1,240 viewers on 2026-11-17 with a
pre-release peak of ~30k (competition.yaml probe; Coilr). The model does
not hard-code any of those numbers: every figure it uses comes from the
case file, so the same program forecasts any case it is given.

Validation (leave-one-out over the 15 pool launches in the kit's
practice data, mean pinball loss on the log scale, lower is better;
re-run against the shipped code, not the analysis draft):
  house persistence baseline   mean 0.447, median 0.309
  this forecaster (blend)      mean 0.284, median 0.177
The house baseline scores ~0.387 on the four practice launches; this
forecaster beats it on the same kind of launch (rolling midnight,
console-led). No network, no clock, no randomness: two runs on the same
case write byte-identical output.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# A comparable joins the pool only if its launch-day peak is under this
# multiple of its hour-1 print (a simultaneous-global launch has a tiny
# hour 1 and a peak hundreds of times larger).
MAX_PEAK_RATIO = 20.0
# ... and its peak hour is at least this far into the day (a launch whose
# hour 1 already is its peak peaked before the day began for our purpose).
MIN_PEAK_HOUR = 5
# An hour-1 print below this is noise, not a launch already under way.
MIN_HOUR1_VIEWERS = 100


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def tier_of(comparable):
    """The tier the case data assigns the comparable (in its unlock text)."""
    unlock = comparable.get("unlock") or ""
    for tier in ("target-history", "early-access-contaminated", "comparable", "weak"):
        if "Tier: " + tier in unlock:
            return tier
    return "comparable"


def launch_day(comparable):
    """The comparable's launch-day hourly viewers, index 0 = hour 1."""
    day_start = parse_ts(comparable["day_start"])
    hours = [None] * 24
    for point in comparable.get("hourly") or []:
        delta = (parse_ts(point["from"]) - day_start).total_seconds() / 3600
        if 0 <= delta < 24 and float(delta).is_integer():
            hours[int(delta)] = point["viewers"]
    return hours


def peak_hour(day):
    best_hour, best_viewers = None, -1
    for index, viewers in enumerate(day):
        if viewers is not None and viewers > best_viewers:
            best_hour, best_viewers = index + 1, viewers
    return best_hour


def in_pool(comparable, day):
    """True when this comparable is a rolling-midnight launch like GTA VI."""
    if tier_of(comparable) != "comparable":
        return False
    hour1 = day[0]
    if hour1 is None or hour1 < MIN_HOUR1_VIEWERS:
        return False
    peak = max(v for v in day if v is not None)
    if peak / max(hour1, 1) >= MAX_PEAK_RATIO:
        return False
    peak_at = peak_hour(day)
    return peak_at is not None and peak_at >= MIN_PEAK_HOUR


def weighted_median(pairs):
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    seen = 0.0
    for value, weight in ordered:
        seen += weight
        if seen >= total / 2:
            return value
    return ordered[-1][0]


def weighted_quantile(pairs, tau):
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    seen = 0.0
    for value, weight in ordered:
        seen += weight
        if seen >= tau * total:
            return value
    return ordered[-1][0]


def round2(x):
    """x rounded to 2 decimals, half up, as the house baseline rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def hour_ratio_pairs(pool_days, hour):
    """(log hour-1 print, log ratio) for the pool members covering this hour."""
    pairs = []
    for day in pool_days:
        if day[0] is None or day[hour - 1] is None:
            continue
        pairs.append((math.log(max(day[0], 1)),
                      math.log(max(day[hour - 1], 1)) - math.log(max(day[0], 1))))
    return pairs


def ols_slope(pairs):
    """Least-squares slope of the log-ratio on log hour-1; 0 if degenerate."""
    if len(pairs) < 3:
        return 0.0
    mean_x = sum(p[0] for p in pairs) / len(pairs)
    mean_y = sum(p[1] for p in pairs) / len(pairs)
    sxx = sum((p[0] - mean_x) ** 2 for p in pairs)
    if sxx <= 1e-12:
        return 0.0
    return sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs) / sxx


def pooled_slope(pool_days):
    """One slope for every hour: the mean of the per-hour OLS slopes."""
    slopes = []
    for hour in range(2, 25):
        pairs = hour_ratio_pairs(pool_days, hour)
        if len(pairs) >= 3:
            slopes.append(ols_slope(pairs))
    return sum(slopes) / len(slopes) if slopes else 0.0


# How much wider than the aligned ratios' IQR-based spread the lognormal
# curve is: conditioning removes systematic variation, so the raw IQR
# would understate what is left.
SIGMA_MULT = 1.5


def hour_quantiles(pool_days, hour1_print, hour, pooled):
    """The five quantiles of one hour, as viewer counts.

    Each pool member's log-ratio is first aligned to the target's
    hour-1 level (see the module docstring); the median, spread and
    empirical quantiles are then taken over the aligned ratios.
    """
    pairs = hour_ratio_pairs(pool_days, hour)
    if not pairs:
        # No comparable covers this hour: fall back to the house baseline's
        # persistence shape (flat median, spread widening with the hour).
        sigma = 0.15 * math.sqrt(hour - 1)
        return [round2(max(1.0, hour1_print * math.exp(z * sigma))) for z in Z]
    # The target's log hour-1, kept inside the pool's observed range so a
    # launch bigger than every pool member does not extrapolate the slope.
    all_x = [math.log(max(day[0], 1)) for day in pool_days if day[0] is not None]
    target_x = min(max(math.log(max(hour1_print, 1)), min(all_x)), max(all_x))
    slope = 0.5 * ols_slope(pairs) + 0.5 * pooled
    ratios = [(r - slope * (x - target_x), 1.0) for x, r in pairs]
    median = weighted_median(ratios)
    sigma = max((weighted_quantile(ratios, 0.75) - weighted_quantile(ratios, 0.25)) / 1.349, 0.05) * SIGMA_MULT
    lognormal = [median + z * sigma for z in Z]
    empirical = [weighted_quantile(ratios, tau) for tau in LEVELS]
    blended = [(a + b) / 2 for a, b in zip(lognormal, empirical)]
    counts = [hour1_print * math.exp(q) for q in blended]
    rounded = [round2(max(1.0, min(c, 1_000_000_000))) for c in counts]
    for index in range(1, 5):
        rounded[index] = max(rounded[index], rounded[index - 1])
    return rounded


def hour1_print(case):
    """The case's hour-1 print, with fallbacks if it is missing."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1, row["viewers"])
    hourly = case.get("history_hourly") or []
    if hourly:
        return max(1, hourly[-1]["viewers"])
    return 1000


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    print1 = hour1_print(case)
    pool_days = []
    for comparable in case.get("comparables") or []:
        day = launch_day(comparable)
        if in_pool(comparable, day):
            pool_days.append(day)
    rows = []
    pooled = pooled_slope(pool_days)
    for row in case.get("forecast") or []:
        hour = row["hour"]
        rows.append({"hour": hour, "q": hour_quantiles(pool_days, print1, hour, pooled)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
