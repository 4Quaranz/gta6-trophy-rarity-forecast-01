"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles, working in logit space. The model is fitted from the case's own
history at run time; nothing is fetched and no value is hard-coded from any
live page.

Idea, per trophy:
  * A trophy's rate is earners / players-tracked. Both counts move, and
    they move differently: players-tracked explodes at launch (dilution
    pushes rates down) while earners catch up as players progress
    (progression pushes rates up). So the forecaster fits two separate
    log-time lines -- log earners and log players-tracked -- and combines
    their ratio with a direct line fit of logit(rate) on log time.
  * Slopes are shrunk toward the cross-trophy median slope, because a
    single trophy's short, steppy history is noisy and trophies share the
    launch's overall shape.
  * Reads that repeat the previous read exactly (Exophase refreshes hours
    apart, so long flat stretches are one observation, not many) are
    de-duplicated before fitting, so flat stretches cannot dominate.
  * Quantile spread grows with the horizon, the fit's residual variance,
    and the slope's uncertainty, and is much wider when the whole history
    is flat (an uninformative history means a refresh is coming unseen).
  * Medians are capped by a loose per-tier ceiling (platinum lowest,
    bronze highest) so a steep log-time extrapolation cannot run away.

Standard library only. Deterministic: same input bytes give same output
bytes. Reads every *.in in MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out per case to MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Loose ceiling (percent) for a trophy's median rate, by tier. These only
# stop runaway extrapolation; ordinary forecasts sit far below them.
TIER_CEILING = {"platinum": 12.0, "gold": 30.0, "silver": 55.0, "bronze": 99.0}
TIER_FALLBACK = {"platinum": 0.05, "gold": 0.2, "silver": 0.5, "bronze": 1.0}


def clamp_pct(x):
    """Clamp a percent into the range the logit can handle."""
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lp):
    if lp >= 0:
        e = math.exp(-lp)
        return 100.0 / (1.0 + e)
    e = math.exp(lp)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median_of(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def fit_line(xs, ys):
    """Plain least-squares y = a + b*x.

    Returns (a, b, resid_var, x_mean, sxx, n). With fewer than two distinct
    x values the slope is 0 and the intercept is the mean of y.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if n < 2 or sxx <= 1e-12:
        resid = sum((y - y_mean) ** 2 for y in ys) / n
        return y_mean, 0.0, resid, x_mean, sxx, n
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = y_mean - b * x_mean
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, resid, x_mean, sxx, n


def distinct_reads(history):
    """Drop reads that repeat the previous kept read exactly.

    Exophase refreshes hours apart, so a run of identical reads is one
    observation of a stale page, not many independent ones.
    """
    kept = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            kept.append(read)
            prev_key = key
    return kept


def forecast_case(case):
    trophies = case["trophies"]
    history = distinct_reads(case.get("history") or [])
    horizon_days = case["horizon_minutes"] / 1440.0

    release = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        release_dt = datetime.fromisoformat(release + "T00:00:00+00:00")
    except ValueError:
        release_dt = parse_time(history[0]["at"]) if history else datetime(
            2026, 11, 19, tzinfo=timezone.utc)
    target_dt = parse_time(case["target"])
    hours_to_target = max(
        0.0, (target_dt - release_dt).total_seconds() / 3600.0)
    x_target = math.log1p(hours_to_target)

    # Per-trophy series, plus the shared players-tracked series.
    series = []  # one entry per trophy: dict with xs, ys, log_e, last_rate
    p_xs, p_ys = [], []
    for idx in range(len(trophies)):
        xs, ys, log_e = [], [], []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            rate = rates[idx]
            last_rate = rate
            at = parse_time(read["at"])
            hours = max(0.0, (at - release_dt).total_seconds() / 3600.0)
            xs.append(math.log1p(hours))
            ys.append(logit_of(rate))
            players = read.get("players_tracked") or 0
            earners = max(1.0, rate / 100.0 * players) if players else 1.0
            log_e.append(math.log(earners))
        series.append({"xs": xs, "ys": ys, "log_e": log_e,
                       "last": last_rate})
    for read in history:
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        at = parse_time(read["at"])
        hours = max(0.0, (at - release_dt).total_seconds() / 3600.0)
        p_xs.append(math.log1p(hours))
        p_ys.append(math.log(players))

    # First pass: raw slopes, so the medians below are the shrinkage target.
    raw_rate_slopes = []
    raw_earn_slopes = []
    fits = []
    for s in series:
        a, b, resid, xm, sxx, n = fit_line(s["xs"], s["ys"])
        ae, be, resid_e, _, _, _ = fit_line(s["xs"], s["log_e"])
        raw_rate_slopes.append(b)
        raw_earn_slopes.append(be)
        fits.append({"a": a, "b": b, "resid": resid, "xm": xm, "sxx": sxx,
                     "n": n, "ae": ae, "be": be, "resid_e": resid_e})
    med_rate_slope = median_of(raw_rate_slopes)
    med_earn_slope = median_of(raw_earn_slopes)

    _, p_slope, _, _, _, _ = fit_line(p_xs, p_ys)
    p_intercept = 0.0
    if p_xs:
        p_intercept = (sum(p_ys) / len(p_ys)
                       - p_slope * (sum(p_xs) / len(p_xs)))

    history_flat = True
    for s in series:
        if s["ys"] and (max(s["ys"]) - min(s["ys"]) > 1e-9):
            history_flat = False
            break

    rows = []
    for trophy, s, fit in zip(trophies, series, fits):
        last = s["last"]
        if last is None:
            last = TIER_FALLBACK.get(trophy.get("tier", "bronze"), 1.0)
        last_logit = logit_of(last)
        ceiling = TIER_CEILING.get(trophy.get("tier", "bronze"), 99.0)

        if fit["n"] >= 3:
            # Shrink this trophy's slopes toward the cross-trophy medians.
            # Three pseudo-observations' worth of pull: noisy trophies get
            # pulled hard, well-observed ones keep their own slope.
            pull = 3.0
            b = (fit["n"] * fit["b"] + pull * med_rate_slope) / (
                fit["n"] + pull)
            be = (fit["n"] * fit["be"] + pull * med_earn_slope) / (
                fit["n"] + pull)
            y_direct = fit["a"] + b * x_target
            log_e_pred = fit["ae"] + be * x_target
            log_p_pred = p_intercept + p_slope * x_target
            rate_struct = math.exp(log_e_pred - log_p_pred) * 100.0
            y_struct = logit_of(rate_struct)
            centre = 0.5 * (y_direct + y_struct)
            resid = min(max(fit["resid"], 0.0), 1.0)
            # Slope uncertainty at the target, from the direct fit.
            if fit["sxx"] > 1e-12 and fit["n"] > 2:
                se_b = math.sqrt(resid / fit["sxx"])
                slope_var = (se_b * abs(x_target - fit["xm"])) ** 2
            else:
                slope_var = 0.09
        else:
            centre = last_logit
            resid = 0.25
            slope_var = 0.25

        if history_flat:
            # A wholly flat history has seen no refresh: the next refresh
            # brings a flood of new, less-progressed players, so rates
            # fall hard. The launch-night practice cases show exactly this:
            # the first refresh cut the typical trophy by ~1.5 logit
            # (e.g. 1.10% -> 0.14%), with big trophy-to-trophy variation,
            # so the centre shifts down that far and the band widens a lot.
            # This branch only fires when the whole history is flat; a
            # real final history has variation and is driven by the fits.
            centre -= 1.5
            flat_extra = 1.5
        else:
            flat_extra = 0.0

        sigma2 = (0.45 ** 2) * horizon_days + resid + slope_var + 0.04 \
            + flat_extra ** 2
        sigma = min(2.0, max(0.40, math.sqrt(sigma2)))

        median_rate = inv_logit(centre)
        if median_rate > ceiling:
            centre = logit_of(ceiling)
            median_rate = ceiling
        if last >= 99.5 and median_rate < 99.0:
            centre = logit_of(99.0)
        if median_rate < 0.01:
            centre = logit_of(0.01)

        qs = []
        prev = -1.0
        for z in Z_SCORES:
            q = round(inv_logit(centre + z * sigma), 4)
            q = min(100.0, max(0.0, q))
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
