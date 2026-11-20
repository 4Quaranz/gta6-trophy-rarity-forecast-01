"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the
case's target, from the house's reads of the trophy page before the case's
checkpoint. Python 3.13, standard library only, offline, deterministic.

Method (logit space throughout)
-------------------------------
1.  Dedup: reads that repeat the page exactly (same players tracked and the
    same rates) carry no new information, so only the first read of each
    distinct page state is kept, at the time it was first seen.
2.  Separate log-time fits: for each trophy, log(earners) against
    x = ln(1 + hours since the first read), and log(players tracked)
    against the same x.  Earners come from the printed rate times the
    players tracked, so the rate forecast is the ratio of the two
    extrapolations, read at the target's x.  Log time (not linear time)
    because launch growth is concave: the player flood and the trophy
    earning both saturate in log time.
3.  Slope shrinkage: what is shrunk is each trophy's log-rate slope
    (its log-earners slope minus the log-players slope), toward the
    cross-trophy median rate slope, with an empirical-Bayes weight from
    its own standard error and the cross-trophy dispersion of rate
    slopes (floored, so a history that shows no variation at all does
    not pretend the slopes are known to be equal).  Shrinking the rate
    slope -- rather than the earners slope alone -- keeps a trophy
    whose earners genuinely track the player base (a flat rate) from
    being dragged down with the diluting median; the universal trophy
    (below) is exempt from shrinkage altogether.  The players slope is
    a single series and is not shrunk.
4.  Refresh-dilution prior: Exophase refreshes a game's counts on its own
    schedule, hours apart, and a refresh brings in a wave of new players
    who have earned almost nothing yet except the trophy everybody gets
    at once (the one the page prints at 100%).  A history that has never
    moved cannot price that: its fitted slopes are all zero with zero
    residuals, which would claim a certainty the data does not support.
    So when the deduped history is uninformative, the median is shifted
    down in logit space by DILUTE * w, where w = 1 / (1 + distinct reads)
    washes the prior out as real reads accumulate, and the shift is
    skipped for the universal trophy (last rate 100%), which a refresh
    leaves at 100%.  With a rich history w is ~0 and the fit rules alone.
5.  Spread: the logit standard deviation combines, in quadrature,
      - a random walk, 0.5 * sqrt(horizon in days) (the house baseline's
        scale),
      - the residual spread of the trophy's log-rate fit around its own
        log-time line (a refresh lands as a step, so steps widen this),
      - slope uncertainty: the shrunk slope standard errors, earners and
        players combined, times the extrapolation distance |x_target -
        x_mean|.  When the history cannot identify a slope at all (fewer
        than 3 distinct reads) a prior standard error of SLOPE_SE_PRIOR
        is used instead of the (meaningless) zero the flat fit reports,
      - a refresh-jump term, REFRESH_SIGMA * w, for the discrete-refresh
        risk a flat history hides.
    Quantiles are median_logit + z_tau * sigma, mapped back with the
    inverse logit, rounded to 4 decimals (half up), clamped to
    [0.01, 99.99] and non-decreasing.

Nothing is random, nothing reads the network, and the input files are
processed in name order, so two runs write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- Tunable constants (fixed in advance; see the docstring) --------------
# Logit shift a refresh costs a non-universal trophy when the history has
# never shown one (w = 1 there is 0.5 on the practice cases, so the shift
# applied is DILUTE / 2).
DILUTE = 3.0
# Prior standard error of a log-time slope (per unit of x) when the history
# cannot identify it.
SLOPE_SE_PRIOR = 0.25
# Floor on the cross-trophy dispersion of slopes used for shrinkage.
TAU_FLOOR = 0.3
# Random-walk spread, logit units per square-root day.
SIGMA_RW = 0.5
# Extra logit spread for discrete-refresh risk, scaled by w.
REFRESH_SIGMA = 0.3
# Floor / cap on the total logit spread.
SIGMA_MIN = 0.2
SIGMA_MAX = 2.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Acklam's approximation to the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b x.  Returns (a, b, resid_std, se_b, x_mean,
    n).  resid_std and se_b are 0.0 when the fit is exact or degenerate."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return my, 0.0, 0.0, 0.0, mx, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0, 0.0, mx, n
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    dof = n - 2
    resid_var = rss / dof
    resid_std = math.sqrt(resid_var)
    se_b = math.sqrt(resid_var / sxx) if sxx > 0 else 0.0
    return a, b, resid_std, se_b, mx, n


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])

    # --- Dedup identical reads, keeping each distinct page state at the
    # time it was first seen. ---
    seen = set()
    reads = []
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key in seen:
            continue
        seen.add(key)
        reads.append(read)
    n_distinct = len(reads)
    w = 1.0 / (1.0 + n_distinct)  # prior weight: 1/2 on a flat history

    if n_distinct == 0:
        # No history at all: say 1% for everything, wide and honest.
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append({"id": trophy["id"], "q": [
                round4(clamp(inv_logit(lg + norm_ppf(t) * 1.0))) for t in LEVELS]})
        return {"trophies": rows}

    t0 = parse_time(reads[0]["at"])
    xs = []
    players = []
    for read in reads:
        hours = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
        xs.append(math.log1p(max(hours, 0.0)))
        players.append(max(float(read.get("players_tracked") or 0), 1.0))
    x_target = math.log1p(max((target - t0).total_seconds() / 3600.0, 0.0))
    x_mean = sum(xs) / len(xs)
    extrap = abs(x_target - x_mean)

    # --- Players fit (one series), pivoted at x_mean so the forecast
    # reads off the series' own mean level. ---
    log_p = [math.log(p) for p in players]
    a_p, b_p, s_p, se_p, _, _ = ols(xs, log_p)
    if n_distinct < 3:
        se_p = SLOPE_SE_PRIOR
    else:
        se_p = max(se_p, 0.08)
    log_p_mean = sum(log_p) / len(log_p)

    # --- Per-trophy earners fits, and the log-rate fit each implies.
    # The forecasted quantity is the rate, so what gets shrunk is the
    # rate's slope d_i = b_earners,i - b_players: the typical trophy's
    # rate dilutes as new players flood in, but a trophy whose earners
    # genuinely track the player base (its rate flat) must not be dragged
    # down with the median.  The log-rate residuals are measured directly
    # from the per-read log rates.
    n_tr = len(trophies)
    fits = []
    d_slopes = []
    for i in range(n_tr):
        ys = []
        lr = []
        for j, read in enumerate(reads):
            rate = read["rates"][i] if i < len(read.get("rates") or []) else None
            if rate is None:
                rate = 0.0
            earners = max(rate, 0.0) / 100.0 * players[j]
            ys.append(math.log(max(earners, 0.5)))
            lr.append(math.log(max(earners, 0.5) / players[j]))
        a_e, b_e, s_e, se_e, _, _ = ols(xs, ys)
        if n_distinct < 3:
            se_e = SLOPE_SE_PRIOR
        else:
            se_e = max(se_e, 0.08)
        _, _, s_r, _, _, _ = ols(xs, lr)
        fits.append((a_e, b_e, s_e, se_e, sum(ys) / len(ys), s_r))
        d_slopes.append(b_e - b_p)

    # --- Shrink rate slopes toward their cross-trophy median. ---
    med_d = median(d_slopes)
    if n_tr > 1:
        mean_d = sum(d_slopes) / n_tr
        var_d = sum((d - mean_d) ** 2 for d in d_slopes) / (n_tr - 1)
        mean_se2 = sum(f[3] ** 2 for f in fits) / n_tr + se_p ** 2
        tau2 = max(var_d - mean_se2, TAU_FLOOR ** 2)
    else:
        tau2 = TAU_FLOOR ** 2
    tau = math.sqrt(tau2)

    # Cross-trophy median residual spread of the log-rate fit, for
    # trophies whose own history is too short to measure one.
    resid_rate = [f[5] for f in fits]
    med_resid = median(resid_rate)

    rows = []
    for i, trophy in enumerate(trophies):
        _, _, _, se_e, y_mean, s_r = fits[i]
        # Last observed rate (for the universal-trophy rule).
        last_rate = None
        for read in reversed(reads):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 1.0
        universal = last_rate >= 99.995

        # Shrink this trophy's rate slope toward the cross-trophy median
        # rate slope -- unless it is the universal trophy, whose rate is
        # pinned at 100% by construction and whose own slope (about zero)
        # is the whole story.
        se_d = math.sqrt(se_e ** 2 + se_p ** 2)
        lam = tau2 / (tau2 + se_d ** 2)
        if universal:
            d_shrunk = d_slopes[i]
            se_d_shrunk = se_d
        else:
            d_shrunk = med_d + lam * (d_slopes[i] - med_d)
            se_d_shrunk = se_d * math.sqrt(lam)
        # Median log rate: the series' own mean level, moved along the
        # (shrunk) rate slope to the target.
        med_logit = (y_mean - log_p_mean) + d_shrunk * (x_target - x_mean)
        # Convert ln(rate fraction) to logit; for small rates they agree,
        # for large ones the exact form matters.
        rate_med = 100.0 * math.exp(med_logit)
        med_logit = logit(rate_med)

        # Refresh-dilution prior: only when the history is uninformative,
        # and never for the trophy everybody already holds.
        if not universal:
            med_logit -= DILUTE * w

        # --- Spread (logit sigma). ---
        sig_rw = SIGMA_RW * math.sqrt(horizon_days)
        s_rate = resid_rate[i]
        if n_distinct < 5:
            s_rate = math.sqrt((n_distinct * s_rate ** 2 + 3 * med_resid ** 2)
                               / (n_distinct + 3))
        sig_slope = se_d_shrunk * extrap
        sig_refresh = REFRESH_SIGMA * w
        sigma = math.sqrt(sig_rw ** 2 + s_rate ** 2 + sig_slope ** 2
                          + sig_refresh ** 2)
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))

        qs = []
        prev = None
        for level in LEVELS:
            q = round4(clamp(inv_logit(med_logit + norm_ppf(level) * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
