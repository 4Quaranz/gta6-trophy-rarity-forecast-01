# GTA VI Trophy Rarity Forecast

[![Trophy Forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/4Quaranz/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/4Quaranz/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250, Top 20 paid): forecast every GTA VI PS5 trophy's Exophase earn rate
at 2026-11-26T00:00:00Z as five quantiles per trophy, scored by pinball loss
on the logit scale.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the
input directory and writes `<id>.out` per case. Deterministic, ~0.1 s per
run (cap 8 s).

## Method

`main.py` fits the model from the case's own history at run time, in logit
space, per trophy. No live value is hard-coded: the only numbers fixed in
the code are structural priors (tier ceilings, shrinkage pull, the
flat-history shift), and every rate forecast is computed from the history
the grader hands the program.

- **Earners vs players, separately.** A trophy's rate is
  earners / players-tracked, and the two move differently at launch:
  players-tracked explodes (dilution pushes rates down) while earners
  catch up as players progress (progression pushes rates up). The
  forecaster fits a log-time line to log earners and one to log
  players-tracked, and averages their implied rate with a direct line fit
  of logit(rate) on log time since release.
- **Shrinkage.** Each trophy's slopes are shrunk toward the cross-trophy
  median slope (three pseudo-observations of pull), because one trophy's
  short, steppy history is noisy and trophies share the launch's shape.
- **De-duplication.** Reads that repeat the previous read exactly are one
  observation, not many: Exophase refreshes hours apart, so flat stretches
  must not dominate the fit.
- **Spread.** Quantile width grows with the horizon, the fit's residual
  variance, and slope uncertainty at the target.
- **Flat-history prior.** A wholly flat history has seen no refresh, so the
  next one brings less-progressed players and rates fall: the centre shifts
  down 1.5 logit and the band widens. (Calibrated on the launch-night
  refresh, where the typical trophy fell ~1.5 logit.)
- **Ceilings.** Loose per-tier caps on the median (platinum 12%, gold 30%,
  silver 55%, bronze 99%) stop runaway log-time extrapolation.

## Backtests (local, see `backtest.py` in the workdir)

Practice histories are completely flat, and the kit carries no practice
answers, so the practice cases can only check format — they cannot be
scored locally. Two scored backtests instead:

| Backtest | Mine | House persistence |
|---|---|---|
| Live page 2026-11-19T14:54Z, trained on p00 (11.9 h horizon) | **0.7413** | 2.3691 |
| Synthetic 2-day steppy history, 4.5-day horizon truth | **0.3584** | 0.5256 |

Live backtest: typical miss 0.177 pp (baseline 0.757), 90% coverage 0.98
(baseline 0.05). Synthetic: typical miss 1.121 pp (baseline 2.667),
coverage 0.98. Both beat persistence; the entry also beats the naive
trend baseline, which equals persistence on a flat history.

## Files

- `main.py` — the whole entry (single entrypoint; `main.ts` removed).
- `LICENSE` — MIT (from the starter kit).
