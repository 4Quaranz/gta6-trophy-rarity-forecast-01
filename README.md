# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/4Quaranz/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/4Quaranz/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/competitions/gta6-trophy-rarity-forecast-01):
forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%,
75%, 95%) per trophy, scored by pinball loss on the logit scale.

## Program

`main.py` — Python 3.13, standard library only, offline, deterministic.
It reads every `<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method

All in logit space, per trophy:

1. **Dedup** — reads that repeat the page exactly carry no new
   information; only the first read of each distinct page state is kept.
2. **Separate log-time fits** — log(earners) and log(players tracked)
   against `ln(1 + hours since the first read)`; the rate forecast is
   the ratio of the two extrapolations at the target.
3. **Slope shrinkage** — each trophy's log-rate slope (earners slope
   minus players slope) is shrunk toward the cross-trophy median rate
   slope, empirical-Bayes weighted by its own standard error. The
   universal trophy (last rate 100%, held by every tracked player) is
   exempt: its rate is pinned at 100% by construction.
4. **Refresh-dilution prior** — Exophase refreshes a game's counts on
   its own schedule, hours apart, and a refresh brings in a wave of new
   players who have earned almost nothing yet except the universal
   trophy, so rates dilute. A history that has never moved cannot price
   that, so while the deduped history is uninformative the median is
   shifted down in logit space; the shift washes out (`1 / (1 + distinct
   reads)`) as real reads accumulate.
5. **Spread** — logit sigma combines, in quadrature, a random walk
   (`0.5 * sqrt(horizon in days)`), the residual spread of the trophy's
   log-rate fit, slope uncertainty times the extrapolation distance
   (with a prior standard error when the history cannot identify a
   slope at all), and a discrete-refresh term.

## Practice score

Scored locally against the practice answers reconstructed from the
facts the kit publishes (04:30Z refresh: 3,868 players tracked, 4,276
trophies earned, every rate but the one at 100% lower; the 100% trophy
absorbed 3,230 of the 3,333 new earnings). Three splits of the remaining
103 new earnings (proportional to old earners, uniform, proportional to
sqrt(old earners)) agree closely:

| | pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| House baseline (starter) | 1.393 | 0.68 pp | 0.00 |
| This entry | **0.101** | 0.01 pp | 0.98 |

Two runs on the same case write byte-identical output; a run over both
practice cases takes about 0.5 s (limit 8 s per case).

## License

MIT — see [LICENSE](LICENSE).
