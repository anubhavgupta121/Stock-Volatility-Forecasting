# Forecasting Next-Week S&P 500 Volatility

ES 335 Machine Learning course project, IIT Gandhinagar
Anubhav Gupta, roll no. 25110040

**Project website:** https://anubhavgupta121.github.io/Stock-Volatility-Forecasting/

## Question

Do machine-learning models (HAR regression, random forest, XGBoost) forecast next-week S&P 500 volatility better than classical baselines (naive, rolling average, EWMA, GARCH(1,1)) on unseen 2023–2026 data, including during volatile periods?

## Result

Every model was tuned on 2019–2022 and scored once on the test years (January 2023 to September 2026). Lower is better on both metrics.

| Model | Test QLIKE | × GARCH | Test RMSE |
|---|---|---|---|
| **XGBoost, trained on the QLIKE loss** | **0.308** | **0.80** | **0.0038** |
| Random forest | 0.323 | 0.84 | 0.0038 |
| XGBoost, squared error | 0.331 | 0.86 | 0.0038 |
| HAR + range + VIX | 0.342 | 0.89 | 0.0038 |
| VIX alone, straight-line fit* | 0.3465 | 0.90 | 0.0038 |
| GARCH(1,1) | 0.384 | 1.00 | 0.0045 |
| HAR | 0.411 | 1.07 | 0.0043 |
| VIX alone, raw* | 0.425 | 1.11 | 0.0048 |
| EWMA | 0.426 | 1.11 | 0.0047 |
| Rolling (22-day) | 0.467 | 1.22 | 0.0049 |
| Naive (5-day) | 0.811 | 2.11 | 0.0049 |

\* Added after a review as an extra baseline (last cells of the notebook); no model choice changed.

Main findings:

- XGBoost trained directly on QLIKE beat GARCH by 20% on the test years, the same edge it had on validation (0.81× GARCH).
- Most of the gain came from information, not model complexity: plain linear HAR given the same inputs (the range and VIX) came within 0.034, and a straight-line fit on VIX alone beat GARCH (0.3465 vs 0.384).
- Each model wins on the metric closest to its training loss: squared-error models lead on RMSE, QLIKE-like models (GARCH, XGBoost-QLIKE) lead on QLIKE.
- The final model has a hard ceiling: it can never forecast above 4.2% a day, less than half the 2020 peak.

Checks added after review (last cells of the notebook; the headline numbers above are unchanged):

- A block bootstrap (22-day blocks, 5,000 resamples) puts XGBoost-QLIKE's gap to GARCH at 0.076 (95% interval 0.042 to 0.113) and to the random forest at 0.014 (−0.003 to 0.033), so the win over GARCH is real and the win over the forest may be luck.
- Small data leak at the split edges: the last 5 training rows before 2019 and before 2023 have targets built from the first days of the next period. Removing them (purging) moves validation and test scores by at most 0.0003.
- The final refit ran to 1,998 trees, while validation stopped at 771. Capped at the validated 772 trees, the refit scores 0.310 (0.81× GARCH).
- Walk-forward over 2019–2022 (refit on all earlier years): XGBoost-QLIKE beat GARCH in every year (0.72–0.88× GARCH), the only model that did.

The website has the full write-up: data, preprocessing, every experiment (including the rejected ones), error analysis and limitations.

## Data

- `^GSPC` (S&P 500) and `^VIX` daily prices from Yahoo Finance via `yfinance`, January 1990 to September 2026.
- No data is stored in the repository; the code downloads it.

## Target and features

The target for day *t* is the volatility of the next five trading days, using daily log returns *r*:

```
target(t) = sqrt( mean( r(t+1)², ..., r(t+5)² ) )
```

`features.py` builds every feature that was tried. The final model uses five:

| Feature | What it is |
|---|---|
| `today_vol` | size of today's move, \|r\| |
| `past_5_days` | volatility of the last 5 days |
| `past_22_days` | volatility of the last 22 days |
| `high_low_1` | today's high–low range (Parkinson estimator) |
| `vix` | VIX / 100 / √252, the market's own forecast |

Tried and dropped: `high_low_5`, `high_low_22`, `neg_return`, `vol_ratio`. Open-price features (`overnight_gap`) were excluded because Yahoo's opening prices are stale before 2008.

## Split

| Part | Years | Use |
|---|---|---|
| Train | 1990–2018 | fitting |
| Validation | 2019–2022 | every modelling choice |
| Test | 2023–Sep 2026 | scored once, after refitting on 1990–2022 |

The split is by date and never shuffled. There is no gap between the parts, so the last 5 rows before each split have targets that use the next period's first days (a small leak; removing them changes scores by at most 0.0003).

## Repository layout

| File | Contents |
|---|---|
| `features.py` | `create_features`: target and all features |
| `baselines.py` | `naive`, `rolling`, `ewm` (EWMA), `garch`, `garch_variance` |
| `models.py` | `HAR`, `HAR_LOG`, `random_forest`, `XGBOOST`, `XGBOOST_QLIKE` (custom QLIKE objective `XG_obj`), `XGBOOST_HYBRID` |
| `metrics.py` | `RMSE`, `QLIKE` |
| `notebooks/checks.ipynb` | all experiments in order: baselines, HAR, random forest, XGBoost, the QLIKE loss, feature selection, the hybrid, the final test, the ceiling check, and the checks added after review (VIX on its own, HAR variants, purged split edges, bootstrap confidence intervals, walk-forward) |
| `notebooks/test_predictions.csv`, `notebooks/val_predictions.csv` | the forecasts the notebook exports (test: every model; validation: GARCH and XGBoost-QLIKE without and with VIX) |
| `requirements.txt` | pinned library versions (Python 3.13) |

## How to reproduce

1. Install the pinned libraries (Python 3.13):

   ```
   pip install -r requirements.txt
   ```

2. Open `notebooks/checks.ipynb` from inside the `notebooks` folder (its first cell adds the parent folder to the path so it can import the `.py` files) and run the cells in order.

Models use `random_state=42`. GARCH is fitted only on data before each cutoff. Yahoo occasionally revises old prices, so re-downloaded data can shift scores slightly.

## AI use

Claude (Anthropic) was used to explain methods, give suggestions, debug code (for example early stopping, the GARCH-anchored hybrid and the final formatted test cell in the notebook), review results, build the project website, and, after a review, write the check cells at the end of the notebook, `requirements.txt` and `.gitignore`. The original experiments, modelling choices and interpretations are all mine, and every number comes from the project's notebook or is computed from the forecasts it exports.
