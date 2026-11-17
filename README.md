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
| HAR + VIX | 0.342 | 0.89 | 0.0038 |
| GARCH(1,1) | 0.384 | 1.00 | 0.0045 |
| HAR | 0.411 | 1.07 | 0.0043 |
| EWMA | 0.426 | 1.11 | 0.0047 |
| Rolling (22-day) | 0.467 | 1.22 | 0.0049 |
| Naive (5-day) | 0.811 | 2.11 | 0.0049 |

Main findings:

- XGBoost trained directly on QLIKE beat GARCH by 20% on the test years, the same edge it had on validation (0.81× GARCH).
- Most of the gain came from information, not model complexity: plain linear HAR given VIX came within 0.034.
- Each model wins on the metric closest to its training loss: squared-error models lead on RMSE, QLIKE-like models (GARCH, XGBoost-QLIKE) lead on QLIKE.
- The final model has a hard ceiling: it can never forecast above 4.2% a day, less than half the 2020 peak.

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

The split is by date and never shuffled.

## Repository layout

| File | Contents |
|---|---|
| `features.py` | `creater_features`: target and all features |
| `baselines.py` | `naive`, `rolling`, `ewm` (EWMA), `garch`, `garch_variance` |
| `models.py` | `HAR`, `HAR_LOG`, `random_forest`, `XGBOOST`, `XGBOOST_QLIKE` (custom QLIKE objective `XG_obj`), `XGBOOST_HYBRID` |
| `metrics.py` | `RMSE`, `QLIKE` |
| `notebooks/checks.ipynb` | all experiments in order: baselines, HAR, random forest, XGBoost, the QLIKE loss, feature selection, the hybrid, the final test and the ceiling check |

## How to reproduce

1. Install the libraries:

   ```
   pip install yfinance pandas numpy statsmodels scikit-learn xgboost arch matplotlib
   ```

2. Open `notebooks/checks.ipynb` from inside the `notebooks` folder (its first cell adds the parent folder to the path so it can import the `.py` files) and run the cells in order.

Models use `random_state=42`. GARCH is fitted only on data before each cutoff. Yahoo occasionally revises old prices, so re-downloaded data can shift scores slightly.

## AI use

Claude (Anthropic) was used to explain methods, give suggestions, debug code (for example early stopping, the GARCH-anchored hybrid and the final formatted test cell in the notebook), review results, and build the project website. The experiments, modelling choices and interpretations are all mine, and every number comes from the project's notebook or is computed from the forecasts it exports.
