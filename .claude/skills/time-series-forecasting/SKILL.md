---
name: time-series-forecasting
description: Predicts future values from historical temporal patterns using statistical models (ARIMA/SARIMA, exponential smoothing), machine learning, and deep learning approaches. Use when the user has time-indexed data and needs forecasts, trend or seasonality analysis, backtesting, or model selection — e.g. demand planning, revenue projection, capacity forecasting, or anomaly detection on a series.
allowed-tools: Read, Write, Edit, Grep, Glob, WebSearch, WebFetch
---

# Time Series Forecasting

You forecast future values from historical temporal patterns. Be practical, specific, and honest: a forecast that cannot be backtested is an opinion, and a point forecast with no interval is usually the wrong deliverable.

## When to use this skill
The user has time-indexed data and needs to predict it, decompose it, or decide which model to use. Typical cases: demand and inventory planning, revenue and headcount projection, capacity and traffic forecasting, budget variance, anomaly detection on a metric.

## Four non-negotiables

These are where most forecasting work goes wrong. Apply them before anything else.

1. **Beat a baseline or don't ship.** Always fit seasonal naive (and naive, and drift) first and report every model against it. A model that loses to "same weekday last week" is not a model. Most production forecasting failures are a complicated model that was never compared to a trivial one.
2. **Backtest on rolling origins, never shuffled cross-validation.** Random k-fold on a time series trains on the future to predict the past. Use expanding or sliding windows evaluated at many origins — a single train/test split tells you almost nothing.
3. **Forecast a distribution, not a point.** Report prediction intervals and check their empirical coverage. If the decision is asymmetric (stockouts cost more than overstock), the deliverable is a quantile, not a mean.
4. **Use only what is knowable at forecast time.** Every feature must be available at the moment the forecast is made, for the whole horizon. This is the single most common source of backtests that look excellent and fail in production.

## Workflow

### 1. Frame the problem
Get these before modelling — they determine the model more than anything in the data:

- **Horizon (h)** and **frequency**: 14 days ahead daily is a different problem from 3 years ahead monthly.
- **The decision it feeds.** Forecasting for an inventory buy, a hiring plan, and a board slide need different quantiles, granularities, and error metrics.
- **Granularity**: per SKU, per store, per SKU-store? Forecast at the level the decision is made, then reconcile.
- **Retraining cadence**: forecast once, or refreshed weekly? This sets the backtest design.
- **Known-future covariates**: holidays, promotions, price changes, contracted volume. These are known ahead and are usually the highest-value features.
- **Cost asymmetry**: if under- and over-forecasting cost differently, say so now. Optimal service level for a newsvendor decision is `Cu / (Cu + Co)` — forecast that quantile.

### 2. Audit the data
Do this explicitly and report what you find. Never skip to modelling.

- **Regular index**: are timestamps evenly spaced? Reindex to a full date range — implicit gaps silently become continuous series and corrupt every lag feature.
- **Missing vs. zero**: a missing day and a zero-sales day are different facts. Distinguish them before imputing.
- **Length vs. seasonality**: you need at least two full seasonal cycles to estimate seasonality, and realistically three or more.
- **Trend, seasonality, multiple seasonalities**: run an STL decomposition and look. Daily retail data often has weekly *and* annual seasonality — single-season models will not capture both.
- **Variance**: if the seasonal swing grows with the level, the pattern is multiplicative — apply a log or Box-Cox transform and model additively.
- **Stationarity**: ADF (null = unit root, i.e. non-stationary) and KPSS (null = stationary) together. They disagree usefully; if both are ambiguous, difference and re-check.
- **Outliers and level shifts**: a promotion, an outage, a pandemic. Flag them as events rather than deleting them, so the model can learn the effect.
- **Data vintage**: are historical values revised after the fact? If so, backtest on the data *as it was known then*, not on the restated series. Financial and operational data is frequently restated.
- **Intermittency**: many zeros with sporadic demand is its own regime — see `references/models.md`.

### 3. Establish baselines
Fit and score: naive (last value), seasonal naive (value from one season ago), drift, and historical mean. These take minutes and define the bar.

### 4. Pick candidate models
See `references/models.md` for the full catalogue and selection logic. Quick guide:

| Situation | Start with |
|---|---|
| One series, clear trend/seasonality, short history | ETS (exponential smoothing), ARIMA, Theta |
| One series, needs explainability | ETS or ARIMA with regressors |
| Daily business data, holidays, multiple seasonality | MSTL, TBATS, or Prophet |
| Many related series, rich covariates | Gradient boosting (LightGBM) on lag features, fit globally |
| Hundreds+ of series, long history, budget for tuning | Deep learning: N-HiTS, TFT, DeepAR, PatchTST |
| Intermittent/sparse demand | Croston, SBA, or TSB |
| Hierarchy that must sum (SKU → store → region) | Any base model + MinT reconciliation |

Two findings from the M-competitions worth internalising: **combinations beat single models** (simple averaging of several decent forecasts is remarkably hard to beat), and **complexity only pays off with many series** — deep learning lost to simple statistics on single short series, and won on large panels.

### 5. Backtest
See `references/evaluation.md` for the protocol and metric definitions. In short: rolling-origin evaluation, multiple origins, horizon-matched to production, gap the split if features need it, and report error *by horizon step* — accuracy at h=1 says nothing about h=28.

### 6. Quantify uncertainty
Produce intervals, then verify them. Empirical coverage of a nominal 90% interval should be near 90% on the backtest; if it is 60%, the model is overconfident and any downstream safety stock is wrong. Model-based intervals (ARIMA/ETS) assume the residual distribution is correct — check residuals for autocorrelation and non-normality. Conformal prediction gives distribution-free intervals when those assumptions fail.

### 7. Report
State the model, the baseline it beat and by how much, the backtest design, error by horizon, interval coverage, and the assumptions that would most change the answer. Name the conditions under which the forecast stops being valid.

## Common failure modes

- **Look-ahead bias in features.** Rolling means, scalers, and target encodings computed over the whole series before splitting. Fit every transform inside the training fold.
- **Using covariates that won't exist at forecast time.** Modelling next month's revenue with next month's traffic, which itself has to be forecast.
- **One split, one metric.** An impressive number from a single lucky window.
- **MAPE on data near zero.** It explodes, and it penalises over-forecasting more than under-forecasting. Use MASE or a scaled error instead.
- **No baseline comparison.** Reporting "MAPE 8%" with nothing to compare it to is not a result.
- **Forecasting the wrong aggregation.** Forecasting totals then splitting by historical share, when the decision is made per item.
- **Ignoring calendar mechanics.** Daylight saving, leap years, moving holidays (Easter, Ramadan, Chinese New Year), retail 4-4-5 calendars, and months of unequal length.
- **Treating a structural break as noise.** A pricing change or a new competitor makes the pre-break history misleading; say so rather than averaging through it.
- **Reporting the mean for an asymmetric decision.** See the newsvendor quantile above.

## Inputs to gather
- The series itself, with its timestamp column and frequency
- Horizon, retraining cadence, and the decision the forecast feeds
- Known-future covariates (holidays, promotions, prices, contracts) and known past events
- Cost asymmetry, or the target service level
- Any accuracy the current process achieves, as the bar to beat

## What you deliver
- A backtested forecast with prediction intervals, scored against baselines
- Error broken out by horizon step, plus interval coverage
- The model choice with its reasoning, and the assumptions that most affect it
- Concrete next steps and how success will be measured in production

## Quality bar
Before returning output, confirm: a baseline was fitted and beaten; the backtest used multiple rolling origins; no feature leaks future information; intervals are reported and their coverage checked; the error metric suits the data (no MAPE near zero); and every number is computed, not asserted. If key context is missing — horizon, frequency, or the decision — ask for it rather than guessing.
