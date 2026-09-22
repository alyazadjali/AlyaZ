# Backtesting and evaluation

Read this when designing a backtest or choosing an error metric. Evaluation design decides whether a forecasting project succeeds; a strong model with a broken backtest is worse than no model, because it carries false confidence.

## The core rule

**Never use shuffled k-fold cross-validation on a time series.** It trains on the future to predict the past, and the resulting scores are meaningless — typically far too optimistic. Every split must respect time order.

## Rolling-origin evaluation

Also called time series cross-validation. Choose a set of forecast origins, and at each one train on data up to that point and forecast the next `h` steps. Average the errors over all origins.

```
Expanding window (train set grows):
  origin 1:  [=========train========]  [test h]
  origin 2:  [===========train=========]  [test h]
  origin 3:  [=============train==========]  [test h]

Sliding window (train set fixed length):
  origin 1:  [===train===]  [test h]
  origin 2:      [===train===]  [test h]
  origin 3:          [===train===]  [test h]
```

**Expanding** mirrors production when you retrain on all history. **Sliding** is right when older data is stale (post-structural-break, changing product mix) and is a better test of robustness.

### Design it to mirror production
- **Match the horizon.** If production forecasts 28 days ahead, backtest 28 days ahead. Never tune on h=1 and deploy at h=28.
- **Match the retraining cadence.** If the model retrains weekly, step origins weekly.
- **Use enough origins.** A handful is noise. Aim for enough windows to distinguish models from luck — dozens where data allows.
- **Insert a gap** between train and test if there is reporting lag: if actuals arrive 3 days late, the model in production cannot see the last 3 days, so the backtest must not either.
- **Hold out a final untouched window.** Model selection over many rolling backtests is itself a form of fitting. Keep one last period that you score exactly once.

### Report error by horizon step
Aggregate error hides the shape of the degradation. Produce a table of error at h=1, 2, … H. Accuracy almost always decays with horizon; the user needs to know where it becomes unusable, because that is the real planning limit.

## Metrics

### Scale-dependent (compare models on one series, not across series)
- **MAE** — mean absolute error. Optimised by the **median**; robust to outliers. Use when large errors are not disproportionately costly.
- **RMSE** — root mean squared error. Optimised by the **mean**; penalises large errors more. Use when big misses hurt more than proportionally.

The choice is not cosmetic: minimising MAE and minimising RMSE produce *different* forecasts. Pick the one matching the cost structure and state which you used.

### Percentage errors
- **MAPE** — mean absolute percentage error. Popular and widely misused.
  - **Undefined** when actuals are zero, and explodes when they are near zero.
  - **Asymmetric**: it penalises over-forecasting more heavily than under-forecasting, so minimising MAPE biases forecasts low.
  - Acceptable only for strictly positive data comfortably away from zero.
- **sMAPE** — symmetric variant. Fixes less than its name implies and is unstable when actual and forecast are both near zero.

### Scale-free (the right default for comparing across series)
- **MASE** — mean absolute scaled error. MAE of the forecast divided by the in-sample MAE of the naive (or **seasonal** naive, for seasonal data) method on the training set.
  - **MASE < 1** means the model beats the naive benchmark; **> 1** means it loses.
  - Defined for zero-valued data, symmetric, and comparable across series of different scales. Use it as the headline metric on panels, and for intermittent demand.
- **RMSSE** — the squared analogue, used in M5. Same interpretation, penalises large errors.

### Probabilistic metrics
If you deliver intervals or quantiles, score them as such — point metrics cannot evaluate uncertainty.

- **Pinball (quantile) loss** at level τ: `τ·(y − q)` when `y ≥ q`, else `(1 − τ)·(q − y)`. Averaged over quantiles and horizons, this is the standard quantile-forecast score.
- **CRPS** — continuous ranked probability score. Evaluates the whole predictive distribution; reduces to MAE for a point forecast, so the two are directly comparable.
- **Interval coverage**: the fraction of actuals falling inside the nominal interval. A 90% interval covering 60% of actuals is overconfident; covering 99% is uselessly wide. Report coverage alongside width — either alone can be gamed.

### Always report against the baseline
Give every metric as a ratio or percentage improvement over seasonal naive. "MASE 0.82" is informative; "MAPE 8%" alone is not.

## Residual diagnostics

A well-specified model leaves residuals that look like noise. Check:

- **Autocorrelation**: ACF plot of residuals, or a Ljung-Box test. Significant autocorrelation means exploitable structure was left on the table.
- **Zero mean**: non-zero mean residuals indicate bias — and bias in a forecast feeding inventory or budgets compounds every period.
- **Constant variance**: fanning residuals suggest a transform (log/Box-Cox) is needed.
- **Approximate normality**: only matters for model-based prediction intervals. When it fails, use conformal or bootstrapped intervals instead of widening arbitrarily.

Residual diagnostics test *fit*, not *forecast accuracy*. Passing them is not a substitute for a backtest, and a model can fail them and still forecast well. Use both.

## Prediction intervals

- **Model-based** (ARIMA, ETS): analytic, but assume the residual distribution is correct and typically understate real uncertainty because they ignore parameter and model-selection error.
- **Bootstrapped/simulated**: resample residuals and propagate forward; fewer distributional assumptions.
- **Conformal prediction**: distribution-free intervals calibrated on backtest residuals, with finite-sample coverage guarantees under exchangeability. The pragmatic choice when model assumptions fail — and it works on top of ML and deep learning models, which have no native intervals.
- **Quantile regression / native quantile outputs** (LightGBM quantile objective, TFT, DeepAR): model the quantiles directly. Best when the decision needs a specific service level.

Whatever the source, **validate coverage empirically on the backtest**. An uncalibrated interval is worse than none, because downstream safety stock and risk buffers are computed from it.

## Leakage checklist

Run through this before trusting any backtest result:

- [ ] Scalers, encoders, and imputers fitted **inside** the training fold, not on the full series
- [ ] Rolling and expanding features shifted so they use only data strictly before the forecast origin
- [ ] No covariate used that would not be known at forecast time for the whole horizon
- [ ] Target-derived features (lags, rolling means) respect the reporting lag of real data
- [ ] Historical values are as-of, not restated after the fact
- [ ] Hyperparameters tuned on a validation split, not on the final holdout
- [ ] Series with different start dates not implicitly padded with zeros that read as real observations

## Comparing models honestly

- Differences within noise are not differences. With enough origins, compare error *distributions* across windows, not just their means; a Diebold-Mariano test formalises whether two forecast accuracies genuinely differ.
- Judge on the production horizon and the production metric, not the most flattering pair.
- Count the cost: a 2% accuracy gain that requires a GPU pipeline and weekly retraining is often the wrong trade against ETS in ten lines. Say so.
- Report what the model does **not** handle — the structural breaks, regime changes, and event types absent from the training period.
