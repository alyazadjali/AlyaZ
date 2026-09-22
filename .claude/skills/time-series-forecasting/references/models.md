# Model catalogue and selection

Read this when choosing between model families. Selection is driven by **number of series**, **history length**, **seasonality structure**, and **whether covariates exist** — not by which method is fashionable.

## Selection logic

Work down this list and stop at the first match:

1. **Demand is intermittent** (many zero periods, sporadic non-zero) → Croston family. Standard metrics and standard models both mislead here.
2. **Only a handful of series, each with limited history** (< ~3 seasonal cycles) → ETS, ARIMA, or Theta. Do not reach for ML; there is nothing to learn from.
3. **Single series, multiple overlapping seasonalities** (e.g. daily data with weekly + annual cycles) → MSTL, TBATS, or Prophet.
4. **Many related series sharing structure, plus covariates** → a single *global* gradient-boosting model over all series with lag and calendar features. This is usually the best accuracy-per-effort on business panels.
5. **Large panel, long history, engineering budget** → global deep learning (N-HiTS, TFT, DeepAR, PatchTST).
6. **Anything else** → ETS and ARIMA as the workhorses, combined.

Then: **combine**. Averaging the forecasts of 3–5 reasonable models is one of the most reliable accuracy gains available and is far more robust than picking a single winner on a backtest.

## Statistical models

### Naive family (always fit these)
- **Naive**: ŷ = last observed value. Surprisingly hard to beat for random-walk-like series (prices, many financial series).
- **Seasonal naive**: ŷ = value from one season ago. The real bar for any seasonal series.
- **Drift**: naive plus the average historical slope.
- **Historical mean**: sanity check for stationary, trendless series.

### Exponential smoothing (ETS)
Weighted averages with exponentially decaying weights. The Hyndman taxonomy classifies each model by **E**rror, **T**rend, **S**easonality, each Additive, Multiplicative, or None (e.g. ETS(A,Ad,M) = additive error, damped additive trend, multiplicative seasonality).

- Fast, robust, few parameters, excellent on short-to-medium single series.
- **Damped trend** is the standard defensive choice for longer horizons — undamped linear trends extrapolate absurdly far.
- Multiplicative seasonality handles seasonal amplitude that grows with level, without transforming.
- Handles one seasonal period only.

### ARIMA / SARIMA
ARIMA(p,d,q) models autocorrelation in a differenced series; SARIMA adds (P,D,Q)ₘ for seasonal period m.

- `d` = differences to reach stationarity; `p`/`q` = autoregressive and moving-average orders.
- Use automatic order selection (AICc-based) as a starting point, then inspect residuals.
- **ARIMAX / regression with ARIMA errors** incorporates covariates while modelling residual autocorrelation — the standard explainable choice when you have drivers.
- Needs stationarity work and a single seasonal period; struggles with long seasonal cycles (m=365).

### Theta
A deceptively simple decomposition method that was the top performer in the M3 competition and remains a strong, nearly free baseline. Fit it whenever you fit ETS.

### STL / MSTL decomposition
Seasonal-Trend decomposition using Loess. Splits the series into trend, seasonal, and remainder. Two uses:
- **Diagnostic**: look at the components before modelling.
- **Forecasting**: seasonally adjust, forecast the remainder with any method, add seasonality back. **MSTL** extends this to multiple seasonal periods and is often the cleanest way to handle daily data with weekly and annual cycles.

### TBATS
Trigonometric seasonality, Box-Cox, ARMA errors, Trend, Seasonal. Handles multiple and non-integer seasonal periods (e.g. m = 365.25/7). Powerful but slow to fit and hard to explain.

### Prophet
Additive decomposable model (piecewise-linear or logistic trend + Fourier seasonalities + holiday effects).

- Strong fit for daily business data with holidays and analyst-supplied changepoints; tolerant of gaps and outliers; interpretable components.
- Weak on short series, sub-daily high-frequency data, and series where autocorrelation matters more than calendar structure. It is a curve-fitting model, not a stochastic process model.
- Always compare it to ETS/ARIMA rather than assuming it wins.

### Intermittent demand
Standard models and MAPE both break down when most periods are zero.
- **Croston's method**: forecasts demand size and inter-arrival interval separately.
- **SBA (Syntetos-Boylan approximation)**: bias-corrects Croston; generally preferred.
- **TSB**: updates demand probability every period, so it handles items that go obsolete.
- Evaluate with MASE or a scaled absolute error, and judge against the inventory decision (fill rate, stockout cost) rather than a percentage error.

## Machine learning

### Gradient boosting (LightGBM, XGBoost, CatBoost)
Reframes forecasting as tabular regression. The winning approach in the M5 competition and the default strong choice for business panels.

**Features that carry most of the signal:**
- Lags of the target at relevant offsets (1, 7, 14, 28, 364 for daily data)
- Rolling statistics of those lags (mean, std, min/max over trailing windows) — computed *only* on data available at forecast time
- Calendar: day of week, day of month, week of year, month, holidays and their proximity, paydays
- Known-future covariates: price, promotion, contracted volume
- Series identifiers and static attributes (category, region, store size)

**Critical mechanics:**
- **Fit one global model over all series**, not one model per series. Pooling is where the accuracy comes from.
- Choose **recursive** (feed predictions back as lags; error compounds) vs. **direct** (a separate model per horizon step; no compounding, more models) deliberately. Direct is safer for long horizons.
- Trees **cannot extrapolate** beyond the range of the training target. On a trending series, either detrend first, model differences, or the forecast will flatten.
- Every rolling feature must be shifted so it uses only past data. This is where leakage enters.

### Linear models
Regularised linear regression on lag and calendar features is a fast, transparent, and often competitive option — worth fitting as a bridge between the statistical baselines and boosting.

## Deep learning

Justified when you have **many series** (hundreds or more) and **long history**. On a single short series they reliably lose to ETS.

- **N-BEATS / N-HiTS**: pure deep stacks of backward/forward residual blocks. N-HiTS adds multi-rate sampling and is markedly cheaper for long horizons. Strong univariate accuracy, no covariates needed.
- **DeepAR**: autoregressive RNN producing a probabilistic forecast by learning distribution parameters. Native support for many related series and covariates.
- **TFT (Temporal Fusion Transformer)**: handles static, known-future, and observed-past covariates separately, with attention-based interpretability and quantile outputs. The most capable when you have rich, mixed covariates.
- **PatchTST** and related transformer variants: competitive on long-horizon multivariate benchmarks.

Practical notes: these need careful scaling per series, ample validation data, and far more tuning than statistical models. Budget for the engineering, and always keep the statistical baseline in the comparison — the M4 winner was a *hybrid* of exponential smoothing and a recurrent network, not a pure neural model.

## Hierarchical and grouped series

When forecasts must sum consistently (SKU → category → region → total), forecasting each level independently produces incoherent numbers.

- **Bottom-up**: forecast the lowest level, sum upward. Coherent, but noisy at the bottom.
- **Top-down**: forecast the total, split by historical proportions. Stable, but loses bottom-level signal.
- **MinT (minimum trace) reconciliation**: optimally combines forecasts from all levels using the error covariance structure. Generally the best choice, and usually improves accuracy at *every* level, not just consistency.

Forecast at every level, then reconcile — don't pick one level and derive the rest by hand.

## Tooling

- **Python**: `statsmodels` (ETS, SARIMAX, STL), `statsforecast` (fast statistical models at scale), `sktime` / `darts` (unified APIs, backtesting), `prophet`, `lightgbm`, `pytorch-forecasting` / `neuralforecast`, `hierarchicalforecast`, `mapie` (conformal intervals).
- **R**: `fable` / `fabletools`, `forecast`, `fpp3`. The reference text is Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., free online) — cite it when the user wants to go deeper.
