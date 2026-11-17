"""
accuracy.py - ForecastWise MR D.I.Y. (Member 3)
Forecast error measures. Error convention: e = Actual - Forecast
(positive e = under-forecast, negative e = over-forecast).
"""
import numpy as np
import pandas as pd


def _arrays(actual, forecast):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape or a.size == 0:
        raise ValueError("Actual and forecast must be non-empty and the same length.")
    return a, f


def mad(actual, forecast):
    """Mean Absolute Deviation = sum|A - F| / n   (RM million)."""
    a, f = _arrays(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def mse(actual, forecast):
    """Mean Squared Error = sum(A - F)^2 / n   (RM million squared)."""
    a, f = _arrays(actual, forecast)
    return float(np.mean((a - f) ** 2))


def rmse(actual, forecast):
    """Root MSE - same units as revenue, used only as an error band."""
    return float(np.sqrt(mse(actual, forecast)))


def mape(actual, forecast):
    """Mean Absolute Percentage Error = (100/n) * sum|A - F|/A   (%). Invalid if any A = 0."""
    a, f = _arrays(actual, forecast)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero.")
    return float(np.mean(np.abs((a - f) / a)) * 100)


def bias(actual, forecast):
    """Mean error (A - F). Positive = method tends to under-forecast."""
    a, f = _arrays(actual, forecast)
    return float(np.mean(a - f))


def accuracy_table(actual, forecasts):
    """forecasts: {method_name: forecast_array}. Returns one row per method."""
    rows = []
    for name, f in forecasts.items():
        rows.append({"Method": name, "MAD": mad(actual, f), "MSE": mse(actual, f),
                     "MAPE (%)": mape(actual, f), "Bias (mean error)": bias(actual, f)})
    return pd.DataFrame(rows).set_index("Method")


def select_method(primary, robustness=None, decision_metrics=("MAD", "MSE")):
    """
    Transparent selection rule:
      1. On the primary validation quarters, the method with the lower value on
         EVERY decision metric wins.
      2. If the metrics disagree, the method that wins every decision metric on the
         longer robustness backtest is chosen.
      3. If still unresolved, the lower primary MSE wins (it penalises large misses).
    Returns (method_name, reason).
    """
    def unanimous(tbl):
        winners = {tbl[m].idxmin() for m in decision_metrics}
        return winners.pop() if len(winners) == 1 else None

    w = unanimous(primary)
    if w:
        return w, f"Lowest {' and '.join(decision_metrics)} on the validation quarters."
    if robustness is not None:
        w = unanimous(robustness)
        if w:
            return w, (f"Validation metrics were split, so the longer backtest decided: "
                       f"{w} has the lowest {' and '.join(decision_metrics)} there.")
    w = primary["MSE"].idxmin()
    return w, "Metrics split on both windows; lower validation MSE used as tie-breaker."
