"""
forecasting.py - ForecastWise MR D.I.Y. (Member 3)
Two taught forecasting methods + rolling one-step-ahead validation.
All revenue values are in RM million. No function uses data after its forecast origin.
"""
import numpy as np
import pandas as pd

MA = "4-Quarter Moving Average"
LS = "Least Squares Trend"


def moving_average_forecast(history, n=4):
    """F(t) = mean of the n most recent actual values in `history`."""
    h = np.asarray(history, dtype=float)
    if n < 1 or len(h) < n:
        raise ValueError(f"Moving average needs at least n={n} observations, got {len(h)}.")
    return float(h[-n:].mean())


def least_squares_fit(history):
    """Fit y = a + b*t with t = 1..len(history). Returns (a, b)."""
    y = np.asarray(history, dtype=float)
    if len(y) < 2:
        raise ValueError("Least squares trend needs at least 2 observations.")
    t = np.arange(1, len(y) + 1, dtype=float)
    n = len(y)
    b = ((t * y).sum() - n * t.mean() * y.mean()) / ((t ** 2).sum() - n * t.mean() ** 2)
    a = y.mean() - b * t.mean()
    return float(a), float(b)


def least_squares_forecast(history, steps_ahead=1):
    """Project the fitted trend to t = len(history) + steps_ahead."""
    a, b = least_squares_fit(history)
    return float(a + b * (len(history) + steps_ahead))


def forecast_next(history, method, n=4):
    """One-step-ahead forecast with the named method (MA or LS)."""
    if method == MA:
        return moving_average_forecast(history, n)
    if method == LS:
        return least_squares_forecast(history)
    raise ValueError(f"Unknown method: {method}")


def rolling_one_step_forecasts(df, start_period, n=4,
                               period_col="period", value_col="revenue_rm_million"):
    """
    For every quarter from `start_period` to the end of `df`, forecast it using
    ONLY the quarters before it (expanding window), then compare with the actual.
    Both methods are evaluated on exactly the same quarters.
    """
    periods = df[period_col].tolist()
    values = df[value_col].to_numpy(dtype=float)
    if start_period not in periods:
        raise ValueError(f"{start_period} not in data.")
    start = periods.index(start_period)
    if start < max(n, 2):
        raise ValueError(f"Need at least {max(n, 2)} quarters before {start_period}.")

    rows = []
    for i in range(start, len(values)):
        hist = values[:i]                       # data available at the forecast origin
        a, b = least_squares_fit(hist)
        ma_f = moving_average_forecast(hist, n)
        ls_f = a + b * (i + 1)                  # t of the target quarter = i + 1
        rows.append({
            "period": periods[i], "t": i + 1, "actual": values[i],
            "ma_forecast": ma_f, "ls_forecast": ls_f, "ls_a": a, "ls_b": b,
            "ma_error": values[i] - ma_f, "ls_error": values[i] - ls_f,
        })
    return pd.DataFrame(rows)


def next_period_label(period):
    """'2026-Q2' -> '2026-Q3'; '2026-Q4' -> '2027-Q1'."""
    year, qtr = period.split("-Q")
    year, qtr = int(year), int(qtr)
    return f"{year + 1}-Q1" if qtr == 4 else f"{year}-Q{qtr + 1}"


def apply_scenario(base_forecast, pct_change):
    """Hypothetical managerial adjustment, e.g. -5 (slowdown) or +10 (promotion)."""
    return float(base_forecast * (1 + pct_change / 100.0))
