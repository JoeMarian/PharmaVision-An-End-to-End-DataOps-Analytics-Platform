import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ------------------------
# Forecasting
# ------------------------
def forecast_sales(df, date_col="Month", value_col="Sales", periods=6):
    """Forecast sales using Prophet (fallback: linear regression)."""
    if df.empty or date_col not in df or value_col not in df:
        return pd.DataFrame({"ds": [], "yhat": []})

    try:
        from prophet import Prophet
        tmp = df[[date_col, value_col]].dropna().rename(columns={date_col: "ds", value_col: "y"})
        model = Prophet()
        model.fit(tmp)
        future = model.make_future_dataframe(periods=periods, freq="M")
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]]
    except Exception as e:
        # fallback to linear regression
        tmp = df[[date_col, value_col]].dropna().copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        tmp["t"] = np.arange(len(tmp))
        X = tmp[["t"]]
        y = tmp[value_col]
        model = LinearRegression()
        model.fit(X, y)
        future_t = np.arange(len(tmp), len(tmp) + periods)
        y_pred = model.predict(future_t.reshape(-1, 1))
        forecast = pd.DataFrame({
            "ds": pd.date_range(start=tmp[date_col].max(), periods=periods+1, freq="M")[1:],
            "yhat": y_pred
        })
        return forecast

def simple_linear_forecast(df, date_col="Month", value_col="Sales", periods=6):
    """Forecast sales using a simple linear regression model."""
    if df.empty or date_col not in df or value_col not in df:
        return pd.DataFrame({"ds": [], "yhat": []})

    tmp = df[[date_col, value_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp["t"] = np.arange(len(tmp))
    X = tmp[["t"]]
    y = tmp[value_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_t = np.arange(len(tmp), len(tmp) + periods)
    y_pred = model.predict(future_t.reshape(-1, 1))
    
    forecast = pd.DataFrame({
        "ds": pd.date_range(start=tmp[date_col].max(), periods=periods + 1, freq="M")[1:],
        "yhat": y_pred
    })
    return forecast

# ------------------------
# Anomaly Detection
# ------------------------
def detect_anomalies(df, value_col="Sales"):
    """Detect anomalies in sales using IsolationForest."""
    if df.empty or value_col not in df:
        return df.copy()

    tmp = df[[value_col]].dropna()
    model = IsolationForest(contamination=0.05, random_state=42)
    tmp["is_anomaly"] = model.fit_predict(tmp)
    tmp["is_anomaly"] = tmp["is_anomaly"].apply(lambda x: x == -1)

    result = df.copy()
    result["is_anomaly"] = tmp["is_anomaly"].reindex(df.index, fill_value=False)
    return result

# ------------------------
# Clustering
# ------------------------
def cluster_drugs(df, n_clusters=3):
    """Cluster drugs by sales and adverse events."""
    if df.empty or "Drug" not in df.columns or "Sales" not in df.columns or "AdverseEvents" not in df.columns:
        return pd.DataFrame()

    agg = df.groupby("Drug")[["Sales", "AdverseEvents"]].mean().reset_index()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    agg["cluster"] = kmeans.fit_predict(agg[["Sales", "AdverseEvents"]])
    return agg

# ------------------------
# LSTM Forecast (optional)
# ------------------------
def lstm_forecast(df, date_col="Month", value_col="Sales", periods=6):
    """Forecast sales using a simple LSTM model (requires TensorFlow)."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        if df.empty or date_col not in df or value_col not in df:
            return pd.DataFrame({"ds": [], "yhat": []})

        tmp = df[[date_col, value_col]].dropna().copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        tmp = tmp.sort_values(date_col)

        series = tmp[value_col].values.astype(float)
        window = 3  # lookback

        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(50, activation="relu", input_shape=(window, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, verbose=0)

        preds = []
        last_window = series[-window:]
        for _ in range(periods):
            x_input = last_window.reshape((1, window, 1))
            yhat = model.predict(x_input, verbose=0)[0][0]
            preds.append(yhat)
            last_window = np.append(last_window[1:], yhat)

        forecast_dates = pd.date_range(start=tmp[date_col].max(), periods=periods+1, freq="M")[1:]
        forecast = pd.DataFrame({"ds": forecast_dates, "yhat": preds})
        return forecast

    except Exception as e:
        return pd.DataFrame({"ds": [], "yhat": []})
