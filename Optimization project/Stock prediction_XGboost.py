import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
import schedule
import time
import joblib
import logging

logging.basicConfig(filename='model_retrain.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def load_data(symbol='JPM', start='2020-01-01', end='2024-01-01'):
    
    data = yf.download(symbol, start=start, end=end)
    
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
    data['Volume'] = data['Volume'].fillna(method='ffill')
    
    data.dropna(inplace=True)
    return data

def train_model():
    data = load_data()
    X = data[['SMA_10', 'EMA_10', 'Lag1', 'Lag2', 'RSI', 'MACD', 'MACD_Signal', 'Volume']]
    y = data['Close']

  
    tscv = TimeSeriesSplit(n_splits=5)

    model = XGBRegressor()
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'alpha': [0, 0.1],
        'lambda': [0, 0.1]
    }

    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'model.pkl')

    
    data['Predicted'] = best_model.predict(X)
    data['Signal'] = np.where(data['Predicted'].shift(-1) > data['Close'], 1, -1)
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

    
    y_pred = data['Predicted']
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    logging.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    plt.figure(figsize=(14, 7))
    plt.plot(data['Cumulative_Return'], label='Strategy Return')
    plt.plot((1 + data['Close'].pct_change()).cumprod(), label='Buy & Hold Return', linestyle='--')
    plt.title('Strategy vs Buy and Hold')
    plt.legend()
    plt.show()

def retrain_schedule():
    schedule.every().day.at("22:01").do(train_model)  
    schedule.run_pending()
    time.sleep(1)

if __name__ == "__main__":
    train_model()  
    