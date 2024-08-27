import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
import schedule
import time
import joblib

def load_data():
    # Load and preprocess data
    symbol = 'JPM'
    # Download historical data for JP Morgan
    data = yf.download(symbol, start='2020-01-01', end='2024-01-01')
    data.head()

    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)  # Added lagged feature
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
    data['Volume'] = data['Volume'].fillna(method='ffill')
    data.dropna(inplace=True)
    return data

def train_model():
    data = load_data()
    X = data[['SMA_10', 'EMA_10', 'Lag1', 'Lag2', 'RSI', 'MACD', 'MACD_Signal', 'Volume']]
    y = data['Close']

    # Split data with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Model selection and tuning
        model = XGBRegressor()
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'alpha': [0, 0.1],  # Regularization parameter
            'lambda': [0, 0.1]  # Regularization parameter
        }
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        # Model evaluation
        y_pred = grid_search.best_estimator_.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R^2: {r2}")
    
        joblib.dump(grid_search.best_estimator_, 'model.pkl')

        # Strategy returns and visualization
        data['Predicted'] = grid_search.best_estimator_.predict(X)
        data['Signal'] = np.where(data['Predicted'].shift(-1) > data['Close'], 1, -1)
        data['Strategy_Return'] = data['Signal'].shift(1) * data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

        plt.figure(figsize=(14, 7))
        plt.plot(data['Cumulative_Return'], label='Strategy Return')
        plt.plot((1 + data['Close'].pct_change()).cumprod(), label='Buy and Hold Return', linestyle='--')
        plt.title('Strategy vs Buy and Hold')
        plt.legend()
        plt.show()

        # Display the last 10 rows of actual vs predicted values
        results = pd.DataFrame(index=y_test.index)
        results['Actual'] = y_test
        results['Predicted'] = y_pred

        print(results.tail(10))  # Display the last 10 rows

        # Plot actual vs predicted prices
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Close'], label='Actual Price')
        plt.plot(data.index, data['Predicted'], label='Predicted Price')
        plt.title('Actual vs Predicted Prices')
        plt.legend()
        plt.show()

# Helper functions
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

def retrain_schedule():
    schedule.every().day.at("22:01").do(train_model)  # Adjust timing as needed

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    retrain_schedule()
