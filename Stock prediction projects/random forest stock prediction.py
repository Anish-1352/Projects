import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


symbol = 'JPM'

# Download historical data for JP Morgan
data = yf.download(symbol, start='2020-01-01', end='2024-01-01')
data.head()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['EMA_10'] = data['Close'].ewm(span=10).mean()
data['Lag1'] = data['Close'].shift(1)
data.dropna(inplace=True)

# Define features and target
X = data[['SMA_10', 'EMA_10', 'Lag1']]
y = data['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and tuning
model = RandomForestRegressor()
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

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
