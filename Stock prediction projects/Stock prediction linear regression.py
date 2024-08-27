import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Download historical stock data
symbol = 'AAPL'  # Example: Apple Inc.
data = yf.download(symbol, start='2020-01-01', end=None)  # For the latest data

# Create moving averages as features
data["MA10"] = data["Close"].rolling(window=10).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# Lag features
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)

# Drop rows with missing values
data.dropna(inplace=True)

# Define target variable (predicting next day close price)
X = data[['MA10', 'MA50', 'Lag1', 'Lag2']]
y = data['Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Create a DataFrame to store actual and predicted values
results = pd.DataFrame(index=y_test.index)
results['Actual'] = y_test
results['Predicted'] = y_pred

# Create a simple trading strategy based on predictions
results['Signal'] = np.where(results['Predicted'] > results['Actual'].shift(1), 1, -1)

# Calculate returns
results['Strategy_Return'] = results['Signal'] * (results['Actual'].pct_change())

# Calculate cumulative returns
results['Cumulative_Return'] = (1 + results['Strategy_Return']).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(results['Cumulative_Return'], label='Strategy Return')
plt.plot((1 + results['Actual'].pct_change()).cumprod(), label='Buy and Hold Return')
plt.title('Cumulative Returns')
plt.legend()
plt.show()

# Print the last 10 actual and predicted prices
print(results[['Actual', 'Predicted']].tail(10))

# Future predictions are not implemented here since the model isn't set up for future dates in this code.
# To predict future prices, you need to prepare `future_X` (features for future dates) and use the model to predict them.
