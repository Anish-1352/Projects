import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Download historical stock data
symbol = 'JPM'  
data = yf.download(symbol, start='2020-01-01', end=None)  # For the latest data

# Create moving averages as features
data["MA10"] = data["Close"].rolling(window=10).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# Lag features
data["Lag1"] = data["Close"].shift(1)
data['Lag2'] = data['Close'].shift(2)

# Drop rows with missing values
data.dropna(inplace=True)

# Define target variable (predicting next day close price)
X = data[['MA10', 'MA50', 'Lag1', 'Lag2']]
y = data['Close']

# Split the data into train and test sets
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
# Buy if the predicted price is higher than today's price
# Sell if it's lower
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

# View the last 10 rows of actual and predicted prices
print(results.tail(10))

# Future predictions (assuming you want to predict the next day's price based on the last available data)
future_X = X_test.iloc[-1:].copy()
future_X.index = [X_test.index[-1] + pd.Timedelta(days=1)]

# Predicting future prices
future_prediction = model.predict(future_X)
print(f"Future Prediction: {future_prediction[0]}")

# Plotting the actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Actual Price')
plt.plot(results.index, results['Predicted'], label='Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
