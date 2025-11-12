import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from LSTM_arch import ARch 
import pandas as pd
from LSTM_preprocess import preprocess

model, history, X_test, y_test, X_train, y_train  = ARch()
X_train, X_test, y_train, y_test, X_data, y_data, scaler, target_col = preprocess()
pred_scaled = model.predict(X_test)
num_total_features = len(scaler.mean_)
dummy_pred = np.zeros((len(pred_scaled), num_total_features))
dummy_pred[:, target_col] = pred_scaled.flatten()
pred_unscaled = scaler.inverse_transform(dummy_pred)[:, target_col]

dummy_actual = np.zeros((len(y_test), num_total_features))
dummy_actual[:, target_col] = y_test.flatten()
actual_unscaled = scaler.inverse_transform(dummy_actual)[:, target_col]

rmse = np.sqrt((mean_squared_error(actual_unscaled, pred_unscaled)))
print(f"RMSE{rmse:.6f}")
R2 = r2_score(actual_unscaled, pred_unscaled)
print(f"R2 {R2:.6f}")


