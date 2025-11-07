from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from LSTM_preprocess import preprocess

model = Sequential()
X_train, X_test, y_train, y_test, X_data, y_data = preprocess()
model.add(LSTM(units=50, return_sequences= True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))

model.add(LSTM(units= 50))
model.add(Dropout(0.1))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(
    X_train, y_train
)



