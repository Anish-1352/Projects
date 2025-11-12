from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from LSTM_preprocess import preprocess


def ARch():
    model = Sequential()
    X_train, X_test, y_train, y_test, X_data, y_data, scaler, target_col = preprocess()
    model.add(LSTM(units=15, return_sequences= True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))

    model.add(LSTM(units= 15))
    model.add(Dropout(0.1))

    model.add(Dense(units=1))

    model.compile(optimizer= Adam(learning_rate= 0.0001), loss='mean_squared_error')
    earlystopping_val= EarlyStopping(monitor ="val_loss", patience =10, restore_best_weights = True)

    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data= (X_test, y_test),
        shuffle = False,
        callbacks= [earlystopping_val]
    )

    return model, history, X_test, y_test, X_train, y_train




