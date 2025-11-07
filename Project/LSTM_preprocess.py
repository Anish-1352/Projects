from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from merged import load_data

def preprocess():
    model_data= load_data()
    scalar = StandardScaler()
    scaled_data =scalar.fit_transform(model_data)

    lookback = 30
    X_data = []
    y_data = []

    num_features = 3 #fama factors i am using to predict
    target_col= 5  #data i want to predict

    for i in range(lookback, len(scaled_data)):
        X_data.append(scaled_data[i-lookback:i, 0:num_features])

        y_data.append(scaled_data[i, target_col])

    X_data= np.array(X_data)
    y_data= np.array(y_data)



    split_ratio = 0.8
    split_index = int(len(X_data)*split_ratio)

    X_train, X_test = X_data[:split_index], X_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    return X_train, X_test, y_train, y_test, X_data, y_data

if __name__ =="__main__":
    X_train, X_test, y_train, y_test, X_data, y_data = preprocess()
    
    print(X_data.shape)
    print(y_data.shape)
    print(len(X_train))
    print(len(X_test))





    
