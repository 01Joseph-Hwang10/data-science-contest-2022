# import optuna
# from optuna.trial import Trial
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from typing import List, Any, Tuple
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.metrics import AUC
from matplotlib import pyplot

# Data Loading
print("Loading data...")
X_model = pd.read_csv('data/X_model.csv')
Y_model = pd.read_csv('data/Y_model.csv')

# Data Preprocessing
print("Preprocessing data...")
def as_timeseries_df(X: pd.DataFrame) -> pd.DataFrame:
    """Converts a DataFrame with `['gender', 'age_code', 'region_code', 'c', 't', 's']` columns and additional time column with timestamps to a time series DataFrame.

    Args:
        X (pd.DataFrame): DataFrame. The data should match the following format:
            gender: 1 (if man) | 2 (if woman)
            age_code: Integer
            region_code: Integer
            r"c2022[0-9]{2}": Number of logins. Integer
            r"t2022[0-9]{2}": Number of logins with money transfer. Integer
            r"s2022[0-9]{2}": Duration of logins. Float
    
    Returns:
        pd.DataFrame: Time series DataFrame. Each column's description:
            gender: 1 (if man) | 2 (if woman)
            age_code: Integer
            region_code: Integer
            c: Number of logins. Integer
            t: Number of logins with money transfer. Integer
            s: Duration of logins. Float
            time: Timestamp.
    """
    # Data cleaning
    print("Cleaning data...")
    X = X.fillna(0)

    # Data transformation
    print("Transforming data...")
    n = len(X)
    timeseries = list(map(
        lambda col: col[1:], 
        filter(lambda col: col.startswith('c2022'), X.columns)
    ))
    data = []
    static_feature_cols = ['gender', 'region_code', 'age_code']
    timeseries_features_col = ['c', 't', 's']
    for i in range(n):
        row = X.iloc[[i]]
        static_features = list(map(lambda col: row[col], static_feature_cols))
        for j in range(len(timeseries)):
            time = pd.Timestamp(timeseries[j])
            timeseries_entry = [row[col + timeseries[j]] for col in timeseries_features_col]
            data.append([time, *static_features, *timeseries_entry])
    # Data scaling
    print("Scaling data...")
    scaler = MinMaxScaler()
    data = np.array(data)
    data[:, 3:] = scaler.fit_transform(data[:, 3:])

    # Data reshaping
    print("Reshaping data...")
    X_timeseries = pd.DataFrame(
        data, 
        columns=['time', *static_feature_cols, *timeseries_features_col]
    )
    return X_timeseries

X_timeseries = as_timeseries_df(X_model)

# Train Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_timeseries, Y_model, test_size=0.2, random_state=42)

# Model training
print("Training model...")
# design network
# References: 
# - https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
# - https://stats.stackexchange.com/questions/242238/what-is-considered-a-normal-quantity-of-outliers
# - https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
hidden_nodes = int(2/3 * (X_train.shape[1] * X_train.shape[2]))
model = Sequential()
model.add(LSTM(hidden_nodes, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(
    loss='mae', 
    optimizer='adam',
    metrics=[AUC()]
)
# fit network
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=72, 
    validation_data=(X_test, y_test), 
    verbose=2, 
    shuffle=False
)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
