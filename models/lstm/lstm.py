# import optuna
# from optuna.trial import Trial
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.metrics import AUC
from matplotlib import pyplot

# Data Loading
print("Loading data...")
X_model = np.load('data/X_timeseries_expanded.npy')
Y_model = pd.read_csv('data/Y_model.csv').values

# Train Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_model, Y_model, test_size=0.2, random_state=42)

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

print("Training model...")
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=100,
    validation_data=(X_test, y_test), 
    verbose=2,
    shuffle=False
)

# plot history
pyplot.plot(history.history['auc'], label='train')
pyplot.plot(history.history['val_auc'], label='test')
pyplot.legend()
pyplot.show()

# Save model with pickle
import pickle
with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)
