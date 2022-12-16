import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.metrics import AUC, Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from matplotlib import pyplot
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def data(filepath):
    return os.path.join('../../data', filepath)

# Data Loading
print("Loading data...")
X_model = np.load(data('X_timeseries_expanded.npy.zip'))
Y_model = pd.read_csv(data('Y_model.csv')).values.astype(np.bool_)

# Train Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_model['X_timeseries_expanded'].astype(np.uint16), 
    Y_model, 
    test_size=0.2
)

# design network
# References: 
# - https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
# - https://stats.stackexchange.com/questions/242238/what-is-considered-a-normal-quantity-of-outliers
# - https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# hidden_nodes = int(2/3 * (X_train.shape[1] * X_train.shape[2]))
LR = 0.01                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate

def lstm(hidden_nodes: int = 16, return_sequences: bool = True):
    return LSTM(
        hidden_nodes,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=return_sequences
    )

model = Sequential()
model.add(lstm())
model.add(BatchNormalization())
model.add(lstm())
model.add(BatchNormalization())
model.add(lstm(return_sequences = False))
model.add(BatchNormalization())
model.add(Dense(1, activation="sigmoid"))
model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(learning_rate=LR),
    metrics=[
        AUC(),
        Accuracy(),
    ]
)

callbacks = []
# Define a learning rate decay method:
callbacks.append(ReduceLROnPlateau(
    monitor='auc', 
    patience=1, 
    verbose=0, 
    factor=0.5, 
    min_lr=1e-8
))
# Define Early Stopping:
callbacks.append(EarlyStopping(
    monitor='val_auc', 
    min_delta=0, 
    patience=30, 
    verbose=1, 
    mode='max',
    baseline=0, 
    restore_best_weights=True
))
callbacks.append(ModelCheckpoint(
    data('lstm.h5'),
    save_best_only=True,
    monitor='val_auc',
    mode='max'
))

print("Training model...")
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test), 
    verbose=1,
    shuffle=False,
    callbacks=callbacks,
)

# plot history
pyplot.plot(history.history['auc'], label='train')
pyplot.plot(history.history['val_auc'], label='test')
pyplot.legend()
pyplot.show()

# Save model with pickle
import pickle
with open(data("model.pkl"), "wb") as f:
    pickle.dump(model, f)
