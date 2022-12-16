import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from resnet import Classifier_RESNET
import pickle

# Data Loading
print("Loading data...")
X_model = np.load('../../data/X_timeseries_expanded.npy.zip')
Y_model = pd.read_csv('../../data/Y_model.csv').values.astype(np.bool_)

# Train Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_model['X_timeseries_expanded'].astype(np.uint16), Y_model, test_size=0.2, random_state=42)

model = Classifier_RESNET(
    output_directory='../../data',
    input_shape=(X_train.shape[1], X_train.shape[2]),
    nb_classes=[0,1]
)

history = model.fit(
    X_train,
    y_train,
    X_test,
    y_test,
    y_test,
)

# plot history
pyplot.plot(history.history['auc'], label='train')
pyplot.plot(history.history['val_auc'], label='test')
pyplot.legend()
pyplot.show()

# Save model with pickle
with open("../../data/model.pkl", "wb") as f:
    pickle.dump(model, f)