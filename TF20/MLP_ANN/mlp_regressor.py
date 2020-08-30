"""
MLP as a regressor
California Housing Dataset
Tensorflow 2.0
"""
import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the dataset
housing = fetch_california_housing()

# 75-25 split
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,
                                                              housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                      y_train_full)

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('X_valid shape: ', X_valid.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)
print('y_valid shape: ', y_valid.shape)

# data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Define a model
model = keras.models.Sequential(
    [
     keras.layers.Dense(30, activation='relu',
                        input_shape=X_train.shape[1:]),
     keras.layers.Dense(1)
    ]
)
model.compile(loss="mean_squared_error", optimizer="sgd")
# train
history = model.fit(X_train_scaled, y_train, epochs=20,
                    validation_data=(X_valid_scaled, y_valid))
# test
mse_test = model.evaluate(X_test_scaled, y_test)
print('MSE Test: ', mse_test)

# Predict
X_new = X_test_scaled[:3]
y_pred = model.predict(X_new)
print('Prediction: ', y_pred)
print('Actual: ', y_test[:3])