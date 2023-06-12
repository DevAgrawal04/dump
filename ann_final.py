import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the hyperparameters
activation = 'linear'
learning_rate = 0.02176
n_layers = 2
n_neurons = 32

# Create the ANN model
model = Sequential()

# Add input layer
model.add(Dense(n_neurons, input_dim=input_dim, activation=activation))

# Add hidden layers
for _ in range(n_layers - 1):
    model.add(Dense(n_neurons, activation=activation))

# Add output layer
model.add(Dense(1, activation='linear'))

# Define optimizer and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)
