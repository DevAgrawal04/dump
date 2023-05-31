import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# Generate dummy regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the search space for hyperparameters
space = {
    'n_layers': hp.choice('n_layers', [1, 2, 3]),
    'n_neurons': hp.choice('n_neurons', [16, 32, 64]),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1))
}


# Define the objective function to minimize (mean squared error)
def objective(params):
    model = Sequential()
    
    # Add input layer
    model.add(Dense(params['n_neurons'], input_dim=X_train.shape[1]))
    model.add(Activation(params['activation']))
    
    # Add hidden layers
    for _ in range(params['n_layers']):
        model.add(Dense(params['n_neurons']))
        model.add(Activation(params['activation']))
    
    # Add output layer with linear activation
    model.add(Dense(1, activation='linear'))
    
    # Compile the model with the specified learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Plot loss vs epochs
    plt.plot(history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    return mse


# Run hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

print('Best hyperparameters:', best)
