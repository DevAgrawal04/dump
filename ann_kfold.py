import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import hp, fmin, tpe, Trials

# Generate a sample regression dataset
X, y = generate_sample_data()

# Define the objective function with cross-validation and loss plot
def objective(params):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(params['units'], activation=params['activation'], input_dim=X.shape[1]))
    
    for i in range(params['num_layers'] - 1):
        model.add(tf.keras.layers.Dense(params['units'], activation=params['activation']))
    
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=params['optimizer'], loss='mse')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    losses = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)
        losses.append(history.history['loss'])
    
    avg_loss = np.mean(losses, axis=0)
    
    # Plot loss vs. epochs
    plt.plot(avg_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.show()
    
    mse = np.mean(losses[-1])
    return mse

# Define the search space for hyperparameters
space = {
    'units': hp.choice('units', [16, 32, 64]),
    'activation': hp.choice('activation', ['relu', 'sigmoid']),
    'optimizer': hp.choice('optimizer', ['adam', 'sgd']),
    'num_layers': hp.choice('num_layers', [1, 2, 3, 4])
}

# Define the optimization function
def optimize(space):
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials(),
        rstate=np.random.RandomState(42)
    )
    return best

# Run the optimization
best_params = optimize(space)
print("Best Parameters:", best_params)
