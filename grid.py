import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generate dummy regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(16,), (32,), (64,)],
    'activation': ['relu', 'sigmoid', 'tanh'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

# Create the MLPRegressor model
model = MLPRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the model with the best hyperparameters
best_model = MLPRegressor(random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print('Best hyperparameters:', best_params)
print('Best score (neg_mean_squared_error):', best_score)
print('Mean squared error on test set:', mse)
