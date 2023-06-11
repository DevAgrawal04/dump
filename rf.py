from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, hp, tpe, Trials

# Define the objective function for hyperopt
def objective(params):
    # Define the model
    model = RandomForestRegressor(**params)

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

    # Compute the average RMSE score
    mean_score = scores.mean()

    return mean_score

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 20)),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.5),
    'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 1.0),
    # Add more hyperparameters here
}

# Create a trials object to store the optimization results
trials = Trials()

# Run the hyperparameter optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# Train the final model with the best hyperparameters
final_model = RandomForestRegressor(**best)
final_model.fit(X_train, y_train)

# Evaluate the RMSE on the test set
y_pred = final_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the RMSE of the final model with the best hyperparameters
print("RMSE: {:.2f}".format(rmse))