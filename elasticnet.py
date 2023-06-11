import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, hp, tpe, Trials

# Generate sample data for regression
np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=10, random_state=0)

# Define the objective function for hyperopt
def objective(params):
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']
    fit_intercept = params['fit_intercept']
    max_iter = params['max_iter']

    # Define the model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, random_state=0)

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

    # Compute the average RMSE score
    mean_score = scores.mean()

    return mean_score

# Define the search space for hyperparameters
space = {
    'alpha': hp.loguniform('alpha', -6, 2),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'max_iter': hp.choice('max_iter', range(100, 10000))
}

# Create a trials object to store the optimization results
trials = Trials()

# Run the hyperparameter optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# Train the final model with the best hyperparameters
final_model = ElasticNet(
    alpha=best['alpha'],
    l1_ratio=best['l1_ratio'],
    fit_intercept=best['fit_intercept'],
    max_iter=best['max_iter'],
    random_state=0
)
final_model.fit(X, y)

# Evaluate the RMSE on the training data
y_pred = final_model.predict(X)
rmse = mean_squared_error(y, y_pred, squared=False)

# Print the RMSE of the final model with the best hyperparameters
print("RMSE: {:.4f}".format(rmse))
