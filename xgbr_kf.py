import numpy as np
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

# Generate a sample regression dataset
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)

# Define the objective function with cross-validation
def objective(params):
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse = np.mean(scores)
    return mse

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
    'max_depth': hp.choice('max_depth', range(1, 10)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
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
