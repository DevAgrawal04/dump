import numpy as np
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate a sample regression dataset
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def objective(params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
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
