import optuna
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the objective function to be minimized (in this case, mean squared error)
def objective(trial):
    # Define the search space for hyperparameters
    param_space = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 10.0)
    }

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    mse_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train the model on the training fold
        model = xgb.XGBRegressor(**param_space)
        model.fit(X_train_fold, y_train_fold)

        # Make predictions on the validation fold
        y_pred = model.predict(X_val_fold)

        # Calculate mean squared error
        mse = mean_squared_error(y_val_fold, y_pred)
        mse_scores.append(mse)

    # Return the average mean squared error across folds
    return np.mean(mse_scores)

# Define your training data (X_train, y_train) and validation data (X_val, y_val) here

# Create a study object and optimize the objective function using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best trial and its hyperparameters
best_trial = study.best_trial
print('Best trial:')
print(f'  Value: {best_trial.value:.4f}')
print('  Params: ')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')
