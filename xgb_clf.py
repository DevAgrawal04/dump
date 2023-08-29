import numpy as np
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, KFold

#Dataset
# X, y = #PUT DATA AND TARGET HERE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 5 fold cv w/ accuracy metric
def objective(params):
    model = xgb.XGBClassifier(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    accuracy = np.mean(scores)
    return -accuracy  # Minimize negative accuracy to maximize accuracy

# # 5 fold cv w/ log_loss/cross_entropy metric
# def objective(params):
#     model = xgb.XGBClassifier(**params)
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_log_loss')  # Use neg_log_loss scoring
#     avg_log_loss = np.mean(scores)
#     return avg_log_loss

#Hyperparameter Search Space
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)), #Narrrow it down post analysis
    'max_depth': hp.choice('max_depth', range(1, 10)), 
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
}

#Optimization function
def optimize(space):
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials(),
        rstate=np.random.RandomState(2)
    )
    return best

#Output
best_params = optimize(space)
print("Best Parameters:", best_params)
