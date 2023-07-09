from sklearn.model_selection import StratifiedKFold

# Define the number of folds
num_folds = 5

# Initialize the cross-validation method
kf = StratifiedKFold(n_splits=num_folds)

# Initialize a list to store the evaluation results
accuracy_scores = []

# Perform cross-validation
for train_index, test_index in kf.split(X, y):
    # Create the train-test split for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train your custom model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy = sum(accuracy_scores) / num_folds

# Print the results
print("Average Accuracy:", average_accuracy)
