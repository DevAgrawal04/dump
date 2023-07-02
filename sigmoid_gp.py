import numpy as np
from sklearn.model_selection import train_test_split
from skopt import gp_minimize

# Define the sigmoid function
def sigmoid(x, k, a, b):
    return k / (1 + np.exp(a + b * x))

# Define the objective function to minimize
def objective(params, features, x, y):
    a_weights = params[:8]
    a_bias = params[8]
    b_weights = params[9:17]
    b_bias = params[17]
    k_weights = params[18:26]
    k_bias = params[26]
    
    a_value = np.dot(features, a_weights) + a_bias
    b_value = np.dot(features, b_weights) + b_bias
    k_value = np.dot(features, k_weights) + k_bias
    
    y_pred = sigmoid(x, k_value, a_value, b_value)
    
    return np.mean((y_pred - y) ** 2)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test, features_train, features_test = train_test_split(
    x, y, features, test_size=0.2, random_state=42
)

# Prepare the training data as numpy arrays
features_train = np.array(features_train)
x_train = np.array(x_train)
y_train = np.array(y_train)

# Define the training objective function
train_objective = lambda params: objective(params, features_train, x_train, y_train)

# Minimize the objective function using GP optimization without bounds
result = gp_minimize(train_objective, n_calls=100, n_random_starts=10, random_state=42)

# Get the optimized weights and biases
optimized_params = result.x

# Prepare the testing data as numpy arrays
features_test = np.array(features_test)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Extract the weights and biases for a, b, and k
a_weights = optimized_params[:8]
a_bias = optimized_params[8]
b_weights = optimized_params[9:17]
b_bias = optimized_params[17]
k_weights = optimized_params[18:26]
k_bias = optimized_params[26]

# Make predictions on the testing data
a_value = np.dot(features_test, a_weights) + a_bias
b_value = np.dot(features_test, b_weights) + b_bias
k_value = np.dot(features_test, k_weights) + k_bias
y_pred = sigmoid(x_test, k_value, a_value, b_value)

# Evaluate the model
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)
