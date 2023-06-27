import numpy as np
from scipy.optimize import minimize

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

# Prepare your data as numpy arrays
features = np.array([f1_values, f2_values, f3_values, f4_values, f5_values, f6_values, f7_values, f8_values])  # Assuming you have numpy arrays for each feature
x = np.array(x_values)  # Assuming you have a numpy array of x values
y = np.array(target_values)  # Assuming you have a numpy array of target values

# Initial guess for weights and biases
initial_params = np.ones(27)  # You can adjust the initial values as needed

# Minimize the objective function using gradient descent
result = minimize(objective, initial_params, args=(features, x, y), method='BFGS')

# Get the optimized weights and biases
optimized_params = result.x

# Extract the weights and biases for a, b, and k
a_weights = optimized_params[:8]
a_bias = optimized_params[8]
b_weights = optimized_params[9:17]
b_bias = optimized_params[17]
k_weights = optimized_params[18:26]
k_bias = optimized_params[26]

# Print the optimized weights and biases
print("a_weights:", a_weights)
print("a_bias:", a_bias)
print("b_weights:", b_weights)
print("b_bias:", b_bias)
print("k_weights:", k_weights)
print("k_bias:", k_bias)
