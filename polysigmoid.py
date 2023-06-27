import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

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

# Define the nonlinear constraints
def constraint(params):
    b_weights = params[9:17]
    k_bias = params[26]
    b_value = np.dot(features.T, b_weights) + params[17]
    return [0.1 - b_value, k_bias - 1]

# Prepare your data as numpy arrays
f1_values = np.array(f1_values)  # Assuming you have a numpy array for f1_values
f2_values = np.array(f2_values)  # Assuming you have a numpy array for f2_values
# ... Repeat for other feature arrays

# Compute polynomial features
degree = 2  # Adjust the degree of the polynomial as needed
poly_features = []
for feature_values in [f1_values, f2_values]:
    poly_feature = np.polyval(np.polyfit(x, feature_values, degree), x)
    poly_features.append(poly_feature)
features = np.array(poly_features).T

# Initial guess for weights and biases
initial_params = np.ones(27)  # You can adjust the initial values as needed

# Create the nonlinear constraint object
nonlinear_constraint = NonlinearConstraint(constraint, lb=[-np.inf, 1], ub=[0, np.inf])

# Minimize the objective function using trust-constr with nonlinear constraints
result = minimize(objective, initial_params, args=(features, x, y), method='trust-constr', constraints=nonlinear_constraint)

# Get the optimized weights and biases
optimized_params = result.x

# Extract the weights and biases for a, b, and k
a_weights = optimized_params[:8]
a_bias = optimized_params[8]
b_weights = optimized_params[9:17]
b_bias = optimized_params[17]
k_weights = optimized_params[18:26]
k_bias = optimized_params[26]

# Calculate a_value using the dot product of features and a_weights
a_value = np.dot(features, a_weights) + a_bias

# Print the optimized weights and biases
print("a_weights:", a_weights)
print("a_bias:", a_bias)
print("b_weights:", b_weights)
print("b_bias:", b_bias)
print("k_weights:", k_weights)
print("k_bias:", k_bias)
print("a_value:", a_value)
