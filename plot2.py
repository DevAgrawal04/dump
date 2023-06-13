import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from xgboost import XGBRegressor

# Create a dummy dataset with 10 features
X, _ = make_classification(n_samples=1000, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

# Select specific indices
indices = [4, 10, 530, 774]
selected_df = df.loc[indices]

# Duplicate rows 1000 times
selected_df = pd.concat([selected_df] * 1000, ignore_index=True)

# Add values to feature_4
selected_df['feature_4'] = np.linspace(-2, 2, num=len(selected_df))

# Initialize the XGBoost regressor
xgb_r = XGBRegressor()

# Store predictions separately
predictions = []

# Make predictions for each dataframe
for i in range(len(selected_df)):
    x = selected_df.iloc[[i]]
    prediction = xgb_r.predict(x)[0]
    predictions.append(prediction)

# Add predictions as a new column to each dataframe
selected_df['prediction'] = predictions

# Get unique dataframe labels
dataframe_labels = selected_df.index // 1000

# Convert feature_4 and predictions to lists
x_values = selected_df['feature_4'].tolist()
y_values = selected_df['prediction'].tolist()

# Create a plot
fig = go.Figure()

# Add scatter traces for each dataframe
for label in set(dataframe_labels):
    df_indices = np.where(dataframe_labels == label)[0]
    name = f'Dataframe {label}'

    x = [x_values[i] for i in df_indices]
    y = [y_values[i] for i in df_indices]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name=name,
        legendgroup=name,  # Assign same legend group to link legend and scatter trace
        showlegend=True,  # Show legend entry for each scatter trace
        legendrank=label,  # Specify legend rank to control order of legend entries
    )
    fig.add_trace(trace)

# Set axis labels and title
fig.update_layout(
    xaxis_title='Feature 4',
    yaxis_title='Predictions',
    title='Predictions vs Feature 4'
)

# Display the plot
fig.show()
