import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Create a dummy dataset with 10 features
np.random.seed(42)
data = np.random.randn(1000, 10)
df = pd.DataFrame(data, columns=['feature_'+str(i+1) for i in range(10)])

# Select specific indices
indices = [4, 10, 530, 774]
df_selected = df.iloc[indices]

# Duplicate rows in each dataframe
df_selected = pd.concat([df_selected] * 1000, ignore_index=True)

# Change feature_4 values
df_selected['feature_4'] = np.linspace(-2, 2, num=len(df_selected))

# Create a plot
fig = go.Figure()

# Add scatter plots for each dataframe
for i in range(len(indices)):
    fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['feature_2'],
                             mode='markers', name='Index '+str(indices[i])))

# Set plot layout and labels
fig.update_layout(title='Feature 2 vs. Feature 4',
                  xaxis_title='Index',
                  yaxis_title='Feature 2',
                  legend_title='Dataframes')

# Show the plot
fig.show()
