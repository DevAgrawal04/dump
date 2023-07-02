import plotly.graph_objects as go
import pandas as pd

# Create dataframes
df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
df2 = pd.DataFrame({'x': [1, 2, 3], 'y': [7, 8, 9]})
df3 = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 11, 12]})
df4 = pd.DataFrame({'x': [1, 2, 3], 'y': [13, 14, 15]})
df5 = pd.DataFrame({'x': [1, 2, 3], 'y': [16, 17, 18]})

dataframes = [df1, df2, df3, df4, df5]

# Create empty list for traces
traces = []

# Create traces for each dataframe and add to the list
for i, df in enumerate(dataframes):
    trace = go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        name=f'Dataframe {i+1}'
    )
    traces.append(trace)

# Create layout
layout = go.Layout(
    title='Multiple Dataframes Plot',
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Create figure and add traces
fig = go.Figure(data=traces, layout=layout)

# Display the plot
fig.show()
