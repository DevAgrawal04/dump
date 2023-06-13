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
        name=name
    )
    fig.add_trace(trace)
