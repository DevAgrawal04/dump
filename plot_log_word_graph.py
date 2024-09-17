import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Improved frequency generator using Counter for optimization
def gen_freq(descriptions):
    count = Counter()
    for desc in descriptions:
        count.update(desc.split())  # Efficiently split and count words in one step
    return count

# Frequency calculation for positive and negative descriptions
pos_freq = gen_freq(df[df['Combined_Output'] == 1]['stemmed_description'])
neg_freq = gen_freq(df[df['Combined_Output'] == 0]['stemmed_description'])

# Combine unique keys from both positive and negative frequencies
all_keys = set(pos_freq.keys()).union(set(neg_freq.keys()))

# Create a DataFrame with word, positive count, and negative count
data = [(key, pos_freq.get(key, 0), neg_freq.get(key, 0)) for key in all_keys]
df_freq = pd.DataFrame(data, columns=['key', 'pos_count', 'neg_count'])

# Function to plot log graph for positive and negative word frequencies
def plot_log_graph(df_freq):
    # Extract data from the DataFrame
    data = df_freq.values.tolist()

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert positive and negative raw counts to Logarithmic scale
    x = np.log([x[1] + 1 for x in data])  # Log of positive count (+1 to avoid log(0))
    y = np.log([x[2] + 1 for x in data])  # Log of negative count (+1 to avoid log(0))

    # Plot a scatter plot for each word
    ax.scatter(x, y)

    # Assign axis labels
    ax.set_xlabel("Log Positive Count", fontsize=14)
    ax.set_ylabel("Log Negative Count", fontsize=14)

    # Annotate the words at the same position as the points
    for i in range(len(data)):
        ax.annotate(data[i][0], (x[i], y[i]), fontsize=10, alpha=0.75)

    # Plot the red line that divides the two areas (y=x line)
    ax.plot([0, max(x)], [0, max(y)], color='red', linewidth=2)

    # Add grid and title
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Log-Log Plot of Word Frequencies (Positive vs Negative)", fontsize=16)

    # Show the plot
    plt.show()

# Call the function to plot the graph
plot_log_graph(df_freq)
