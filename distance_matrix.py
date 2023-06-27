import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Assuming your dataframe is called 'df' and contains the data

# Compute the pairwise distances between data points
distances = pdist(df.values)

# Convert the distances to a square matrix
distance_matrix = squareform(distances)

# Set the threshold for distance below which data points will be removed
threshold = 0.5  # Adjust this value as per your requirement

# Find the indices of data points to be removed
indices_to_remove = []
removed_indices = set()  # To keep track of the removed indices
for i, row in enumerate(distance_matrix):
    # Exclude the diagonal element and already removed indices
    close_indices = [j for j, d in enumerate(row) if j != i and d < threshold and j not in removed_indices]
    if close_indices:
        indices_to_remove.append(close_indices[0])  # Remove only the first close point
        removed_indices.add(close_indices[0])  # Add the removed index to the set

# Remove the data points from the dataframe
df = df.drop(df.index[indices_to_remove])

# Reset the index of the modified dataframe
df = df.reset_index(drop=True)
