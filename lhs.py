import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyDOE import lhs


df = pd.DataFrame(...)  # Replace ... with your data
scaler = StandardScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

num_samples = 10000  # Replace with the desired number of samples

samples = lhs(normalized_df.shape[1], samples=num_samples)

original_samples = scaler.inverse_transform(samples)

sampled_df = pd.DataFrame(original_samples, columns=df.columns)
