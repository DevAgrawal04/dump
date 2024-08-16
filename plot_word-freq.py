import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Sample DataFrame
data = {
    'Processed_Report': [
        'system failure error occurred',
        'error in system reboot',
        'failure in network connection',
        'network issue and system error',
    ],
    'MI_Incident': ['High', 'Medium', 'High', 'Low']
}

df = pd.DataFrame(data)

# Step 1: Tokenize and count word frequencies, grouped by MI_Incident
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Processed_Report'])
words = vectorizer.get_feature_names_out()

# Create a DataFrame with word frequencies
word_counts = pd.DataFrame(X.toarray(), columns=words)
word_counts['MI_Incident'] = df['MI_Incident']

# Step 2: Melt the DataFrame to get a long format
word_counts_long = word_counts.melt(id_vars=['MI_Incident'], var_name='Word', value_name='Frequency')
word_counts_long = word_counts_long[word_counts_long['Frequency'] > 0]  # Filter out zero frequencies

# Step 3: Group by word and MI_Incident and sum frequencies
word_frequencies = word_counts_long.groupby(['Word', 'MI_Incident']).sum().reset_index()

# Step 4: Plot the word frequency bar chart
plt.figure(figsize=(14, 8))
sns.barplot(data=word_frequencies, x='Word', y='Frequency', hue='MI_Incident')

# Customize the plot
plt.title('Word Frequency by MI_Incident Category')
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.xticks(rotation=45, ha='right')
plt.legend(title='MI_Incident')
plt.show()
