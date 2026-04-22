"""
t-SNE dimensionality reduction for protein array data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# File path and name (replace 'xxx' with your system path)
file_path = r'C:\Users\xxx\Downloads'
file_name = 'Protein_Array.csv'
full_file_path = f'{file_path}\\{file_name}'

# Load data
original_df = pd.read_csv(full_file_path)

# Extract features
X = original_df.iloc[:, 1:]

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE dimensionality reduction
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create DataFrame
tsne_df = pd.DataFrame(
    X_tsne,
    columns=[
        'Dimension_1_protein_array',
        'Dimension_2_protein_array',
        'Dimension_3_protein_array'
    ]
)

# Add identifier
tsne_df['Identifier'] = original_df.iloc[:, 0]

# Reorder columns
tsne_df = tsne_df[
    ['Identifier',
     'Dimension_1_protein_array',
     'Dimension_2_protein_array',
     'Dimension_3_protein_array']
]

# Save output
output_file = f'{file_path}\\{file_name}_tsne_results.csv'
tsne_df.to_csv(output_file, index=False)

print("t-SNE reduction completed and saved.")