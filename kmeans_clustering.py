"""
K-means clustering on  t-SNE reduced features.

"""

import os
os.environ['OMP_NUM_THREADS'] = '3'

import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
import warnings

# Database credentials (update as needed)
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Password',
    'database': 'drug_response_predict'
}

# Create database connection
engine = create_engine(
    f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

try:
    # Load integrated dataset
    query = "SELECT * FROM drug_proexp_mutcnv_mirna_meta_tsne_result"
    df = pd.read_sql(query, con=engine)

    # Define feature groups
    mirna_columns = ['mirna_TSNE1', 'mirna_TSNE2']
    metabolomics_columns = ['metabolomics_TSNE1', 'metabolomics_TSNE2']
    copynumber = ['Dimension_1_cnv', 'Dimension_2_cnv', 'Dimension_3_cnv']
    expression = ['Dimension_1_expression', 'Dimension_2_expression', 'Dimension_3_expression']
    mutation = ['Dimension_1_mutation', 'Dimension_2_mutation', 'Dimension_3_mutation']
    protein_array = ['Dimension_1_protein_array', 'Dimension_2_protein_array', 'Dimension_3_protein_array']

    # Select single or multi-omics feature set(s) for clustering
    feature_columns = (
        mirna_columns +
        metabolomics_columns +
        copynumber +
        expression +
        mutation +
        protein_array
    )

    # Prepare data for clustering
    X_cluster = df[feature_columns]

    # Apply k-means clustering
    k = 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_cluster)

    # Add cluster labels
    df['Cluster'] = clusters

    print("Clustering completed.")

    # Save each cluster as separate table
    for cluster_id in range(k):
        cluster_data = df[df['Cluster'] == cluster_id]
        table_name = f'cluster_{cluster_id}_data'

        cluster_data.to_sql(
            name=table_name,
            con=engine,
            index=False,
            if_exists='replace'
        )

        print(f"Cluster {cluster_id} saved to table: {table_name}")

except Exception as e:
    print("Error:", e)

finally:
    engine.dispose()