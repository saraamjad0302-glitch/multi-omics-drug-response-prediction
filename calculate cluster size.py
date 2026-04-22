# -*- coding: utf-8 -*-
"""
cluster quality evaluation
"""


import os
os.environ['OMP_NUM_THREADS'] = '3'
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sqlalchemy import create_engine
import warnings

# Database credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Password',
    'database': 'drug_response_predict'
}

# Establish a connection to the database
engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
try:
    #  SQL query to select table
    query = "SELECT * FROM drug_proexp_mutcnv_mirna_meta_tsne_result"
   

    # Read data into a DataFrame
    df = pd.read_sql(query, con=engine)

    mirna_columns = ['mirna_TSNE1', 'mirna_TSNE2']
    metabolomics_columns = ['metabolomics_TSNE1', 'metabolomics_TSNE2']
    copynumber = ['Dimension_1_cnv', 'Dimension_2_cnv', 'Dimension_3_cnv']
    expression = ['Dimension_1_expression', 'Dimension_2_expression', 'Dimension_3_expression']
    mutation = ['Dimension_1_mutation', 'Dimension_2_mutation', 'Dimension_3_mutation']
    protein_array = ['Dimension_1_protein_array', 'Dimension_2_protein_array', 'Dimension_3_protein_array']
    # Select Feature
    feature_columns = mirna_columns+  metabolomics_columns +copynumber + expression + mutation  + protein_array
    

    # Apply k-means clustering
    k = 5 # Number of clusters
    X_cluster = df[feature_columns]

    # Suppress KMeans memory leak warning temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_cluster)

    # Add cluster labels to the DataFrame
    df['Cluster'] = clusters

    # Calculate cluster sizes
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    print("Cluster Sizes:")
    print(cluster_sizes)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(X_cluster, clusters)
    print(f"Silhouette Score: {silhouette_avg}")

    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(X_cluster, clusters)
    print(f"Davies-Bouldin Index: {db_index}")

    # Check if there are more than one cluster before calculating silhouette scores
    if len(set(clusters)) > 1:
        # Calculate Silhouette Score for each data point
        silhouette_scores = silhouette_samples(X_cluster, clusters)

        # Create a DataFrame with the cluster labels and corresponding silhouette scores
        silhouette_df = pd.DataFrame({'Cluster': clusters, 'Silhouette_Score': silhouette_scores})

        # Calculate Average Silhouette Score for each cluster
        average_silhouette_scores = silhouette_df.groupby('Cluster')['Silhouette_Score'].mean()

        # Display average silhouette scores
        print("Average Silhouette Scores for Each Cluster:")
        print(average_silhouette_scores)
    else:
        print("Not enough clusters to calculate silhouette scores.")

except Exception as e:
    print("Error:", e)

finally:
    # Close the SQLAlchemy engine
    engine.dispose()
