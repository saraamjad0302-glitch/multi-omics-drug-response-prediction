# -*- coding: utf-8 -*-

"""
Random Forest-based drug response prediction 
"""

import os
os.environ['OMP_NUM_THREADS'] = '3'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Database credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Password',
    'database': 'drug_response_predict'
}

# Database connection
engine = create_engine(
    f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

try:
    # Load clustered data
    query = "SELECT * FROM cluster_1_data"
    df = pd.read_sql(query, con=engine)

    # Feature groups
    mirna_columns = ['mirna_TSNE1', 'mirna_TSNE2']
    metabolomics_columns = ['metabolomics_TSNE1', 'metabolomics_TSNE2']
    copynumber = ['Dimension_1_cnv', 'Dimension_2_cnv', 'Dimension_3_cnv']
    expression = ['Dimension_1_expression', 'Dimension_2_expression', 'Dimension_3_expression']
    mutation = ['Dimension_1_mutation', 'Dimension_2_mutation', 'Dimension_3_mutation']
    protein_array = ['Dimension_1_protein_array', 'Dimension_2_protein_array', 'Dimension_3_protein_array']

    # Select features
    feature_columns = (
        mirna_columns +
        metabolomics_columns +
        copynumber +
        expression +
        mutation +
        protein_array
    )

    # Drug list provided separately
    target_variables = [...]  

    # Model
    model = RandomForestClassifier(n_estimators=300, random_state=42)

    accuracies = []

    print("\nModel: Random Forest")

    for target_var in target_variables:
        print(f"\nTarget Variable: {target_var}")

        df_target = df[df[target_var] != 0]

        X = df_target[feature_columns]
        y = df_target[target_var]

        # Binary classification
        threshold = y.mean()
        y_binary = (y >= threshold).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.2f}")
        accuracies.append(acc)

    # Split for plotting
    target_split = [target_variables[i:i+30] for i in range(0, len(target_variables), 30)]
    acc_split = [accuracies[i:i+30] for i in range(0, len(accuracies), 30)]

    # Accuracy plots
    for i in range(len(target_split)):
        plt.figure(figsize=(10, 10))
        bars = plt.bar(target_split[i], acc_split[i])

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Plot - Part {i+1}")

        for bar, acc in zip(bars, acc_split[i]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{acc:.2f}", fontsize=8)

        plt.tight_layout()
        plt.show()

    # High accuracy drugs (>0.75)
    mean_high_acc = []

    plt.rc('font', family='serif', size=8)
    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(target_split)):
        high_drugs = [
            drug for drug, acc in zip(target_split[i], acc_split[i])
            if acc > 0.75
        ]
        high_acc = [acc for acc in acc_split[i] if acc > 0.75]

        if high_acc:
            bars = ax.bar(high_drugs, high_acc)

            for bar, acc, drug in zip(bars, high_acc, high_drugs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{acc:.2f}", fontsize=8)
                print(drug)

            mean_val = np.mean(high_acc)
            mean_high_acc.append(mean_val)
            print(f"\nMean Accuracy (Part {i+1}): {mean_val:.4f}")

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Prediction Accuracy")
    plt.title("High Accuracy Drugs")
    plt.tight_layout()
    plt.savefig('your_graph.tiff', dpi=300)
    plt.show()

    if mean_high_acc:
        print(f"\nOverall Mean Accuracy: {np.mean(mean_high_acc):.4f}")
    else:
        print("\nNo high accuracy drugs found.")

except Exception as e:
    print("Error:", e)

finally:
    engine.dispose()