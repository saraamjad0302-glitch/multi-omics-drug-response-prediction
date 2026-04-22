# Multi-Omics Feature Strength Evaluation Framework

## Overview

This repository contains code for evaluating the predictive strength of multi-omics and single-omics feature sets in cancer drug response prediction.

The framework integrates multiple omics layers, performs clustering, and applies machine learning to assess feature utility.

---

## Workflow

1. **Dimensionality Reduction**
   t-SNE applied to each omics dataset.

2. **Data Integration**
   Iterative integration of t-SNE reduced omics features and drug response (AUC).

3. **Clustering**
   K-means clustering on integrated features.

4. **Cluster Evaluation**
   Silhouette Score and Davies–Bouldin Index.

5. **Prediction**
   Random Forest model for drug response prediction.

---

## Files

* `excel_tsne_dimensionality_reduction.py` – t-SNE reduction
* `kmeans_clustering.py` – clustering
* `calculate cluster size.py` – cluster evaluation
* `rf_prediction.py` – prediction model
* `sql code.txt` – data integration
* `Primary and Validation dataset drugs.docx` – drug lists

---

## Notes

* Replace database credentials and file paths before running.
* Drug lists can be directly copied into the code.
* Supports both single-omics and multi-omics feature configurations.

---

## Authors
- Sara Amjad  
- M. M. Sufyan Beg  
- Mohd Azhar Aziz  
