# pca

## About

Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction while preserving as much variability as possible. The main idea behind PCA is to identify patterns in data, express the data in such a way to highlight their similarities and differences, and, most importantly, to reduce the number of variables used in an analysis.

## Main Idea

PCA transforms the original variables into a new set of variables, which are linear combinations of the original variables. These new variables, called principal components (PCs), are ordered by the amount of variance they capture from the data. The first principal component captures the maximum variance, the second captures the largest remaining variance under the constraint that it is orthogonal to the first, and so on.

## When It Is Used

* Visualization: When dealing with high-dimensional data, PCA is used to simplify the complexity in high-dimensional data while retaining trends and patterns.
* Noise Reduction: By keeping only the significant principal components and ignoring the lower ones, which might represent noise, PCA can help in clarifying the true signals in data.
* Feature Extraction and Engineering: PCA can be used to derive new features that can be useful for machine learning models, especially when dealing with correlated variables.
* Efficiency: Reducing the dimension of data using PCA can decrease the computational costs for processing data and can improve the efficiency of machine learning algorithms.

