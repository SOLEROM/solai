# svd (pca)

## Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique used to reduce the number of features in a dataset while retaining most of the variance. PCA achieves this by identifying the directions (principal components) along which the variance of the data is maximized. Singular Value Decomposition (SVD) is a mathematical method used in the computation of PCA.

## Singular Value Decomposition (SVD)

SVD is a factorization of a real or complex matrix. For a given matrix $ \mathbf{X} $ of size $ m \times n $, SVD can be represented as:

$$
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

Where:
- $ \mathbf{U} $ is an $ m \times m $ orthogonal matrix.
- $ \mathbf{\Sigma} $ is an $ m \times n $ diagonal matrix with non-negative real numbers on the diagonal, known as singular values.
- $ \mathbf{V} $ is an $ n \times n $ orthogonal matrix.

## Relationship Between PCA and SVD

PCA can be performed using SVD. The steps to perform PCA using SVD are as follows:

1. **Standardize the Data**: Ensure that the dataset is centered around the mean (mean of each feature is 0).

   $$
   \mathbf{X}_{\text{standardized}} = \mathbf{X} - \mathbf{\mu}
   $$

2. **Compute SVD**: Apply SVD to the standardized data matrix $ \mathbf{X}_{\text{standardized}} $.

   $$
   \mathbf{X}_{\text{standardized}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
   $$

3. **Principal Components**: The principal components are given by the columns of $ \mathbf{V} $.

4. **Explained Variance**: The variance explained by each principal component is given by the singular values $ \mathbf{\Sigma} $.

   $$
   \text{Explained Variance} = \frac{\mathbf{\Sigma}^2}{n - 1}
   $$

## Real-World Example

Consider an image processing application where you have a dataset of images, each represented by a large number of pixels (features). Directly working with such high-dimensional data can be computationally expensive and may lead to overfitting in machine learning models.

Using PCA, you can reduce the dimensionality of the images while preserving the essential information. Here’s a practical workflow:

1. **Standardize the Data**: Subtract the mean of each pixel value across all images.
2. **Apply SVD**: Decompose the standardized image matrix.
3. **Select Principal Components**: Choose the top $ k $ principal components that capture the most variance.
4. **Transform Data**: Project the original high-dimensional data onto the new $ k $-dimensional subspace.

By using PCA, you can reduce the complexity of your dataset, making it easier to analyze and process while still retaining most of the original information.

## Example in Python (Using PyTorch)

Here is a code example demonstrating PCA using SVD in PyTorch:

```python
import torch

# Generate a random dataset with 100 samples and 50 features
X = torch.randn(100, 50)

# Center the data
X_centered = X - X.mean(dim=0)

# Perform SVD
U, S, V = torch.svd(X_centered)

# Select the top k principal components
k = 10
V_k = V[:, :k]

# Transform the data to the new k-dimensional subspace
X_pca = torch.matmul(X_centered, V_k)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_pca.shape}")
```

This code snippet demonstrates how to perform PCA using SVD in PyTorch, reducing the dimensionality of the dataset from 50 features to 10 principal components.