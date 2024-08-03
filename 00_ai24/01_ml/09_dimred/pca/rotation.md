# Rotation 

# PCA Rotation with Change of Basis

Principal Component Analysis (PCA) is a technique used for dimensionality reduction and feature extraction in machine learning. It works by identifying the directions (principal components) in which the data varies the most. These principal components are then used to reorient the data along new axes.

## General Overview

PCA involves two main steps:
1. **Finding the Principal Components:** These are the eigenvectors of the covariance matrix of the data.
2. **Projecting the Data:** The data is projected onto the principal components to reduce its dimensionality.

### Change of Basis

When we talk about the "change of basis" in PCA, we are referring to transforming the original coordinate system of the data to a new coordinate system defined by the principal components. This is essentially a rotation of the data along the new axes.

### Mathematical Explanation

1. **Standardize the Data:** Ensure the data has a mean of zero and a standard deviation of one.
   $$
   X_{\text{standardized}} = \frac{X - \mu}{\sigma}
   $$

2. **Compute the Covariance Matrix:**
   $$
   \Sigma = \frac{1}{n-1} X_{\text{standardized}}^T X_{\text{standardized}}
   $$

3. **Compute the Eigenvalues and Eigenvectors of the Covariance Matrix:**
   $$
   \Sigma v = \lambda v
   $$
   Here, \(\lambda\) are the eigenvalues and \(v\) are the eigenvectors.

4. **Form the Feature Vector:** This is a matrix of the eigenvectors.
   $$
   \text{Feature Vector} = [v_1, v_2, \ldots, v_k]
   $$

5. **Transform the Data:** Multiply the original data by the feature vector to get the new coordinates (principal components).
   $$
   X_{\text{transformed}} = X_{\text{standardized}} \times \text{Feature Vector}
   $$

### Real-World Example

Consider a dataset with measurements of different types of flowers. Each type of measurement (e.g., petal length, petal width) can be considered a dimension. If we want to visualize this data in 2D, we can use PCA to reduce the dimensionality from, say, 4D to 2D. The principal components will be the new axes along which the variance of the data is maximized.

### Practical Implications

- **Noise Reduction:** By focusing on the principal components that capture the most variance, we can often ignore components that are likely to be noise.
- **Visualization:** Reducing data to 2 or 3 dimensions using PCA allows us to visualize it, which is invaluable for understanding the underlying structure.
- **Feature Extraction:** PCA helps in extracting the most important features that contribute to the variance in the data.

### Python Code Example

Here is a simple Python example using PyTorch to perform PCA:

```python
import torch
import numpy as np

# Generate some data
np.random.seed(42)
data = np.random.randn(100, 3)

# Standardize the data
data_mean = torch.mean(torch.tensor(data), dim=0)
data_std = torch.std(torch.tensor(data), dim=0)
data_standardized = (torch.tensor(data) - data_mean) / data_std

# Compute the covariance matrix
cov_matrix = torch.matmul(data_standardized.T, data_standardized) / (data_standardized.shape[0] - 1)

# Eigen decomposition
eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
sorted_indices = torch.argsort(eigenvalues, descending=True)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the new basis
data_transformed = torch.matmul(data_standardized, eigenvectors[:, :2])

print("Transformed Data:\n", data_transformed)
```

In this code, we standardize the data, compute the covariance matrix, perform eigen decomposition, sort the eigenvalues and eigenvectors, and finally project the data onto the new basis defined by the top principal components.

By changing the basis of the data to the principal components, PCA helps in simplifying the data, reducing noise, and making it easier to visualize and understand.
