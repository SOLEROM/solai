# oose 

## Out-of-Sample Extension in PCA

### Overview
Principal Component Analysis (PCA) is a widely used technique for dimensionality reduction and feature extraction. It transforms the original data into a new coordinate system where the greatest variances by any projection of the data come to lie on the first coordinates (called principal components), the second greatest variances on the second coordinates, and so on.

However, PCA is typically applied to a fixed dataset, and the principal components are computed based on this dataset. When we encounter new data (out-of-sample data), we need to project this new data onto the existing principal components. This process is known as **out-of-sample extension**.

### Mathematical Explanation
Given a dataset $ X \in \mathbb{R}^{n \times d} $ with $ n $ samples and $ d $ features, we perform PCA to obtain the principal components. The steps are as follows:

1. **Center the Data**: Subtract the mean of each feature from the data.
    $$
    X_{\text{centered}} = X - \mu
    $$
    where $ \mu $ is the mean vector of $ X $.

2. **Compute the Covariance Matrix**:
    $$
    \Sigma = \frac{1}{n-1} X_{\text{centered}}^\top X_{\text{centered}}
    $$

3. **Eigen Decomposition**: Perform eigen decomposition of the covariance matrix to obtain eigenvalues and eigenvectors.
    $$
    \Sigma = V \Lambda V^\top
    $$
    where $ V $ contains the eigenvectors (principal components) and $ \Lambda $ is a diagonal matrix of eigenvalues.

4. **Project the Data**: Project the centered data onto the principal components.
    $$
    Z = X_{\text{centered}} V
    $$
    where $ Z $ is the transformed data in the principal component space.

For an out-of-sample data point $ x_{\text{new}} $, the steps are:

1. **Center the New Data Point**: Subtract the mean $ \mu $ of the original training data.
    $$
    x_{\text{new, centered}} = x_{\text{new}} - \mu
    $$

2. **Project onto Principal Components**: Use the previously computed eigenvectors $ V $ to project the centered new data point.
    $$
    z_{\text{new}} = x_{\text{new, centered}} V
    $$

### Real-World Example
Consider a face recognition system trained on a dataset of faces. The PCA has already been performed on the training dataset to obtain the principal components. Now, we want to recognize a new face that was not part of the training dataset.

1. **Center the New Face**: Subtract the mean face (computed from the training data) from the new face image.
2. **Project onto Principal Components**: Use the principal components (eigenfaces) computed from the training data to project the centered new face image.

This allows the new face image to be represented in the same principal component space as the training data, enabling the recognition system to compare and recognize the new face effectively.

### Python Example (if requested)
```python
import numpy as np
from sklearn.decomposition import PCA

# Assuming X_train is the training data
X_train = np.random.rand(100, 50)  # Example training data
pca = PCA(n_components=10)
pca.fit(X_train)

# Mean of the training data
mu = np.mean(X_train, axis=0)

# New out-of-sample data
X_new = np.random.rand(1, 50)  # Example new data point

# Center the new data
X_new_centered = X_new - mu

# Project the new data
X_new_pca = pca.transform(X_new_centered)

print(X_new_pca)
```

This example uses `sklearn`'s PCA implementation to demonstrate how to project new data onto the principal components obtained from the training data. The new data point `X_new` is centered using the mean of the training data and then transformed using the PCA model.

