# Full PCA

## Centring and Rotation

Principal Component Analysis (PCA) is a technique used for dimensionality reduction while preserving as much variance as possible. The core idea of PCA involves centring the data and then rotating it to align with the directions of maximum variance.

1. **Centring the Data**: This involves subtracting the mean of each feature from the data to shift the origin to the center of the data.

2. **Rotation**: This involves rotating the data such that the axes are aligned with the directions of maximum variance. These directions are called principal components.

## PCA Steps

The PCA process can be summarized in the following steps:

1. **Standardization**: Subtract the mean and divide by the standard deviation for each feature.
   
2. **Covariance Matrix Computation**: Calculate the covariance matrix of the standardized data.

3. **Eigen Decomposition**: Compute the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors represent the directions of maximum variance (principal components), and the eigenvalues represent the magnitude of variance in these directions.

4. **Feature Vector**: Form a feature vector by selecting the top $ k $ eigenvectors.

5. **Recast the Data**: Transform the original data into the new subspace using the feature vector.

### Mathematical Formulation

Let $ \mathbf{X} $ be the $ m \times n $ data matrix where $ m $ is the number of samples and $ n $ is the number of features.

1. **Centring the Data**:
   $$
   \mathbf{X}_{centered} = \mathbf{X} - \mathbf{\mu}
   $$
   where $ \mathbf{\mu} $ is the mean vector of $ \mathbf{X} $.

2. **Covariance Matrix**:
   $$
   \mathbf{C} = \frac{1}{m-1} \mathbf{X}_{centered}^T \mathbf{X}_{centered}
   $$

3. **Eigen Decomposition**:
   $$
   \mathbf{C} \mathbf{v}_i = \lambda_i \mathbf{v}_i
   $$
   where $ \lambda_i $ are the eigenvalues and $ \mathbf{v}_i $ are the corresponding eigenvectors.

4. **Transformation**:
   $$
   \mathbf{X}_{transformed} = \mathbf{X}_{centered} \mathbf{V}_k
   $$
   where $ \mathbf{V}_k $ is the matrix of the top $ k $ eigenvectors.

## Examples

Consider a dataset with features that are highly correlated. PCA can be used to reduce the dimensionality by finding new axes (principal components) that capture the most variance.

### Example: Iris Dataset

The Iris dataset has four features (sepal length, sepal width, petal length, petal width). Applying PCA to this dataset can reduce it to two principal components that capture most of the variance.

### Real-World Applications

1. **Image Compression**: Reducing the dimensionality of image data for storage and processing efficiency.
2. **Face Recognition**: Using PCA to identify key features in facial recognition systems.
3. **Genomics**: Analyzing high-dimensional genetic data to identify key genetic variations.

## Reconstruction Error

PCA minimizes the reconstruction error, which is the difference between the original data and its projection onto the principal components. This is achieved by selecting the top $ k $ components that capture the most variance.

### Mathematical Representation

The reconstruction error is given by:

$$
\text{Error} = \| \mathbf{X} - \mathbf{X}_{reconstructed} \|^2
$$

where $ \mathbf{X}_{reconstructed} = \mathbf{X}_{transformed} \mathbf{V}_k^T $.

### Non-Unique Solutions

The solution to PCA is not unique because the direction of the principal components can be flipped (eigenvectors can be multiplied by -1 without changing their direction). Additionally, the components are determined up to a rotation if their corresponding eigenvalues are equal.

