# Eigen Decomposition



### Principal Component Analysis (PCA) and Eigen Decomposition

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information in the large set. One of the key mathematical concepts behind PCA is eigen decomposition. Let's delve into the subtopics to understand this process better.

#### The Covariance Matrix

The covariance matrix is a square matrix that contains the covariances between pairs of variables in a dataset. If we have a dataset with $ n $ variables, the covariance matrix will be an $ n \times n $ matrix.

- **Definition**: The covariance between two variables $ X $ and $ Y $ is given by:
  $$
  \text{cov}(X, Y) = \frac{1}{N-1} \sum_{i=1}^N (X_i - \mu_X)(Y_i - \mu_Y)
  $$
  where $ \mu_X $ and $ \mu_Y $ are the means of $ X $ and $ Y $, respectively, and $ N $ is the number of data points.
- **Purpose**: The covariance matrix provides a measure of how much each variable in the dataset varies from the mean with respect to each other variable.

#### Variance of a Random Variable

Variance measures the spread of a set of numbers. For a random variable $ X $, the variance is the expectation of the squared deviation of $ X $ from its mean $ \mu $.

- **Definition**: The variance of $ X $ is given by:
  $$
  \text{Var}(X) = \mathbb{E}[(X - \mu_X)^2]
  $$
  where $ \mathbb{E} $ denotes the expected value.
- **Importance in PCA**: In PCA, the variance of each principal component (new variable) helps to understand how much of the total variability is captured by that component.

#### Covariance - Eigenvalues

Eigenvalues ($ \lambda $) are scalars associated with a square matrix in linear algebra. In the context of PCA, eigenvalues of the covariance matrix indicate the magnitude of the variance captured by each principal component.

- **Computation**: For a covariance matrix $ \Sigma $, an eigenvalue $ \lambda $ satisfies:
  $$
  \Sigma \mathbf{v} = \lambda \mathbf{v}
  $$
  where $ \mathbf{v} $ is the corresponding eigenvector.
- **Significance**: The eigenvalues indicate the amount of variance explained by each principal component. Larger eigenvalues correspond to components that capture more variance.

#### Covariance - Eigenvectors

Eigenvectors ($ \mathbf{v} $) are non-zero vectors that, when multiplied by the covariance matrix, yield a scalar multiple of themselves (the eigenvalue).

- **Computation**: As described above, the eigenvectors satisfy the equation:
  $$
  \Sigma \mathbf{v} = \lambda \mathbf{v}
  $$
- **Role in PCA**: The eigenvectors of the covariance matrix define the directions of the new feature space (principal components). Each eigenvector points in the direction of maximum variance in the data.

#### Covariance Decomposition

Covariance decomposition involves breaking down the covariance matrix into its eigenvalues and eigenvectors, which is the essence of eigen decomposition.

- **Process**: The covariance matrix $ \Sigma $ can be decomposed as:
  $$
  \Sigma = Q \Lambda Q^{-1}
  $$
  where $ Q $ is the matrix of eigenvectors and $ \Lambda $ is the diagonal matrix of eigenvalues.
- **Interpretation**: This decomposition allows us to understand the original data in terms of its principal components, simplifying analysis and dimensionality reduction.

#### Geometric Viewpoint

From a geometric perspective, PCA transforms the data into a new coordinate system where the axes (principal components) are aligned with the directions of maximum variance.

- **Transformation**: The data is projected onto the eigenvectors of the covariance matrix. Each data point $ \mathbf{x} $ is represented as:
  $$
  \mathbf{x}' = Q^T \mathbf{x}
  $$
  where $ \mathbf{x}' $ is the transformed data point in the new coordinate system.
- **Visualization**: In 2D or 3D, PCA can be visualized as rotating the coordinate system to align with the directions of highest variance, effectively flattening the data along the axes with the least variance.

### Summary

Eigen decomposition of the covariance matrix is central to PCA. The covariance matrix captures the pairwise variability of the dataset. Eigenvalues and eigenvectors derived from the covariance matrix reveal the principal components, which represent the directions and magnitudes of maximum variance in the data. Covariance decomposition breaks down the covariance matrix into these fundamental components, providing a clear geometric interpretation of the data's structure.