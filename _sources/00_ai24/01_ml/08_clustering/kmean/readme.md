## K-Means 

K-Means is a popular clustering algorithm used in machine learning to partition data points into $ K $ distinct clusters. The objective is to minimize the variance within each cluster.

### Objective Function

The goal of K-Means is to minimize the following objective function:

$$
J = \sum_{k=1}^{K} \sum_{i \in C_k} \| \mathbf{x}_i - \mathbf{\mu}_k \|^2
$$

where:
- $ \mathbf{x}_i $ is a data point,
- $ \mathbf{\mu}_k $ is the centroid of cluster $ k $,
- $ C_k $ is the set of points assigned to cluster $ k $,
- $ \| \cdot \| $ denotes the Euclidean distance.

### Algorithm

The K-Means algorithm proceeds as follows:

1. **Initialization**: Choose $ K $ initial centroids randomly.
2. **Assignment Step**: Assign each data point to the nearest centroid.
3. **Update Step**: Recompute the centroids as the mean of all points assigned to each centroid.
4. **Convergence Check**: Repeat the assignment and update steps until convergence (i.e., the centroids no longer change significantly).

### Detailed Steps

#### Step 1: Initialization

The initial centroids are chosen randomly from the data points or using the K-Means++ algorithm, which spreads out the initial centroids to improve convergence.

#### Step 2: Assignment

Each data point $ \mathbf{x}_i $ is assigned to the nearest centroid $ \mathbf{\mu}_k $:

$$
C_k = \{ \mathbf{x}_i : \| \mathbf{x}_i - \mathbf{\mu}_k \|^2 \leq \| \mathbf{x}_i - \mathbf{\mu}_j \|^2 \text{ for all } j \neq k \}
$$

#### Step 3: Update

The centroids are updated as the mean of the points assigned to each cluster:

$$
\mathbf{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
$$

### Initialization by K-Means++

K-Means++ improves the initialization step by spreading out the initial centroids. The steps are:

1. Randomly choose the first centroid from the data points.
2. For each remaining centroid, choose a point from the data set with probability proportional to its distance squared from the nearest centroid already chosen.
3. Repeat until $ K $ centroids are chosen.

### Extension: K-Medoids

K-Medoids is a variant of K-Means that is more robust to noise and outliers. Instead of using the mean of the points in a cluster, K-Medoids uses the median. This makes it more suitable for scenarios with non-Euclidean distances.

#### Algorithm

1. **Initialization**: Choose $ K $ initial medoids randomly.
2. **Assignment Step**: Assign each data point to the nearest medoid.
3. **Update Step**: For each cluster, choose the medoid that minimizes the sum of distances to other points in the cluster.
4. **Convergence Check**: Repeat the assignment and update steps until convergence.

#### Objective Function for K-Medoids

The objective function for K-Medoids is to minimize the following:

$$
J = \sum_{k=1}^{K} \sum_{i \in C_k} d(\mathbf{x}_i, \mathbf{m}_k)
$$

where $ \mathbf{m}_k $ is the medoid of cluster $ k $, and $ d(\cdot, \cdot) $ is a general distance metric.

### Real-World Examples

- **Customer Segmentation**: K-Means is often used in marketing to segment customers into different groups based on their purchasing behavior.
- **Image Compression**: By clustering pixel colors, K-Means can reduce the number of colors in an image, leading to compression.
- **Anomaly Detection**: K-Means can help identify unusual patterns in data, such as detecting fraud in financial transactions.

K-Means and its variants are powerful tools in the machine learning toolkit, providing simple yet effective ways to analyze and group data.