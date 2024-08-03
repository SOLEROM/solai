# Hierarchical clustering

# Hierarchical Clustering

Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. This approach can be divided into two main types: divisive (top-down) and agglomerative (bottom-up). Here's a more detailed look into each type and various linkage methods.

## Top Down - Divisive Clustering

Divisive clustering, also known as top-down clustering, starts with all data points in a single cluster and recursively splits them into smaller clusters. This approach is less common compared to agglomerative clustering and often uses parametric methods which come with their disadvantages, such as higher computational cost and complexity.

## Bottom Up - Agglomerative Clustering

Agglomerative clustering, or bottom-up clustering, starts with each data point as a single cluster and merges the closest pairs of clusters recursively until only one cluster is left or a stopping criterion is met. This method is more commonly used and offers various ways to define the distance (dissimilarity) between clusters.

### Linkage Methods

Linkage methods determine how the distance between clusters is computed. Different linkage methods can produce different results, even on the same dataset.

#### Single Linkage
- **Definition**: The minimum distance between points in the two clusters.
- **Advantages**: Simple and intuitive.
- **Disadvantages**: Sensitive to noise and outliers, can create long, “chain-like” clusters.
- **Example**: Clustering of city locations where the closest points are connected first.

#### Complete Linkage
- **Definition**: The maximum distance between points in the two clusters.
- **Advantages**: Less sensitive to noise compared to single linkage.
- **Disadvantages**: Can still create long clusters, may ignore cluster shape.
- **Example**: Grouping documents where the most dissimilar documents are considered for clustering boundaries.

#### Centroid Linkage
- **Definition**: The distance between the centroids (mean points) of the two clusters.
- **Advantages**: Considers the overall position of clusters.
- **Disadvantages**: Can sometimes merge dissimilar clusters if their centroids are close.
- **Example**: Clustering of consumer behavior data, where centroids represent average consumer profiles.

#### Ward Linkage
- **Definition**: Minimizes the variance within the clusters being merged.
- **Advantages**: Often produces more compact and spherical clusters.
- **Disadvantages**: Computationally intensive.
- **Example**: Grouping genes with similar expression profiles to minimize intra-cluster variance.

### Lance Williams Algorithm

The Lance Williams algorithm provides a general formula to update the distances between clusters for agglomerative clustering and can be adapted to any linkage method. This algorithm ensures that hierarchical clustering can be computed efficiently.

### Mathematical Representation

For a given set of clusters, the distance between clusters $ A $ and $ B $ can be updated when clusters $ A $ and $ C $ are merged. The general formula for the Lance Williams update is:

$$ d(A \cup C, B) = \alpha_A d(A, B) + \alpha_C d(C, B) + \beta d(A, C) + \gamma |d(A, B) - d(C, B)| $$

Where:
- $ \alpha_A $, $ \alpha_C $, $ \beta $, and $ \gamma $ are parameters that depend on the specific linkage method.

### Practical Example

Consider you have a dataset of customer purchase histories. Using hierarchical clustering, you can group customers with similar purchasing patterns. Depending on the chosen linkage method, you may get different groupings:
- **Single Linkage**: Customers who made their purchases on similar dates might cluster together first.
- **Complete Linkage**: Customers with the most dissimilar purchase dates form boundaries of clusters.
- **Centroid Linkage**: Customers whose average purchase amount is similar will be grouped.
- **Ward Linkage**: Customers whose purchase variance within clusters is minimized will form the final clusters.

Hierarchical clustering provides flexibility in how you define and discover natural groupings within your data, making it a powerful tool in exploratory data analysis.