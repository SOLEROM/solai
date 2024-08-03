# KNN

### Overview of K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for classification and regression tasks in machine learning. It operates on the principle that data points which are close to each other in feature space are likely to have similar labels. KNN is non-parametric, meaning it makes no explicit assumptions about the underlying data distribution.

### Key Concepts

1. **Distance Metric**: The choice of distance metric is crucial in KNN. Commonly used metrics include:
   - **Euclidean Distance**: The straight-line distance between two points in Euclidean space.
   - **Manhattan Distance**: The sum of the absolute differences of the coordinates.
   - **Cosine Distance**: Measures the cosine of the angle between two vectors, useful for high-dimensional data where the magnitude of the vectors may be less important than their direction.

2. **Number of Neighbors (k)**: The parameter $ k $ specifies the number of nearest neighbors to consider when making a prediction.
   - **$ k = 1 $**: The algorithm assigns the label of the single closest neighbor to the new data point.
   - **$ k > 1 $**: The algorithm assigns the label based on the majority vote or average (in case of regression) of the $ k $ nearest neighbors.

3. **Shortest Path**: In some contexts, especially graph-based approaches, KNN can be used to find the shortest path between nodes based on some distance metric, though this is not its primary use case.

### Detailed Discussion

#### Case of $ k = 1 $

When $ k = 1 $, the algorithm is referred to as the 1-Nearest Neighbor (1-NN). For each test point, the 1-NN algorithm:
- Computes the distance between the test point and all training points.
- Selects the training point with the smallest distance.
- Assigns the label of this closest point to the test point.

    K=1 is the best overfit on the trainning data !!!

**Advantages**:
- Simple and intuitive.
- Can capture complex decision boundaries.

**Disadvantages**:
- Sensitive to noise (outliers can significantly affect the prediction).
- Prone to overfitting, especially in noisy datasets.

#### Case of $ k > 1 $

When $ k > 1 $, the algorithm considers the $ k $ nearest neighbors:
- Computes the distance between the test point and all training points.
- Selects the $ k $ training points with the smallest distances.
- For classification, assigns the most common label among the $ k $ neighbors (majority voting).
- For regression, assigns the average value of the labels of the $ k $ neighbors.

**Advantages**:
- More robust to noise compared to $ k = 1 $.
- Smoother decision boundaries.

**Disadvantages**:
- Choosing the optimal $ k $ can be non-trivial and typically requires cross-validation.
- Higher computational cost due to the need to compute distances to all training points.

#### Cosine Distance

Cosine distance (or cosine similarity) measures the cosine of the angle between two non-zero vectors. It is defined as:
\[ \text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} \]
Where $ \mathbf{A} $ and $ \mathbf{B} $ are vectors, $ \cdot $ denotes the dot product, and $ \|\mathbf{A}\| $ is the magnitude of vector $ \mathbf{A} $.

**Application**:
- Particularly useful in text mining and information retrieval where the focus is on the direction of the data points rather than their magnitude.

**Advantages**:
- Effective for high-dimensional data where the magnitude of vectors may vary significantly.
- Captures the orientation of the vectors.

**Disadvantages**:
- May not perform well if the magnitude of the vectors carries important information.

### Applications of KNN

- **Classification**: Used in applications like handwriting recognition, image classification, and anomaly detection.
- **Regression**: Applied in forecasting problems like stock price prediction and weather forecasting.
- **Recommender Systems**: Suggests items to users based on the preferences of similar users.
- **Imputation of Missing Values**: Estimates missing values in datasets by averaging the values of nearest neighbors.

### Advantages and Disadvantages

**Advantages**:
- Simple to understand and implement.
- No training phase, making it efficient for small datasets.
- Adaptable to various types of distance metrics.

**Disadvantages**:
- Computationally intensive for large datasets due to distance calculations.
- Memory intensive as it stores all training data.
- Sensitive to irrelevant features and the scale of data.
- Requires careful feature scaling and selection of the appropriate distance metric.

### Conclusion

KNN is a versatile algorithm suitable for both classification and regression tasks. Its performance heavily depends on the choice of $ k $, the distance metric, and the preprocessing of data. While it is simple and effective for small datasets, its computational cost and sensitivity to irrelevant features pose challenges for larger and more complex datasets.