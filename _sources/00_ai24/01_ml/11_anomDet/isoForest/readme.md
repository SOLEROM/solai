# Isolation forest

### Overview

**Isolation Forest** is an algorithm used for anomaly detection. Unlike other methods that profile normal points, Isolation Forest explicitly isolates anomalies. It is based on the premise that anomalies are few and different, making them easier to isolate.

### How Isolation Forest Works

Isolation Forest works by creating an ensemble of trees (i.e., forest). It recursively divides the dataset by randomly selecting a feature and then randomly selecting a split value between the minimum and maximum values of the selected feature. This process is repeated until each data point is isolated.

#### Steps:
1. **Random Subsampling**: The algorithm randomly selects a subset of the data.
2. **Tree Construction**: For each subset, a tree is built by recursively partitioning the data.
3. **Isolation**: The partitioning continues until each point is isolated (i.e., each point resides in a leaf node of the tree).

The idea is that anomalies, being distinct and few, will require fewer partitions to isolate.

### Anomaly Score Calculation

The anomaly score in Isolation Forest is derived from the path length, i.e., the number of edges an instance traverses from the root node to the terminating node. For an instance $ x $:

- $ h(x) $ is the path length from the root to the leaf.
- $ E(h(x)) $ is the expected path length.

The anomaly score $ s(x, n) $ for an instance $ x $ with $ n $ being the number of data points is given by:

$$
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
$$

where $ c(n) $ is the average path length of unsuccessful searches in a Binary Search Tree, approximated by:

$$
c(n) \approx 2H(n-1) - \frac{2(n-1)}{n}
$$

with $ H(i) $ being the harmonic number, which can be approximated by $ \ln(i) + 0.5772156649 $ (Euler's constant).

- A score close to 1 indicates an anomaly.
- A score much smaller than 0.5 indicates a normal point.

### Real-World Example

Imagine a bank detecting fraudulent transactions. Legitimate transactions follow regular patterns, but fraudulent ones often differ significantly. By using Isolation Forest, the bank can isolate these anomalous transactions more efficiently than with profiling-based methods.

### Python Code Example

Here is how you might implement Isolation Forest using PyTorch:

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Sample data: 1000 normal points and 50 outliers
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(1000, 2)
X = np.r_[X + 2, X - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(50, 2))
X = np.r_[X, X_outliers]

# Fit the model
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X)

# Predict
y_pred_train = clf.predict(X)
y_pred_train = [1 if x == -1 else 0 for x in y_pred_train]

# Anomaly scores
anomaly_scores = clf.decision_function(X)

print("Anomaly Scores: ", anomaly_scores)
```

In this example:
- We create a dataset with normal points and outliers.
- We fit the Isolation Forest model.
- We predict anomalies in the dataset.
- We obtain anomaly scores for further analysis.

This approach allows the detection of anomalous data points effectively, which can be critical in various applications such as fraud detection, network security, and fault detection in machinery.
