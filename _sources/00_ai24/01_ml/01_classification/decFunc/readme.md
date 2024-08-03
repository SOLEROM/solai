# multi-class

### Multi-Class KNN (K-Nearest Neighbors)

#### General Overview

K-Nearest Neighbors (KNN) is a simple and intuitive algorithm used for both classification and regression. For multi-class classification, KNN assigns a class to a sample based on the majority class among its $ k $ nearest neighbors.

Given a sample $ x $, the KNN algorithm works as follows:

1. Compute the distance between $ x $ and all samples in the training set.
2. Select the $ k $ samples that are closest to $ x $.
3. Determine the majority class among these $ k $ neighbors.
4. Assign $ x $ to this majority class.

The distance metric commonly used is the Euclidean distance:

$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

#### Real-World Example

Imagine you're developing a system to classify different types of fruits based on features like color, size, and weight. You have a labeled dataset with examples of apples, oranges, and bananas. Using KNN, you can classify a new fruit sample by looking at its closest neighbors in the feature space.

### Decision Function and Classification Score

#### General Overview

The decision function in classification algorithms, including KNN, is used to evaluate how well the model separates different classes. For KNN, this involves computing the distances to all points in the training set and determining the nearest neighbors.

The classification score for KNN is typically based on the accuracy of the predicted labels compared to the true labels in a validation set. Other metrics such as precision, recall, and F1-score can also be used, especially in cases of imbalanced datasets.

#### Real-World Example

Continuing with the fruit classification example, after training your KNN model, you evaluate its performance by measuring how accurately it classifies a set of validation samples. If the model correctly identifies 90 out of 100 samples, the accuracy score would be 90%.

### Multi-Class Classification:

#### General Overview

In multi-class classification, strategies like One-vs-All (OvA) and One-vs-One (OvO) are often used to extend binary classifiers to handle multiple classes.

- **One-vs-All (OvA)**: This strategy involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. For $ k $ classes, $ k $ classifiers are trained. The class with the highest classification score for a given sample is chosen as the predicted class.

- **One-vs-One (OvO)**: This strategy involves training a binary classifier for every possible pair of classes. For $ k $ classes, $ \frac{k(k-1)}{2} $ classifiers are trained. Each classifier predicts which of the two classes a given sample belongs to, and a voting scheme is used to determine the final predicted class.

#### Real-World Example

In a handwriting recognition system, you want to classify handwritten digits (0-9). Using OvA, you train 10 classifiers, each distinguishing one digit from the rest. Using OvO, you train 45 classifiers, each distinguishing between a pair of digits. The system then uses the results from these classifiers to predict the digit.

### Using `sklearn.multiclass`

Scikit-learn provides tools for implementing multi-class strategies with various classifiers.

#### One-vs-Rest (OvR) Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Apply One-vs-Rest
ovr = OneVsRestClassifier(knn)
ovr.fit(X_train, y_train)

# Predict and evaluate
y_pred = ovr.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

#### One-vs-One (OvO) Example

```python
from sklearn.multiclass import OneVsOneClassifier

# Apply One-vs-One
ovo = OneVsOneClassifier(knn)
ovo.fit(X_train, y_train)

# Predict and evaluate
y_pred = ovo.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

These examples demonstrate how to use Scikit-learn's `OneVsRestClassifier` and `OneVsOneClassifier` wrappers with a KNN classifier to handle multi-class classification problems.


# using

* for svc we have that support

![alt text](image-1.png)

* for sklearn.multiclass  - input : classify function; can deal with both options;