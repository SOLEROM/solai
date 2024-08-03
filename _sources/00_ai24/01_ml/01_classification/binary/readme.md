# Binary Classification

Binary classification involves predicting one of two possible classes for a given input.

## Training Set

The training set consists of pairs $(x_i, y_i)$ where $x_i$ is the input and $y_i \in \{0, 1\}$ is the corresponding class label.

## Test Set

The test set is used to evaluate the model's performance on unseen data.

## Objective

Given a new input $x_0$, predict the class label $y_0$. This can be expressed as:
$$y_0 = f(x_0)$$
where $f$ is the learned classification function.

## Example

A visual representation might show different points corresponding to different classes, and the task is to correctly classify a new point based on its position relative to the training points.

## Accuracy

Accuracy is the proportion of correctly classified instances out of the total instances:
$$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} $$

Minimize the training set error, especially if the test set is not yet available. The training error can be computed as:
$$ \text{Training Error} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(f(x_i) \neq y_i) $$
where $\mathbb{1}(\cdot)$ is the indicator function.

## Hamming Loss

The Hamming loss for binary classification is the fraction of incorrect labels:
$$ \text{Hamming Loss} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(f(x_i) \neq y_i) $$

The training error in terms of Hamming loss is equivalent to the formula above.

## Defining the Model

Since it is not feasible to define a model for all possible inputs, we select a hypothesis space $H$ and restrict our search to this set. This helps in managing complexity and improving generalization.

$$ H = \{ h: X \rightarrow \{0, 1\} \mid h \text{ is a member of the hypothesis class} \} $$

Example model types include:

- **Linear Models**: Models that predict the class based on a linear combination of input features.
- **Polynomial Models**: Extensions of linear models that include polynomial terms of the input features.
- **K-Nearest Neighbors (KNN)**: Models that classify a point based on the majority class among its $k$ nearest neighbors.
- **Support Vector Machines (SVM)**: Models that find a hyperplane that maximizes the margin between the two classes.
- **Decision Trees**: Models that split the input space into regions based on feature values, leading to a tree structure for decision making.

By selecting an appropriate hypothesis class $H$, we can efficiently search for the best model that minimizes the training error and generalizes well to unseen data.