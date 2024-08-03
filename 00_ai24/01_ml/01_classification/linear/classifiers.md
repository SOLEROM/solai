# Linear Classifier


A linear classifier uses a linear function of the input features to make predictions. The decision rule is based on the weighted sum of the input features. Mathematically, this can be expressed as:

$$ y_i = w^T x_i $$

where:
- $ y_i $ is the predicted value for the $ i $-th input.
- $ w $ is the weight vector.
- $ x_i $ is the input feature vector.

## Decision Boundary

The decision boundary of a linear classifier is defined by the sign of the linear function. This determines on which side of the decision boundary (a hyperplane in high-dimensional space) a given input point is located.

## Role of $ b $ (Bias Term)

The bias term $ b $ shifts the decision boundary. Including $ b $, the linear function becomes:

$$ y_i = w^T x_i + b $$

The role of $ b $ is to allow the decision boundary to be positioned more flexibly, not necessarily passing through the origin.

## Selecting $ w $ and $ b $

The selection of $ w $ and $ b $ is critical for the performance of the classifier. This process typically involves minimizing a loss function that measures how well the classifier predicts the training data. Common loss functions include:

- **Mean Squared Error (MSE)** for regression tasks.
- **Cross-Entropy Loss** for classification tasks.

Optimization algorithms, such as gradient descent, are used to adjust $ w $ and $ b $ to minimize the loss function.

## Non-Linear Data

For data that is not linearly separable, linear classifiers can struggle. To address this, techniques such as:

- **Kernel Trick**: Transforming the input features into a higher-dimensional space where a linear decision boundary can be found.
- **Feature Engineering**: Creating new features based on the original ones to improve linear separability.


