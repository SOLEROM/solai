# Outline

### Linear Classification recap

#### Overview
Linear classification involves finding a linear decision boundary that separates data points into different classes. This method is effective when data is linearly separable, meaning that the classes can be separated by a straight line (in 2D) or a hyperplane (in higher dimensions).

#### Key Concepts
- **Linear Decision Boundary**: A hyperplane defined by the equation $ \mathbf{w} \cdot \mathbf{x} + b = 0 $, where $ \mathbf{w} $ is the weight vector and $ b $ is the bias term.
- **Classification Rule**: A data point $ \mathbf{x} $ is classified based on the sign of $ \mathbf{w} \cdot \mathbf{x} + b $. If the result is positive, it belongs to one class; if negative, it belongs to the other class.
- **Loss Functions**: Functions such as hinge loss or logistic loss are used to measure the error of the classifier on the training data.
- **Optimization**: The goal is to find the optimal $ \mathbf{w} $ and $ b $ that minimize the chosen loss function, often subject to regularization to prevent overfitting.

- **Disadvantages**:
  - Limited to linear boundaries
  - Can perform poorly on complex datasets with nonlinear relationships

### The Dual Problem in Linear Classification

#### Overview
The dual problem arises from the primal optimization problem in linear classification. It is derived using Lagrange multipliers and provides an alternative formulation that can be more efficient to solve, especially when dealing with constraints.

#### Key Concepts
- **Primal Problem**: The original optimization problem, often minimizing a loss function subject to constraints. For Support Vector Machines (SVMs), it typically looks like:
  $$
  \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
  $$
  subject to $ y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i $ and $ \xi_i \geq 0 $.
- **Dual Problem**: Obtained by introducing Lagrange multipliers $ \alpha_i $ for each constraint. The dual problem for SVMs is:
  $$
  \max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j
  $$
  subject to $ \sum_{i=1}^{n} \alpha_i y_i = 0 $ and $ 0 \leq \alpha_i \leq C $.
- **KKT Conditions**: The Karush-Kuhn-Tucker conditions are necessary for a solution to be optimal and provide a set of criteria that the solution must satisfy.

#### Applications
- Support Vector Machines (SVMs)
- Kernelized algorithms (via kernel trick to handle nonlinearity)
- Constrained optimization problems

#### Advantages and Disadvantages
- **Advantages**:
  - Simplifies solving problems with constraints
  - Facilitates the use of kernel methods
  - Can be more computationally efficient for large datasets
- **Disadvantages**:
  - Interpretation of dual variables can be less intuitive
  - May require complex mathematical transformations

### Kernel Tricks

#### Overview
Kernel tricks allow linear classifiers to solve nonlinear problems by implicitly mapping the input features into a higher-dimensional space where a linear separator can be found.

#### Key Concepts
- **Kernel Function**: A function $ k(\mathbf{x}_i, \mathbf{x}_j) $ that computes the dot product in the higher-dimensional space without explicitly mapping the data points.
- **Common Kernels**:
  - Linear Kernel: $ k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j $
  - Polynomial Kernel: $ k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d $
  - Gaussian (RBF) Kernel: $ k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) $
- **Kernelized SVM**: Uses a kernel function to find a linear separator in the transformed space, allowing for nonlinear decision boundaries in the original space.

#### Applications
- Image and signal processing
- Bioinformatics
- Pattern recognition

#### Advantages and Disadvantages
- **Advantages**:
  - Enables handling of nonlinear data
  - Flexibility in choosing different kernel functions
- **Disadvantages**:
  - Choice of kernel and parameters can be complex
  - Computationally intensive for large datasets

This overview provides a foundation for understanding linear classification and its dual problem, along with the benefits of kernel tricks for handling nonlinearity.