# Regression Tree

A **Regression Tree** is a type of decision tree that is used for predicting a continuous-valued target variable. It is a tree structure where each internal node represents a decision on an attribute, each branch represents the outcome of the decision, and each leaf node represents a predicted value of the target variable.

### Key Concepts of Regression Trees

1. **Split Criteria**:
   - The process of creating splits or partitions in a regression tree involves selecting attributes and split points that minimize the variance within each subset of the data.
   - Commonly used split criteria in regression trees include:
     - **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values. A split is chosen to minimize the MSE.
     - **Mean Absolute Error (MAE)**: Measures the average absolute difference between the actual and predicted values. A split is chosen to minimize the MAE.
   - The process typically involves evaluating all possible splits and selecting the one that results in the greatest reduction in error (i.e., the best split).

2. **Regularization**:
   - Regularization techniques are used to prevent overfitting, where the model becomes too complex and captures noise in the training data rather than the underlying patterns.
   - Common regularization methods for regression trees include:
     - **Pruning**: This involves removing branches that have little importance. Pruning can be done by setting a minimum number of samples required to be at a leaf node (min_samples_leaf) or a minimum number of samples required to split an internal node (min_samples_split).
     - **Maximum Depth**: Limiting the maximum depth of the tree can control the complexity of the model. Shallow trees are less likely to overfit.
     - **Minimum Split Improvement**: Setting a threshold for the minimum improvement in error required to consider a split. If the improvement is below this threshold, the split is not made.
     - **Regularization Parameters (like in Gradient Boosting)**: Techniques such as L1 (Lasso) and L2 (Ridge) regularization can be applied to the leaf values in the context of ensemble methods like Gradient Boosting Trees.

3. **Multivariate Regression Trees**:
   - In multivariate regression trees, the goal is to predict multiple target variables simultaneously.
   - Each leaf node in the tree represents a vector of predicted values, one for each target variable.
   - The splitting criteria need to account for multiple outputs, often by extending the variance reduction criteria to a multivariate context.
   - Techniques like Multivariate Regression Tree (MRT) involve calculating the sum of squared deviations for each target variable and finding splits that minimize the overall deviation across all targets.

### How Regression Trees Work

1. **Building the Tree**:
   - Start with all the data at the root.
   - Choose the best split according to the split criteria (e.g., MSE, MAE) to partition the data into subsets.
   - Recursively repeat the splitting process for each subset until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or no further reduction in error).

2. **Making Predictions**:
   - For a given input, traverse the tree from the root to a leaf node by following the decision rules.
   - The value at the leaf node is the predicted value for the input.

### Example

Let's say we have a dataset with a single input variable $ X $ and a target variable $ Y $.

1. **Initial Split**:
   - Calculate the split points for $ X $ that result in the smallest MSE for $ Y $.
   - Assume the best split is at $ X = 5 $.

2. **Create Nodes**:
   - Create two branches, one for $ X \leq 5 $ and one for $ X > 5 $.

3. **Recursively Split**:
   - For each subset, repeat the process of finding the best split until a stopping criterion is met.

4. **Final Tree**:
   - Each leaf node represents a predicted value for $ Y $.

### Regularization Example

- Suppose we have set a maximum depth of 3 and a minimum number of samples per leaf as 5.
- These constraints prevent the tree from growing too deep and capturing noise in the data.

### Multivariate Example

- Suppose we have two target variables $ Y1 $ and $ Y2 $.
- The tree would consider splits that reduce the overall error for both $ Y1 $ and $ Y2 $.
- Each leaf node will contain a vector of predicted values $[Y1, Y2]$.

Regression trees are powerful tools for modeling complex relationships, and understanding split criteria, regularization, and multivariate aspects are key to effectively using them.

## summary

* useful when categorical features are present
* useful when no other metric
* weak regressor with large variance
* can be improved using bagging and random forests