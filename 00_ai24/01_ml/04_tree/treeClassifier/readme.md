# Decision Trees

A decision tree is a supervised machine learning algorithm used for both classification and regression tasks. It operates by recursively splitting the dataset into subsets based on the values of input features. This process results in a tree-like model where:

- **Nodes**: Represent features or attributes.
- **Edges**: Represent decision rules.
- **Leaves**: Represent outcomes or target values.

#### Impurity

Impurity measures how mixed the classes are within a node. Lower impurity indicates a more homogeneous node, which is desirable. The goal when constructing a decision tree is to reduce impurity at each split, leading to more homogeneous child nodes.

#### Gini Index and Entropy

1. **Gini Index**: Used by the CART (Classification and Regression Tree) algorithm, the Gini Index (or Gini Impurity) measures node impurity. It is calculated as:

    $$ Gini = 1 - \sum_{i=1}^{n} p_i^2 $$

    where $ p_i $ is the probability of an element being classified into a particular class. A Gini index of 0 indicates a pure node where all elements belong to a single class.

2. **Entropy**: Used by the ID3, C4.5, and C5.0 algorithms, entropy is another measure of impurity based on information theory. It is calculated as:

    $$ Entropy = -\sum_{i=1}^{n} p_i \log_2(p_i) $$

    where $ p_i $ is the probability of an element being classified into a particular class. An entropy of 0 indicates a pure node.

#### Regularization

Regularization techniques prevent overfitting, where a model learns noise rather than actual patterns. Regularization methods include:

- **Pruning**: Removing branches that have little importance.
- **Setting maximum depth**: Limiting the depth of the tree.
- **Minimum samples per leaf**: Setting a minimum number of samples required to be at a leaf node.
- **Minimum samples per split**: Setting a minimum number of samples required to split a node.

#### Stop Criteria

Stop criteria determine when the tree should stop growing. Common criteria include:

- **Maximum depth**: Limiting the number of levels in the tree.
- **Minimum samples per leaf**: Ensuring each leaf has a minimum number of samples.
- **Minimum samples per split**: Ensuring each split has a minimum number of samples.
- **No further impurity reduction**: Halting when further splits do not significantly reduce impurity.

#### Leaves

Leaves, or leaf nodes, are the terminal nodes of a decision tree. They represent the final output (class label or regression value) after a data point has been fully processed by the tree. Each path from the root to a leaf constitutes a classification rule.

### Advantages and Disadvantages

**Advantages**:
- Intuitive and easy to interpret.
- Handles both numerical and categorical data.
- Requires minimal data preprocessing.

**Disadvantages**:
- Prone to overfitting.
- Can be biased towards dominant classes.
- Less powerful alone compared to ensemble methods like Random Forest or Gradient Boosting.

### Example

Consider a simple dataset with features like "Weather" and "Temperature" to decide whether to play tennis:

```
| Weather  | Temperature | PlayTennis |
|----------|-------------|------------|
| Sunny    | Hot         | No         |
| Overcast | Hot         | Yes        |
| Rainy    | Mild        | Yes        |
| Sunny    | Mild        | Yes        |
```

A decision tree might first split on "Weather". If the weather is "Sunny", it might then split on "Temperature". Finally, it would make a decision (PlayTennis: Yes/No) at the leaves.

### Conclusion

Decision trees are a powerful and intuitive tool for classification and regression tasks. They model complex relationships in data and serve as a fundamental component in many advanced ensemble methods.