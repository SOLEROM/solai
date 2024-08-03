# Bagging


Bagging (Bootstrap Aggregating) is like asking each of your friends to guess the candies in the jar using only a random sample of all the candies. Each friend uses a slightly different set of candies to make their guess. This way, each guess is a bit different, and combining them helps reduce variance (inconsistent predictions) without increasing bias too much.

* **Bagging** (Bootstrap Aggregating) is a technique to reduce the variance of a model by training multiple models on different samples of the training data and aggregating their predictions.


## Bagging (Bootstrap Aggregating)

### Overview
Bagging, or Bootstrap Aggregating, is an ensemble learning technique designed to improve the stability and accuracy of machine learning algorithms. It involves generating multiple versions of a predictor and using these to get an aggregated result.

### Key Concepts
1. **Bootstrap Sampling**:
   - Multiple datasets are created by randomly sampling with replacement from the original training dataset.
   - Each of these datasets is called a bootstrap sample and is usually the same size as the original dataset but may contain duplicate instances.

2. **Model Training**:
   - A separate model is trained on each bootstrap sample.
   - These models are often referred to as base learners or weak learners.

3. **Aggregation**:
   - For regression tasks, the predictions from each model are averaged.
   - For classification tasks, a majority vote is taken among the models' predictions.

### Applications
- Commonly used with decision trees to create Random Forests.
- Can be applied to various types of models, including neural networks, linear regression, and support vector machines.

### Advantages
- **Reduction in Overfitting**: By averaging multiple models, bagging reduces the variance without increasing the bias, thereby preventing overfitting.
- **Improved Accuracy**: Often results in better predictive performance compared to individual models.
- **Simple Implementation**: Straightforward to implement and understand.

### Disadvantages
- **Increased Computational Cost**: Training multiple models requires more computational resources.
- **Storage Requirements**: Requires storing multiple versions of the model, which can be memory-intensive.

