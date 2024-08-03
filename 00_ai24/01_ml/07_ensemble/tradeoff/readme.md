# Bias Variance Tradeoff

* Bias is like consistently hitting the same spot, but not necessarily the bullseye. A high bias means you're not even hitting close to the bullseye.
* Variance is when your shots are spread all over the target. High variance means your shots are very inconsistent.

In machine learning, a perfect model would have low bias (accurately hitting the bullseye) and low variance (consistently hitting the same spot). However, usually reducing one increases the other. So, there's a trade-off.


## Overview

In machine learning, the concepts of bias, variance, and the tradeoff between them are crucial for understanding model performance. Additionally, underfitting and overfitting are related issues that arise due to this tradeoff.

### Bias

Bias refers to the error introduced by approximating a real-world problem, which may be complex, by a simpler model. A model with high bias pays very little attention to the training data and oversimplifies the model. Consequently, such a model might perform poorly on both training and test data.

Mathematically, bias is defined as:

$$
\text{Bias} = E[\hat{f}(x)] - f(x)
$$

where $ \hat{f}(x) $ is the predicted value and $ f(x) $ is the true value.

### Variance

Variance refers to the error introduced by the model's sensitivity to small fluctuations in the training set. A model with high variance pays too much attention to the training data, including noise, and as a result, it performs well on training data but poorly on test data.

Mathematically, variance is defined as:

$$
\text{Variance} = E[\hat{f}(x) - E[\hat{f}(x)]]^2
$$

### Bias-Variance Tradeoff

The bias-variance tradeoff is the balance between the error introduced by bias and the error introduced by variance. Ideally, we want to minimize both, but in practice, there is a tradeoff. Reducing bias typically increases variance and vice versa.

The total error can be expressed as:

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

### Underfitting and Overfitting

- **Underfitting:** This occurs when the model is too simple to capture the underlying pattern of the data. An underfitted model will have high bias and low variance. It fails to perform well on both training and test datasets.

- **Overfitting:** This occurs when the model is too complex and captures noise in the training data. An overfitted model will have low bias and high variance. It performs well on training data but poorly on test data.

## Real-World Examples

### Bias Example: Linear Regression on Non-Linear Data

If you use a linear regression model to fit a dataset that has a quadratic relationship, the model will have high bias. It will fail to capture the curve and will perform poorly on both the training and test sets.

### Variance Example: High-Degree Polynomial Regression

If you fit a high-degree polynomial to a small dataset, the model will likely capture noise in the training data, resulting in high variance. It will perform very well on the training set but poorly on unseen data.

### Underfitting Example: Simple Model for Complex Data

Imagine trying to predict house prices using only one feature, like the number of rooms, when the actual price is influenced by many factors like location, size, and age. This model will underfit the data, leading to high bias.

### Overfitting Example: Complex Model for Simple Data

Using a deep neural network to fit a dataset with only a few features and limited samples can lead to overfitting. The model will learn the noise in the training data, leading to high variance.

## Summary

Understanding the bias-variance tradeoff and recognizing underfitting and overfitting are essential for building effective machine learning models. Balancing bias and variance is key to achieving a model that generalizes well to new data.

If you have specific questions or need code examples in PyTorch, feel free to ask!





## notes

* green line is the ground truth
* blue line is the model of linear fit

### example 1

different data - the model fit is almost the same
![alt text](image-1.png)

### example 2

now with polyfit model of rank 8 :

![alt text](image-2.png)

    !!! for best fit we want low bias and low variance !!!


