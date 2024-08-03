# ReParametrization

In machine learning, reparameterization is a technique used to simplify the optimization of certain types of models by transforming the variables. This can make the optimization landscape smoother and help gradient-based methods converge more effectively. Here we dive into the details of reparameterization and its implications.

## Error Function

The error function, which we aim to minimize, is typically given as:

$ E(\theta) = \frac{1}{2} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $

Where $ y_i $ is the true label and $ \hat{y}_i $ is the predicted label.

## Objective

Our objective is to minimize this error function. This involves finding the parameter set $ \theta $ that results in the smallest possible error:

$ \theta^* = \arg \min_{\theta} E(\theta) $

## Approximation Using Sigmoid Function

To approximate the objective function, we can use the sigmoid function, which is defined as:

$ \sigma(x) = \frac{1}{1 + e^{-x}} $

The sigmoid function is useful in binary classification tasks because it maps any real-valued number into the range (0, 1), which can be interpreted as a probability.

## Sigmoid Function Details

The sigmoid function and its derivative are crucial in understanding the optimization process:

$ \sigma(x) = \frac{1}{1 + e^{-x}} $

The derivative of the sigmoid function is:

$ \sigma'(x) = \sigma(x) (1 - \sigma(x)) $

## Gradient

To minimize the loss function, we need to compute the gradient of the error with respect to the parameters. The loss function, often denoted as $ L $, can be written in terms of the predicted values $ \hat{y}_i $:

$ L(\theta) = \frac{1}{2} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $

The gradient of this loss function with respect to the parameters $ \theta $ is:

$ \nabla_{\theta} L(\theta) = \frac{\partial L}{\partial \theta} $

Using the chain rule, this gradient can be expressed as:

$ \nabla_{\theta} L(\theta) = \sum_{i=1}^{N} (\hat{y}_i - y_i) \nabla_{\theta} \hat{y}_i $

For a neural network using the sigmoid function, the gradient will involve the derivative of the sigmoid function:

$ \nabla_{\theta} \hat{y}_i = \hat{y}_i (1 - \hat{y}_i) $

## Summary

The loss function and gradient are central to the training process of a machine learning model. The accuracy function and gradient descent steps are as follows:

1. **Loss Function**: Measures the error between predicted and actual values.
2. **Gradient**: Provides the direction to update the parameters to minimize the loss.
3. **Accuracy Function**: Evaluates the performance of the model.
4. **Gradient Descent Step**: Updates the parameters iteratively to minimize the loss.

### Loss Function and Gradient

$ L(\theta) = \frac{1}{2} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $

$ \nabla_{\theta} L(\theta) = \sum_{i=1}^{N} (\hat{y}_i - y_i) \hat{y}_i (1 - \hat{y}_i) $

### Gradient Descent Step

$ \theta := \theta - \eta \nabla_{\theta} L(\theta) $

Where $ \eta $ is the learning rate.

### Accuracy Function

Accuracy is typically calculated as the proportion of correct predictions out of the total predictions.

By understanding and implementing these concepts, one can effectively train machine learning models, especially neural networks, to achieve high accuracy and generalize well to new data.