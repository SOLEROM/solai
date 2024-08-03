# huber

* [huber loss](https://en.wikipedia.org/wiki/Huber_loss)
* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html


## Huber Loss

### Overview

Huber Loss is a robust loss function used in regression problems to handle outliers more effectively than traditional loss functions like Mean Squared Error (MSE). It combines the best properties of MSE and Mean Absolute Error (MAE) by being quadratic for small errors and linear for large errors.

The Huber loss is defined as:

$$
L_\delta(y, f(x)) = 
\begin{cases} 
\frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)| \leq \delta \\
\delta |y - f(x)| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

where \( \delta \) is a threshold parameter that determines the point where the loss function transitions from quadratic to linear.

### Properties

- **Quadratic Behavior for Small Errors**: For errors smaller than \(\delta\), the loss function behaves like MSE, providing smooth gradients which are useful for optimization.
- **Linear Behavior for Large Errors**: For errors larger than \(\delta\), the loss function behaves like MAE, reducing the influence of outliers.

### Real-World Example

Imagine you are developing a regression model to predict the prices of houses. In your dataset, there might be some outliers due to incorrect data entries or extremely unusual houses. Using Huber Loss can help reduce the impact of these outliers on the model.

### Example

Here’s how you can implement Huber Loss using Python and PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data with outliers
torch.manual_seed(42)
X = torch.linspace(-10, 10, 100).reshape(-1, 1)
y = X + 3 * torch.randn(X.size())
y[::10] += 20  # Add outliers

# Define a simple linear regression model
model = nn.Linear(1, 1)

# Define Huber Loss with delta = 1.0
criterion = nn.HuberLoss(delta=1.0)

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Predict
model.eval()
with torch.no_grad():
    predictions = model(X)

# Plot results
import matplotlib.pyplot as plt

plt.scatter(X.numpy(), y.numpy(), color='blue', label='Data with Outliers')
plt.plot(X.numpy(), predictions.numpy(), color='red', label='Huber Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Huber Loss Regression')
plt.legend()
plt.show()
```

### Practical Use

- **Regression with Outliers**: Huber Loss is particularly useful when dealing with datasets containing outliers, as it mitigates their impact on the model.
- **Robust Optimization**: It provides a balance between the sensitivity of MSE and the robustness of MAE, making it suitable for various regression tasks.
- **Computer Vision**: Used in applications like image alignment and tracking, where robustness to outliers is essential.

### Conclusion

Huber Loss is a robust and efficient loss function for regression tasks, offering a compromise between Mean Squared Error and Mean Absolute Error. By combining quadratic and linear behaviors, it effectively handles outliers while providing smooth gradients for optimization. This makes it a valuable tool in scenarios where data quality is a concern.