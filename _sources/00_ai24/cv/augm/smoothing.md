# Smoothing

### Motivation

Label Smoothing is a technique used to mitigate numerical issues related to the logarithm function in the Cross Entropy loss. It helps improve model generalization and calibration by making the model less confident about its predictions, which can be particularly useful in the presence of noisy labels.

### Contributions of Label Smoothing

1. **Handling Noisy Labels**:
   - Reduces the impact of incorrect labels by distributing part of the probability mass to all other labels.
   - This limits the loss contributed by noisy labels, leading to more robust training.

2. **Regularizing Overfitting**:
   - Prevents the model from becoming too confident about its predictions on the training data, which can help reduce overfitting.
   - The model learns to be less certain, thus avoiding over-reliance on specific training examples.

3. **Improving Model Calibration**:
   - Encourages the model to produce probabilities that better reflect true likelihoods, leading to better-calibrated models.
   - This can be particularly important for downstream tasks that rely on probability estimates, such as decision making or risk assessment.

### How it Works

Label smoothing modifies the target distribution by mixing the ground truth labels with a uniform distribution. For a given class $ c $, the modified target probability $ q(c) $ is given by:

$ q(c) = (1 - \epsilon) \cdot \delta_{c,c^*} + \frac{\epsilon}{K} $

where:
- $ \epsilon $ is the smoothing parameter.
- $ \delta_{c,c^*} $ is the Kronecker delta, which is 1 if $ c = c^* $ (the true class) and 0 otherwise.
- $ K $ is the number of classes.

### Practical Implementation in PyTorch

PyTorch's `CrossEntropyLoss` class includes a `label_smoothing` parameter to easily implement label smoothing.

Example:
```python
import torch.nn as nn

# Define the loss function with label smoothing
epsilon = 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=epsilon)

# Example usage in a training loop
for inputs, targets in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### Notes

- **Effectiveness in Binary Classification**:
  - Label smoothing is less effective in binary classification tasks because it primarily helps by distributing probability mass across multiple incorrect labels. In binary classification, there are only two classes, so the effect is minimal.
  - The main benefit of clustering wrong labels together with equal probability doesn't apply as strongly when there's only one incorrect label.

### References

- [Understanding Label Smoothing](https://leimao.github.io/blog/Label-Smoothing/)
- [Label Smoothing in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

