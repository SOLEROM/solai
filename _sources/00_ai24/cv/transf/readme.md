# Transfer Learning

## Motivation

### In Classic Learning

In traditional machine learning, we often use feature extractors that can be reused for similar tasks. For instance, if we have a feature extractor that works well for classifying types of flowers, it might also work well for classifying types of fruits. The main advantage is that for new data, we only need to modify the learning algorithm block rather than designing a new feature extractor from scratch.

### In Deep Learning

In deep learning, the structure of neural networks is typically divided into several layers with distinct roles:

1. **First Layers (Low-Level Features)**: These layers capture fundamental features such as edges, textures, and basic shapes.
2. **Mid Layers (High-Level Features)**: These layers identify more complex patterns and structures, combining the low-level features to recognize parts of objects.
3. **Last Layer (Output Layer)**: This layer learns to combine the high-level features to make final predictions.

The idea of transfer learning is to leverage a pre-trained model, where the initial layers (low and high-level feature extractors) have already been trained on a large dataset. We then fine-tune the last layer or a few of the top layers to adapt the model to the new task, significantly reducing the amount of data and computational resources required.

## Example

Imagine we have a deep neural network trained on the ImageNet dataset, which contains millions of images across a thousand different categories. We can use this pre-trained model to classify medical images by making the following adjustments:

1. **Retain the Early Layers**: Keep the first layers that have learned to extract basic visual features.
2. **Modify the Last Layer**: Replace the output layer with a new one specific to our medical imaging task (e.g., detecting tumors).
3. **Fine-Tune the Model**: Train the modified model on our specific dataset, ensuring that the early layers retain their pre-learned features while the new layers learn task-specific features.

## Learning Rate

When fine-tuning a model in transfer learning, adjusting the learning rate is crucial:

1. **Gradual Learning Rate Increase**: Start with a learning rate of zero and gradually increase it to the desired level for the new layers. This ensures that the model's weights are updated in a controlled manner, preventing drastic changes that could disrupt the pre-learned features.
2. **Freezing Weights**: Alternatively, you can freeze the weights of the initial layers (preventing them from updating) and only train the new layers. This approach is useful when the new dataset is small or the initial layers have very reliable feature extraction capabilities.

For detailed information on setting learning rates and other optimization techniques in PyTorch, you can refer to the [PyTorch Optimization Documentation](https://pytorch.org/docs/stable/optim.html#per-parameter-options).

```python
# Example: Fine-tuning a pre-trained model in PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Replace the last fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # num_classes is the number of output classes

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Define the optimizer with a specific learning rate for the last layer
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Alternatively, use different learning rates for different layers
optimizer = optim.SGD([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
], momentum=0.9)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    # Training code here
    pass
```

In this example, we load a pre-trained ResNet18 model, replace its last fully connected layer, and fine-tune it on a new dataset. We freeze the weights of the initial layers to preserve their learned features and set a specific learning rate for the new layer. This approach effectively transfers the knowledge from the pre-trained model to the new task.