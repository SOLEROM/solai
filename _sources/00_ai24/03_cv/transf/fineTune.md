# Fine-Tuning


TBD:

https://d2l.ai/chapter_computer-vision/fine-tuning.html
https://stats.stackexchange.com/questions/343763/fine-tuning-vs-transferlearning-vs-learning-from-scratch
### Ultimate Guide to Fine ??????????? TBD
https://scribe.rip/8990194b71e
https://scribe.rip/b0f8f447546b
https://scribe.rip/a533a58051ef




## Overview

Fine-tuning a pre-trained model involves two main strategies:

1. **Freezing the Network**: Initially, most of the layers are frozen, meaning their weights are not updated during training. This helps in retaining the valuable features learned from the pre-trained model.
2. **Unfreezing and Training Again**: Gradually unfreezing the layers, starting from the last layers to the first ones, and fine-tuning them. This allows the model to adapt progressively to the new task.

## Layer-wise Fine-Tuning

Layer-wise fine-tuning is a method where you start by training only the last few layers (or the new layers you've added). After some epochs, you progressively unfreeze more layers and continue training. This approach ensures that the model does not lose the beneficial features learned from the original dataset while gradually adapting to the new data.

### Steps for Layer-wise Fine-Tuning

1. **Load the Pre-trained Model**: Use a model that has been pre-trained on a large dataset (e.g., ImageNet).
2. **Replace the Last Layer(s)**: Modify the final layer(s) to match the number of classes in your new task.
3. **Freeze Initial Layers**: Freeze most of the layers initially to preserve their learned features.
4. **Train the Last Layer(s)**: Train the newly added layers with a higher learning rate.
5. **Gradually Unfreeze Layers**: Unfreeze more layers layer by layer, reducing the learning rate progressively.
6. **Fine-Tune the Entire Network**: Fine-tune the whole network once the model starts to perform well on the new task.

### Example in PyTorch

```python
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

# Define the optimizer for the last layer
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop for the last layer
for epoch in range(num_epochs):
    # Training code here for the last layer
    pass

# Unfreeze the last few layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Define a new optimizer with different learning rates for different layers
optimizer = optim.SGD([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
], momentum=0.9)

# Continue training loop with more layers unfrozen
for epoch in range(num_epochs):
    # Training code here for unfrozen layers
    pass

# Optionally, unfreeze more layers and repeat the process
# until the entire network is fine-tuned

```

### Tips for Fine-Tuning

- **Start with a Lower Learning Rate**: When unfreezing layers, use a lower learning rate to prevent drastic changes to the pre-trained weights.
- **Monitor Performance**: Continuously monitor the model's performance on a validation set to ensure that fine-tuning is improving the model.
- **Data Augmentation**: Use data augmentation techniques to increase the diversity of the training data, helping the model generalize better to the new task.
- **Early Stopping**: Implement early stopping to prevent overfitting, stopping the training process if the performance on the validation set starts to degrade.

By following these steps, you can effectively fine-tune a pre-trained model to adapt it to your specific task, leveraging the knowledge it has already acquired while making it more suitable for the new data.