# Freeze Layers

Freezing layers in a neural network is a common technique used to speed up training and to leverage pre-trained models. When layers are frozen, their weights are not updated during training. This can be particularly useful when you are fine-tuning a model on a new dataset, where you want to preserve the learned features from a pre-trained model while only updating the weights of the final layers. There are two primary methods for freezing layers:

1. **Setting the Learning Rate to 0**: By setting the learning rate of certain layers to 0, you effectively freeze them because their weights will not be updated during backpropagation.

2. **Disabling Gradients**: By disabling gradients for certain layers, you prevent their weights from being updated during backpropagation. This is done by setting `requires_grad` to `False` for the parameters of the layers you want to freeze.

Let's explore these methods in more detail.

## Method 1: Setting Learning Rate to 0

In this method, we adjust the learning rate of the parameters we want to freeze. This is typically done using an optimizer that allows per-parameter learning rates.

```python
# Assuming optimizer is already defined
for param_group in optimizer.param_groups:
    if 'fc' not in param_group['params']:
        param_group['lr'] = 0
```

This code sets the learning rate of all parameters except those in the fully connected layers to 0, effectively freezing them.

## Method 2: Disabling Gradients

In this method, we disable the gradients for the parameters we want to freeze. This is achieved by setting the `requires_grad` attribute to `False`.

```python
for paramName, oPrm in oModelPreTrn.named_parameters():
    if not ('fc' in paramName):
        oPrm.requires_grad = False
```

This code iterates over all parameters of the pre-trained model (`oModelPreTrn`). If the parameter name does not include 'fc', the parameter's `requires_grad` attribute is set to `False`, freezing it.

## Real-World Examples

### Example 1: Fine-Tuning a Pre-trained CNN

Suppose you are using a pre-trained ResNet model for image classification on a new dataset. You might want to freeze the convolutional layers to preserve the learned features and only train the final fully connected layers.

```python
import torchvision.models as models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# Replace the final fully connected layer to match the number of classes in the new dataset
num_classes = 10  # Example number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Now only the final fully connected layer will be trained
```

### Example 2: Transfer Learning with BERT

When fine-tuning BERT for a specific NLP task, you might want to freeze all layers except the last few transformer blocks and the final classification layer.

```python
from transformers import BertModel

# Load a pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# Freeze all layers except the last 2 transformer blocks and the classification layer
for name, param in model.named_parameters():
    if not any(n in name for n in ['encoder.layer.10', 'encoder.layer.11', 'pooler']):
        param.requires_grad = False
```

In both examples, the non-frozen layers will be fine-tuned to adapt the pre-trained model to the new task, while the frozen layers retain their pre-trained weights.

## Conclusion

Freezing layers is a powerful technique in transfer learning and fine-tuning that allows you to leverage pre-trained models efficiently. By setting the learning rate to 0 or disabling gradients, you can control which layers of your model are updated during training. This helps in preserving useful features from pre-trained models and speeding up the training process.