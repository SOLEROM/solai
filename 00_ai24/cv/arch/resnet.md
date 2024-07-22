# resnet

https://d2l.ai/chapter_convolutional-modern/resnet.html

# ResNet (Residual Networks)

## Overview

ResNet, or Residual Networks, is a groundbreaking deep learning architecture introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in 2015. ResNet won the ILSVRC 2015 competition by a significant margin and addressed the problem of training very deep networks, which often suffer from vanishing gradients and degradation. The key innovation of ResNet is the introduction of "residual blocks," which allow the network to learn residual functions instead of direct mappings, enabling the training of much deeper networks.

## Architecture

### Residual Block

A residual block is the fundamental building block of ResNet. It consists of two or more convolutional layers and an identity shortcut connection that skips one or more layers. This shortcut connection helps mitigate the vanishing gradient problem by allowing gradients to flow directly through the network.

The basic residual block structure can be described as follows:

1. Input: $ \mathbf{x} $
2. Convolutional layer with Batch Normalization and ReLU: $ \mathbf{F}(\mathbf{x}) $
3. Another convolutional layer with Batch Normalization: $ \mathbf{G}(\mathbf{x}) $
4. Identity shortcut connection: $ \mathbf{x} $
5. Addition: $ \mathbf{y} = \mathbf{G}(\mathbf{x}) + \mathbf{x} $
6. ReLU activation: $ \text{ReLU}(\mathbf{y}) $

### ResNet Variants

ResNet comes in several variants with different depths:
- **ResNet-18**: 18 layers
- **ResNet-34**: 34 layers
- **ResNet-50**: 50 layers (uses bottleneck blocks)
- **ResNet-101**: 101 layers (uses bottleneck blocks)
- **ResNet-152**: 152 layers (uses bottleneck blocks)

### Detailed Layer Breakdown

For **ResNet-50**, the layers are as follows:

1. **Input Layer**: $224 \times 224 \times 3$ input image.
2. **Conv Layer 1**: $7 \times 7$ convolution with 64 filters, stride 2, followed by max-pooling.
3. **Conv Layer 2**: 3 bottleneck blocks with $64$ filters.
4. **Conv Layer 3**: 4 bottleneck blocks with $128$ filters.
5. **Conv Layer 4**: 6 bottleneck blocks with $256$ filters.
6. **Conv Layer 5**: 3 bottleneck blocks with $512$ filters.
7. **Global Average Pooling**: Reduces the spatial dimensions to 1x1.
8. **Fully Connected Layer**: A dense layer with 1000 outputs for classification.

## Real-World Example

ResNet has been widely adopted in various fields due to its robustness and ability to train very deep networks effectively. For example:

1. **Medical Image Analysis**: Enhancing the detection of diseases from medical images.
2. **Autonomous Vehicles**: Improving the accuracy of object detection and recognition systems.
3. **Natural Language Processing**: Serving as a backbone for models that process text data.

### Practical Application

ResNet's residual blocks are used in transfer learning, where a pre-trained ResNet model on ImageNet can be fine-tuned for specific tasks, leveraging its deep feature representations.

## Python Code (PyTorch)

Here's a simplified implementation of a ResNet model using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Example usage
model = resnet50(num_classes=1000)
print(model)
```

In this implementation, the `BasicBlock` and `Bottleneck` classes define the residual blocks. The `ResNet` class assembles the complete network using these blocks. The `resnet50` function creates an instance of the ResNet-50 model.

## Conclusion

ResNet introduced the concept of residual learning, enabling the training of very deep networks by addressing the vanishing gradient problem. This innovation significantly improved the performance and training of deep neural networks, making ResNet one of the most influential architectures in the field of deep learning.