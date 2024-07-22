## Layers in Convolutional Neural Networks

### Overview

In Convolutional Neural Networks (CNNs), multiple filters are learned simultaneously during training to capture various features of the input data. Each layer in a CNN transforms the input data through convolution operations, followed by optional addition of bias, non-linear activation, and pooling operations. The process involves both forward and backward passes.

![alt text](image-1.png)

### Learning Multiple Filters

- **Filters (Kernels)**: In each convolutional layer, multiple filters (or kernels) are learned simultaneously. These filters are used to extract different features from the input data, such as edges, textures, and patterns.

### Forward Pass

During the forward pass, data moves from the input layer through several convolutional layers, each applying a set of filters and optional biases, and then through activation functions.

#### First Layer
- **Input**: Raw input data (e.g., an image).
- **Operation**: Each filter convolves with the input data to produce feature maps.

#### Second Layer
- **Input**: Feature maps from the first layer.
- **Operation**: Each filter convolves with the input feature maps to produce new feature maps.

### Adding Bias
- **Bias Term**: To add bias to the convolution operation, replace \( x * h_i \) with \( x * h_i + b \), where \( x \) is the input, \( h_i \) is the filter, and \( b \) is the bias term.

### Backward Pass

The backward pass, or backpropagation, involves calculating the gradients of the loss with respect to each parameter (filters and biases) and updating the parameters to minimize the loss.
