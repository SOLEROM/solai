# Fully Connected Drawbacks

Fully connected layers, also known as dense layers, are fundamental components in neural networks where each neuron in one layer is connected to every neuron in the next layer. Despite their utility, they come with significant drawbacks:

1. **Large Number of Parameters**: 
   - Fully connected layers have a high number of parameters, which increases with the input size and the number of neurons. This can lead to overfitting, where the model performs well on training data but poorly on unseen data, and makes the network computationally expensive and memory-intensive.
   
2. **Fixed Input Size**:
   - Fully connected layers require a fixed input size, which means that the model cannot handle variable-sized inputs. This limitation is problematic for tasks like image and text processing, where input sizes can vary.
   
3. **Lack of Emphasis on Internal Structure**:
   - These layers do not take into account the spatial or temporal structure of the data. For example, in an image, nearby pixels are often related, but fully connected layers treat all pixels as independent features. This ignorance of data structure limits the model's ability to learn and generalize effectively.

## Internal Structure

Understanding and exploiting the internal structure of the data is crucial for improving model performance. Modules that leverage this structure can significantly outperform naive fully connected implementations.

internal structure can be:
* time series   - temporal stcuct
* images        - spatial struct
* videos        - spatila + temporal
* text          - text struct

# Modules Exploiting Data Structure

## Convolutional Layers
   - Convolutional layers are specifically designed to exploit the spatial structure of image data. By using filters that slide over the input data, convolutional layers can capture local patterns, such as edges and textures, and build hierarchical representations. This approach reduces the number of parameters compared to fully connected layers and enhances the model's ability to generalize from the data.



## Recurrent Neural Networks (RNNs)

### Overview:
- **Purpose**: Handle sequential data.
- **Structure**: Maintains a hidden state to capture information from previous steps.
- **Applications**: Time-series prediction, natural language processing, speech recognition.
- **Advantage**: Captures temporal dependencies in data.

## Long Short-Term Memory Networks (LSTMs)

### Overview:
- **Purpose**: Address the limitations of RNNs in capturing long-term dependencies.
- **Structure**: Uses gates (input, output, forget) to control the flow of information.
- **Applications**: Similar to RNNs but more effective for longer sequences.
- **Advantage**: Mitigates the vanishing gradient problem, allowing for better handling of long-term dependencies.

## Graph Neural Networks (GNNs)

### Overview:
- **Purpose**: Handle graph-structured data.
- **Structure**: Nodes and edges with relationships between them.
- **Applications**: Social network analysis, recommendation systems, bioinformatics.
- **Advantage**: Captures relational information between data points, leveraging the graph structure.

## Attention Mechanisms and Transformers

### Overview:
- **Purpose**: Focus on specific parts of the input data.
- **Structure**: Uses self-attention to process sequences in parallel.
- **Applications**: Natural language processing (e.g., translation, text generation), image processing.
- **Advantage**: Efficiently captures dependencies across the entire sequence, leading to significant performance improvements.

## Capsule Networks

### Overview:
- **Purpose**: Capture spatial hierarchies more effectively than traditional convolutional networks.
- **Structure**: Uses capsules (groups of neurons) to capture various properties of objects.
- **Applications**: Image recognition, especially for capturing pose and spatial relationships.
- **Advantage**: Preserves hierarchical relationships and improves robustness to affine transformations.

## Autoencoders

### Overview:
- **Purpose**: Learn efficient representations of data.
- **Structure**: Consists of an encoder to compress data and a decoder to reconstruct it.
- **Applications**: Dimensionality reduction, denoising, anomaly detection.
- **Advantage**: Captures the most important features of the data, enabling effective compression and reconstruction.

## Variational Autoencoders (VAEs)

### Overview:
- **Purpose**: Generate new data points similar to the training data.
- **Structure**: Similar to autoencoders but includes a probabilistic component.
- **Applications**: Data generation, anomaly detection.
- **Advantage**: Models the underlying distribution of the data, allowing for realistic data generation.

These modules and techniques are designed to exploit various internal structures and dependencies within data, leading to more efficient and effective models tailored to specific types of data and tasks.