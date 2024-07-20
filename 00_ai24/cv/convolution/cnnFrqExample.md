# CNN for Frequency Estimation Problem 

explore a basic example of using a Convolutional Neural Network (CNN) to solve a frequency estimation problem. This type of problem involves estimating the frequency of a signal from input data, which is common in signal processing and time-series analysis.

#### Problem Description

Given a series of input signals, the goal is to estimate the frequency of each signal. This is particularly useful in applications like audio signal processing, where identifying the frequency can help in recognizing musical notes, speech processing, and other tasks.

#### Model Architecture

A basic CNN architecture for this problem typically consists of the following layers:

1. **Input Layer**: This layer accepts the raw signal data. For simplicity, assume the input signals are 1D arrays.

2. **Convolutional Layers**: These layers apply convolutional filters to the input signal to extract relevant features. Each filter slides over the input signal to detect patterns. Common configurations include:
   - **Number of Filters**: Determines how many different features are learned.
   - **Filter Size**: Defines the size of the filters that slide over the input signal.
   - **Stride**: Specifies how much the filter moves at each step.
   - **Padding**: Decides whether the input signal is padded to maintain the same output size.

3. **Pooling Layers**: These layers reduce the dimensionality of the feature maps, typically using max pooling. This helps in reducing the computational complexity and extracting dominant features.

4. **Fully Connected Layers**: After several convolutional and pooling layers, the output is flattened and passed through one or more fully connected layers. These layers act as a classifier or a regressor depending on the task.

5. **Output Layer**: For frequency estimation, the output layer typically has one neuron with a linear activation function to predict the frequency.

#### Loss Function

The loss function used in this example is the Root Mean Squared Error (RMSE). RMSE measures the average magnitude of the errors between predicted and actual frequencies, providing a clear indication of the model's performance.

\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]


where \( y_i \) is the actual frequency, \( \hat{y}_i \) is the predicted frequency, and \( n \) is the number of samples.

#### Example Code

Below is a simple example using Python and Keras to build and train a CNN for frequency estimation.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.losses import MeanSquaredError

# Generate synthetic data for the example
def generate_synthetic_data(num_samples, length):
    X = np.random.randn(num_samples, length)
    y = np.random.uniform(low=0.1, high=10.0, size=num_samples)
    return X, y

# Prepare data
num_samples = 1000
length = 100
X_train, y_train = generate_synthetic_data(num_samples, length)
X_test, y_test = generate_synthetic_data(num_samples // 10, length)

# Build the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(length, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)
print(f'Test RMSE: {rmse}')

# Predicting on new data
predictions = model.predict(X_test[:5])
print(f'Predicted frequencies: {predictions.flatten()}')
```

#### Explanation

1. **Data Generation**: Synthetic data is generated to simulate the frequency estimation problem. The signals (`X`) are random noise, and the frequencies (`y`) are uniformly distributed between 0.1 and 10.0.
2. **Model Building**: A sequential model is built with two convolutional layers, each followed by a max-pooling layer. The output is flattened and passed through a dense layer to produce the final frequency prediction.
3. **Training**: The model is compiled with the Adam optimizer and Mean Squared Error loss function, then trained on the synthetic data.
4. **Evaluation**: The model's performance is evaluated on a test set, and the Root Mean Squared Error (RMSE) is printed.
5. **Prediction**: The model makes predictions on new data to demonstrate its frequency estimation capability.

This example provides a foundation for understanding how CNNs can be applied to frequency estimation problems. The architecture and parameters can be further tuned based on specific requirements and data characteristics.