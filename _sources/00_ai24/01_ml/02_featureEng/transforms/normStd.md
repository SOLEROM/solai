# Normalizing and Standardizing

Feature engineering is a crucial step in the machine learning pipeline, and normalizing and standardizing are two essential techniques for preparing data before feeding it into models. These processes help improve the performance and convergence speed of machine learning algorithms by ensuring that features have similar scales.

## Normalizing

Normalization is the process of scaling individual samples to have unit norm. This can be particularly useful when using machine learning algorithms that rely on distance calculations, such as k-nearest neighbors (KNN) or support vector machines (SVM).

**Formula:**

For a feature vector $ x $, normalization typically transforms it as follows:

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

This scales the values of $ x $ to a range of [0, 1].

**Example:**

Consider a dataset with a feature representing the age of individuals, ranging from 18 to 90 years. Without normalization, age values will dominate over other features like height (measured in centimeters) when calculating distances. Normalizing the age values to a [0, 1] range ensures that no single feature unduly influences the model.

Real-world Example:
- **Image Processing:** In image processing tasks, pixel values are often normalized to a range of [0, 1] to ensure that the neural network treats all pixel intensities equally.

## Standardizing

Standardization transforms features to have a mean of zero and a standard deviation of one. This is particularly useful for algorithms like linear regression, logistic regression, and neural networks that assume normally distributed data.

**Formula:**

For a feature vector $ x $, standardization typically transforms it as follows:

$$
x' = \frac{x - \mu}{\sigma}
$$

where $ \mu $ is the mean of the feature and $ \sigma $ is the standard deviation.

**Example:**

Consider a dataset with a feature representing annual income in dollars, which might range from a few thousand to hundreds of thousands of dollars. Standardizing the income values ensures that they have a mean of zero and a standard deviation of one, making the data more suitable for algorithms that assume normal distribution.

Real-world Example:
- **Finance:** In financial modeling, standardizing features like stock returns ensures that they have a common scale, facilitating comparison and modeling.

## Practical Considerations

- **Choosing Between Normalizing and Standardizing:** The choice depends on the specific algorithm and the nature of the data. Normalizing is preferred for algorithms that rely on distances, while standardizing is generally better for algorithms that assume normally distributed data.
- **Handling Outliers:** Both normalizing and standardizing can be sensitive to outliers. Robust scaling methods, such as using the interquartile range, can be considered when outliers are present.
- **Implementation:** In practice, libraries like scikit-learn provide convenient functions for normalizing and standardizing features, ensuring that this preprocessing step is straightforward.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample data
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
```
