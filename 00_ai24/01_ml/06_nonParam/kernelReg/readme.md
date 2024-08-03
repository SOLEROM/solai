#  kernel regression

## Kernel Regression

### Overview
Kernel regression is a non-parametric technique used to estimate the conditional expectation of a random variable. It is a type of smoothing method that aims to estimate the relationship between variables without assuming a specific form for the function that represents this relationship.

### Key Concepts

1. **Kernel Function**: The kernel function $ K $ is a weighting function used in kernel regression. It determines the weight assigned to each observation based on its distance from the point where the estimation is being made. Common kernel functions include:
   - **Gaussian Kernel**: $ K(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} $
   - **Epanechnikov Kernel**: $ K(x) = \frac{3}{4}(1-x^2) $ for $ |x| \leq 1 $
   - **Uniform Kernel**: $ K(x) = \frac{1}{2} $ for $ |x| \leq 1 $
   - **Triangular Kernel**: $ K(x) = 1 - |x| $ for $ |x| \leq 1 $

2. **Bandwidth (h)**: The bandwidth, also known as the smoothing parameter, controls the width of the kernel function. It determines how smooth the resulting estimate will be. A smaller bandwidth can capture more detail but might overfit the data, while a larger bandwidth produces a smoother estimate but might underfit the data.

3. **Nadaraya-Watson Estimator**: A common approach in kernel regression is the Nadaraya-Watson estimator, defined as:
   $$
   \hat{m}(x) = \frac{\sum_{i=1}^n K\left(\frac{x - x_i}{h}\right) y_i}{\sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)}
   $$
   where $ x_i $ and $ y_i $ are the observed data points, and $ h $ is the bandwidth.

### Example of Common Kernels
1. **Gaussian Kernel**:
   $$
   K(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
   $$
   This kernel gives weights that decrease exponentially as the distance from the target point increases.

2. **Epanechnikov Kernel**:
   $$
   K(x) = \frac{3}{4}(1-x^2) \quad \text{for} \quad |x| \leq 1
   $$
   This kernel is optimal in a mean square error sense and assigns zero weight to points farther than one bandwidth away.

3. **Uniform Kernel**:
   $$
   K(x) = \frac{1}{2} \quad \text{for} \quad |x| \leq 1
   $$
   This kernel gives equal weight to all points within the bandwidth and zero weight to those outside.

4. **Triangular Kernel**:
   $$
   K(x) = 1 - |x| \quad \text{for} \quad |x| \leq 1
   $$
   This kernel linearly decreases the weight as the distance increases.

### Silverman's Rule of Thumb
Silverman's rule of thumb is a method for selecting the bandwidth $ h $ for kernel density estimation. It provides a practical approach to determine a suitable bandwidth, which balances the trade-off between bias and variance.

For univariate data, Silverman's rule of thumb is given by:
$$
h = 0.9 \times \min(\sigma, \text{IQR}/1.34) \times n^{-1/5}
$$
where:
- $ \sigma $ is the standard deviation of the data.
- IQR is the interquartile range of the data.
- $ n $ is the number of observations.

### Applications
Kernel regression is widely used in various fields such as:
- **Econometrics**: To estimate demand curves or production functions.
- **Biostatistics**: For smoothing survival curves or estimating hazard rates.
- **Machine Learning**: As a smoothing method in non-linear regression and density estimation.

### Advantages and Disadvantages

**Advantages**:
- **Flexibility**: Does not assume a specific functional form for the relationship between variables.
- **Intuitive Interpretation**: The resulting estimates are easy to interpret and visualize.

**Disadvantages**:
- **Computationally Intensive**: Kernel regression can be slow, especially for large datasets, due to the need to compute weights for each observation.
- **Bandwidth Selection**: Choosing an appropriate bandwidth is crucial and can be challenging.
- **Boundary Bias**: Kernel regression can suffer from bias at the boundaries of the data range.

Kernel regression, with the appropriate choice of kernel and bandwidth, is a powerful tool for estimating relationships in data without imposing strict parametric assumptions.


## scikit kernel density estimation

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity

fit usually learn the parameters of the model
if the model have no parameters to learn - it will just prepare for estimation;

