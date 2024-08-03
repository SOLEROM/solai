# Weighted Least Squares

* the ability to assign different weights to different data points to show importance! 

### Weighted Least Squares (WLS)

#### Overview
Weighted Least Squares (WLS) is an extension of the Ordinary Least Squares (OLS) regression method. In WLS, each data point is assigned a weight, allowing for differential treatment of data points based on their importance or reliability. This technique is particularly useful when the variance of the errors differs across observations, a condition known as heteroscedasticity.

#### Key Concepts

1. **Weight Assignment**: 
    - Each data point $(x_i, y_i)$ in the dataset is associated with a weight $w_i$. These weights reflect the relative importance or reliability of each observation.
    - Higher weights indicate more reliable or important data points, while lower weights suggest less reliable data points.

2. **Weighted Sum of Squares**:
    - The WLS method minimizes the weighted sum of squared residuals:
    $$
    \text{Minimize} \sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2
    $$
    where $y_i$ is the observed value, $\hat{y}_i$ is the predicted value, and $w_i$ is the weight assigned to the $i$-th data point.

3. **Matrix Formulation**:
    - In matrix terms, if $W$ is a diagonal matrix with weights $w_i$ on the diagonal, the WLS estimator $\hat{\beta}$ is given by:
    $$
    \hat{\beta} = (X^T W X)^{-1} X^T W y
    $$
    where $X$ is the design matrix of input features and $y$ is the vector of observed outputs.

#### Applications

1. **Heteroscedasticity**:
    - WLS is used when the assumption of homoscedasticity (constant variance of errors) is violated in OLS regression. It adjusts for varying levels of error variance.

2. **Data Quality**:
    - In datasets where the quality of observations varies, WLS allows more accurate modeling by giving higher weights to more reliable data points and lower weights to less reliable ones.

3. **Time Series and Longitudinal Data**:
    - In time series or longitudinal data, observations might have different levels of importance due to changing conditions over time. WLS can account for these differences.

#### Advantages

1. **Flexibility**:
    - WLS provides a flexible framework for dealing with heteroscedasticity and varying data quality, leading to more accurate parameter estimates.

2. **Improved Estimates**:
    - By appropriately weighting the data points, WLS can produce better estimates of regression coefficients compared to OLS, especially when the error variances are not constant.

3. **Bias Reduction**:
    - In scenarios with measurement errors or varying reliability, WLS helps reduce bias in the parameter estimates.

#### Disadvantages

1. **Weight Determination**:
    - The effectiveness of WLS depends on the proper choice of weights. Incorrect weights can lead to biased estimates and suboptimal model performance.

2. **Complexity**:
    - Implementing WLS is more complex than OLS due to the need to determine and apply weights.

3. **Computational Cost**:
    - The computation of the weighted regression estimates involves matrix operations that can be computationally intensive, especially for large datasets.

### Conclusion
Weighted Least Squares is a powerful tool for regression analysis, allowing for the assignment of different weights to data points to reflect their importance or reliability. It addresses issues of heteroscedasticity and varying data quality, leading to more accurate and unbiased parameter estimates. However, its effectiveness relies on the correct determination of weights, and it comes with increased complexity and computational cost compared to OLS.



![alt text](image-4.png)

## notes
![alt text](image-3.png)





