# LS outiner

## Least Squares Outlier Detection

### Overview

Least Squares (LS) regression is sensitive to outliers, which are data points that deviate significantly from other observations. Outliers can disproportionately influence the regression line, leading to skewed results and reduced model performance.

### Identifying Outliers in LS Regression

Outliers in LS regression can be detected using several methods, including:

1. **Standardized Residuals**: Residuals are the differences between the observed and predicted values. Standardized residuals (residuals divided by their standard deviation) can identify outliers. A common rule of thumb is that standardized residuals greater than ±3 are potential outliers.

2. **Cook's Distance**: Cook's Distance measures the influence of each data point on the fitted regression model. Points with a Cook's Distance greater than 4/n (where n is the number of observations) are often considered influential.

3. **Leverage**: Leverage points are those with extreme predictor values. High leverage points can disproportionately affect the regression model.

### Real-World Example

Imagine you are developing a regression model to predict housing prices based on features like size, number of bedrooms, age, and location. Some data points might have unusual combinations of these features, making them outliers.

### Example

Let's consider how to detect outliers in a housing price dataset using Python and scikit-learn.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample data
housing_data = {
    'size': [1500, 1600, 1700, 1800, 2000, 5000],  # The last entry is an outlier
    'bedrooms': [3, 3, 3, 3, 4, 8],
    'age': [10, 15, 20, 25, 5, 50],
    'price': [300000, 320000, 340000, 360000, 400000, 1000000]
}

df = pd.DataFrame(housing_data)
X = df[['size', 'bedrooms', 'age']]
y = df['price']

# Add constant for intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()
influence = model.get_influence()

# Standardized Residuals
standardized_residuals = influence.resid_studentized_internal
df['standardized_residuals'] = standardized_residuals

# Cook's Distance
cooks_d = influence.cooks_distance[0]
df['cooks_d'] = cooks_d

# Leverage
leverage = influence.hat_matrix_diag
df['leverage'] = leverage

# Identify outliers
outliers = df[(np.abs(df['standardized_residuals']) > 3) | (df['cooks_d'] > 4/len(df))]

print("Outliers:")
print(outliers)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['standardized_residuals'], label='Standardized Residuals')
plt.hlines(y=3, xmin=0, xmax=len(df)-1, color='r', linestyles='dashed')
plt.hlines(y=-3, xmin=0, xmax=len(df)-1, color='r', linestyles='dashed')
plt.xlabel('Index')
plt.ylabel('Standardized Residuals')
plt.title('Standardized Residuals Plot')
plt.legend()
plt.show()
```

### Practical Use

- **Data Cleaning**: Detecting and addressing outliers is essential for accurate modeling. Outliers can be investigated further to determine if they are data entry errors or genuine variations.
- **Model Improvement**: Removing or adjusting for outliers can improve the robustness and accuracy of the model.

### Conclusion

Outliers can significantly impact Least Squares regression models. By using techniques like standardized residuals, Cook's Distance, and leverage, we can identify and address these outliers to enhance model performance and reliability. Always consider the context of outliers to make informed decisions about their treatment.


