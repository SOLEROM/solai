# R2 Score


### Overview

The R² score, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables in a regression model. It is a key indicator of the model's goodness of fit.

The R² score is defined as:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

where:
- $ y_i $ are the actual values,
- $ \hat{y}_i $ are the predicted values,
- $ \bar{y} $ is the mean of the actual values,
- $ n $ is the number of data points.

### Interpretation

- **R² = 1**: The model perfectly explains the variance in the data.
- **R² = 0**: The model does not explain any of the variance in the data.
- **R² < 0**: The model is worse than a horizontal line (mean of the target values).

### Real-World Example

Consider you're developing a model to predict house prices based on features like size, number of bedrooms, age, and location. After training your model, you calculate the R² score to understand how well your model captures the variability in house prices.

### Example

Here's how you can calculate the R² score using Python and scikit-learn, assuming you already have a trained model and test data.

```python
import numpy as np
from sklearn.metrics import r2_score

# Assume y_test and y_pred are the actual and predicted values from the test set
y_test = np.array([300000, 450000, 200000, 500000, 700000])
y_pred = np.array([310000, 430000, 210000, 480000, 690000])

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")
```

### Practical Use

The R² score helps in comparing different models. For instance, you might want to compare a Ridge Regression model and a Lasso Regression model to determine which one better explains the variance in house prices. While a higher R² score generally indicates a better fit, it's essential to be cautious of overfitting, where the model performs exceptionally well on training data but poorly on unseen data.

### Limitations

1. **Does not indicate model accuracy**: A high R² score doesn't mean the model is accurate; it just means the model explains a large portion of the variance.
2. **Sensitive to outliers**: Outliers can significantly affect the R² score, potentially leading to misleading conclusions.
3. **Not suitable for non-linear models**: R² score assumes a linear relationship between the dependent and independent variables. For non-linear models, other metrics might be more appropriate.

### Conclusion

The R² score is a valuable metric for evaluating the performance of a regression model, indicating how well the model explains the variance in the target variable. However, it should be used in conjunction with other evaluation metrics to ensure a comprehensive assessment of model performance.


## Notes

for normalized MSE = 1 => the prediction is as good as the mean of the target.

we want MSE <1 ;


* R2 = 0 good as the mean of the target.
* R2 = 1 perfect prediction.
* R2 < 0 worse than the mean of the target.



in time series - forcating - use the mean may be a very good prediction.

