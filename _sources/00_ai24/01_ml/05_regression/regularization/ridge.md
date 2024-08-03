# Ridge Regression 


## Tikhonov Regularization (L2)

### Overview

Ridge Regression, also known as Tikhonov Regularization, is a technique used to address multicollinearity in linear regression models. It adds a penalty to the size of the coefficients to reduce their variance, improving the model's generalization to new data.

The ridge regression model minimizes the following cost function:

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^{p} w_j^2
$$

Here, $ \lambda $ is the regularization parameter, $ \mathbf{w} $ are the model coefficients, $ y_i $ are the target values, and $ \mathbf{x}_i $ are the feature vectors. The second term is the L2 penalty, which discourages large coefficient values.

### Real-World Example

Imagine you're predicting the price of houses based on features like size, number of bedrooms, age, and location. With standard linear regression, if some features are highly correlated (e.g., size and number of bedrooms), it can lead to large variances in the coefficient estimates. Ridge Regression helps mitigate this by shrinking the coefficients, thus making the model more robust and reducing overfitting.

### Example

Consider a dataset `housing_data` with features `size`, `bedrooms`, `age`, and `location`.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = housing_data[['size', 'bedrooms', 'age', 'location']].values
y = housing_data['price'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression model
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Predictions
y_pred = ridge_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Lasso Regression (L1)

### Overview

Lasso Regression, short for Least Absolute Shrinkage and Selection Operator, is another form of regularization that adds a penalty equivalent to the absolute value of the coefficients. Unlike Ridge Regression, Lasso can shrink some coefficients to zero, effectively performing feature selection.

The Lasso regression model minimizes the following cost function:

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^{p} |w_j|
$$

Here, $ \lambda $ is the regularization parameter, $ \mathbf{w} $ are the model coefficients, $ y_i $ are the target values, and $ \mathbf{x}_i $ are the feature vectors. The second term is the L1 penalty, which encourages sparsity in the model coefficients.

### Real-World Example

Using the same housing price prediction scenario, Lasso Regression can be particularly useful if you suspect that some features are irrelevant. By driving these irrelevant feature coefficients to zero, Lasso simplifies the model and enhances interpretability.

### Example

Consider the same `housing_data` dataset with features `size`, `bedrooms`, `age`, and `location`.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = housing_data[['size', 'bedrooms', 'age', 'location']].values
y = housing_data['price'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso Regression model
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)

# Predictions
y_pred = lasso_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Display coefficients
print(f"Coefficients: {lasso_reg.coef_}")
```

## Conclusion

Both Ridge and Lasso Regression are powerful tools for preventing overfitting and enhancing model performance when dealing with multicollinear or irrelevant features. Ridge is suitable when you want to retain all features but reduce their impact, while Lasso is ideal for feature selection by eliminating insignificant predictors.

Feel free to ask for more details or specific code snippets!