# Gradient boosting

#### Overview
Gradient Boosting is a machine learning technique used for regression and classification problems. It builds models sequentially, where each new model aims to correct the errors of the previous ones. This technique is powerful because it combines multiple weak learners (typically decision trees) to form a strong predictor.

#### Key Concepts

1. **Weak Learner**: Typically, decision trees are used as weak learners in gradient boosting. These trees are shallow, meaning they have a limited number of splits.

2. **Additive Model**: Gradient boosting builds an ensemble model in an additive manner. Each new tree is added to the ensemble to reduce the overall prediction error.

3. **Gradient Descent**: The "gradient" in gradient boosting refers to the gradient descent optimization algorithm, which is used to minimize the loss function. The new model is trained to predict the negative gradient of the loss function with respect to the current model's predictions.

4. **Loss Function**: The choice of the loss function depends on the problem at hand (e.g., mean squared error for regression, log-loss for classification). The loss function measures the difference between the actual and predicted values.

5. **Learning Rate**: A hyperparameter that controls the contribution of each new model. A smaller learning rate requires more trees but can lead to better generalization.

#### Applications

- **Regression**: Predicting continuous values, such as house prices or stock prices.
- **Classification**: Predicting categorical outcomes, such as spam detection or image classification.
- **Ranking**: Applications in search engines and recommendation systems.

#### Advantages

- **Accuracy**: High predictive performance on both regression and classification tasks.
- **Flexibility**: Can optimize a wide range of loss functions and work with different types of data.
- **Feature Importance**: Provides insights into feature importance, helping with feature selection and understanding the model.

#### Disadvantages

- **Complexity**: Can be computationally intensive and require careful tuning of hyperparameters.
- **Overfitting**: Prone to overfitting if the model is too complex or the learning rate is too high.
- **Training Time**: Longer training times compared to simpler models, especially for large datasets.

### Least Squares Boosting

#### Overview
Least Squares Boosting is a specific type of gradient boosting where the loss function used is the least squares error. This approach is primarily used for regression problems.

#### Key Concepts

1. **Least Squares Error**: The loss function is the mean squared error (MSE), which measures the average of the squares of the errors between the actual and predicted values.

2. **Sequential Learning**: Similar to general gradient boosting, models are added sequentially, with each new model attempting to correct the residuals (errors) of the previous model.

3. **Residuals**: The difference between the actual values and the predictions of the current model. Each new model is trained to predict these residuals.

4. **Fitting Trees**: In each iteration, a new decision tree is fitted to the residuals, and the predictions are updated by adding the new tree's predictions, scaled by the learning rate.

#### Applications

- **Regression Tasks**: Any task requiring the prediction of continuous values can benefit from least squares boosting, such as financial forecasting, environmental modeling, and quality control.

#### Advantages

- **Simplicity**: The use of least squares error makes the algorithm straightforward to implement and understand.
- **Effectiveness**: Can produce highly accurate models, especially with proper tuning and sufficient data.
- **Interpretability**: The step-by-step correction of residuals can be easier to interpret compared to other complex models.

#### Disadvantages

- **Sensitivity to Outliers**: Least squares error is sensitive to outliers, which can adversely affect the model's performance.
- **Computational Cost**: Requires significant computational resources, especially with large datasets and numerous iterations.
- **Risk of Overfitting**: As with other gradient boosting methods, there is a risk of overfitting, particularly if the model is too complex or not properly regularized.

### Conclusion
Gradient boosting and least squares boosting are powerful techniques for building predictive models. They combine the strengths of multiple weak learners to achieve high accuracy and flexibility. However, they require careful tuning and computational resources to avoid pitfalls like overfitting and high training times. Understanding these techniques' underlying concepts, applications, advantages, and disadvantages is crucial for effectively applying them to real-world problems.


## notes

Gradient Boosting is more like a team game where each successive friend tries to correct the mistake of the previous one. So, the first friend guesses, the next one looks at what was wrong with that guess and tries to fix it, and so on. Each step tries to reduce the error from the previous step, aiming to get as close to the right answer as possible.

* use sequence of models to reduce bias
* each model will correct the errors of the previous model;
* the final model is the sum of all models;


The Gradient Boosting approach is currently considered the _go to_ approach when working on tabular data






## scikit

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html


## optimized implementation

- [XGBoost](https://github.com/dmlc/xgboost).
- [LightGBM](https://github.com/microsoft/LightGBM).
- [CatBoost](https://github.com/catboost/catboost). 