# Data Split

#### Overview
In machine learning, the dataset is typically split into three parts: training set, validation set, and test set. This splitting is crucial for building, tuning, and evaluating a model effectively.

#### Key Concepts

1. **Training Set:**
   - **Purpose:** Used to train the machine learning model.
   - **Description:** The model learns the patterns, features, and parameters from this subset.
   - **Proportion:** Usually the largest portion, often around 60-70% of the total dataset.

2. **Validation Set:**
   - **Purpose:** Used for model validation during training.
   - **Description:** Helps in tuning hyperparameters and selecting the best model. The model is not trained on this data but rather validated to see how well it generalizes to new, unseen data.
   - **Proportion:** Typically around 15-20% of the dataset.

3. **Test Set:**
   - **Purpose:** Used for final evaluation after the model has been trained and validated.
   - **Description:** Provides an unbiased evaluation of the model’s performance.
   - **Proportion:** Usually around 15-20% of the dataset.

#### Applications
- **Training Set:** Used during the learning phase to fit the model.
- **Validation Set:** Employed during model selection and hyperparameter tuning to prevent overfitting.
- **Test Set:** Used for the final assessment of the model to ensure it performs well on completely unseen data.

#### Advantages
- **Training Set:** Enables the model to learn and adapt to the data.
- **Validation Set:** Helps in selecting the best model and fine-tuning it, ensuring better generalization.
- **Test Set:** Provides an objective measure of model performance on new data, ensuring that the model is not overfitting to the training data.

#### Disadvantages
- **Training Set:** If too large, might leave insufficient data for validation and testing.
- **Validation Set:** Might lead to some information loss since it's not used in training but is crucial for hyperparameter tuning.
- **Test Set:** If too small, might not provide a reliable estimate of the model's performance.

