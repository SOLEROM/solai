# CalibratedClassifierCV

https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html


CalibratedClassifierCV from scikit-learn is used for calibrating probabilities in classification models that do not naturally output well-calibrated probabilities or when the model output probabilities are not reliable. It's particularly useful when:


* Improved Probability Estimates: Some models, like Support Vector Machines (SVMs), inherently do not output probabilities, and other models might output probabilities that are not aligned with the actual likelihood of an event. CalibratedClassifierCV adjusts these probabilities to better reflect reality, which can be crucial in applications where decision thresholds, risk assessments, and probabilities are directly utilized.
* Model Comparison: When comparing models, especially in terms of probabilistic outputs, calibrated probabilities ensure a fair comparison under metrics like Brier score or log loss.
* Decision Making: In scenarios where decisions are made based on the probability threshold (e.g., finance, healthcare), having well-calibrated probabilities ensures that decisions are made on more accurate and reliable estimates.


## What Does CalibratedClassifierCV Do?

It applies a probability calibration to a classification model through two main methods:

* Platt Scaling: A logistic regression model is fitted to the model's outputs. It's more effective when the decision function is sigmoid.
* Isotonic Regression: A non-parametric, piecewise linear model that estimates a more flexible calibration map. It's well-suited for larger datasets as it can capture more complex shapes in the score-to-probability relationship.


## When to Use CalibratedClassifierCV?

* Post-Model Training: It is typically used after training a classifier to adjust the output probabilities.
* Non-Probabilistic Models: For models that do not natively output probabilities, such as SVM or k-nearest neighbors with a uniform weight.
* When Original Probabilities Are Poor: For models like RandomForest or neural networks, where the calibration of the probabilities might not be reliable.


```

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM model
svm = SVC(C=1.0, probability=True, random_state=42)
svm.fit(X_train, y_train)

# Calibrate probabilities on the training set
calibrated_svm = CalibratedClassifierCV(base_estimator=svm, method='sigmoid', cv='prefit')
calibrated_svm.fit(X_train, y_train)

# Predict probabilities on the test set
prob_pos_uncalibrated = svm.predict_proba(X_test)[:, 1]
prob_pos_calibrated = calibrated_svm.predict_proba(X_test)[:, 1]

# Compare Brier scores
score_uncalibrated = brier_score_loss(y_test, prob_pos_uncalibrated)
score_calibrated = brier_score_loss(y_test, prob_pos_calibrated)

print(f"Brier score (uncalibrated): {score_uncalibrated:.4f}")
print(f"Brier score (calibrated): {score_calibrated:.4f}")


```