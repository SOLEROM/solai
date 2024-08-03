# Decision function

```
vScore         = oSVM.decision_function(mX) 
            ##  Values proportional to distance from the separating hyperplane
```


#### Real-World Example

In a system recognizing handwritten digits (0-9), using OvA, you'd have 10 decision functions (one for each digit). For a given sample, each decision function would provide a score, and the digit with the highest score would be the predicted class.

### Using `sklearn.svm.SVC` and `decision_function`

Scikit-learn's `SVC` (Support Vector Classification) class provides the `decision_function` method to retrieve decision function scores.

#### Example Code

Here's an example using `SVC` for multi-class classification with the Iris dataset:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy as np

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SVM with linear kernel
svm = SVC(kernel='linear', decision_function_shape='ovr')

# One-vs-Rest Classifier
ovr = OneVsRestClassifier(svm)
ovr.fit(X_train, y_train)

# Get decision function scores for One-vs-Rest
decision_scores_ovr = ovr.decision_function(X_test)
print(f"One-vs-Rest Decision Scores:\n{decision_scores_ovr}")

# One-vs-One Classifier
ovo = OneVsOneClassifier(svm)
ovo.fit(X_train, y_train)

# Get decision function scores for One-vs-One
decision_scores_ovo = ovo.decision_function(X_test)
print(f"One-vs-One Decision Scores:\n{decision_scores_ovo}")
```

### Explanation

- **Training**: The SVM model is trained using both OvA and OvO strategies.
- **Decision Function Scores**: After training, the `decision_function` method is used to get the scores for the test set. These scores indicate the confidence of each classifier in the OvA and OvO settings.

#### Interpreting Scores

- **OvA Scores**: Each row in `decision_scores_ovr` corresponds to a sample, and each column to a class. The values represent the confidence of the classifier that a sample belongs to each class. The class with the highest score is chosen as the predicted class.
  
- **OvO Scores**: Each row in `decision_scores_ovo` corresponds to a sample, and the columns represent the confidence scores for each pair of classes. Voting among these scores determines the final predicted class.

### Conclusion

The `decision_function` in scikit-learn's SVM provides valuable insights into the confidence of classifications, essential for understanding and improving model performance. By extending this to multi-class problems with OvA and OvO strategies, one can efficiently handle complex classification tasks.