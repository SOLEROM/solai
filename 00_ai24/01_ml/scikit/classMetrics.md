# classification metrics

### 1. `classification_report`
The `classification_report` function provides a comprehensive summary of precision, recall, F1 score, and support for each class. It can also include average metrics if specified.

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]

report = classification_report(y_true, y_pred)
print(report)
```

### 2. `confusion_matrix`
The `confusion_matrix` function computes the confusion matrix, which can be used to derive other metrics manually.

```python
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

### 3. `balanced_accuracy_score`
The `balanced_accuracy_score` function computes the balanced accuracy directly.

```python
from sklearn.metrics import balanced_accuracy_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(balanced_accuracy)
```

### 4. `accuracy_score`
The `accuracy_score` function computes the standard accuracy, which is the ratio of correctly predicted instances to the total instances.

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
```

### 5. `precision_score`, `recall_score`, `f1_score`
These functions calculate precision, recall, and F1 score individually, allowing for more fine-grained control over the calculations.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```

### 6. `roc_auc_score`
The `roc_auc_score` function computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC), which is a performance measurement for classification problems at various threshold settings.

```python
from sklearn.metrics import roc_auc_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred_proba = [0.1, 0.4, 0.8, 0.2, 0.9, 0.7]  # Example probabilities

roc_auc = roc_auc_score(y_true, y_pred_proba)
print(roc_auc)
```

### 7. `average_precision_score`
The `average_precision_score` function calculates the average precision score, which summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold.

```python
from sklearn.metrics import average_precision_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred_proba = [0.1, 0.4, 0.8, 0.2, 0.9, 0.7]

average_precision = average_precision_score(y_true, y_pred_proba)
print(average_precision)
```


### 8. ` fscoreSupport`


```
# Calculating the Scores
vHatY                    = oSVM.predict(mX)
precision, recall, f1, support = precision_recall_fscore_support(vY, vHatY, pos_label = 1, average = 'binary')


print(f'Precision = {precision:0.3f}')
print(f'Recall    = {recall:0.3f}'   )
print(f'f1        = {f1:0.3f}'       )
print(f'Support   = {support}'  )

```