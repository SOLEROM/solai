# classification_report

```
classification_report(vY, vYpred)
```

## svm example

```python
from sklearn.metrics import classification_report
## print classification_report
vYpred = oSVM.predict(mX)
print(classification_report(vY, vYpred))
```

```
              precision    recall  f1-score   support

           0       0.97      1.00      0.99       950
           1       0.96      0.50      0.66        50

    accuracy                           0.97      1000
   macro avg       0.97      0.75      0.82      1000
weighted avg       0.97      0.97      0.97      1000


```