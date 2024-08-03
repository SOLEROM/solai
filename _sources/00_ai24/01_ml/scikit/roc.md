# roc curve

### calc

```
vScore         = oSVM.decision_function(mX) #<! Values proportional to distance from the separating hyperplane

vFP, vTP, vThr = roc_curve(vY, vScore, pos_label = 1)

AUC            = auc(vFP, vTP)


```

### plot 
```
hA = vHA.flat[0]
hA.plot(vFP, vTP, color = 'b', lw = 2, label = f'ROC Curve, AUC = {AUC:.3f}')
hA.plot([0, 1], [0, 1], color = 'k', lw = 2, linestyle = '--')
hA.set_xlabel('False Positive Rate')
hA.set_ylabel('True Positive Rate')
hA.set_title('ROC')
hA.grid()
hA.legend()
```

## what to use for curve:

When calculating the Receiver Operating Characteristic (ROC) curve for a Support Vector Machine (SVM) model in scikit-learn, you essentially need a score or probability estimate for each instance to determine its position relative to the ROC curve. The choice between oSVM.decision_function and oSVM.predict_proba depends on the nature of your SVM model and what you're trying to measure:


When to Use What?

* Decision Function (oSVM.decision_function): Use this when your SVM model is not configured to estimate probabilities (probability=False), or when you're specifically interested in how distances from the decision boundary (confidence scores) rank the instances. It's also useful when comparing against models where probability estimation isn't directly available or meaningful.

    
* Probability Estimates (oSVM.predict_proba): Use this when your model is configured to output probabilities (probability=True), and you want your ROC analysis to reflect the model's probabilistic confidence in its predictions. This is especially useful for binary classification tasks where interpreting the probability of belonging to the positive class is straightforward and directly comparable to other probabilistic classifiers.