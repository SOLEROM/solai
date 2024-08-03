# SVC

* https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

## simple SVC

```
        oSvmCls     = SVC(C = C, kernel = 'linear')

    ...
    oSvmCls     = oSvmCls.fit(mXTrain, vYTrain)
    accScore    = oSvmCls.score(mXTest, vYTest)
    ...

```

## kernels

![TBD ](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png)

## LinearSVC

* `LinearSVC` class which optimized `SVC` with kernel `linear` as it fits for larger data sets

When using `LinearSVC`:
    *   If #Samples > #Features -> Set `dual = False`.
    *   If #Samples < #Features -> Set `dual = True` (Default).

```
        oSvmCls     = LinearSVC(C = C, max_iter = maxItr, dual = False)
```


## plot area

```
# Grid of the data support
v0       = np.linspace(mX[:, 0].min(), mX[:, 0].max(), numGridPts)
v1       = np.linspace(mX[:, 1].min(), mX[:, 1].max(), numGridPts)
XX0, XX1 = np.meshgrid(v0, v1)
XX       = np.c_[XX0.ravel(), XX1.ravel()]

Z = oSVM.predict(XX)
Z = Z.reshape(XX0.shape)

hF, hA = plt.subplots(figsize = FIG_SIZE_DEF)
hA.contourf(XX0, XX1, Z, colors = CLASS_COLOR, alpha = 0.3, levels = [-0.5, 0.5, 1.5])

```


# option:: probability

* in order to have the probability per class on the _SVC_ class we need to set `probability = True`.
* `probability = True` is not always consistent with the `decision_function()` method. Hence it is better to use it in the case of the `SVC`

https://scikit-learn.org/stable/modules/svm.html#scores-probabilities

* enables the estimation of class probabilities for the predictions
* This option allows you to call the predict_proba and predict_log_proba methods of the SVC -  which give you the probability estimates for each class.

Use Cases:

* Binary and Multiclass Classification: Probability estimates can be particularly useful in binary classification for understanding the likelihood of the positive class. They are also valuable in multiclass scenarios to assess confidence across multiple classes.
* Threshold Adjustment: In applications where the decision threshold needs to be adjusted away from the default 0.5 (for binary classification), having probability estimates allows for more nuanced control over classification decisions.
* Risk Assessment: In fields like finance or healthcare, understanding the probability of certain outcomes can be as important as the outcomes themselves for risk assessment and decision-making processes.
* Ensemble Methods: Probabilities can be used in ensemble learning methods that require them, such as stacking classifiers, where the predictions of several models are combined.