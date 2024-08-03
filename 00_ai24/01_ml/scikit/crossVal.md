# cross validation

* K fold
* see [k fold](../01_classification/training/dataDivide.md)
* see [mnist example](../01_classification/training/0033ConfMatCrossValidation.ipynb)


* cross_val_score() provides a measure of model performance through scores, while cross_val_predict() provides the actual predictions made during the cross-validation.

* The scores from cross_val_score() can be directly used to estimate model performance, whereas the predictions from cross_val_predict() require further analysis (e.g., calculating accuracy, precision, recall manually) to understand model performance.

## cross_val_predict

* Purpose: To generate cross-validated estimates for each input data point. Similar to cross_val_score(), the data is split into folds, and each fold is used once as a validation while the others form the training set. However, instead of returning the evaluation scores, it returns the predictions that were made for each element when it was in the test set.
* Output: An array of predictions that has the same length as the number of input samples. Each prediction corresponds to the output predicted by the model when that sample was in the test fold.
* Common Use Case: Useful for further analysis, such as calculating confusion matrices, other custom performance metrics, or for visualization of model performance. It allows you to see how the model predicts on unseen data throughout the cross-validation process.

```
numFold = ?? 

vYTrainPred = cross_val_predict(KNeighborsClassifier(n_neighbors = K), mXTrain, vYTrain, cv = KFold(numFold, shuffle = True))

vYTrainPred = cross_val_predict(KNeighborsClassifier(n_neighbors = K), mX, vY, cv = StratifiedKFold(numFold, shuffle = True))
```

## cross_val_score

* Purpose: To evaluate a score by cross-validation. It splits the data into several folds (parts), systematically using one fold for validation and the others for training, and then calculates scores for each split. It is primarily used for model evaluation.
* Output: A list of scores obtained for each cross-validation fold. You get as many scores as there are folds in the cross-validation.
* Common Use Case: When you want to get an estimate of a model's performance metrics (e.g., accuracy, precision, recall) using cross-validation. It's especially useful for comparing the performance of different models or configurations.

```
 vAccuracy = cross_val_score(modelCls, mX, vY, cv = KFold(mX.shape[0], shuffle = False)) #<! Leave One Out
    accuracy = np.mean(vAccuracy)

```