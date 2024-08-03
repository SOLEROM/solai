# Ensemble Methods

* ensemble - combine multiple models to improve performance
* usually used with trees;


Imagine you're trying to guess the number of candies in a jar. Instead of just taking a guess yourself, you ask a bunch of your friends to guess too. Then, you combine all those guesses to come up with a final answer. The idea is that, while some of your friends might guess too high and others too low, overall their combined guesses are likely to be closer to the right answer than just one person guessing alone. This is the basic idea behind ensemble learning in machine learning.


In machine learning, instead of friends guessing candies, we use different models (like decision trees, logistic regressors, etc.) to make predictions. By combining their predictions, we usually end up with a better, more stable result.



## topics

* [bias variance tradeoff](tradeoff/readme.md)

each model has high variance and small bias - reduce by averaging and use bagging:

* [bagging](./tradeoff/bagging.md)
* [OOB](./tradeoff/oob.md)

* [random forest](./randomForest/readme.md)
* [mdi](./randomForest/mdi.md)
* [permutation](./randomForest/permutation.md)
    * [lab demo random forest](./randomForest/0054EnsembleRandomForests.ipynb)


each model has high bias and low variance - reduce the bias by using a sequence of models with :

* [gradient boosting](./boosting/readme.md)
    * [lab GradientBoosting](./boosting/0055EnsembleGradientBoosting.ipynb)
* [ada boost](./boosting/adaBoost.md)
    * [lab AdaptiveBoosting](./boosting/0056EnsembleAdaptiveBoosting.ipynb)

* [lightgbm](./optimizedLib/lightgbm/readme.md)
* [xgboost](./optimizedLib/xgboost/readme.md)

## summary

![alt text](image-1.png)