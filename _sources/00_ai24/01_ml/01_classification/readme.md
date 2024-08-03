# Classification

application

```
    classification :
        image classification
        identify fraud detection
        customer retention
        diagnostics
```

### topics

* [binary](./binary/readme.md) ; decision boundary ; accuracy ; hamming loss ; 

* [linear](./linear/readme.md) ; linear boundary - the role of w and b ; reparametrization and approximation using sigmod to define loss function ; gradien and accuracy to perform gradient descent ; validate using complex trick;

* [svm](./svm/readme.md) ;  maximize the margin ; define gap problem ; hard and soft margins(C - hyperparameter) ; train svm to show the effect of C ; show score and accuracy ; optimize C param;

* [knn](./kNear/readme.md) ; k=1 euclidean distance (best overfit) ; cosine dist ; shortest path ; the course of dimensionality

* full [training](./training/readme.md) process ; mnist example ; confusion matrix ; hyper params under/over/fit ; how to divide the data ; cross validation - K fold ; 

* [resample](./resample/readme.md)

### Performance Scores

* [decision function](./decFunc/readme.md) multi class performance  evaluation ; one vs all ; one vs one ;  ([scikit-func](../scikit/decisionFunction.md))
* [recall](./recall/readme.md) ; precision ; f1 ; ([scikit-func](../scikit/fscoreSupport.md))
* [Balanced Accuracy](./recall/BalAcc.md) ;
* [roc](./roc/readme.md) ; auc ; ([scikit-func](../scikit/roc.md))
    * [understand auc](./roc/auc_demo.ipynb) lab
* [cost](./cost/readme.md) ; loss ;  
* [scikit-classificationReport](../scikit/classificationReport.md)
* [scikit classification metrics](../scikit/classMetrics.md)


## labs

* [lab : mnist knn](./labs/0033ConfMatCrossValidation.ipynb)
* [lab : mnist svm](./labs/0034ConfMatCrossValidation.ipynb)
* [lab :score demo](./labs/scoreDemo.ipynb) ; Classification Report
* [lab :decFunc](./labs/decFunc.ipynb) ; decision function ; predict_proba
* [lab : perf score](./labs/0035PerformanceScoreMetrics.ipynb)
* [lab : cal imbalanced](./labs/0036PerformanceScoreMetrics.ipynb)



