# pipeline

* https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html


Pipeline allows you to sequentially apply a list of transformers to preprocess the data and, if desired, conclude the sequence with a final predictor for predictive modeling.

Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods.

The final estimator only needs to implement fit.


```
>>> # The pipeline can be used as any other estimator
>>> # and avoids leaking the test set into the train set
>>> pipe.fit(X_train, y_train).score(X_test, y_test)
```

* [see combining pipe with grid search](../play/gridSearchPipeline.ipynb)