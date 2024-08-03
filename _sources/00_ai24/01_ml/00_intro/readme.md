# Intro

* understand why human is still relevant ; [xor challange](xorChallengeExample.md)
* [about ML](aboutML.md)
* [types](types.md)
* [applications](applications.md)
* [pipeline](pipeline.md)
* [hyper-params](hyperparams.md)



## is problem is classification or regression?

    demo: rain prediction

* classification: the rain mesurement is a continuous value,
for example from 0 to 500;

* regression: the rain mesurement is a by ranges for city management : from 0-25 do nothing; for 25-200 do something; for 200-500 close the city...


## train
When training a model, we optimize it vs. a loss function

* Measures the “deviation” / “distance” of the model output to an input from the ground truth data or other measure.
* Serves the optimization (Minimization / Optimization) step of  the model’s parameters.
* Usually is smooth and differentiable (Sub Gradient).
* Examples: MSE, MAE, Cross Entropy, ...



## eval

When evaluating a model, we measure its performance using metrics / scores.

* Measures the fit of the model to real world measures of  its performance.
* Serves the evaluation step of a trained model or the progress of the training 
* Has no limitations but being computable.
* Examples: Accuracy, Number of outliers, AUC, ...

