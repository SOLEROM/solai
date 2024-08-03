# Anomaly detection


## scores
* [zscore](scores/zscore.md)
    * Measures the distance of a data point from the mean in terms of standard deviations, assuming a single variable.
* [mahalanobis](./scores/mahalanobis.md)
    * Measures the distance of a data point from the mean of a multivariate distribution, taking into account the correlations between variables.
* [time series](./scores/timeSeries.md)

## methods

* [local_outlier_factor](lof/readme.md)
    * [moon lof lab](./lof/0070AnomalyDetectionLocalOutlierFactor.ipynb)
    * [taxi lab](./lof/0071AnomalyDetectionLocalOutlierFactor.ipynb)
* [isolation_forest](isoForest/readme.md)
    * [lab compare outlinear of super vs unsuper](./isoForest/0072AnomalyDetectionIsolationForest.ipynb)
    * [lab isolation on test train](./isoForest/0073AnomalyDetectionIsolationForest.ipynb)