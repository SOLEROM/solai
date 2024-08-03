# preprocessing

https://scikit-learn.org/stable/modules/preprocessing.html


https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

```
preprocessing.Binarizer(*[, threshold, copy]) - Binarize data (set feature values to 0 or 1) according to a threshold.

preprocessing.FunctionTransformer([func, ...]) - Constructs a transformer from an arbitrary callable.

preprocessing.KBinsDiscretizer([n_bins, ...]) - Bin continuous data into intervals.

preprocessing.KernelCenterer() - Center an arbitrary kernel matrix.

preprocessing.LabelBinarizer(*[, neg_label, ...]) - Binarize labels in a one-vs-all fashion.

preprocessing.LabelEncoder() - Encode target labels with value between 0 and n_classes-1.

preprocessing.MultiLabelBinarizer(*[, ...]) - Transform between iterable of iterables and a multilabel format.

preprocessing.MaxAbsScaler(*[, copy]) - Scale each feature by its maximum absolute value.

preprocessing.MinMaxScaler([feature_range, ...]) - Transform features by scaling each feature to a given range.

preprocessing.Normalizer([norm, copy]) - Normalize samples individually to unit norm.

preprocessing.OneHotEncoder(*[, categories, ...]) - Encode categorical features as a one-hot numeric array.

preprocessing.OrdinalEncoder(*[, ...]) - Encode categorical features as an integer array.

preprocessing.PolynomialFeatures([degree, ...]) - Generate polynomial and interaction features.

preprocessing.PowerTransformer([method, ...]) - Apply a power transform featurewise to make data more Gaussian-like.

preprocessing.QuantileTransformer(*[, ...]) - Transform features using quantiles information.

preprocessing.RobustScaler(*[, ...]) - Scale features using statistics that are robust to outliers.

preprocessing.SplineTransformer([n_knots, ...]) - Generate univariate B-spline bases for features.

preprocessing.StandardScaler(*[, copy, ...]) - Standardize features by removing the mean and scaling to unit variance.

preprocessing.TargetEncoder([categories, ...]) - Target Encoder for regression and classification targets.

preprocessing.add_dummy_feature(X[, value]) - Augment dataset with an additional dummy feature.

preprocessing.binarize(X, *[, threshold, copy]) - Boolean thresholding of array-like or scipy.sparse matrix.

preprocessing.label_binarize(y, *, classes) - Binarize labels in a one-vs-all fashion.

preprocessing.maxabs_scale(X, *[, axis, copy]) - Scale each feature to the [-1, 1] range without breaking the sparsity.

preprocessing.minmax_scale(X[, ...]) - Transform features by scaling each feature to a given range.

preprocessing.normalize(X[, norm, axis, ...])  - Scale input vectors individually to unit norm (vector length).

preprocessing.quantile_transform(X, *[, ...]) - Transform features using quantiles information.

preprocessing.robust_scale(X, *[, axis, ...]) - Standardize a dataset along any axis.

preprocessing.scale(X, *[, axis, with_mean, ...]) - Standardize a dataset along any axis.

preprocessing.power_transform(X[, method, ...]) - Parametric, monotonic transformation to make data more Gaussian-like.

```
