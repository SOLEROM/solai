# Training Process Comparison

compare the lazy learning of kNN with the eager learning of SVM:

* Learning Approach: kNN is a non-parametric, instance-based, lazy learning algorithm, meaning it doesn't learn a generalized model but instead memorizes the training dataset. SVM is a parametric, eager learning algorithm that actively constructs a model (the optimal hyperplane) during the training process.
* Computation: For kNN, the major computational workload is during prediction, requiring distance calculations between test instances and all training instances. For SVM, the computational effort is upfront during the training phase, where it solves an optimization problem to find the decision boundary.
* Performance in High Dimensions: While both kNN and SVM can work with high-dimensional data, SVMs, especially with appropriate kernel functions, are generally better suited for dealing with the curse of dimensionality and finding complex patterns in high-dimensional spaces.

* In summary, fitting an SVM involves a complex process of optimization and model building that actively learns from the training data, in contrast to the simpler, data-storing process of fitting a kNN model.

