# ada boost

### AdaBoost: Overview

AdaBoost, short for Adaptive Boosting, is an ensemble learning algorithm primarily used for classification tasks. It was introduced by Yoav Freund and Robert Schapire in 1995. The central idea of AdaBoost is to combine multiple weak classifiers to form a strong classifier. Each weak classifier is trained sequentially, with a focus on the mistakes made by the previous classifiers. This adaptive process ensures that the model improves iteratively.

### Key Concepts

1. **Weak Classifiers**: These are models that perform slightly better than random guessing. In the context of AdaBoost, decision stumps (one-level decision trees) are often used as weak classifiers.

2. **Weights**: AdaBoost assigns weights to each training instance. Initially, all weights are equal, but they are adjusted after each round of training to emphasize the misclassified instances.

3. **Boosting Process**: The algorithm iteratively trains weak classifiers, adjusting the weights of misclassified instances to improve the subsequent classifier's focus on hard-to-classify examples.

4. **Combination of Classifiers**: The final model is a weighted sum of the weak classifiers. Each classifier's contribution is proportional to its accuracy.

### Algorithm Steps

1. **Initialize Weights**: Start with weights $ w_i = \frac{1}{N} $ for $ i = 1, \ldots, N $, where $ N $ is the number of training samples.

2. **For each iteration $ t $ from 1 to $ T $**:
    - **Train a Weak Classifier**: Train a weak classifier $ h_t(x) $ using the weighted training data.
    - **Compute the Error**: Calculate the error $ \epsilon_t $ of the classifier $ h_t $ as the sum of the weights of the misclassified instances:
      $$
      \epsilon_t = \sum_{i=1}^{N} w_i \cdot I(y_i \neq h_t(x_i))
      $$
      where $ I $ is the indicator function.
    - **Compute Classifier Weight**: Calculate the weight $ \alpha_t $ of the classifier:
      $$
      \alpha_t = \frac{1}{2} \ln \left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
      $$
    - **Update Weights**: Update the weights for the next iteration:
      $$
      w_i \leftarrow w_i \cdot \exp(\alpha_t \cdot I(y_i \neq h_t(x_i)))
      $$
      Normalize the weights so they sum to 1:
      $$
      w_i \leftarrow \frac{w_i}{\sum_{j=1}^{N} w_j}
      $$

3. **Final Model**: The final strong classifier is a weighted majority vote of the $ T $ weak classifiers:
   $$
   H(x) = \text{sign} \left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)
   $$

### Applications

- **Text Classification**: AdaBoost is used in spam detection, sentiment analysis, and topic categorization.
- **Face Detection**: It plays a crucial role in object detection frameworks, such as the Viola-Jones face detector.
- **Medical Diagnosis**: Used for classifying medical conditions based on patient data.
- **Fraud Detection**: Identifying fraudulent transactions in financial services.

### Advantages

1. **Improved Accuracy**: Combines multiple weak classifiers to form a strong classifier with better accuracy.
2. **Adaptability**: Focuses on hard-to-classify instances, improving the model's performance iteratively.
3. **Simplicity**: Conceptually simple and can be used with various types of weak classifiers.

### Disadvantages

1. **Sensitivity to Noisy Data**: Performance can degrade if the training data contains significant noise or outliers.
2. **Overfitting**: Although generally robust, AdaBoost can overfit if the number of iterations is too high.
3. **Computational Complexity**: Sequential training of classifiers can be computationally intensive, especially with large datasets.

AdaBoost remains a powerful and widely used algorithm in machine learning due to its effectiveness and versatility in improving the performance of weak classifiers.


## sci kit
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html


## choose the estimator

![alt text](image-7.png)

combine estimators hyper params with cv in ada-boost

![alt text](image-8.png)
