# cost

## Cost in Machine Learning

### Weighted Loss for General Loss Function

In machine learning, the cost (or loss) function measures how well a model's predictions match the actual data. A general loss function can be weighted to account for different types of errors or to emphasize certain examples more than others. This can be particularly useful in cases where some errors are more costly than others or when dealing with imbalanced datasets.

#### General Overview

A general loss function $ L $ can be represented as:

$$ L(y, \hat{y}) $$

where $ y $ is the true label and $ \hat{y} $ is the predicted label. To apply weights, we introduce a weight $ w $ that can vary for each instance:

$$ L_{\text{weighted}}(y, \hat{y}) = w \cdot L(y, \hat{y}) $$

Weights $ w $ can be assigned based on various criteria, such as the importance of the sample or the frequency of the class in the dataset.

#### Real-World Examples

1. **Imbalanced Dataset**: In a medical diagnosis task, false negatives might be more costly than false positives. By assigning a higher weight to false negatives, the model can be trained to minimize these more aggressively.
2. **Class Imbalance**: In a fraud detection system, fraudulent transactions might be much rarer than legitimate ones. Weighting the loss function can ensure the model pays more attention to the rare but important fraud cases.

### Hamming Loss

Hamming loss is a specific type of loss function used in multi-label classification tasks, where each instance can belong to multiple classes simultaneously.

#### General Overview

Hamming loss is the fraction of incorrect labels. For a set of true labels $ y $ and predicted labels $ \hat{y} $, the Hamming loss is defined as:

$$ L_{\text{Hamming}}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n \mathbb{1}(y_i \neq \hat{y}_i) $$

where $ \mathbb{1} $ is the indicator function that returns 1 if the condition is true and 0 otherwise, and $ n $ is the total number of labels.

#### Real-World Examples

1. **Document Tagging**: In a document tagging system where a document can have multiple tags, the Hamming loss helps measure how many tags are incorrectly predicted or missed.
2. **Image Annotation**: In an image annotation task where each image can have multiple labels (e.g., "dog", "outdoor", "sunny"), the Hamming loss gives a straightforward measure of label prediction accuracy.

By understanding and applying these loss functions correctly, you can improve your machine learning models, especially in scenarios with imbalanced data or multi-label classification tasks.




## Balanced Weighing

* Weight per Class -  Applies the weighing on the samples according to their class.   
* Usually applied in SciKit Learn under `class_weight`.
* has a `balanced` option which tries to balance imbalanced data.

### auto

```
oSVM  = SVC(C = paramC, kernel = kernelType, class_weight = 'balanced').fit(mX, vY)

```

### manual

```
# SVM Linear Model - Manual Weighing
# exmaple : class 0 to 1, and class 1 to 1000.

#===========================Fill This===========================#
dClassWeight = {0: 1, 1: 1000} #<! Weighing dictionary
#===========================Fill This===========================#

oSVM  = SVC(C = paramC, kernel = kernelType, class_weight = dClassWeight).fit(mX, vY)
```
