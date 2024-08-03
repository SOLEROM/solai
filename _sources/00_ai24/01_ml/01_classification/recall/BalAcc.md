# Balanced Accuracy


* https://en.wikipedia.org/wiki/Sensitivity_and_specificity

Balanced accuracy is a performance metric used to evaluate classification models, particularly in cases where the classes are imbalanced. It provides a more comprehensive measure of the classifier's ability to handle both the majority and minority classes correctly. Balanced accuracy is defined as the average of the recall (or sensitivity) obtained on each class.

### Formula

Balanced accuracy can be calculated as follows:

$$ \text{Balanced Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right) $$

Where:
- **True Positive (TP)**: The number of correctly predicted positive instances.
- **False Negative (FN)**: The number of actual positive instances that were incorrectly predicted as negative.
- **True Negative (TN)**: The number of correctly predicted negative instances.
- **False Positive (FP)**: The number of incorrectly predicted negative instances.

Alternatively, it can be expressed as the average of recall for each class:

$$ \text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} $$

Where:
- **Sensitivity (Recall)**: $  \frac{TP}{TP + FN} $ 
- **Specificity**: $  \frac{TN}{TN + FP} $ 

### Example

Let's consider a binary classification problem with the following confusion matrix:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | TP = 50            | FN = 10            |
| Actual Negative | FP = 40            | TN = 100           |

Calculate the sensitivity (recall) and specificity:

- **Sensitivity (Recall)**: $  \frac{TP}{TP + FN} = \frac{50}{50 + 10} = 0.833 $ 
- **Specificity**: $  \frac{TN}{TN + FP} = \frac{100}{100 + 40} = 0.714 $ 

Now, calculate the balanced accuracy:

$$ \text{Balanced Accuracy} = \frac{0.833 + 0.714}{2} = 0.774 $$

### Interpretation

Balanced accuracy provides a more nuanced view of classifier performance compared to traditional accuracy, which can be misleading in the presence of class imbalance. By averaging the recall of both classes, balanced accuracy ensures that the performance on the minority class is given equal importance as the performance on the majority class.

### Key Points

- **Useful for imbalanced datasets**: Balanced accuracy accounts for the skewed distribution of classes, providing a fair evaluation of model performance.
- **Average of recall and specificity**: It balances the model's ability to correctly identify positive cases and negative cases.
- **Complementary to other metrics**: While balanced accuracy provides a more balanced view, it should be used alongside other metrics like precision, recall, and F1 score for a comprehensive evaluation.

Balanced accuracy is especially beneficial in domains where missing a minority class instance (false negative) or incorrectly predicting a majority class instance (false positive) has significant consequences, such as in medical diagnoses or fraud detection.


### method1:

calc by parts

```
_, recall, _ , _       = precision_recall_fscore_support(vY, vHatY, pos_label = 1, average = 'binary')

_, specificity, _, _   = precision_recall_fscore_support(vY, vHatY, pos_label = 0, average = 'binary')  

bAcc = 0.5 * (recall + specificity)

print(f'Balanced Accuracy = {bAcc:0.2%}')



```


### method 2: 

use method

```
# SciKit Learn Balanced Accuracy
# The `balanced_accuracy_score` can be used in binary and multi class cases.

print(f'Balanced Accuracy = {balanced_accuracy_score(vY, vHatY):0.2%}')

```