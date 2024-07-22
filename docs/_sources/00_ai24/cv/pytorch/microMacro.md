# micro macro









## Micro vs. Macro Averaging in Classification Metrics

In classification tasks, especially when dealing with imbalanced datasets, choosing the right averaging method is crucial for meaningful evaluation. The two common averaging methods are **micro** and **macro** averaging. 


```

from torchmetrics.classification import MulticlassAccuracy

hS = MulticlassAccuracy(num_classes = len(lClass), average = 'micro')
                                                            !!!!!!!!!!

```
* The averaging mode `macro` averages samples per class and average the result of each class.
* The averaging mode `micro` averages all samples.

### Micro Averaging

Micro averaging computes metrics globally by counting the total true positives, false negatives, and false positives. In essence, it treats each individual sample equally, regardless of the class it belongs to.

### Macro Averaging

Macro averaging, on the other hand, computes metrics for each class independently and then takes the average. This approach treats all classes equally, irrespective of their sample size, making it more suitable for imbalanced datasets.


### example1

Given 8 samples of class `A` with 6 predictions being correct and 2 samples of class `B` with 1 being correct.  
  What will be the _macro average_? What will be the _micro average_?


```
A (6/8) =>              macro (75 + 50 ) / 2 = 62.5%
B (1/2) = >             micro 7/10 = 70%
```

* for inbalanced dataset, macro is better ; 


### Example2

Given 8 samples of class `A` with 6 correct predictions and 2 samples of class `B` with 1 correct prediction, let's compute both the macro and micro averages.

#### Class A:
- Total samples: 8
- Correct predictions: 6
- Accuracy for class A: $ \frac{6}{8} = 75\% $

- Accuracy for class A: $ \frac{6}{8} = 75\% $


#### Class B:
- Total samples: 2
- Correct predictions: 1
- Accuracy for class B: $ \frac{1}{2} = 50\% $

### Macro Average

To calculate the macro average, we take the mean of the individual accuracies for each class:

$$
\text{Macro Average} = \frac{\text{Accuracy for class A} + \text{Accuracy for class B}}{2} = \frac{75\% + 50\%}{2} = 62.5\%
$$

### Micro Average

For the micro average, we sum the correct predictions and divide by the total number of samples:

$$
\text{Micro Average} = \frac{\text{Total correct predictions}}{\text{Total samples}} = \frac{6 + 1}{8 + 2} = \frac{7}{10} = 70\%
$$

### Summary

- **Macro Average**: Treats each class equally, better for imbalanced datasets.
- **Micro Average**: Treats each sample equally, better for balanced datasets.

### Python Code with PyTorch

Below is a demonstration using PyTorch to calculate these metrics using `torchmetrics`:

```python
from torchmetrics.classification import MulticlassAccuracy

# Example class distribution
lClass = ['A', 'B']

# Initializing MulticlassAccuracy
hS_micro = MulticlassAccuracy(num_classes=len(lClass), average='micro')
hS_macro = MulticlassAccuracy(num_classes=len(lClass), average='macro')

# Dummy predictions and targets
predictions = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 0, 0])  # Predicted classes
targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # True classes

# Calculate micro and macro accuracy
micro_accuracy = hS_micro(predictions, targets)
macro_accuracy = hS_macro(predictions, targets)

print(f"Micro Average Accuracy: {micro_accuracy * 100:.2f}%")
print(f"Macro Average Accuracy: {macro_accuracy * 100:.2f}%")
```

### Real-World Application

In practice, if you're working with a dataset where some classes are underrepresented, the macro average will give you a better sense of performance across all classes. On the other hand, for a balanced dataset, the micro average is often more representative of the overall performance.



















