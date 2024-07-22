## Mean Average Precision (mAP)

### Overview

Mean Average Precision (mAP) is a comprehensive metric used to evaluate the performance of object detection models. It involves calculating precision and recall for each class, plotting precision-recall curves, and then averaging the results to obtain a single score that reflects the model's ability to detect and localize objects accurately.

### Key Concepts

#### True Positives (TP)
A detection is a true positive if it correctly identifies an object and its bounding box sufficiently overlaps with the ground truth bounding box.

#### False Positives (FP)
A detection is a false positive if it incorrectly identifies an object (either no object is present or it doesn't overlap sufficiently with any ground truth box).

#### False Negatives (FN)
A ground truth object is considered a false negative if it is not detected by the model.

#### True Negatives (TN)
In the context of object detection, true negatives are usually not considered since the focus is on detecting objects within a region of interest.

### Precision and Recall

- **Precision**: The ratio of true positive detections to the total number of detections.
  $
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $

- **Recall**: The ratio of true positive detections to the total number of ground truth objects.
  $
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $

### Precision-Recall Curve

A precision-recall curve is a graph that plots precision (\(y\)-axis) against recall (\(x\)-axis) at various threshold settings. It provides a comprehensive view of the model's performance across different levels of confidence.

### Steps to Compute mAP

1. **Compute Precision and Recall**: For each class, compute precision and recall at various thresholds.
2. **Plot Precision-Recall Curve**: Plot the precision against recall for each class.
3. **Calculate Average Precision (AP)**: Calculate the area under the precision-recall curve for each class.
4. **Mean Average Precision (mAP)**: Average the AP values across all classes.

### Average Precision (AP)

AP is calculated as the area under the precision-recall curve. One common approach is to use interpolation to ensure that precision does not increase as recall increases.

### Example of Calculating AP

Consider precision and recall values at different thresholds:

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.9       | 1.0       | 0.1    |
| 0.8       | 0.9       | 0.2    |
| 0.7       | 0.8       | 0.3    |
| 0.6       | 0.7       | 0.4    |
| 0.5       | 0.6       | 0.5    |

To calculate AP, you would plot these points on a precision-recall graph and compute the area under the curve.

### Mean Average Precision (mAP)

Once the AP for each class is calculated, the mAP is obtained by averaging these AP values. For \( N \) classes, mAP is given by:

$
\text{mAP} = \frac{1}{N} \sum_{i=1}^N \text{AP}_i
$

### Real-World Example

Suppose you are evaluating an object detection model on a dataset containing three classes: cars, pedestrians, and bicycles. After running the model, you get the following AP values:

- **Cars**: 0.85
- **Pedestrians**: 0.75
- **Bicycles**: 0.65

The mAP for the model would be:

$
\text{mAP} = \frac{1}{3} (0.85 + 0.75 + 0.65) = 0.75
$

### Importance in Object Detection

- **Comprehensive Evaluation**: mAP provides a comprehensive evaluation of both detection accuracy and localization precision.
- **Class-wise Performance**: By considering AP for each class, mAP helps in understanding the performance of the model across different object categories.
- **Benchmarking**: mAP is widely used for benchmarking object detection models on standard datasets like COCO, PASCAL VOC, etc.

### Example Calculation of TP, FP, FN

Consider an object detection task with the following results:

- **Ground Truth Objects**: 10
- **Detected Objects**: 12
  - **Correctly Detected (TP)**: 8
  - **Incorrectly Detected (FP)**: 4
  - **Missed Detections (FN)**: 2

| Metric           | Value |
|------------------|-------|
| True Positives (TP) | 8     |
| False Positives (FP) | 4     |
| False Negatives (FN) | 2     |

### Precision and Recall Calculation

$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{8}{8 + 4} = \frac{8}{12} = 0.67
$

$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{8}{8 + 2} = \frac{8}{10} = 0.80
$

### Precision-Recall Curve and AP Calculation

The precision-recall curve can be plotted using the precision and recall values at various thresholds. The AP is the area under this curve.

### Python Code Example

Here's a basic implementation of the mAP calculation in Python using PyTorch:

```python
import numpy as np

def compute_ap(precision, recall):
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def compute_map(detections, ground_truths, iou_threshold=0.5):
    aps = []
    for class_id in range(num_classes):
        class_detections = [d for d in detections if d['class_id'] == class_id]
        class_ground_truths = [g for g in ground_truths if g['class_id'] == class_id]

        # Sort detections by confidence
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        gt_detected = []

        for i, detection in enumerate(class_detections):
            max_iou = 0
            max_gt_idx = -1
            for j, gt in enumerate(class_ground_truths):
                if gt['image_id'] == detection['image_id']:
                    iou = calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = j

            if max_iou >= iou_threshold and max_gt_idx not in gt_detected:
                tp[i] = 1
                gt_detected.append(max_gt_idx)
            else:
                fp[i] = 1

        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        recall = np.cumsum(tp) / len(class_ground_truths)
        ap = compute_ap(precision, recall)
        aps.append(ap)

    return np.mean(aps)

# Example usage
num_classes = 3
detections = [
    {'image_id': 1, 'class_id': 0, 'bbox': [50, 50, 100, 100], 'confidence': 0.9},
    # Add more detections
]
ground_truths = [
    {'image_id': 1, 'class_id': 0, 'bbox': [48, 48, 102, 102]},
    # Add more ground truths
]

mAP = compute_map(detections, ground_truths)
print(f"mAP: {mAP:.4f}")
```
