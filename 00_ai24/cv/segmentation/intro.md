# intro


## def

segmentation task:

    * classifiaction - what is the object
    * where is the object

![alt text](image.png)

## types

![alt text](image-1.png)
![alt text](image-2.png)



```
Semantic: Pixel (Each) is labeled by its texture and other image related properties.
    
Instance: Pixel is labeled as part of a predefined set of objects. Each object is uniquely identified (Can be counted).
    
Panoptic: Pixel (Each) is labeled by its texture and object.

```

## output

![alt text](image-3.png)

output of L is num of classes;

argmax on each L channel will yield with 

![alt text](image-4.png)

![alt text](image-5.png)


## score

* by definition is mostly imbalanced
* so the Imbalanced classification Scores can be:
    * Balanced Accuracy.
    * Recall, Precision.
    * Dice / F1.
    * Confusion Matrix
* Object Scores
    * IoU.
    * mAP.

![alt text](image-15.png)

![alt text](image-16.png)


### resources
* Understanding Evaluation Metrics in Medical Image Segmentation  - https://scribe.rip/d289a373a3f
* Evaluating Image Segmentation Models - https://www.jeremyjordan.me/evaluating-image-segmentation-models/
* Image Segmentation — Choosing the Correct Metric - https://scribe.rip/aa21fd5751af
* miseval: A Metric Library for Medical Image Segmentation EVALuation - https://github.com/frankkramer-lab/miseval
* Kaggle: All the Segmentation Metrics, Understanding Dice Coefficient, Visual Guide To Understanding Segmentation Metrics - https://www.kaggle.com/code/yassinealouini/all-the-segmentation-metrics


# The Loss Function

* Cross Entropy Loss.
* Cross Entropy Loss + Label Smoothing.
* Balanced Cross Entropy / Focal Loss.
* Gradient Friendly Region / Boundary Loss.

### resources
* Loss Functions for Image Segmentation - https://github.com/JunMa11/SegLossOdyssey
* 3 Common Loss Functions for Image Segmentation https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o
* Instance segmentation loss functions - https://softwaremill.com/instance-segmentation-loss-functions/
* Focal Loss: An Efficient Way of Handling Class Imbalance - https://scribe.rip/4855ae1db4cb




