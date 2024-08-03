# Localization

Localization is a critical task in computer vision that involves determining the location of objects within an image. This is typically achieved using bounding boxes. 


## Bounding Box

* https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

![alt text](image-1.png)


### Conventions:

Bounding boxes are rectangular boxes used to define the position and size of an object in an image. They are usually represented by the coordinates of their corners or by their center coordinates along with their width and height.

- **Standard Bounding Box**: Defined by the coordinates of the top-left corner (x_min, y_min) and the bottom-right corner (x_max, y_max).

### Rotated Bounding Boxes:

To solve for a rotated bounding box, you can take the input image, rotate it, and check for the minimal size of the bounding box. This process involves finding the smallest enclosing rectangle that can cover the object after rotation.

**Steps to find a rotated bounding box:**
1. Rotate the input image.
2. Calculate the bounding box for the rotated image.
3. Determine the minimal bounding box size.

## Intersection over Union (IoU)

Intersection over Union (IoU) is a metric used to evaluate the accuracy of an object detector on a particular dataset. It is defined as the area of overlap between the predicted bounding box and the ground truth bounding box divided by the area of their union.


![alt text](image-2.png)

* score above 0.5 is consider good;

### IoU Formula:

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

IoU is used as a score to measure how well the predicted bounding box matches the ground truth bounding box.

### Why Not Use IoU as a Loss Function?

Although IoU is a great metric for evaluating model performance, it is not commonly used as a loss function for training object detectors. The primary reason is that IoU is often zero for most predictions, especially during the early stages of training when the predicted boxes do not overlap with the ground truth boxes. This results in a gradient of zero, which hinders the learning process.

## Loss Function for Bounding Box Regression

In practice, loss functions like Mean Squared Error (MSE) or Smooth L1 Loss are used for bounding box regression. These loss functions are more stable and provide better gradients for optimization.

### MSE and IoU Correlation:

Mean Squared Error (MSE) measures the average squared difference between the predicted coordinates and the ground truth coordinates. However, MSE is not directly correlated with IoU because it measures the error in the coordinate space rather than the overlap between boxes.

### Smooth L1 Loss:

Smooth L1 Loss, also known as Huber Loss, is a combination of L1 Loss and L2 Loss, providing a balance between robustness and sensitivity. It is less sensitive to outliers than MSE and provides smoother gradients, making it a popular choice for object detection tasks.

### Smooth L1 Loss Formula:

$$
\text{Smooth L1 Loss}(x) = 
\begin{cases} 
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}
$$

Using Smooth L1 Loss helps in achieving better convergence and more accurate bounding box predictions compared to MSE.

![alt text](image-3.png)

* https://pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou_loss.html