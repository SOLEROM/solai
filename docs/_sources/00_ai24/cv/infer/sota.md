# sota - state-of-the-art

* https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/


```
+ LR optimizations 	
+ TrivialAugment 	
+ Long Training 	
+ Random Erasing 	
+ Label Smoothing 
+ Mixup 	
+ Cutmix 	
+ Weight Decay tuning 	
+ FixRes mitigations 	
+ EMA 	
+ Inference Resize tuning 
+ Repeated Augmentation 
```



### LR Optimizations
**Learning Rate (LR) Optimization** involves adjusting the learning rate during training to improve convergence and performance. Techniques like learning rate schedules, warm restarts, and adaptive learning rates (e.g., using optimizers like Adam, RMSprop) help in finding a better minimum of the loss function.
- **How it helps**: Proper LR adjustments prevent the model from getting stuck in local minima and ensure faster and more stable convergence.

### TrivialAugment
**TrivialAugment** is a simple data augmentation technique that applies a random transformation from a predefined set of augmentations to the input data.
- **How it helps**: By introducing random variations, the model becomes more robust to changes and improves generalization.

### Long Training
**Long Training** refers to extending the training duration, often with a smaller learning rate, to allow the model to fine-tune its parameters more precisely.
- **How it helps**: Prolonged training with a reduced learning rate can lead to better convergence and improved performance.

### Random Erasing
**Random Erasing** is a data augmentation technique where random patches of input images are masked out during training.
- **How it helps**: This makes the model more resilient to missing parts of the image, improving robustness and generalization.

### Label Smoothing
**Label Smoothing** involves softening the labels by assigning a small probability to all classes instead of assigning a probability of 1 to the correct class and 0 to others.
- **How it helps**: It prevents the model from becoming overconfident, leading to better calibration and improved generalization.

### Mixup
**Mixup** creates new training examples by combining pairs of examples and their labels with a certain mixing ratio.
- **How it helps**: This encourages the model to behave linearly in-between training examples, improving robustness and generalization.

### Cutmix
**Cutmix** combines two images by cutting a patch from one image and pasting it into another, while mixing their labels proportionally.
- **How it helps**: Similar to Mixup, it promotes learning more robust features and improves generalization by providing more varied training examples.

### Weight Decay Tuning
**Weight Decay Tuning** involves adjusting the regularization parameter that penalizes large weights to prevent overfitting.
- **How it helps**: Proper weight decay tuning helps in controlling the complexity of the model, leading to better generalization.

### FixRes Mitigations
**FixRes (Fixed Resolution) Mitigations** involve adjusting the resolution of input images during different stages of training and testing to improve accuracy.
- **How it helps**: It helps in training models at higher resolutions, making them more robust to different scales and improving performance.

### EMA (Exponential Moving Average)
**Exponential Moving Average (EMA)** maintains a moving average of the model parameters during training.
- **How it helps**: EMA can stabilize training and improve final model performance by averaging out the noise in parameter updates.

### Inference Resize Tuning
**Inference Resize Tuning** involves adjusting the size of input images at inference time to optimize performance.
- **How it helps**: Properly tuning the input size can balance accuracy and computational efficiency, leading to better model performance during inference.

### Repeated Augmentation
**Repeated Augmentation** applies the same augmentation multiple times to the same image within a batch.
- **How it helps**: It increases the diversity of the training data seen by the model, enhancing robustness and generalization.


### Batch Normalization
**Batch Normalization (BatchNorm)** normalizes the activations of each layer for each mini-batch during training.
- **How it helps**: It reduces internal covariate shift, leading to faster convergence and higher stability during training.

### Dropout
**Dropout** randomly drops neurons during training to prevent overfitting.
- **How it helps**: By randomly omitting neurons, dropout forces the network to learn more robust features and prevents overfitting.

### Gradient Clipping
**Gradient Clipping** involves capping the gradients during backpropagation to prevent the problem of exploding gradients.
- **How it helps**: It stabilizes training by ensuring that the gradient updates are not excessively large, which can otherwise destabilize training.

### Transfer Learning
**Transfer Learning** leverages pre-trained models on large datasets to initialize a model for a new task.
- **How it helps**: It speeds up training and often leads to better performance by starting with a model that already has learned useful features.

### Cosine Annealing
**Cosine Annealing** is a learning rate schedule that reduces the learning rate following a cosine function.
- **How it helps**: It allows the model to converge more smoothly by reducing the learning rate gradually and cyclically.

### Early Stopping
**Early Stopping** monitors the model’s performance on a validation set and stops training when performance stops improving.
- **How it helps**: It prevents overfitting by halting training once the model starts to overfit the training data.

### Warmup Learning Rate
**Warmup Learning Rate** starts training with a very small learning rate and gradually increases it to the initial value.
- **How it helps**: It helps in stabilizing training in the initial epochs, especially for very deep networks.

### Knowledge Distillation
**Knowledge Distillation** involves training a smaller, student model to mimic the outputs of a larger, teacher model.
- **How it helps**: The student model can achieve competitive performance with reduced complexity and better generalization.

### Data Augmentation
**Data Augmentation** involves generating new training samples through transformations like rotation, scaling, and flipping.
- **How it helps**: It artificially enlarges the training dataset, helping the model generalize better by learning from more diverse examples.

### AutoAugment
**AutoAugment** is a data augmentation technique that uses reinforcement learning to find the best augmentation policies.
- **How it helps**: It automatically finds effective augmentation strategies that improve model performance.

### Stochastic Depth
**Stochastic Depth** randomly drops entire layers during training.
- **How it helps**: It acts as a form of regularization, making the network more robust and reducing overfitting.

### Ghost Batch Normalization
**Ghost Batch Normalization** splits a large batch into smaller "ghost" batches for batch normalization.
- **How it helps**: It reduces memory usage and makes training more stable, especially for very large batches.

### Label Propagation
**Label Propagation** uses unlabeled data by propagating labels from labeled data points.
- **How it helps**: It leverages both labeled and unlabeled data, improving the model's performance in semi-supervised learning settings.

### Gradient Accumulation
**Gradient Accumulation** accumulates gradients over multiple mini-batches before performing an update.
- **How it helps**: It allows the use of larger effective batch sizes without increasing memory requirements.

