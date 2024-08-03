# Augmentation

* https://pytorch.org/vision/stable/transforms.html


### Motivation

In machine learning, particularly with deep learning models, a common problem is overfitting. This occurs when a model performs exceptionally well on training data but fails to generalize to unseen data. This usually happens when the model has memorized the training data, capturing noise and details that do not generalize to new data points.

## Data Augmentation

Data augmentation is a powerful technique to address overfitting by artificially increasing the size and diversity of the training dataset. It involves creating synthetic variations of existing data, which helps the model generalize better.

### Key Techniques

1. **Geometric Transformations**: 
   - **Rotation**: Slightly rotating images within a certain range.
   - **Translation**: Shifting images horizontally or vertically.
   - **Scaling**: Zooming in or out.
   - **Flipping**: Horizontally flipping images.

2. **Color Space Transformations**:
   - **Brightness**: Adjusting the brightness of the images.
   - **Contrast**: Modifying the contrast levels.
   - **Saturation**: Changing the intensity of colors.
   - **Hue**: Altering the color balance.

3. **Noise Injection**:
   - Adding random noise to the images to make the model robust to noisy inputs.

4. **Random Erasing**:
   - Randomly erasing a part of the image to simulate occlusions.

5. **Cutout**:
   - Cutting out a part of the image and filling it with a constant value, usually zero.

### Real-World Examples

1. **Image Classification**:
   - In computer vision tasks like image classification, augmentation techniques like rotation, flipping, and scaling are commonly used. For instance, in a dataset of cat images, we might rotate the images by ±15 degrees and flip them horizontally to create new training samples.

2. **Object Detection**:
   - For object detection tasks, in addition to geometric transformations, techniques like random cropping and random resizing can be applied to ensure the model learns to detect objects at various scales and positions.

3. **Natural Language Processing**:
   - In text data, augmentation can be done by paraphrasing sentences, replacing words with synonyms, or even using back-translation (translating a sentence to another language and back to the original language).

4. **Speech Recognition**:
   - For audio data, techniques like time stretching, pitch shifting, and adding background noise can help create a more robust model.

### Practical Implementation in PyTorch

To implement data augmentation in PyTorch, you can use the `torchvision.transforms` module. Here’s a basic example for image data:

```python
import torchvision.transforms as transforms

# Define the transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

# Apply the transformations to the dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

dataset = ImageFolder(root='path_to_your_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the dataloader
for images, labels in dataloader:
    # Your training loop here
    pass
```

In this example, we combine several augmentation techniques such as horizontal flipping, rotation, and color jittering to increase the diversity of the training data.