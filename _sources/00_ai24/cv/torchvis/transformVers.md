# torchvision.transforms


TBD ???

https://pytorch.org/vision/stable/transforms.html

https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html

https://pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html


### Key Changes from v1 to v2:

1. **ToImage and ToDtype Transformations**:
    - **v2** introduces `ToImage()` and `ToDtype()` transformations.
    - `ToImage()` converts the input to an image.
    - `ToDtype(dtype, scale=True)` ensures the data type of the tensor and optionally scales it.
    - In **v1**, this was managed implicitly within `ToTensor()`, which converts a `PIL` image or `numpy.ndarray` to a tensor.

2. **Image Size Handling**:
    - **v2** explicitly addresses different dimensions and ensures the size matches the input size of ImageNet (224x224) using `Resize(224)` and `CenterCrop(224)`.
    - **v1** handles resizing and cropping similarly but lacks explicit control over data types before normalization.

3. **Normalization**:
    - Both versions use `Normalize(mean, std)` for standardizing the dataset based on mean and standard deviation values.

4. **Data Augmentation**:
    - Both **v1** and **v2** include `RandomHorizontalFlip(p=0.5)` for data augmentation.

Here is a more detailed breakdown with examples:

### v2 Transformation Pipeline

```python
import torch
import torchvision.transforms as TorchVisionTrns

# Update Transforms using v2
oDataTrnsTrain = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),  # Convert input to image
    TorchVisionTrns.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale
    TorchVisionTrns.Resize(224),  # Resize to 224x224
    TorchVisionTrns.CenterCrop(224),  # Center crop to 224x224
    TorchVisionTrns.RandomHorizontalFlip(p=0.5),  # Apply random horizontal flip
    TorchVisionTrns.Normalize(mean=vMean, std=vStd),  # Normalize with mean and std
])

oDataTrnsVal = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),
    TorchVisionTrns.ToDtype(torch.float32, scale=True),
    TorchVisionTrns.Resize(224),
    TorchVisionTrns.CenterCrop(224),
    TorchVisionTrns.Normalize(mean=vMean, std=vStd),
])
```

### v1 Transformation Pipeline

```python
import torchvision.transforms as transforms

# Using v1 Transforms
oDataTrnsTrain = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Apply random horizontal flip
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=vMean, std=vStd),  # Normalize with mean and std
])

oDataTrnsVal = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=vMean, std=vStd),
])
```

### Update Dataset Transforms

```python
# Update the dataset transformer
dsTrain.transform = oDataTrnsTrain
dsVal.transform = oDataTrnsVal
```

### Summary of Differences:

- **Data Type and Scaling**: v2 explicitly handles data type conversion and scaling with `ToDtype()`.
- **Image Conversion**: v2 uses `ToImage()` for image conversion.
- **Consistency**: Both pipelines maintain resizing, cropping, flipping, and normalization.

### Real-World Examples:

- **Image Classification Tasks**: Using these transformations ensures that images are consistently pre-processed, which is crucial for tasks like image classification with models pre-trained on ImageNet.
- **Data Augmentation**: The `RandomHorizontalFlip` helps in augmenting the dataset, improving the model's generalization.

If you have any specific part of this process that you would like to see implemented in code, or if you have further questions, feel free to ask!