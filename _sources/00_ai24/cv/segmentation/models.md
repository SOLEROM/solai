# models

Modern models are based on U-Net like shape -  using high receptive field by encoder decoder model with skip connection methods;

![alt text](image-6.png)

## Resources

* Comparative Study of Image Segmentation Architectures Using Deep Learning - https://scribe.rip/3743875fd608
* Image Segmentation: Architectures, Losses, Datasets and Frameworks - https://neptune.ai/blog/image-segmentation
* Complete Guide to Semantic Segmentation - https://www.superannotate.com/blog/guide-to-semantic-segmentation


## UpSample method

### for Signal Processing
* insert zero
* apply low pass filter

![alt text](image-7.png)

One could generalize the model by using any given interpolation method instead of applying Low Pass Filter.

### for Image Processing 

* increase zero
* Apply Interpolation

![alt text](image-8.png)

![alt text](image-9.png)

## upsample layer

* The conventional upsampling methods are not adaptive to the loss.
* use learned filter coefficients (LPF).
* The concept of Transposed Convolution was introduced in Fully Convolutional Networks for Semantic Segmentation.
* in pytorch:
    * https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

* Adjoint Operator
![alt text](image-10.png)

![alt text](image-11.png)



## Resources

* What is Transposed Convolutional Layer - https://scribe.rip/40e5e6e31c11
* Understand Transposed Convolutions - https://scribe.rip/4f5d97b2967
* Dive into Deep Learning - Computer Vision - Transposed Convolution - https://d2l.ai/chapter_computer-vision/transposed-conv.html 


  