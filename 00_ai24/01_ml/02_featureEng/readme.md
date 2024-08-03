# Feature Engineering

## methods
* [normalizing and standardizing](./transforms/normStd.md)
* [non separable problem](./transforms/nonSepr.md)
* [technics list](./transforms/techList.md)
    * [lab Features Transform case1](./transforms/0037FeaturesTransform.ipynb)
    * [lab Features Transform case2](./transforms/0038FeaturesTransform.ipynb)

## kernels 

The idea behind the kernel trick is to implicitly map data to a higher-dimensional space where it becomes linearly separable, enabling linear algorithms to solve nonlinear problems. This mapping is done through a kernel function, which computes the inner product between the images of two data points in this higher-dimensional space. Essentially, the kernel function measures the similarity between pairs of data points.

* [recap dual](./kernelTrick/outline.md)
* [define kernel](./kernelTrick/dual.md)
* [lab kernel types demo ](./kernelTrick/0039ClassifierKernelSVM.ipynb)
* [mnist example](./mnist/readme.md)
* [lab fashion-mnist](./mnist/0040ClassifierKernelSVM.ipynb)

## Feature Engineering Tools
* [preprocessing](../scikit/preprocessing.md)
* [pipe](../scikit/pipe.md)
* [grid search demo](../play/gridSearchPipeline.ipynb)
* [autouml](./autouml/readme.md)


## summary

```
* continuous features
    * combination of features 
        * polynominal
        * normalization and standardization
        * change of coordinates
    * transforms:
        * kernels
        * STFT
        * wavelet
        * dictionaries

* discrete features
    * encoding (one-hot)
    * grouping (valid combinations)    

* missing value
    * dropping (features/ samples)
    * imputer by a model - nearest / interpolation / classifier / regressor

* data / time
    * day of the week/month/year

* text
    * steamming
    * lemmatization

* feature selction

* dimensionality reduction

* automatic feature generation (autoML)

* stemming vs lemmatiztion
```

## TBD
https://www.kaggle.com/code/willkoehrsen/automated-feature-engineering-tutorial
https://www.kaggle.com/code/willkoehrsen/automated-feature-engineering-basics

