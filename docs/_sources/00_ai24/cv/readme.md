# AI Computer Vision 

## intro
* motivation [Exploiting Data Structure](./convolution/fullyConnectedDrawback.md)
* [cnn example for Frequency Estimation](./convolution/cnnFrqExample.ipynb)
    * :green_book: [lab Convolution for Frequency Estimation](./convolution/0085DeepLearning1DConvFreqEst.ipynb) show also when model dont learn to generalize
* [Basic convulsion](./convolution/readme.md) padding and basic kernel examples
* [2d convolution](./convolution/2d.md)


## basic features
* [layers](./convolution/layers.md)
* [pooling](./convolution/pooling.md)
* [batch Norm](./convolution/batchNorm.md)
    * [Norm demo](./convolution/lab_batchNorm.ipynb)
* [filters](./convolution/filters.md)
    * [laplacian](./convolution/filter_laplacian.md)


### pytorch layers
* [common layers](./pytorch/common.md)
* [custom nn](./pytorch/custom.md)
* [MulticlassAccuracy](./pytorch/microMacro.md) micro macro modes


### lib
* [torchvision lib](./torchvis/readme.md)
* [v1/v2.transforms ](./torchvis/transformVers.md)

## concepts
* [receptive field](./concept/recfield.md)
    * :green_book: [lab calc RF](./concept/recField.ipynb)
* [encode decode](./concept/encodeDecode.md)
* [skip connections](./concept/skipConnections.md)


## augmentation
* [about augmentation](./augm/readme.md)
* :green_book: [basic img effects](./augm/0094DeepLearningImageAugmentation.ipynb)
* [cut effects](./augm/cutMethods.md)
    * :green_book: [lab demo cut effects](./augm/0095DeepLearningImageAugmentation.ipynb)
* [smoothing](./augm/smoothing.md)
    * :green_book: [lab smoothing demo](./augm/0096DeepLearningLabelSmoothing.ipynb)
* :green_book: [lab combine cutting with smooth](./augm/0097DeepLearningRegularizedTraining.ipynb)



## demo arch
* :green_book: [lab cifar](./convolution/0086DeepLearningConv2DCifar10.ipynb)
* :green_book: [lab mnist](./convolution/0087DeepLearningConv2DFashionMNIST.ipynb) - multiple models with tensor board
* :green_book: [lab student model](./concept/studentTeacherDemo.ipynb) - Knowledge Distillation


## common arch
* [compare](./arch/readme.md)
* [darknet53](./arch/darknet53.md)
* [alexnet](./arch/alexnet.md)
* [vgg](./arch/vgg.md)
* [googleNet](./arch/googleNet.md)
    * [Inception Modules](./arch/inception.md)
* [resnet](./arch/resnet.md)
    * :green_book: [lab resnet model](./arch/0092DeepLearningResNet.ipynb)
* [yolo](./arch/yolo.md)
* [minirocket](./arch/minirocket.md)
    * :green_book: [lab minirocket demo](./arch/lab_minirocket_basicDemo.ipynb)


## inference
* [preDefine models](./infer/readme.md)
* [pytorch inference](./infer/pyInter.md)
* [sota](./infer/sota.md)
* [zoo](./infer/zoo.md)
* :green_book: [lab pre train compare](./infer/0091DeepLearningPreTrainedModels.ipynb)



## transfer learning
* [about transfer learning](./transf/readme.md)
* [compare methods](./transf/compare.md) 
* [fineTune](./transf/fineTune.md)
* [freeze](./transf/freeze.md)
* :green_book: [lab Transfer Learning of resnet](./transf/0093DeepLearningTransferLearning.ipynb)
* [overfit](./transf/overfit.md)
    * :green_book: [lab reduce demo](./transf/lab_reducude_demo.ipynb)  TBD
* [transformers](./transf/transformers.md)
    * :green_book: [lab transofmers basic](./transf/lab_transofmers_basic.ipynb)  TBD

## debug
visualize cnn
https://jithinjk.github.io/blog/nn_visualized.md.html




* [location](./aaa/location.md)
* [lab pretrain](./aaa/lab_pretrain_demo.ipynb)

* [localization](./aaa/localization.md)
    * [lab localization](./aaa/0098DeepLearningObjectLocalization.ipynb)

* [](./aaa/lab_plot_transforms_e2e.ipynb)

* [detection](./aaa/detection.md)
* [NMS](./aaa/hms.md) Non Maximum Suppression
* [MAP](./aaa/map.md)


## detection


## localization