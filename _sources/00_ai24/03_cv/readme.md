# AI Computer Vision 

## intro
* motivation [Exploiting Data Structure](./convolution/fullyConnectedDrawback.md)
* [cnn example for Frequency Estimation](./convolution/cnnFrqExample.ipynb)
    * [lab Convolution for Frequency Estimation](./convolution/0085DeepLearning1DConvFreqEst.ipynb) 📔show also when model dont learn to generalize
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
    * [lab calc RF](./concept/recField.ipynb)📔
* [encode decode](./concept/encodeDecode.md)
* [skip connections](./concept/skipConnections.md)


## augmentation
* [about augmentation](./augm/readme.md)
* [basic img effects](./augm/0094DeepLearningImageAugmentation.ipynb)📔
* [cut effects](./augm/cutMethods.md)
    * [lab demo cut effects](./augm/0095DeepLearningImageAugmentation.ipynb)📔
* [smoothing](./augm/smoothing.md)
    * [lab smoothing demo](./augm/0096DeepLearningLabelSmoothing.ipynb)📔
* [lab combine cutting with smooth](./augm/0097DeepLearningRegularizedTraining.ipynb)📔



## demo arch
* [lab cifar](./convolution/0086DeepLearningConv2DCifar10.ipynb)📔
* [lab mnist](./convolution/0087DeepLearningConv2DFashionMNIST.ipynb) 📔- multiple models with tensor board
* [lab student model](./concept/studentTeacherDemo.ipynb) 📔- Knowledge Distillation


## common arch
* [compare](./arch/readme.md)
* [darknet53](./arch/darknet53.md)
* [alexnet](./arch/alexnet.md)
* [vgg](./arch/vgg.md)
* [googleNet](./arch/googleNet.md)
    * [Inception Modules](./arch/inception.md)
* [resnet](./arch/resnet.md)
    * [lab resnet model](./arch/0092DeepLearningResNet.ipynb)📔
* [yolo](./arch/yolo.md)
* [minirocket](./arch/minirocket.md)
    * [lab minirocket demo](./arch/lab_minirocket_basicDemo.ipynb)📔


## inference
* [preDefine models](./infer/readme.md)
* [pytorch inference](./infer/pyInter.md)
* [sota](./infer/sota.md)
* [zoo](./infer/zoo.md)
* [lab pre train compare](./infer/0091DeepLearningPreTrainedModels.ipynb)📔



## transfer learning
* [about transfer learning](./transf/readme.md)
* [compare methods](./transf/compare.md) 
* [fineTune](./transf/fineTune.md)
* [freeze](./transf/freeze.md)
* [lab Transfer Learning of resnet](./transf/0093DeepLearningTransferLearning.ipynb)📔
* [overfit](./transf/overfit.md)
    * [lab reduce demo](./transf/lab_reducude_demo.ipynb)📔  TBD
* [transformers](./transf/transformers.md)
    * [lab transofmers basic](./transf/lab_transofmers_basic.ipynb)📔   TBD

* [adaptation](./transf/adaptation.md)

## debug by visualize

* [methods](./visualize/readme.md)

## location

```
(object) location
    
    localization                - single object in the image
        classification          - what the object
        regression              - where the object
    
    detection                   -no knowledge of how many objects are in the image
        Segmentation
            - classification    - what the object
            - pixel regression  - where the object
```

* [about location](./location/readme.md)
    * [lab pretrain](./location/lab_pretrain_demo.ipynb)📔 

#### localization
* [localization](./location/localization.md)
    * [lab localization](./location/0098DeepLearningObjectLocalization.ipynb)📔
#### detection
* [detection](./location/detection.md)
    * [lab pretrain detection demo](./location/lab_plot_transforms_e2e.ipynb)📔
* [common arch](./location/commonArch.md)
* [NMS](./location/hms.md) Non Maximum Suppression
* [MAP](./location/map.md)

* [workshop object Detection with Yolov8](./wrkshp_yolo/readme.md)
    * [labelme](./wrkshp_yolo/labelme/readme.md)
    * [0001BoundingBoxFormat](./wrkshp_yolo/0001BoundingBoxFormat.ipynb)
    * [0002Dataset](./wrkshp_yolo/0002Dataset.py)
    * [0003PreProcessTiles](./wrkshp_yolo/0003PreProcessTiles.py)
    * [0004PreProcessTrainTestSplit](./wrkshp_yolo/0004PreProcessTrainTestSplit.py)
    * [0005TrainYolo](./wrkshp_yolo/0005TrainYolo.py)

#### Segmentation

* [about Segmentation](./segmentation/readme.md)
    * [intro](./segmentation/intro.md)
    * [data](./segmentation/data.md)
    * [models](./segmentation/models.md)
    * [unet](./segmentation/unet.md)
    * [workshop segmentation UNET](./wrkshp_unet/readme.md)
        * [0001Dataset](./wrkshp_unet/0001Dataset.py)
        * [0002TrainModel](./wrkshp_unet/0002TrainModelScript.py)
        * [0003InferModel](./wrkshp_unet/0003InferModel.py)


