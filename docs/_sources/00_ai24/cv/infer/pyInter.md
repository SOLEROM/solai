# pytorch Inference

* load the models
```
# Model
# Name, Constructor, Weights
lModels = [('AlexNet', torchvision.models.alexnet, torchvision.models.AlexNet_Weights.IMAGENET1K_V1),
           ('VGG16', torchvision.models.vgg16, torchvision.models.VGG16_Weights.IMAGENET1K_V1),
           ('InceptionV3', torchvision.models.inception_v3, torchvision.models.Inception_V3_Weights.IMAGENET1K_V1),
           ('ResNet152', torchvision.models.resnet152, torchvision.models.ResNet152_Weights.IMAGENET1K_V2),
           ]

```

* do the inference

```
oModel = modelClass(weights = modelWeights)         ## load the mode = load the weights !!!!
oModel = oModel.eval()          #<! Batch Norm / Dropout Layers
oModel = oModel.to('cpu')       ## Inference on the cpu 
with torch.inference_mode():
    vYHat = oModel(tI)      ## ti is the test image

```

* get result and probability

```
        vProb   = torch.softmax(vYHat, dim = 0) #<! Probabilities  
        clsIdx  = torch.argmax(vYHat)
        clsProb = vProb[clsIdx] #<! Probability of the class

```


