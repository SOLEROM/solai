# the task

Given an Image, Classify each pixel into 3 classes:

    Segment Dog pixels.
    Segment Cat pixel.
    Segment Background pixels


## venv:
    EnvImageSegmentation.yml

## dataset

* https://www.robots.ox.ac.uk/~vgg/data/pets/

```
Each image is an RGB image.
Image dimensions is non constant.
Masks are in the range [1, 2, 3]
```

## 01 prepare
* run 0001Dataset.py
* Ensure the data in the folder Data.
* label is black because it is 1,2,3 out of 256 which are all grey
* if we draw by 3 colors only we will get:
![alt text](image-15.png)
* for argmax we need 0,1,2 - convert to that ;


## 02 train

*  run 0002TrainModelScript.py

## help functions

```
DL
├── DataLoader.py
├── Training.py
└── UNetModule.py
```

## 03 inference


