# Cut Effects

## CutOut

CutOut is a data augmentation technique where a random square patch is cut out from an image during training. This method helps in improving the robustness of the model by simulating occlusions and forcing the model to learn features that are distributed across the image.

### How it works

- A square patch of a given size is randomly removed from the image.
- The removed patch is usually filled with zeros or the mean pixel value of the dataset.
- This augmentation helps prevent the model from becoming overly reliant on specific parts of the image.

Example:
```python
import torch
import torchvision.transforms as transforms
import numpy as np

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

transform = transforms.Compose([
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])
```

## MixUp

MixUp is a data augmentation technique where two images and their corresponding labels are mixed together to create a new training sample. This technique can improve the accuracy of the model by around 5%, as it encourages the model to behave linearly in-between training examples.

### How it works

- Two images $ x_i $ and $ x_j $ are mixed together using a mixing factor $ \lambda $, which is sampled from a Beta distribution.
- The corresponding labels $ y_i $ and $ y_j $ are also mixed using the same $ \lambda $.

Mathematically:

$ \tilde{x} = \lambda x_i + (1 - \lambda) x_j $
$ \tilde{y} = \lambda y_i + (1 - \lambda) y_j $

Example:
```python
import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## CutMix

CutMix is an augmentation technique that combines CutOut and MixUp. In CutMix, a part of one image is cut and pasted onto another image, and the labels are mixed proportionally to the area of the patches.

### How it works

- A patch from one image is cut and pasted onto another image.
- The labels are mixed based on the area of the patches involved.

Mathematically:
$ \tilde{x} = M \odot x_i + (1 - M) \odot x_j $
$ \tilde{y} = \lambda y_i + (1 - \lambda) y_j $

where $ M $ is a binary mask indicating the cut region and $ \lambda $ is proportional to the area of the patches.

Example usage:
```python
import torchvision.transforms.v2 as transforms

cutmix_transform = transforms.CutMix(
    num_classes=10, 
    alpha=1.0
)
```

## Where to Use

### After the DataLoader

Applying CutMix and MixUp after the DataLoader is straightforward but doesn't utilize the DataLoader's multi-processing capabilities.

### As Part of the Collation Function

Incorporating these augmentations in the collation function takes advantage of the DataLoader's multi-processing. Here’s how you can do it:

```python
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
```

### References

- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn)
- [PyTorch CutMix Transform](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.CutMix.html)
- [Example Implementation](https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html)