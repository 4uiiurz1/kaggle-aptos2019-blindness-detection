# kaggle-aptos2019-blindness-detection
**14th** place solution for APTOS 2019 Blindness Detection on Kaggle (https://www.kaggle.com/c/aptos2019-blindness-detection).

## Solution
### Preprocessing
I used only Ben's crop.

### Augmentation
Output image size is 256x256.
```python
train_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.RandomAffine(
        degrees=(-180, 180),
        scale=(0.8889, 1.0),
        shear=(-36, 36)),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

### 1st-level models (run on local)
- Models: SE-ResNeXt50\_32x4d, SE-ResNeXt101\_32x4d, SENet154
- Loss: MSE
- Optimizer: SGD (momentum=0.9)
- LR scheduler: CosineAnnealingLR (lr=1e-3 -&gt; 1e-5)
- 30 epochs
- Dataset: 2019 train dataset (5-folds cv) + 2015 dataset (like https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/97860#581042)

### 2nd-level models (run on [kernel](https://www.kaggle.com/uiiurz1/aptos-2019-14th-place-solution))
- Models: SE-ResNeXt50\_32x4d, SE-ResNeXt101\_32x4d (1st-level models' weights)
- Loss: MSE
- Optimizer: RAdam
- LR scheduler: CosineAnnealingLR (lr=1e-3 -&gt; 1e-5)
- 10 epochs
- Dataset: 2019 train dataset (5-folds cv) + 2019 test dataset (**public + private**,  divided into 5 and used different data each fold. )
- Pseudo labels: weighted average of 1st-level models

### Ensemble
Finally, averaged 2nd-level models' predictions.

- PublicLB: 0.826
- PrivateLB: 0.930

## Train 1st-level models
To train 1st-level models, run:

```
python train.py --arch se_resnext50_32x4d
python train.py --arch se_resnext101_32x4d --batch_size 24
python train.py --arch senet154 --batch_size 16
```

## Train 2nd-level models and ensemble
https://www.kaggle.com/uiiurz1/aptos-2019-14th-place-solution
