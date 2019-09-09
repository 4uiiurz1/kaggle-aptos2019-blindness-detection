# Experiments
## Preprocessing
- resnet34
- cross entropy
- Adam
- 5epochs
- no scheduler
- no augmentation

| scale | norm | pad | remove | val loss | val score |
|:-----:|:----:|:---:|:------:|:--------:|:---------:|
|       |      |     |        | 0.5877   | 0.8276    |
| o     |      |     |        | 0.5519   | 0.8561    |
|       | o    |     |        | 0.6209   | 0.8874    |
| o     |      | o   |        | 0.5546   | 0.8692    |
| o     | o    |     |        |**0.5348**| 0.8616    |
| o     | o    | o   |        | 0.5453   | 0.8408    |
| o     | o    |     | o      | 0.5479   | 0.8508    |

## Augmentation
- resnet34
- cross entropy
- Adam
- 10epochs
- no scheduler
- preprocessing: scale
- rotate (0, 360)
- rescale (0.9, 1.1)
- shear (-36, 36)
- translate (0.1, 0.1)
- flip (0.5)

| rotate | rescale | shear | translate | flip | val loss | val score |
|:------:|:-------:|:-----:|:---------:|:----:|:--------:|:---------:|
|        |         |       |           |      | 0.5519   | 0.8561    |
| o      |         |       |           |      | 0.5211   | 0.8769    |
|        | o       |       |           |      | 0.5997   | 0.8194    |
|        |         | o     |           |      | 0.5669   | 0.8490    |
|        |         |       | o         |      | 0.5742   | 0.8285    |
|        |         |       |           | o    | 0.5422   | 0.8650    |
| o      | o       |       |           |      | 0.5331   | 0.8591    |
| o      |         | o     |           |      | 0.5195   | 0.8812    |
| o      |         |       | o         |      | 0.5390   | 0.8664    |
| o      |         |       |           | o    | 0.5207   | 0.8760    |
|        | o       | o     |           |      | 0.5785   | 0.8349    |
|        | o       |       | o         |      | 0.5427   | 0.8732    |
|        | o       |       |           | o    | 0.5422   | 0.8787    |
|        |         | o     | o         |      | 0.5960   | 0.8601    |
|        |         |       | o         | o    | 0.5227   | 0.8759    |
| o      |         | o     |           | o    |**0.4966**| 0.8703    |

## Image size
- resnet34
- cross entropy
- SGD
- 30epochs
- CosineAnnealingLR (5e-3 -> 1e-4)
- rotate (0, 360)
- shear (-36, 36)
- flip (0.5)


- 512
  - 0.4429438032037077,0.8809532394548247

## Best parameters (2019/07/16)
- resnet34
- cross entropy
- SGD
- momentum: 0.9
- weight_decay: 0.0001
- 10epochs
- Batch size: 32
- CosineAnnealingLR (5e-3 -> 1e-4)
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- Input size: 224x224
- scale + norm
- classification

### Local CV
| fold | best_loss | best_score |
|:-----|:---------:|:----------:|
|1     | 0.4424    | 0.8971     |
|2     | 0.4401    | 0.9058     |
|3     | 0.4566    | 0.8894     |
|4     | 0.4781    | 0.8829     |
|5     | 0.4738    | 0.9055     |

## Local vs PublicLB
| Model           | Local  | PublicLB |
|:----------------|:------:|:--------:|
| ResNet34_072223 | 0.8994 | 0.657    |
| ResNet34_072517 | 0.8797 | 0.685    |

## Preprocessing
- resnet34
- regression
- SGD
- 30epochs
- CosineAnnealingLR(1e-3, 1e-5)
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)

| Model           | scale | remove | norm | val loss | val score | PublicLB |
|:---------------:|:-----:|:------:|:----:|:--------:|:---------:|:--------:|
| resnet34_080306 |       |        |      | 0.2777   | 0.8998    | 0.768    |
| resnet34_080214 | o     |        |      | **0.2718**   | **0.9049**    | **0.784**    |
| resnet34_080206 | o     | o      |      | 0.2917   | 0.8920    |          |
| resnet34_073012 | o     | o      | o    | 0.3016   | 0.8883    | 0.770    |

- 効果があるのはscale_radiusのみ。他はやらないほうが良い。

## Contrast
- resnet34
- regression
- SGD
- 30epochs
- CosineAnnealingLR(1e-3, 1e-5)
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)

| Model           | contrast | val loss | val score | PublicLB |
|:---------------:|:--------:|:--------:|:---------:|:--------:|
| resnet34_080214 |          | 0.2718   | 0.9049    | 0.784    |
| resnet34_080320 |(0.9, 1.1)| **0.2639**   | **0.9062**    | **0.788**    |

## Rescale
- resnet34
- regression
- SGD
- epochs: 30
- CosineAnnealingLR (1e-3, 1e-5)
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)

| Model           | rescale    | val loss | val score | PublicLB |
|:---------------:|:----------:|:--------:|:---------:|:--------:|
| resnet34_080320 |            | **0.2639**   | **0.9062**    | 0.788    |
| resnet34_080320 |(1.0, 1.125)| 0.2771   | 0.9016    | 0.789    |
| resnet34_080806 | 288 -> rescale (0.8889, 1.0) -> centercrop(256) | 0.2720 | 0.9031 | **0.793** |
| resnet34_081200 | 320 -> rescale (0.9, 1.0) -> centercrop(288) | 0.2829 | 0.8967 | 0.785 |
| resnet34_081417 | 320 -> rescale (0.8889, 1.0) -> centercrop(288)  | 0.2705 | 0.8962 | 0.781 |
| resnet34_081602 | 252 -> rescale (0.8889, 1.0) -> centercrop(224)  | 0.2810 | 0.8953 | 0.784 |

- LocalCVとPublicLBが相関していないのは、テストデータには拡大したような画像が多いがバリデーションデータにはほとんど見られないためだと推測。

## Freeze Batch Normalization
- resnet34
- regression
- SGD
- epochs: 30
- CosineAnnealingLR (1e-3, 1e-5)
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)
- 288 -> rescale (0.8889, 1.0) -> centercrop(256)

| Model           | freeze   | val loss | val score | PublicLB |
|:---------------:|:--------:|:--------:|:---------:|:--------:|
| resnet34_080806 | o        | 0.2720 | 0.9031 | **0.793** |
| resnet34_081711 | x        | 0.2652 | 0.9025 | 0.791 |

- 誤差程度。

## Multitask learning
- resnet34
- regression
- SGD
- epochs: 30
- CosineAnnealingLR (1e-3, 1e-5)
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)
- 288 -> rescale (0.8889, 1.0) -> centercrop(256)

| Model           | multitask   | val loss | val score | PublicLB |
|:---------------:|:-----------:|:--------:|:---------:|:--------:|
| resnet34_080806 | x           | 0.2720 | 0.9031 | **0.793** |
| resnet34_081820 | o           | 0.7281 | 0.9067 | 0.791 |

## Best models
### resnet34_080806
- regression
- SGD
- epochs: 30
- CosineAnnealingLR (1e-3, 1e-5)
- preprocess: scale_radius
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)
- 288 -> rescale (0.8889, 1.0) -> centercrop(256)

| Model           | val loss | val score | PublicLB  |
|:---------------:|:--------:|:---------:|:---------:|
| resnet34_080806 | 0.2720   | 0.9031    | **0.793** |

### se_resnext50_32x4d_080922
- regression
- SGD
- epochs: 30
- CosineAnnealingLR (1e-3, 1e-5)
- preprocess: scale_radius
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)
- 288 -> rescale (0.8889, 1.0) -> centercrop(256)

| Model           | val loss | val score | PublicLB  |
|:---------------:|:--------:|:---------:|:---------:|
| se_resnext50_32x4d_080922 | 0.2273   | 0.9159    | **0.811** |

### se_resnext101_32x4d_081208
- regression
- SGD
- epochs: 30
- CosineAnnealingLR (1e-3, 1e-5)
- preprocess: scale_radius
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)
- 288 -> rescale (0.8889, 1.0) -> centercrop(256)

| Model           | val loss | val score | PublicLB  |
|:---------------:|:--------:|:---------:|:---------:|
| se_resnext101_32x4d_081208 | 0.2157   | 0.9186    | **0.807** |

### se_resnext50_32x4d_082413
- regression
- RAdam
- epochs: 10
- CosineAnnealingLR (1e-3, 1e-5)
- preprocess: scale_radius
- rotate (-180, 180)
- shear (-36, 36)
- flip (0.5)
- contrast (0.9, 1.1)
- 288 -> rescale (0.8889, 1.0) -> centercrop(256)
- pretrained_model: se_resnext50_32x4d_080922
- pseudo_labels: se_resnext50_32x4d_080922
- train_dataset: aptos2019

| Model           | val loss | val score | PublicLB  |
|:---------------:|:--------:|:---------:|:---------:|
| se_resnext50_32x4d_082413 | 0.1934   | 0.9254    | **0.817** |

## EfficientNetのパラメータ
```python
params_dict = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
}
```
