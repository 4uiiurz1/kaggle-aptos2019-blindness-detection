# Experiments
## Preprocessing
- ResNet34
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
- ResNet34
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
- ResNet34
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
- ResNet34
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
- ResNet34
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
- ResNet34
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

## メモ
- ```
package_dir = "../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/"
sys.path.insert(0, package_dir)
```
- epochs=30が最適
