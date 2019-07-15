# iMet Collection 2019 - FGVC6
- 美術品のクラス分類問題。
- 美術品の属性を表す1103のラベルがあって、そのうちの**少なくとも1つ**のラベルが一つの画像に付けられている。
-

## 1st Solution
### Stage 0. The same part for all stages:
#### Models:
- 使用したモデルは以下の3つ (すべてcadeneのリポジトリのpretrained model)
  - SENet154
  - PNasNet-5
  - SE-ResNeXt101

#### CV:
- 5 fold (multilabel stratify)

#### Data Augmentations
```
HorizontalFlip(p=0.5),
OneOf([
    RandomBrightness(0.1, p=1),
    RandomContrast(0.1, p=1),
], p=0.3),
ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
IAAAdditiveGaussianNoise(p=0.3),
```

#### Data Preprocessing
- ここで述べる後処理によって収束が早くなるとともに、スコアが大幅に改善した。
- 何も考えずに画像をクロップすると、あるラベルに対応した領域（例えばpersonというラベルに対して人間が写っている領域）が画像から抜けてしまうということが起こりうる。そうするとデータがとてもnoisyになってしまう。
- そこで以下のコードのように必要最低限のクロップのみ行うようにした。
```
class RandomCropIfNeeded(RandomCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        h, w, _ = img.shape
        return F.random_crop(img, min(self.height, h), min(self.width, w), h_start, w_start)
```
- 以下のように使用。
```
RandomCropIfNeeded(SIZE * 2, SIZE * 2),
Resize(SIZE, SIZE)
```
- SEResNext101 / SENet154: SIZE = 320
- PNasNet-5: SIZE = 331

#### TTA
- HorizontalFlipのみ

#### Scheduler
- merticsとlossの下がり方をplotして最適なスケジューリングを求めた。
- lr=5e-3からスタートして数epoch経つたびに1/5にする。

### Stage 1. Training the zoo:
- Loss: Focal
- 対数重みによるサンプリング (Sampling with logarithmic weights)
- Batch size: 1000-1500 (accumulation 10-20 times)

### Stage 2. Filtering predictions:
- Stage1において、OOFに対する推定値の誤差が大きすぎるサンプルについては、very noisyかつ誤ったラベルが付いたサンプルとして除去。
- Loss: Focal
- Hard negative mining (各エポックにおいて上位5%の難しさのサンプルをサンプリング)
- スクラッチからモデルを再学習。

### Stage 3. Pseudo labeling:
- Loss: Focal
- 確信度が高いテストデータを学習データに追加。```np.mean(np.abs(probabilities - 0.5))```
- スクラッチからモデルを再学習。

#### Stage 4. Culture and tags separately:
- tagとcultureについて以下の二つの重要なことを発見。
  1. cultureよりもtagの方がnoisy
  2. いくつかのtagクラスはimagenetのクラスによく似てる。
- そこで、既に学習済みのモデルの重みを使ってtagの705クラスのみ推定するモデルを学習。
- LossをFocalからBCEに変更。(大きく寄与)

#### Stage 5. Second-level model
- 各画像が各クラスに属しているかどうかというバイナリ分類を行うLightGBMを学習。
- 以下の特徴量を用いる。
  - probabilities of each models, sum / division / multiplication of each pair / triple / .. of models
  - mean / median / std / max / min of each channel
  - brightness / colorness of each image (you can say me that NN can easily detect it — yes, but here i can do it without cropping and resizing — it is less noisy)
  - Max side size and binary flag — height more than width or no (it is a little bit better for tree boosting than just height + width in case of lower side == 300)
  - Aaaaand the secret sauce: ImageNet predictions ;) As I already mentioned — some tags classes similar to ImageNet classes, but ImageNet much bigger, pretrained models much more generalized. So, I add all 1000 (number of ImageNet classes) predictions to this dataset

#### Hints / Postprocessing:
- culture modelとtags modelで異なるしきい値を用いる。
- cultureのラベル付けは不完全なので以下のコードで予測を二値化。
```python
culture_predictions = binarize_predictions(predictions[:, :398], threshold=0.28, min_samples=0, max_samples=3)
tags_predictions = binarize_predictions(predictions[:, 398:], threshold=0.1, min_samples=1, max_samples=8)
predictions = np.hstack([culture_predictions, tags_predictions])
```
