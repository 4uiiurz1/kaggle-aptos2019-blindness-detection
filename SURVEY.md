# 過去コンペ調査

## Diabetic Retinopathy Detection
- 似たようなコンペ。4年前のコンペなので少し古い。
- 問題は全く同じで糖尿病性網膜症を眼底画像から早期発見するというもの。
- 重症度について使っているスケールも0~4の同じもの。

### 1st Solution
#### Cohen's Kappa
- コンペの評価関数はCohen's Kappa。重み付きカッパ係数。
- 回帰で予測。最終的には浮動小数点数から整数にする必要がある。
- 重症度の高いサンプルに対するペナルティが大きいため、単純にroundするよりも適切なしきい値を設定するべき。

#### Image preprocessing
- OpenCVを使って前処理。
  1. 同じ半径 (300px or 500px) になるように画像をリスケール。画像の中央にあたるベクトルを抜き出して、平均値以上の値を持つピクセルの数をカウントすることで半径を求める。
  2. 局所領域の平均値が127.5になるように、画像から局所領域の平均値を減じた。
  3. "boundary effects"を除くために、眼の領域の面積がオリジナルの円の90%になるように境界付近の領域を削除。
- 画像間の照明条件やカメラの解像度による差をなくす意図でやった。
- pdfにpythonコードあり。

#### Data augmentation
- OpenCVのwarpAffine関数を用いる。
  1. リスケール (±10%)
  2. 回転 (0°~360°)
  3. スキュー (せん断) (±0.2)
- TTAは回転のみ。
- 色を変化させるようなData augmentationは効果が無かった。

#### Network configurations
- fractional max-pooling (?) を用いた3つのモデルを採用。
- 5-class softmaxで0~4のクラスを推定。
- ネットワークの出力のprobabilityをランダムフォレストの入力として用いる。

#### Predictions: From softmax to integer scores
- 複数ネットワーク + TTAから得られたprobabilityを平均。
- より大きな特徴ベクトルを作るために以下のメタデータを追加。
  - 左右の眼のprobability。
  - オリジナル画像のサイズ。
  - オリジナル画像の分散。
  - 前処理後の画像の分散。
- ランダムフォレストで回帰？。しきい値処理。{0.57, 1.37, 2.30, 3.12}
- ランダムフォレストはオーバーキルで、メタデータ追加なしで線形回帰をかけるだけでもうまくいくと思う。

### 2nd Solution
#### Preprocessing
- 眼が収まる最小の正方形領域を選択。
- (512, 256, 128ピクセル)にリサイズ。
- 正規化 (平均0, 分散1)

#### Augmentation
- 360度の回転、並進、スケーリング、ストレッチング。
- Krizhevsky color augmentation (gaussian)

#### Network Configurations
```
net A       |          net B
units   filter stride size  |  filter stride size
1 Input                           448  |     4           448
2 Conv        32     5       2    224  |     4     2     224
3 Conv        32     3            224  |     4           225
4 MaxPool            3       2    111  |     3     2     112
5 Conv        64     5       2     56  |     4     2      56
6 Conv        64     3             56  |     4            57
7 Conv        64     3             56  |     4            56
8 MaxPool            3       2     27  |     3     2      27
9 Conv       128     3             27  |     4            28
10 Conv       128     3             27  |     4            27
11 Conv       128     3             27  |     4            28
12 MaxPool            3       2     13  |     3     2      13
13 Conv       256     3             13  |     4            14
14 Conv       256     3             13  |     4            13
15 Conv       256     3             13  |     4            14
16 MaxPool            3       2      6  |     3     2       6
17 Conv       512     3              6  |     4             5
18 Conv       512     3              6  |   n/a           n/a
19 RMSPool            3       3      2  |     3     2       2
20 Dropout
21 Dense     1024
22 Maxout    512
23 Dropout
24 Dense     1024
25 Maxout    512
```
- LeakyReLU (0.01)
- weight_decay 5e-4
- MSE loss

#### Training
- 128 px images -> layers 1 - 11 and 20 to 25.
- 256 px images -> layers 1 - 15 and 20 to 25. Weights of layer 1 - 11 initialized
with weights from above.
- 512 px images -> all layers. Weights of layers 1 - 15 initialized with
weights from above.
- 10%をバリデーションデータとして用いる。
- すべてのクラスのサンプル比が均等になるようにサンプリング。
- 徐々に割合を変えていって最終的にclass 0の比率が1に対して他のすべてのクラスが2になるように調整。
- nesterov momentum (0.9)
  - epoch 0: 0.003
  - epoch 150: 0.0003
  - epoch 220: 0.00003

#### "Per Patient" Blend
- ベストモデルのRMSPoolレイヤの出力の平均と標準偏差を抽出 (TTAあり)。
- それぞれの眼について以下の特徴量を追加。
```
[this_eye_mean, other_eye_mean, this_eye_stddev, other_eye_stddev, left_eye_indicator]
```
- これらの特徴量を正規化したものを用いてさらに以下のネットワークを学習。
```
Input        8193
Dense          32
Maxout         16
Dense          32
Maxout         16
```
- 最初のレイヤにはL1正則化 (2e-5)
- weight_decay (5e-3)
- Adam Updates with fixed learning rate schedule over 100 epochs.
  - epoch 0: 5e-4
  - epoch 60: 5e-5
  - epoch 80: 5e-6
  - epoch 90: 5e-7
- MSE loss
- 基本的なバッチサイズは128で、確率で以下のバッチサイズに置き換える。
  - 0.2: クラス間のサンプル数が均等になるようなサンプリング。
  - 0.5: ランダムにサンプリング。
- 回帰によって得られる連続値に対してしきい値処理。[0.5, 1.5, 2.5, 3.5]

#### Notes
- カッパ係数が最大になるようにしきい値を最適化するとprivate LBはわずかに改善した。
- しかし、Public LBは改善しなかったため結局final submissionには採用しなかった。

### 3rd Solution
#### Model Details
- 384x384~1024x1024の様々な入力サイズのモデルを学習。
- ほとんどのモデルはMSE lossで学習。
- 4つのシグモイド関数を用いた学習も行い、性能はMSEとほぼ同じ。
- モデルの詳細
  - 片眼ずつ推定。
  - data augmentationはクロップ (85\~95%), horizontal flip, 回転 (0\~360°) (オフライン)
  - チャネルごとにコントラスト正規化。
  - LeakyReLU (0.1)
  - Nesterov momentum (0.9)
- 学習を安定させるためにかなり低い学習率からスタート (warm-upみたいな感じ)
- 3-5 epochは1e-4 or 1e-5。
- 0.003 for 100 epochs
- 0.001 for 30 epochs
- 0.0001 for 20 epochs.

#### Post-Processing
- 1-stage modelの両目に対する予測から線形モデルによって最終的な予測値を推定。
- 学習には1-stage modelの学習時にバリデーションデータとして用いた10%のデータを使う。
- 2-stage modelにニューラルネットワークを試してみたがうまくいかなかった。
- バリデーションデータに対するカッパ係数が最大になるようなしきい値をグリッドサーチで探索。

#### Ensemble Details
- 9つのモデルの予測値に対する上記の後処理後のint値を入力特徴量として、L1/L2正則化をかけた線形回帰モデルを学習。
- 連続値はPost-Processingと同様の方法でintに変換。

#### Discussion
- 両目の予測を統合する処理は1st and 2nd solutionほど効果的ではなかったと思う。
- 10%のhold outsetで線形モデルを学習させたがおそらくオーバーフィッティングしてる。
- やるなら学習データ全体で学習させるべきだった。
- 入力サイズを大きくするのは効果的だった。385→767で0.808から0.843に向上した。
- その一方で1024よりも大きい入力サイズのモデルは過学習してしまったため、採用しなかった。
