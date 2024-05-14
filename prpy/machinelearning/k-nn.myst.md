

# k近傍法(k-nearest neighbor algorithm, k-NN）

## k-NNとは？

### 概要
k-NNとは，教師あり学習の一種です．主にクラス分類に利用されます．

●k-NNの立ち位置

- 機械学習
    - 教師あり学習
        - クラス分類
            - 二クラス分類
            - 他クラス分類
                - k-NN
        - 回帰
            - k-NN
            
今回は説明しませんが，k-NNは回帰問題にも活用できます．

さて，教師あり機械学習では
1. `fit`:パラメータの訓練ステップ
2. `predict`:訓練で得たパラメータを使って未知のデータのラベルを予測するステップ

の2ステップが主に存在しますが，k-NNではパラメータと呼べるものはありません．そのため，predictステップのみを実装すればOKです．k-NN（のpredictステップ）では以下の手順でクラスラベルを予測します．

::::{important}
k-NNのpredictステップの手順：  
1. 訓練データと訓練ラベルを保持し，テストデータ（未知データ）が与えられた時に，テストデータのそれぞれから最も近い距離にあるデータk個をピックアップします．  
2. ピックアップされたデータの持つラベルの中で，最も頻出する訓練ラベルをそれぞれのテストデータの予測ラベルとします．
::::

アルゴリズムの説明は上の二行で終わりです．

### より具体的に

下に参考になりそうな動画を貼っておきます．

%%html
<iframe width="560" height="315" src="https://www.youtube.com/embed/7HEQy4BoBiQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

