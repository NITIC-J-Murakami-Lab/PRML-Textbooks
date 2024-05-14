# 自己組織化マップ (Self-Organizing Map; SOM) の実装

## SOMクラスの実装

```py
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from IPython.display import display, Markdown, HTML

class SelfOrganizingMap:
    def __init__(self, n_x:int,n_y:int,n_z:int, max_epoch=100):
        # 二次元の配列のそれぞれの要素がベクトルであるイメージ。つまり実際には三次元配列になる。
        self.weight = np.random.random([n_xaxis,n_yaxis,n_channels])
        self.max_epoch = max_epoch
        self._weights = []
    
    def fit(self, X):
        self._weights.append(self.weight.copy())
        for i in range(self.max_epoch):
            for color_vec in X:
                self._partial_fit(color_vec)
            self._weights.append(self.weight.copy())
        return self
    
    def _partial_fit(self, color_vec):
        """Self-Organizing Mapの学習可能パラメータ（weight）の更新を行う関数。
        データを一つ一つ受け取り、最も類似度の高いニューロンとその周辺（前後左右各2マス分）のパラメータを更新する。
        ただし、簡単のために近傍関数はステップ関数にしている。
        """
        # 入力データ（color_vec）と最も近い座標を特定する。
        min_index = np.argmin(((self.weight - color_vec)**2).sum(axis=2))

        # ただし、二次元座標が欲しいので変換する。
        _, n_yaxis, _ = self.weight.shape
        mini = int(min_index / n_yaxis)
        minj = int(min_index % n_yaxis)

        # 選ばれたニューロンの近傍（前後左右2マス）の重みを更新する。
        for i in range(-2,3): # -2, -1, 0, 1, 2
            for j in range(-2,3): # -2, -1, 0, 1, 2
                try:
                    self.weight[mini+i,minj+j] += alpha * (color_vec - self.weight[mini+i,minj+j])
                except:
                    pass
        return self

if __name__ == "__main__":
    som = SelfOrganizingMap(30,30,3)
    som.fit(demo_data)

    save_path = "simply_som.gif"
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    def get_frames(weights):
        imgs = []
        for w in weights:
            imgs.append([plt.imshow(w, interpolation="none")])
        plt.close()
        return imgs

    imgs = get_frames(som._weights)
    ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat_delay=1000)
    ani.save(save_path, writer='Pillow')
    display(HTML(ani.to_jshtml()))
```

## 出力結果

som.pyで作成した「自己組織化の様子」を表すgif：  
![](simply_som.gif)