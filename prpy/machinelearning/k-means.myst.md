# k-meansクラス


```py
class KMeansClustering():
    def __init__(self,k:int=3,max_epochs:int=100, n_init:int=1, delta:float=1e-15, seed:int|None=None):
        """初期化

        Parameters
        ----------
        k : int, optional
            クラスタ数, by default 3
        max_epochs : int, optional
            最大更新回数, by default 100
        n_init : int, optional
            初期インスタンス生成数, by default 1
        delta : float, optional
            許容誤差, by default 1e-15
        seed : int, optional
            疑似乱数シード, by default None
        """
        self.k = k 
        self.max_epochs = max_epochs
        self.delta = delta # Tolerance for stopping criterion. 
        self.seed = seed if isinstance(seed, int) else np.random.randint(2**20)
        self.rng = np.random.default_rng(self.seed)
        self.n_init=n_init
        
    def fit(self, X,y=None):
        num_data, num_features = X.shape
        
        # initialize centroids
        """
        self.centroids = np.zeros((self.k, num_features))
        for i in range(num_features):
            max_value = np.max(X[:,i])
            min_value = np.min(X[:,i])
            centroid = self.rng.uniform(min_value,max_value, size=(self.k,))
            self.centroids[:,i] = centroid
        """
        # initialize labels
        labels = self.rng.integers(self.k, size=(num_data))
        
        # initialize centroids
        centroids = []
        for label in range(self.k):
            current_members = X[labels == label]
            new_centroid = current_members.mean(axis=0)
            centroids.append(new_centroid)
        self.centroids = np.vstack(centroids)
        self._labels = [labels]
        # initialize labels  
        labels = self.predict(X)
        self._labels.append(labels)
        
        for epoch in range(self.max_epochs):
            delta = np.zeros(self.k)
            # update labels  
            labels = self.predict(X)
            self._labels.append(labels)
            
            # update centroids
            for label in range(self.k):
                mask = labels == label
                assert mask.sum()!=0, f"更新{epoch}回目でクラスタ{label}が消滅した. {mask.sum()}"
                
                current_members = X[mask]
                new_centroid = current_members.mean(axis=0)
                delta[label] = np.linalg.norm(self.centroids[label] - new_centroid)
                self.centroids[label] = new_centroid
                
            # breaking condition
            if self.delta >= delta.mean():
                #print(f"breaking at epoch-{epoch}")
                #labels = self.predict(X)
                #self._labels.append(labels)
                break
                
        return self
    
    def predict(self,X):
        # update labels 
        labels = np.zeros(X.shape[0], dtype=int)
        for index, x in enumerate(X):
            distance = np.linalg.norm(self.centroids - x, axis=1)
            labels[index] = np.argmin(distance)
        return labels
    
    def transform(self, X):
        embed = []
        for index, x in enumerate(X):
            distance = np.linalg.norm(self.centroids - x, axis=1)
            embed.append(distance)
        
        embed = np.concatenate(embed)
        return embed
```

```py
# ------ 訓練と推論の実行
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X = scaler.transform(X_train)

model = KMeansClustering(k=3)
model.fit(X)

print("予測したクラスタ",model.predict(X))


# ------ 学習中のクラスタの変化を可視化

from sklearn.decomposition import PCA

pca = PCA(2)
X_embed = pca.fit_transform(X)

tmp = []
for i,l in enumerate(model._labels):
    df= pd.DataFrame(X_embed, columns=["PC1", "PC2"])
    #df= pd.DataFrame(X_train, columns=iris.feature_names)
    df["cluster_index"] = l
    df["epoch"] = i
    tmp.append(df)

df = pd.concat(tmp)

#px.scatter(df, x='sepal length (cm)', y='sepal width (cm)',color="cluster_index",
px.scatter(df, x='PC1', y='PC2',color="cluster_index",
           width=700, height=600, 
           animation_frame="epoch"
          )

```