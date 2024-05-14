Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
```py
b = float(rng.normal(0,1, 1))
```
```py
# scalarが欲しい場合：
b = rng.normal(0,1, 1)[0]
# 配列のままで良い場合：
b = rng.normal(0,1, 1)
```

MovieWriter ffmpeg unavailable; using Pillow instead.  
```python
    animation = ArtistAnimation(fig2, animation_frames, interval=150)
    if save_path is not None:
        animation.save(save_path)
```

```python
    animation = ArtistAnimation(fig2, animation_frames, interval=150)
    if save_path is not None:
        animation.save(save_path, writer='PillowWriter')
```

japanize_matplotlibがpython 3.12に未対応 2024/04/23
[ここ](https://qiita.com/take_me/items/7d1a8823b99951210efa)と[ここ](https://github.com/ciffelia/matplotlib-fontja)を参考に，matplotlib-fontjaで代用．

