import numpy as np
from sklearn.decomposition import PCA

def compress_tile_pca(tile, retained_variance=0.95):
    flat = tile.reshape(-1, 1)
    pca = PCA(n_components=retained_variance, svd_solver='full')
    flat_transformed = pca.fit_transform(flat)
    flat_restored = pca.inverse_transform(flat_transformed)
    restored = flat_restored.reshape(tile.shape)
    return np.clip(restored, 0, 255).astype(np.uint8)