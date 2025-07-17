import numpy as np
from sklearn.cluster import MiniBatchKMeans

def compress_tile_kmeans(tile, n_colors=32):
    flat = tile.flatten().reshape(-1, 1)
    km = MiniBatchKMeans(n_clusters=n_colors, n_init=1, random_state=42)
    km.fit(flat)
    compressed_flat = km.cluster_centers_[km.labels_].flatten()
    compressed = compressed_flat.reshape(tile.shape)
    return np.clip(compressed, 0, 255).astype(np.uint8)