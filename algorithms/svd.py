import numpy as np

def compress_tile_svd(tile, rank):
    # Compresses a grayscale tile using SVD with the given rank.
    
    U, S, VT = np.linalg.svd(tile, full_matrices=False)
    S_reduced = np.diag(S[:rank])
    U_reduced = U[:, :rank]
    VT_reduced = VT[:rank, :]
    compressed = np.dot(U_reduced, np.dot(S_reduced, VT_reduced))
    return np.clip(compressed, 0, 255).astype(np.uint8)