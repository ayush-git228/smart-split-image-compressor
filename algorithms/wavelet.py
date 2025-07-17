import pywt
import numpy as np

def compress_tile_wavelet(tile, level=1, wavelet='haar', keep_ratio=0.2):
    coeffs = pywt.wavedec2(tile, wavelet, level=level)
    # Zero out the detail coefficients for higher compression
    thresh = []
    for arr in coeffs[1:]:
        new_arr = tuple([np.where(np.abs(c) < np.max(np.abs(c))*keep_ratio, 0, c) for c in arr])
        thresh.append(new_arr)
    thresh = [coeffs[0]] + thresh
    restored = pywt.waverec2(thresh, wavelet)
    return np.clip(restored[:tile.shape[0], :tile.shape[1]], 0, 255).astype(np.uint8)