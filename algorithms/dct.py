import cv2
import numpy as np

def compress_tile_dct(tile, quality=60):
    
    # Compresses a grayscale tile using JPEG-style (DCT-based) compression.
    # Args: tile: np.ndarray, shape (H, W), quality: int, 1-100
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', tile, encode_param)
    if not result or encimg is None:
        return tile
    tile_dec = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

    # Defensive: check that tile_dec loaded successfully before using shape
    if tile_dec is None:
        return tile
    
    # Resize only if shapes don't match
    if tile_dec.shape != tile.shape:
        # cv2.resize wants (width, height)
        h, w = tile.shape
        tile_dec = cv2.resize(tile_dec, (w, h), interpolation=cv2.INTER_LINEAR)
    return tile_dec