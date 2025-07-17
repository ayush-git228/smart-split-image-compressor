import cv2
import numpy as np
from algorithms.svd import compress_tile_svd
from algorithms.dct import compress_tile_dct
from config import TILE_SIZE

def compress_image_cli(image, regions=None, algorithm='SVD', high_rank=25, low_rank=5):
    # Compress image using SVD or DCT on tile basis.
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    compressed = np.zeros_like(gray)
    tiles_x, tiles_y = h // TILE_SIZE, w // TILE_SIZE

    # If no regions provided, compress the entire image at low rank
    for i in range(tiles_x):
        for j in range(tiles_y):
            y1, y2 = i * TILE_SIZE, (i + 1) * TILE_SIZE
            x1, x2 = j * TILE_SIZE, (j + 1) * TILE_SIZE
            tile = gray[y1:y2, x1:x2]
            if regions and any(
                x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1
                for (rx1, ry1, rx2, ry2) in regions
            ):
                value = high_rank
            else:
                value = low_rank
            if algorithm.upper() == 'SVD':
                compressed_tile = compress_tile_svd(tile, value)
            elif algorithm.upper() == 'DCT':
                compressed_tile = compress_tile_dct(tile, value)
            else:
                raise ValueError('Unknown algorithm')
            compressed[y1:y2, x1:x2] = compressed_tile

    return compressed

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python compression.py input.jpg output.jpg [SVD|DCT]")
        sys.exit(1)
    input_file, output_file = sys.argv[1:3]
    algo = sys.argv[3].upper() if len(sys.argv) > 3 else 'SVD'
    image = cv2.imread(input_file)
    out = compress_image_cli(image, algorithm=algo)
    cv2.imwrite(output_file, out)
    print(f"Compressed image saved to {output_file}")