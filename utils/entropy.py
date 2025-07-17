import numpy as np
from scipy.stats import entropy
import cv2

def calculate_tile_entropy(tile):
    # Computes entropy of a grayscale tile using histogram.
    
    histogram, _ = np.histogram(tile, bins=256, range=(0, 255))
    histogram = histogram / np.sum(histogram)  # Normalize to get probabilities
    return entropy(histogram)

def compute_entropy_map(image, tile_size=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    entropy_map = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = gray[y:y+tile_size, x:x+tile_size]
            e = int(calculate_tile_entropy(tile) * 255 / np.log(256))
            entropy_map[y:y+tile_size, x:x+tile_size] = e
    return entropy_map

def generate_rank_map(image, tile_size, min_rank=5, max_rank=50):
    
    # Generates a 2D rank map for an image based on entropy of each tile.
   
    h, w = image.shape
    tiles_x, tiles_y = h // tile_size, w // tile_size
    rank_map = []

    for i in range(tiles_x):
        row = []
        for j in range(tiles_y):
            y1, y2 = i * tile_size, (i + 1) * tile_size
            x1, x2 = j * tile_size, (j + 1) * tile_size
            tile = image[y1:y2, x1:x2]
            tile_entropy = calculate_tile_entropy(tile)
            normalized_entropy = tile_entropy / np.log(256)  # Normalize between 0 and 1
            rank = int(min_rank + (max_rank - min_rank) * normalized_entropy)
            row.append(rank)
        rank_map.append(row)

    return rank_map