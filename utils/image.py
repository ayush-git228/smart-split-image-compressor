import os
import cv2
import numpy as np
from algorithms.svd import compress_tile_svd
from algorithms.dct import compress_tile_dct
from algorithms.pca import compress_tile_pca
from algorithms.kmeans import compress_tile_kmeans
from algorithms.ae import compress_tile_ae
from algorithms.wavelet import compress_tile_wavelet

# def save_image(image, path):
#     # Save an image to disk, creating directories as needed.
#     directory = os.path.dirname(path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory, exist_ok=True)
#     cv2.imwrite(path, image)

def overlay_rectangles(image, regions, color=(0, 255, 0), thickness=2):

    # Return a copy of the image with rectangles drawn for each region (x1, y1, x2, y2).
    clone = image.copy()
    for (x1, y1, x2, y2) in regions:
        cv2.rectangle(clone, (x1, y1), (x2, y2), color, thickness)
    return clone

def compress_image_by_tiles(image, rank_map, algorithm='SVD'):
    # Compress an image by applying the selected algorithm to each tile, using tile-specific parameters from rank_map.
    
    tile_size = len(rank_map[0])
    height, width = image.shape
    compressed_img = np.zeros_like(image)
    tiles_x = len(rank_map)
    tiles_y = len(rank_map[0])

    for i in range(tiles_x):
        for j in range(tiles_y):
            y, x = i * tile_size, j * tile_size
            tile = image[y:y+tile_size, x:x+tile_size]
            if algorithm == 'SVD':
                v = rank_map[i][j]
                compressed_tile = compress_tile_svd(tile, v)
            elif algorithm == 'DCT':
                v = rank_map[i][j]
                compressed_tile = compress_tile_dct(tile, v)
            elif algorithm == 'PCA':
                v = rank_map[i][j] / 100.0
                compressed_tile = compress_tile_pca(tile, retained_variance=v)
            elif algorithm == 'KMeans':
                v = rank_map[i][j]
                compressed_tile = compress_tile_kmeans(tile, n_colors=max(2, v))
            elif algorithm == 'Wavelet':
                keep = rank_map[i][j] / 100.0
                compressed_tile = compress_tile_wavelet(tile, keep_ratio=keep)
            elif algorithm == 'Autoencoder':
                compressed_tile = compress_tile_ae(tile)
            else:
                compressed_tile = tile
            compressed_img[y:y+tile_size, x:x+tile_size] = compressed_tile

    return compressed_img

def compress_image_pil(image, quality=60):

    # Compress an image using Pillow (JPEG) and return as a numpy array.
    # Lossy JPEG compression, 'quality' in [1, 95] (higher is better).
    from PIL import Image as PILImage
    import io
    pil_img = PILImage.fromarray(image)
    with io.BytesIO() as output:
        pil_img.save(output, format="JPEG", quality=quality, optimize=True)
        output.seek(0)
        decoded = PILImage.open(output)
        return np.array(decoded)

def load_image_gray(input_image):
    # Load an image as grayscale.
    # Accepts: filepath or BGR numpy array.
    # Returns: numpy array (uint8, grayscale).
    
    if isinstance(input_image, np.ndarray):
        # Convert BGR color to grayscale if necessary
        if len(input_image.shape) == 2:
            return input_image
        return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    elif isinstance(input_image, (str, os.PathLike)):
        image = cv2.imread(str(input_image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image from: {input_image}")
        return image
    else:
        raise TypeError("Unsupported input type. Must be NumPy array or file path.")