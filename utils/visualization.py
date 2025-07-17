import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_side_by_side(original: np.ndarray, compressed: np.ndarray) -> None:
    # Display two grayscale images side by side for comparison.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(compressed, cmap='gray')
    axs[1].set_title("Compressed")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

def save_image(path: str, img: np.ndarray) -> None:
    # Save an image to disk using OpenCV.
    cv2.imwrite(path, img)

def gen_heatmap_from_rankmap(rank_map: np.ndarray, image_shape: tuple) -> np.ndarray:
    # Generate a color heatmap from the tilewise rank map, resized to match image shape.
    # Args: rank_map: 2D array of per-tile “compression rank” (higher = less compression), 
    #     image_shape: Shape of the full image (h, w[, c])

    # Returns: 3D np.ndarray (h, w, 3) in uint8 suitable for imshow or cv2.

    norm = np.array(rank_map, dtype=np.float32)
    # Use np.ptp (not norm.ptp()) to avoid type checker warnings
    norm = (norm - norm.min()) / (np.ptp(norm) + 1e-6)
    # Resize (will broadcast to proper shape)
    heatmap = cv2.resize(norm, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    colormap = plt.get_cmap('jet')
    colored = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return colored

def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:

    # Overlay a heatmap onto an image with a given transparency. If image is grayscale, converts to color.
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image
    blended = cv2.addWeighted(img_color, 1 - alpha, heatmap, alpha, 0)
    return blended