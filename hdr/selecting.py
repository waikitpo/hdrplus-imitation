import cv2
import numpy as np
from typing import List


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute a sharpness score for a single image using OpenCV Sobel gradient magnitude.

    Args:
        image (np.ndarray): Input image in HxW or HxWxC format (uint8/uint16/float32).

    Returns:
        float: Mean gradient magnitude as sharpness score.
    """
    # Convert to grayscale if needed
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = image[:, :, 1]  # use green channel
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Use mean magnitude as sharpness score
    return float(np.mean(grad_mag))


def select_reference_frame(burst: List[np.ndarray]) -> int:
    """
    Select the index of the sharpest frame in a burst using OpenCV-based sharpness.

    Args:
        burst (List[np.ndarray]): List of images as numpy arrays.

    Returns:
        int: Index of the reference frame (sharpest).
    """
    # Compute sharpness score for each frame
    scores = [compute_sharpness(frame) for frame in burst]

    # Select the index with the highest score
    ref_idx = int(np.argmax(scores))
    return ref_idx
