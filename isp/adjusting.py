import cv2
import numpy as np


def chromatic_aberration_correction(img: np.ndarray,
                                    edge_thresh=50) -> np.ndarray:
    """
    Simple chromatic aberration correction by replacing chroma channels at edges.

    This method replaces the a/b chroma components at detected high-contrast edges
    with their local median, effectively suppressing color fringing artifacts.

    Args:
        img: Input BGR uint8 image.
        edge_thresh: Edge detection threshold. Larger → fewer edges considered.

    Returns:
        Corrected BGR uint8 image.
    """
    # Convert to Lab color space to separate luminance (L) and chrominance (a/b)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    # Compute gradient magnitude of luminance channel using Sobel filters
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Create edge mask by thresholding gradient magnitude
    edge_mask = grad_mag > edge_thresh

    # Median-filter the chroma channels (a/b) in 3x3 neighborhood
    # This provides a local smooth reference for replacement
    a_median = cv2.medianBlur(a, 3)
    b_median = cv2.medianBlur(b, 3)

    # Replace chroma values at edge pixels with median values
    a[edge_mask] = a_median[edge_mask]
    b[edge_mask] = b_median[edge_mask]

    # Merge back the corrected Lab channels and convert to BGR
    lab_corrected = cv2.merge([L, a, b])
    bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)

    return bgr_corrected
