import numpy as np


def apply_ccm(image, ccm):
    """
    Apply a 3x3 Color Correction Matrix (CCM) to an RGB image.

    This operation is typically used in ISP (Image Signal Processing) to correct
    color reproduction by transforming the RGB space to match a desired target space.

    Args:
        image: Input RGB image, shape (H, W, 3), usually in float32.
        ccm: Color Correction Matrix, shape (3, 3).

    Returns:
        Corrected RGB image, shape (H, W, 3), with values clipped to non-negative.
    """
    # Reshape the image to a 2D array (N, 3) for matrix multiplication
    h, w, _ = image.shape
    reshaped = image.reshape(-1, 3)

    # Apply color correction matrix (dot product)
    # Note: matrix is transposed to match row vector convention
    corrected = np.dot(reshaped, ccm.T)

    # Clip negative values to 0 (no negative RGB)
    corrected = np.clip(corrected, 0, None)

    # Reshape back to (H, W, 3)
    return corrected.reshape(h, w, 3)
