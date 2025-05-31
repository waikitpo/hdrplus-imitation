import cv2
import numpy as np


def bilateral_filter_chroma_only(bgr_img: np.ndarray,
                                 d: int = 9,
                                 sigma_color: float = 75,
                                 sigma_space: float = 75) -> np.ndarray:
    """
    Perform bilateral filtering only on chroma channels (U/V) in the YUV color space.

    This preserves luminance (Y) details while denoising color information, which
    can be particularly helpful for video or photo color smoothing without losing sharpness.

    Args:
        bgr_img: Input BGR image, uint8 format.
        d: Diameter of each pixel neighborhood. Larger d → stronger smoothing.
        sigma_color: Filter sigma in the color space. Larger → more colors mixed together.
        sigma_space: Filter sigma in the coordinate space. Larger → farther pixels considered.

    Returns:
        BGR image (uint8), with chroma denoised by bilateral filtering.
    """
    # Convert the input BGR image to YUV color space.
    yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)

    # Split Y, U, V channels
    Y, U, V = cv2.split(yuv)

    # Perform bilateral filtering on U and V channels separately.
    # Note: Luminance (Y) is left untouched to preserve image sharpness.
    U_denoised = cv2.bilateralFilter(U, d, sigma_color, sigma_space)
    V_denoised = cv2.bilateralFilter(V, d, sigma_color, sigma_space)

    # Merge the original Y and denoised U/V channels back into a single YUV image.
    yuv_denoised = cv2.merge([Y, U_denoised, V_denoised])

    # Convert back to BGR color space for output.
    bgr_denoised = cv2.cvtColor(yuv_denoised, cv2.COLOR_YUV2BGR)

    return bgr_denoised
