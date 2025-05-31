import cv2
import numpy as np


def multiscale_unsharp_masking(img: np.ndarray,
                               sigmas=[1.0, 2.0, 4.0],
                               weights=[0.5, 0.3, 0.2],
                               amount=1.0) -> np.ndarray:
    """
    Multi-scale unsharp masking using a 3-level Gaussian pyramid.
    This enhances details at multiple scales for a more natural sharpening effect.

    Args:
        img: Input BGR uint8 image.
        sigmas: List of Gaussian blur sigmas for 3 levels.
        weights: Corresponding blend weights for each blurred image.
        amount: Overall sharpening strength (scales the detail enhancement).

    Returns:
        BGR uint8 sharpened image.
    """
    # Convert to float32 in range [0, 1]
    img_f = img.astype(np.float32) / 255.0

    # Initialize accumulation buffer for blurred images
    blurred_sum = np.zeros_like(img_f)

    # Apply Gaussian blur at 3 scales and accumulate weighted sum
    for sigma, w in zip(sigmas, weights):
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        blurred_sum += w * blurred

    # Compute detail image (residual): difference from the blurred version
    detail = img_f - blurred_sum

    # Enhance details by adding back weighted residual
    sharpened = img_f + amount * detail

    # Clip to valid [0,1] range
    sharpened = np.clip(sharpened, 0, 1)

    # Convert back to uint8 for output
    return (sharpened * 255).round().astype(np.uint8)


def dog_sharpen_y_channel(bgr_img: np.ndarray,
                          sigma_small: float = 1.0,
                          sigma_large: float = 2.0,
                          strength: float = 1.0) -> np.ndarray:
    """
    Difference of Gaussians (DoG)-based sharpening applied to the luminance (Y) channel only.
    This enhances details while preserving chroma (color) consistency.

    Args:
        bgr_img: Input BGR uint8 image.
        sigma_small: Sigma for smaller Gaussian blur (captures fine details).
        sigma_large: Sigma for larger Gaussian blur (captures coarse structures).
        strength: Sharpening strength (scales the DoG residual added to luminance).

    Returns:
        BGR uint8 sharpened image with enhanced luminance.
    """
    # Convert input to YUV color space for separating luminance
    yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
    yuv_f = yuv.astype(np.float32)

    # Extract Y (luminance) channel
    Y = yuv_f[:, :, 0]

    # Apply two Gaussian blurs: fine (small) and coarse (large)
    small_blurred = cv2.GaussianBlur(Y, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small)
    large_blurred = cv2.GaussianBlur(small_blurred, (0, 0), sigmaX=sigma_large, sigmaY=sigma_large)

    # Compute Difference of Gaussians (DoG) - detail enhancement
    dog = small_blurred - large_blurred

    # Enhance Y channel by adding scaled DoG residual
    Y_sharpened = Y + strength * dog
    Y_sharpened = np.clip(Y_sharpened, 0, 255)

    # Merge back to YUV and convert to BGR
    yuv_f[:, :, 0] = Y_sharpened
    yuv_sharpened = yuv_f.astype(np.uint8)
    bgr_sharpened = cv2.cvtColor(yuv_sharpened, cv2.COLOR_YUV2BGR)

    return bgr_sharpened
