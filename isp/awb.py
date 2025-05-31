import cv2
import numpy as np


def apply_wb(image, wb_gains):
    """
    Apply white balance gains to an RGB image.

    Args:
        image: Input RGB image, shape (H, W, 3), float32 or uint8.
        wb_gains: List/tuple of 3 white balance gains for R, G, B channels.

    Returns:
        White-balanced RGB image, same shape and dtype as input.
    """
    gains = np.array(wb_gains).reshape(1, 1, 3)
    return image * gains


def auto_wb_joint_statistics(img: np.ndarray) -> np.ndarray:
    """
    Automatic white balance based on joint image statistics: combining
    Gray World (global average) and Gray Edge (gradient-based) estimations.

    This method computes white balance gains by balancing both the global
    color distribution and the gradient magnitude distribution across the image.

    Args:
        img: (H, W, 3) RGB image, float32, range [0,1].

    Returns:
        White-balanced RGB image, float32, range [0,1].
    """
    # -------------------------------
    # (1) Gray World Component (assumes avg RGB should be neutral gray)
    avg_rgb = np.mean(img, axis=(0, 1))  # Mean per-channel RGB
    gray_val = np.mean(avg_rgb)          # Expected neutral gray value
    gain_gray_world = gray_val / avg_rgb # Per-channel gains to make average neutral

    # -------------------------------
    # (2) Gray Edge Component (assumes avg gradient magnitude per channel should be neutral)
    def gradient_magnitude(channel):
        # Compute gradient magnitude using Sobel filters
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(dx ** 2 + dy ** 2)

    grad_R = gradient_magnitude(img[:, :, 2])
    grad_G = gradient_magnitude(img[:, :, 1])
    grad_B = gradient_magnitude(img[:, :, 0])

    # Average gradient magnitudes per channel
    avg_grad = np.array([
        np.mean(grad_B),
        np.mean(grad_G),
        np.mean(grad_R)
    ])
    gray_val_grad = np.mean(avg_grad)          # Expected neutral gradient magnitude
    gain_gray_edge = gray_val_grad / avg_grad  # Per-channel gains

    # -------------------------------
    # (3) Joint Estimation: Weighted combination of Gray World and Gray Edge
    alpha = 0.5  # Blending factor; adjust as needed (0=Gray World only, 1=Gray Edge only)
    gain = (1 - alpha) * gain_gray_world + alpha * gain_gray_edge

    print(f"AWB gains (joint statistics): {gain}")

    # -------------------------------
    # (4) Apply gains and clip to valid range
    balanced = img * gain.reshape(1, 1, 3)
    balanced = np.clip(balanced, 0, 1)

    return balanced
