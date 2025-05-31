import numpy as np


def s_curve(x, strength=5.0):
    """
    Apply a simple S-shaped curve adjustment based on tanh function.

    This enhances contrast by steepening mid-tones while compressing highlights and shadows.

    Args:
        x: Input float32 array, expected in range [0, 1].
        strength: S-curve strength. Larger → more pronounced S-shape.

    Returns:
        Output float32 array, range [0, 1].
    """
    return 0.5 + 0.5 * np.tanh(strength * (x - 0.5))


def apply_srgb_gamma(x):
    """
    Apply sRGB gamma correction to linear RGB data.

    sRGB standard approximates human visual response to luminance.

    Args:
        x: Input float32 array, expected in range [0, 1].

    Returns:
        Gamma-corrected float32 array, range [0, 1].
    """
    a = 0.055
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    (1 + a) * np.power(x, 1 / 2.4) - a)


def global_tone_adjustment(img_bgr: np.ndarray,
                           s_strength=5.0) -> np.ndarray:
    """
    Simulate a global tone adjustment effect: S-shaped contrast boost + sRGB gamma correction.

    This is commonly used in image pipelines (ISP / HDR) for global tonemapping and contrast enhancement.

    Args:
        img_bgr: Input BGR uint8 image.
        s_strength: Strength of S-curve adjustment. Larger → stronger contrast boost.

    Returns:
        BGR uint8 image after tone adjustment.
    """
    # Convert to float32 in [0, 1] range
    img_f = img_bgr.astype(np.float32) / 255.0

    # Apply S-shaped contrast curve
    img_s_curve = s_curve(img_f, strength=s_strength)

    # Apply sRGB gamma correction for display
    img_srgb = apply_srgb_gamma(img_s_curve)

    # Convert back to uint8 for display or saving
    out = (img_srgb * 255).round().clip(0, 255).astype(np.uint8)
    return out
