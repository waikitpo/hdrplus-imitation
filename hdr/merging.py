import cv2
import numpy as np


def merge_hdrplus_color(ref: np.ndarray, aligned_list: list, lambdaS=0.5, lambdaR=20.0, tile_size=16,
                        motion_thresh=0.05) -> np.ndarray:
    """
    Multi-frame HDR+ color image merging with motion mask-based ghosting suppression.

    Args:
        ref: Reference RGB image, shape (H, W, 3), float32 in [0, 1].
        aligned_list: List of aligned RGB images, each shape (H, W, 3).
        lambdaS: Spatial weight kernel scale factor (smaller → sharper kernel).
        lambdaR: Temporal weight parameter for intensity difference.
        tile_size: Size of square tile for spatial kernel and merging.
        motion_thresh: Motion mask threshold (larger → more tolerant of motion).

    Returns:
        Merged HDR+ RGB image, shape (H, W, 3), float32 in [0, 1].
    """
    h, w, _ = ref.shape
    merged = np.zeros_like(ref, dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    # Combine reference frame with aligned alternate frames
    frames = [ref] + aligned_list

    # Create spatial Gaussian kernel (tile_size x tile_size)
    gauss = cv2.getGaussianKernel(tile_size, tile_size / lambdaS)
    spatial_kernel = gauss @ gauss.T  # 2D spatial weight kernel

    # Compute per-frame motion mask (based on green channel)
    ref_gray = ref[:, :, 1]  # Use green channel for luminance stability
    aligned_gray_stack = np.stack([f[:, :, 1] for f in frames], axis=0)  # Shape: (N, H, W)
    median = np.median(aligned_gray_stack, axis=0)

    # Motion mask: if deviation from median is larger than threshold → dynamic → mask out
    motion_mask_stack = (np.abs(aligned_gray_stack - median) < motion_thresh).astype(np.float32)

    # Loop over tiles
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, x1 = min(y + tile_size, h), min(x + tile_size, w)
            ref_tile = ref[y:y1, x:x1, :]

            for i, f in enumerate(frames):
                f_tile = f[y:y1, x:x1, :]
                # Temporal weight based on pixel-wise intensity difference (green channel)
                diff2 = (f_tile[:, :, 1] - ref_tile[:, :, 1]) ** 2
                temporal_w = np.exp(-diff2 / (2 * lambdaR ** 2))

                # Spatial Gaussian weight (tile window)
                spatial_w = spatial_kernel[:y1 - y, :x1 - x]

                # Motion mask in this tile
                motion_mask_tile = motion_mask_stack[i, y:y1, x:x1]

                # Final per-frame, per-pixel weight
                w_tile = temporal_w * spatial_w * motion_mask_tile

                # Weighted summation for each color channel
                for ch in range(3):
                    merged[y:y1, x:x1, ch] += f_tile[:, :, ch] * w_tile

                # Accumulate total weights for normalization
                weight_sum[y:y1, x:x1] += w_tile

    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-6)

    # Normalize the final result by total weights
    for ch in range(3):
        merged[:, :, ch] /= weight_sum

    return merged
