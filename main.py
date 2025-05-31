import os
import cv2
import numpy as np
import rawpy


from hdr.selecting import select_reference_frame
from hdr.aligning import align_multi_level, remap_flow_color
from hdr.merging import merge_hdrplus_color

from isp.awb import auto_wb_joint_statistics
from isp.ccm import apply_ccm
from isp.mapping import global_tone_adjustment

from isp.denoising import bilateral_filter_chroma_only
from isp.sharpening import multiscale_unsharp_masking
from isp.adjusting import chromatic_aberration_correction


def load_burst_rgb(folder: str):
    """
    Load a burst sequence of RGB images from a folder containing DNG files.
    Each image is demosaicked and converted to BGR float32 format for further processing.

    Args:
        folder: Path to the folder containing the burst of DNG images.

    Returns:
        burst: List of BGR float32 images, each normalized to [0,1].
        paths: List of DNG file paths, sorted.
    """
    # Get sorted list of DNG file paths
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dng')])

    burst = []
    for p in paths:
        # Read and postprocess each raw DNG file
        with rawpy.imread(p) as raw:
            # Use rawpy postprocess to get demosaicked RGB image (8 bits per channel)
            rgb = raw.postprocess(output_bps=8, no_auto_bright=True)
            # Convert to BGR (OpenCV convention) and normalize to float32 [0,1]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
            burst.append(rgb)

    return burst, paths


def read_raw(dng_path: str):
    """
    Read raw Bayer data and associated metadata (CCM) from a single DNG file.

    Args:
        dng_path: Path to the DNG file.

    Returns:
        raw_image: RAW Bayer image as float32.
        ccm: 3x3 Color Correction Matrix (CCM) for RGB color transform.
    """
    with rawpy.imread(dng_path) as raw:
        # Extract raw Bayer data (visible area only)
        raw_image = raw.raw_image_visible.astype(np.float32)

        # Get color correction matrix (CCM), 3x3 part only
        ccm = raw.color_matrix[:, :3]
        print(f"CCM:\n{ccm}")

    return raw_image, ccm


def main():
    # Locate burst folder
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'testing_dataset'))
    subfolders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subfolders:
        raise RuntimeError('No subfolders in testing_dataset')
    folder = subfolders[0]
    print(f'Processing burst folder: {folder}')

    # Load burst sequence (demosaicked RGB images)
    burst_rgb, paths = load_burst_rgb(folder)
    if len(burst_rgb) < 2:
        raise RuntimeError('Burst must contain at least 2 frames!')

    # Select reference frame based on sharpness (green channel)
    burst_gray = [frame[:, :, 1] for frame in burst_rgb]
    ref_idx = select_reference_frame(burst_gray)
    ref_rgb = burst_rgb[ref_idx]
    print(f'Reference frame index: {ref_idx}')

    # Align other frames to the reference using multi-scale alignment (Luma-based)
    aligned_list = []
    for i, frame in enumerate(burst_rgb):
        if i == ref_idx:
            continue
        flow = align_multi_level(ref_rgb[:, :, 1], frame[:, :, 1], levels=3, tile_size=16, search_radius=4)
        aligned = remap_flow_color(frame, flow, tile_size=16)
        cv2.imwrite(f'testing_output/aligned_rgb_{i}.png', (aligned*255).clip(0,255).astype(np.uint8))
        aligned_list.append(aligned)

    # Merge aligned frames (HDR+ style) with motion-aware weighted blending
    merged = merge_hdrplus_color(ref_rgb, aligned_list, lambdaS=0.5, lambdaR=20.0, tile_size=16, motion_thresh=0.3)
    cv2.imwrite('testing_output/merged_color_hdrplus.png', (merged*255).clip(0,255).astype(np.uint8))
    print("HDR+ fusion complete!")

    # Read raw metadata for further color correction
    ref_raw_image, ref_ccm = read_raw(paths[ref_idx])

    # Apply automatic white balance (joint gray-world & gray-edge estimation)
    awb_img = auto_wb_joint_statistics(merged)
    out_img = (awb_img * 255).clip(0, 255).astype(np.uint8)

    # Apply color correction matrix (CCM)
    ccm_img = apply_ccm(out_img, ref_ccm)

    # Apply global tone mapping (S-curve + sRGB gamma)
    map_img = global_tone_adjustment(ccm_img)
    cv2.imwrite('testing_output/hdr_tone_mapped.jpg', map_img)

    # ISP-like post-processing: chroma denoise → multi-scale sharpen → chromatic aberration correction
    img_denoise = bilateral_filter_chroma_only(map_img)
    img_sharpen = multiscale_unsharp_masking(img_denoise)
    img_correct = chromatic_aberration_correction(img_sharpen)

    # Save final processed image
    cv2.imwrite('testing_output/hdr_finalised.jpg', img_correct)

    # Visualize the reference frame (linear post-processed RGB)
    with rawpy.imread(paths[ref_idx]) as raw:
        ref_rgb_image = raw.postprocess()
    cv2.imwrite('testing_output/ref.jpg', cv2.cvtColor(ref_rgb_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
