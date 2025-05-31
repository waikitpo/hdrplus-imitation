import cv2
import numpy as np


def build_pyramid(image, levels):
    """
    Build a Gaussian pyramid of the input image.

    Args:
        image: Input grayscale image.
        levels: Number of pyramid levels.

    Returns:
        A list of images, from coarsest to finest (reverse order).
    """
    pyr = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyr.append(image)
    return pyr[::-1]  # reverse order: coarse → fine


def compute_tile_ssd(ref_tile, search_region):
    """
    Compute the minimum SSD (Sum of Squared Differences) displacement for a given tile
    within a search region, and refine it using a quadratic (parabolic) fit.

    Args:
        ref_tile: Reference tile (template), size (tile_size, tile_size).
        search_region: Larger region to search within.

    Returns:
        Sub-pixel refined displacement (dx, dy) relative to top-left of search_region.
    """
    # Compute the SSD map using template matching
    res = cv2.matchTemplate(search_region, ref_tile, cv2.TM_SQDIFF)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    iy, ix = min_loc[1], min_loc[0]

    # Parabolic sub-pixel refinement using 3x3 quadratic fitting
    if 1 <= iy < res.shape[0] - 1 and 1 <= ix < res.shape[1] - 1:
        window = res[iy - 1:iy + 2, ix - 1:ix + 2].astype(np.float32)
        coords = np.array([[-1, -1], [0, -1], [1, -1],
                           [-1, 0], [0, 0], [1, 0],
                           [-1, 1], [0, 1], [1, 1]], np.float32)
        A = np.column_stack([
            coords[:, 0] ** 2, coords[:, 0] * coords[:, 1],
            coords[:, 1] ** 2, coords[:, 0], coords[:, 1], np.ones(9)
        ])
        b = window.flatten()
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b2, c, d, e, _ = params
        H = np.array([[2 * a, b2], [b2, 2 * c]], np.float32)
        g = np.array([d, e], np.float32)
        if np.linalg.det(H) != 0:
            sub = -np.linalg.inv(H).dot(g)
            if abs(sub[0]) <= 1 and abs(sub[1]) <= 1:
                return ix + sub[0], iy + sub[1]
    return ix, iy


def align_layer(ref, alt, prev_flow, tile_size, search_radius, min_search, max_search):
    """
    Perform tile-based alignment for a single resolution layer.

    Args:
        ref: Reference grayscale image.
        alt: Alternate grayscale image to align to reference.
        prev_flow: Previous flow field (from coarser layer).
        tile_size: Size of the square tile.
        search_radius: Radius of search window around initial guess.
        min_search, max_search: Constraints for motion range.

    Returns:
        Updated flow field for this layer.
    """
    h, w = ref.shape
    tile_h, tile_w = prev_flow.shape[:2]
    flow = np.zeros_like(prev_flow)

    # Iterate over tiles with half-tile overlap
    for y in range(0, h, tile_size // 2):
        for x in range(0, w, tile_size // 2):
            iy, ix = y // (tile_size // 2), x // (tile_size // 2)
            iy = min(iy, tile_h - 1)
            ix = min(ix, tile_w - 1)
            prev_dx, prev_dy = prev_flow[iy, ix]

            # Clamp previous displacement to min/max search range
            prev_dx = np.clip(prev_dx, min_search[0], max_search[0])
            prev_dy = np.clip(prev_dy, min_search[1], max_search[1])

            # Extract the reference tile
            y0, x0 = y, x
            ref_tile = ref[y0:y0 + tile_size, x0:x0 + tile_size]
            if ref_tile.shape[0] != tile_size or ref_tile.shape[1] != tile_size:
                continue  # skip incomplete tiles on boundaries

            # Extract search region in alternate image
            ys, ye = y0 + int(prev_dy) - search_radius, y0 + int(prev_dy) + search_radius + tile_size
            xs, xe = x0 + int(prev_dx) - search_radius, x0 + int(prev_dx) + search_radius + tile_size
            ys, ye = max(0, ys), min(h, ye)
            xs, xe = max(0, xs), min(w, xe)
            alt_region = alt[ys:ye, xs:xe]
            if alt_region.shape[0] < tile_size or alt_region.shape[1] < tile_size:
                continue

            # Compute refined displacement using SSD matching
            dx, dy = compute_tile_ssd(ref_tile, alt_region)
            flow[iy, ix, 0] = prev_dx + dx - search_radius
            flow[iy, ix, 1] = prev_dy + dy - search_radius
    return flow


def align_multi_level(ref, alt, levels=3, tile_size=16, search_radius=4):
    """
    Multi-level (coarse-to-fine) alignment using Gaussian pyramid.

    Args:
        ref: Reference grayscale image.
        alt: Alternate grayscale image to align.
        levels: Number of pyramid levels.
        tile_size: Tile size for matching.
        search_radius: Search radius for each level.

    Returns:
        Final flow field for the finest layer.
    """
    min_search = (-4, -4)
    max_search = (3, 3)

    # Build Gaussian pyramids (coarse to fine)
    ref_pyr = build_pyramid(ref, levels)
    alt_pyr = build_pyramid(alt, levels)

    # Initialize flow at coarsest level
    flow = np.zeros((ref_pyr[0].shape[0] // (tile_size // 2), ref_pyr[0].shape[1] // (tile_size // 2), 2),
                    dtype=np.float32)

    # Align from coarsest to finest
    for lvl in range(levels):
        flow = align_layer(ref_pyr[lvl], alt_pyr[lvl], flow, tile_size, search_radius, min_search, max_search)
        if lvl < levels - 1:
            # Upsample flow field to next finer layer
            new_shape = (ref_pyr[lvl + 1].shape[1] // (tile_size // 2), ref_pyr[lvl + 1].shape[0] // (tile_size // 2))
            flow = cv2.resize(flow, new_shape, interpolation=cv2.INTER_LINEAR) * 2.0
    return flow


def remap_flow_color(color_image, flow, tile_size):
    """
    Warp the input color image using the computed flow field.

    Args:
        color_image: Input color image (H, W, 3).
        flow: Optical flow field (H/tile, W/tile, 2).
        tile_size: Tile size used for alignment.

    Returns:
        Aligned color image.
    """
    h, w = color_image.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    # Upsample the flow field to full resolution
    flow_upsampled = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply remapping (displacement)
    mx = (gx + flow_upsampled[..., 0]).astype(np.float32)
    my = (gy + flow_upsampled[..., 1]).astype(np.float32)
    aligned = cv2.remap(color_image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned
