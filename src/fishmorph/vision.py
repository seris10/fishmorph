"""CPU segmentation assistance. A foreground proposal is never an anatomical label."""
from __future__ import annotations
import numpy as np
import cv2


def image_indicators(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "width_px": int(rgb.shape[1]), "height_px": int(rgb.shape[0]),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "clipped_channel_fraction": float(np.any(rgb == 255, axis=2).mean()),
        "black_pixel_fraction": float(np.all(rgb == 0, axis=2).mean()),
        "note": "Whole-image diagnostics; no universal blur threshold, fish pose or occlusion classification implied.",
    }


def grabcut(rgb, rectangle, iterations=5, foreground=None, background=None):
    """Box-guided GrabCut with optional foreground/background seed pixels.

    Returned polygons approximate the external foreground contour. Interior holes
    are returned separately and must remain excluded. Review every proposal.
    """
    h, w = rgb.shape[:2]
    x, y, rw, rh = map(int, rectangle)
    if rw < 3 or rh < 3 or x < 0 or y < 0 or x + rw > w or y + rh > h:
        raise ValueError("GrabCut rectangle must fit inside the image and be at least 3×3 pixels.")
    if x == 0 and y == 0 and rw == w and rh == h:
        raise ValueError("Leave some background outside the GrabCut box.")
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD
    if foreground is not None:
        mask[np.asarray(foreground, bool)] = cv2.GC_FGD
    if background is not None:
        mask[np.asarray(background, bool)] = cv2.GC_BGD
    cv2.setRNGSeed(2026)
    b, f = np.zeros((1, 65), float), np.zeros((1, 65), float)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, (x, y, rw, rh), b, f,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise ValueError("GrabCut could not separate this image; use manual polygons.") from exc
    binary = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons, holes = [], []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 4:
                continue
            simplified = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2).tolist()
            if len(simplified) >= 3:
                (holes if hierarchy[0, i, 3] >= 0 else polygons).append(simplified)
    return {"polygons": polygons, "holes": holes, "method": "GrabCut; unreviewed foreground proposal"}
