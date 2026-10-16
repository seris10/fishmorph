"""Observed-image color, anatomically registered with explicit missing data.

These are descriptive sRGB/D65 observations, not spectra or fish-vision models.
No white balancing or inferred hidden pigment is silently applied.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.color import rgb2lab, rgb2hsv, deltaE_ciede2000
from skimage.measure import label, regionprops
from .geometry import (layers, polygon_mask, visible_points, Frame, body_profile,
                       geometry_reasons, interpolate_supported)
from .models import Annotation, AnalysisConfig
from .util import clean


def linearize_srgb(rgb):
    rgb = np.asarray(rgb, float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def summarize_pixels(rgb, mask):
    pix = rgb[mask]
    if len(pix) == 0:
        return {"n_source_pixels": 0, "mean_lab": None, "median_lab": None}
    lab = rgb2lab(pix.reshape(-1, 1, 3)).reshape(-1, 3)
    hsv = rgb2hsv(pix.reshape(-1, 1, 3)).reshape(-1, 3)
    return clean({
        "n_source_pixels": len(pix), "mean_rgb": pix.mean(axis=0), "median_rgb": np.median(pix, axis=0),
        "mean_linear_rgb": linearize_srgb(pix).mean(axis=0),
        "mean_lab": lab.mean(axis=0), "median_lab": np.median(lab, axis=0),
        "sd_lab": lab.std(axis=0, ddof=1) if len(lab) > 1 else None,
        "quantiles_lab": np.quantile(lab, [.05, .25, .5, .75, .95], axis=0),
        "median_saturation_value": np.median(hsv[:, 1:], axis=0),
        "hue_circular_mean": float((np.angle(np.mean(np.exp(2j * np.pi * hsv[:, 0]))) / (2 * np.pi)) % 1),
        "hue_note": "Exploratory, unreliable at low saturation; not a linear average of hue angles.",
    })


def _sample_grid(rgb, annotation, config, profile, frame, valid_source):
    aw, ah = config.atlas_width, config.atlas_height
    u = (np.arange(aw) + .5) / aw
    v = (np.arange(ah) + .5) / ah
    top = interpolate_supported(u, profile["u"], profile["dorsal"])
    bottom = interpolate_supported(u, profile["u"], profile["ventral"])
    uu, vv = np.meshgrid(u, v)
    yy = top[None, :] + vv * (bottom - top)[None, :]
    xy = frame.inverse(np.stack([uu, yy], axis=-1))
    finite = np.all(np.isfinite(xy), axis=2)
    ix = np.rint(np.nan_to_num(xy[..., 0], nan=-1)).astype(int)
    iy = np.rint(np.nan_to_num(xy[..., 1], nan=-1)).astype(int)
    ok = finite & (ix >= 0) & (ix < rgb.shape[1]) & (iy >= 0) & (iy < rgb.shape[0])
    ix_safe, iy_safe = np.clip(ix, 0, rgb.shape[1] - 1), np.clip(iy, 0, rgb.shape[0] - 1)
    ok &= valid_source[iy_safe, ix_safe]
    atlas_rgb = rgb[iy_safe, ix_safe].copy()
    atlas_rgb[~ok] = np.nan
    # Record exact sampled source pixel indices, not the nominal interpolated location.
    source_xy = np.stack([ix, iy], axis=-1)
    source_xy[~ok] = -1
    return {"u": u, "v": v, "rgb": atlas_rgb, "lab": rgb2lab(np.nan_to_num(atlas_rgb)),
            "valid": ok, "source_xy": source_xy,
            "projected_pixel_area_per_cell": ((bottom - top) * frame.length**2 / (aw * ah))}


def candidate_markings(rgb, valid, frame, config, profile=None):
    """Local Lab-lightness contrast candidates, not a trained pigmentation classifier.

    Components touching missing pixels are retained as truncated observations but
    excluded from counts. Features are detected in source pixels, never on a
    stretched atlas, so area/shape are not registration artifacts.
    """
    lab = rgb2lab(rgb)
    sigma = max(1.0, config.spot_background_sigma_sl * frame.length)
    weights = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    background = gaussian_filter(lab[..., 0] * valid, sigma, mode="nearest") / np.maximum(weights, 1e-9)
    residual = background - lab[..., 0]
    min_area = max(3, int(np.ceil(config.spot_min_area_sl2 * frame.length**2)))
    max_area = config.spot_max_area_sl2 * frame.length**2
    edge = binary_dilation(~valid, iterations=1)
    items = []
    marking_mask = np.zeros(valid.shape, bool)
    densities = []
    for kind, detected in (("dark", residual >= config.dark_spot_contrast),
                           ("light", residual <= -config.light_spot_contrast)):
        components = label(detected & valid, connectivity=2)
        for region in regionprops(components):
            if not min_area <= region.area <= max_area:
                continue
            coords = region.coords
            xy = coords[:, ::-1].astype(float)
            norm = frame.normalize(xy)
            uv_center = norm.mean(axis=0)
            truncated = bool(edge[coords[:, 0], coords[:, 1]].any())
            centered = norm - norm.mean(axis=0)
            cov = centered.T @ centered / max(1, len(centered) - 1)
            vals, vecs = np.linalg.eigh(cov)
            main = vecs[:, np.argmax(vals)]
            orientation = float(np.degrees(np.arctan2(main[1], main[0])))
            v_atlas = None
            if profile is not None:
                x = np.array([uv_center[0]])
                t = interpolate_supported(x, profile["u"], profile["dorsal"])[0]
                b = interpolate_supported(x, profile["u"], profile["ventral"])[0]
                if np.isfinite(t + b) and b > t:
                    v_atlas = float((uv_center[1] - t) / (b - t))
            items.append({"kind": kind, "source_centroid_xy": xy.mean(axis=0), "u": uv_center[0],
                          "y_sl": uv_center[1], "v": v_atlas, "area_px": int(region.area),
                          "area_sl2": float(region.area / frame.length**2),
                          "eccentricity": float(region.eccentricity), "orientation_degrees_ventral_positive": orientation,
                          "mean_lab": lab[coords[:, 0], coords[:, 1]].mean(axis=0),
                          "mean_local_contrast_L": float(np.abs(residual[coords[:, 0], coords[:, 1]]).mean()),
                          "truncated": truncated, "detection_confidence": None,
                          "status": "heuristic_candidate_requires_review"})
            marking_mask[coords[:, 0], coords[:, 1]] = True
    yy, xx = np.nonzero(valid)
    longitudinal = frame.normalize(np.column_stack([xx, yy]))[:, 0]
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        area = float(((longitudinal >= lo) & (longitudinal < hi)).sum() / frame.length**2)
        row = {"u_interval": [float(lo), float(hi)], "observed_area_sl2": area}
        for kind in ("dark", "light"):
            count = sum(not item["truncated"] and item["kind"] == kind and lo <= item["u"] < hi for item in items)
            row[kind + "_count"] = count
            row[kind + "_density_per_sl2"] = count / area if area > 0 else None
        densities.append(row)
    return clean(items), marking_mask, densities


def color_analysis(rgb_uint8, annotation: Annotation, config: AnalysisConfig | None = None):
    cfg = config or AnalysisConfig()
    rgb = np.asarray(rgb_uint8, float) / 255.0
    mm = layers(annotation, rgb.shape[:2])
    clipped = np.any(rgb_uint8 == 255, axis=2) | np.all(rgb_uint8 == 0, axis=2)
    visible = mm["visible"] & ~mm["occlusion"] & ~mm["uncertain"] & ~mm["artifact"]
    valid = visible & (~clipped if cfg.mask_clipped_colors else True)
    base = {"status": "unavailable", "regions": {}, "atlas": None, "markings": [], "density_bins": [],
            "appearance_only": True, "color_space": "display sRGB to CIELAB D65/2 degrees",
            "reviewed": annotation.quality.color_reviewed,
            "visible_source_pixels": int(visible.sum()), "usable_source_pixels": int(valid.sum()),
            "clipped_visible_fraction": float((clipped & visible).sum() / max(1, visible.sum())),
            "limits": ["Apparent camera color, not absolute biological reflectance.",
                       "Clipping is a diagnostic, not proof of specular reflection; manual artifact masks remain necessary.",
                       "Lateral upper/lower margins are not observed dorsal/ventral 3D surfaces."]}
    if not annotation.quality.color_reviewed:
        base["reason"] = "color_not_reviewed"
        return base
    if valid.sum() == 0:
        base["reason"] = "no_usable_visible_fish_pixels"
        return base
    base["status"] = "observed_image_color"
    base["regions"]["whole_visible_fish"] = summarize_pixels(rgb, valid)
    for name, polys in annotation.regions.items():
        base["regions"][name] = summarize_pixels(rgb, polygon_mask(polys, rgb.shape[:2]) & valid)
    reasons = geometry_reasons(annotation)
    if reasons:
        base["atlas_reason"] = "; ".join(reasons)
        return clean(base)
    try:
        profile, frame = body_profile(annotation, rgb.shape[:2], cfg.profile_samples, cfg.boundary_guard_px)
    except ValueError as exc:
        base["atlas_reason"] = str(exc)
        return clean(base)
    # Fin pixels are excluded from the body atlas/pattern analysis; user regions can still quantify them.
    body_valid = valid & mm["body"]
    atlas = _sample_grid(rgb, annotation, cfg, profile, frame, body_valid)
    atlas["lab"][~atlas["valid"]] = np.nan
    yy, xx = np.nonzero(body_valid)
    if len(xx):
        norm = frame.normalize(np.column_stack([xx, yy]))
        top = interpolate_supported(norm[:, 0], profile["u"], profile["dorsal"])
        bottom = interpolate_supported(norm[:, 0], profile["u"], profile["ventral"])
        vv = (norm[:, 1] - top) / (bottom - top)
        for name, lo, hi in (("upper_lateral", 0, 1/3), ("mid_lateral", 1/3, 2/3), ("lower_lateral", 2/3, 1)):
            m = np.zeros(valid.shape, bool)
            keep = (vv >= lo) & (vv < hi)
            m[yy[keep], xx[keep]] = True
            base["regions"][name] = summarize_pixels(rgb, m)
    marks, markmask, densities = candidate_markings(rgb, body_valid, frame, cfg, profile)
    source = atlas["source_xy"]
    idx_x = np.clip(source[..., 0], 0, rgb.shape[1] - 1)
    idx_y = np.clip(source[..., 1], 0, rgb.shape[0] - 1)
    atlas["candidate_marking"] = markmask[idx_y, idx_x] & atlas["valid"]
    # Keep a non-pattern field for descriptive aggregation. No filling of removed markings.
    field = atlas["lab"].copy()
    field[atlas["candidate_marking"]] = np.nan
    atlas["nonmarking_lab"] = field
    atlas["coverage"] = float(atlas["valid"].mean())
    atlas["coordinate_definition"] = "u=snout-to-caudal-base fraction; v=0 dorsal body edge, v=1 ventral body edge; lateral body only"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        along = np.nanmedian(field, axis=0)
        across = np.nanmedian(field, axis=1)
    # Differences remain null next to missing samples, not zero-gradient filler.
    base["gradients"] = {"longitudinal_median_lab": along, "dorsoventral_median_lab": across,
                         "longitudinal_dlab_du": np.diff(along, axis=0) * cfg.atlas_width,
                         "dorsoventral_dlab_dv": np.diff(across, axis=0) * cfg.atlas_height,
                         "note": "Exploratory registered-image gradients after heuristic marking exclusion; no lighting correction."}
    base.update(atlas=atlas, markings=marks, density_bins=densities)
    return clean(base)


def regional_delta_e(reference, candidate):
    """Descriptive regional CIEDE2000 only, with caller-enforced photometric compatibility."""
    result = {}
    for name in sorted(set(reference) & set(candidate)):
        a, b = reference[name].get("mean_lab"), candidate[name].get("mean_lab")
        if a is not None and b is not None:
            result[name] = float(deltaE_ciede2000(np.asarray(a), np.asarray(b)))
    return result
