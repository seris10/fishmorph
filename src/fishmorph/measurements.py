"""Measurement eligibility and first-order, conditional annotation uncertainty."""
from __future__ import annotations
import numpy as np
from .anatomy import RATIOS, FIN_LANDMARKS
from .geometry import Frame, visible_points, geometry_reasons, body_profile, interpolate_supported
from .models import Annotation, AnalysisConfig
from .util import clean


def conditional_sd(fn, points, annotation, required):
    """Delta method with independent isotropic point error, shared endpoints retained.

    This is annotation uncertainty ONLY. No inference about lens distortion,
    true 3D anatomy, landmark bias, or unknown per-point correlations is made.
    """
    if any(annotation.landmarks[k].sigma_px is None for k in required):
        return None
    var = 0.0
    step = max(np.linalg.norm(points["caudal_base"] - points["snout"]) * 1e-5, 1e-5)
    for name in sorted(required):
        sigma = annotation.landmarks[name].sigma_px
        for axis in (0, 1):
            p1, p2 = {k: v.copy() for k, v in points.items()}, {k: v.copy() for k, v in points.items()}
            p1[name][axis] += step
            p2[name][axis] -= step
            grad = (fn(p1) - fn(p2)) / (2 * step)
            var += float(grad * grad * sigma * sigma)
    return float(np.sqrt(var))


def measure(annotation: Annotation, image_shape, config: AnalysisConfig | None = None, standard_length_mm=None):
    cfg = config or AnalysisConfig()
    points = visible_points(annotation, image_shape)
    common_reasons = geometry_reasons(annotation)
    results = {}

    def add(name, required, fn, label, unit="ratio", extra_reasons=None):
        reasons = list(common_reasons) + list(extra_reasons or [])
        missing = sorted(set(required) - points.keys())
        if missing:
            reasons.append("missing_or_unobserved:" + ",".join(missing))
        if reasons:
            results[name] = {"value": None, "unit": unit, "label": label,
                             "status": "unavailable", "reasons": reasons, "sd_annotation": None}
            return
        try:
            value = float(fn(points))
            if not np.isfinite(value):
                raise ValueError("Nonfinite measurement")
            sd = conditional_sd(fn, points, annotation, set(required))
            results[name] = {"value": value, "unit": unit, "label": label, "status": "observed",
                             "reasons": [], "sd_annotation": sd,
                             "uncertainty_method": "conditional independent-point delta method" if sd is not None else "not quantified"}
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            results[name] = {"value": None, "unit": unit, "label": label,
                             "status": "unavailable", "reasons": [str(exc)], "sd_annotation": None}

    def distance(p, ends):
        return np.linalg.norm(p[ends[1]] - p[ends[0]])

    add("standard_length_px", ["snout", "caudal_base"], lambda p: distance(p, ("snout", "caudal_base")),
        "Photographed standard-length chord", "px")
    for name, (a, b, label) in RATIOS.items():
        def ratio(p, a=a, b=b):
            den = distance(p, b)
            if den < 1e-8:
                raise ValueError("Degenerate denominator")
            return distance(p, a) / den
        pose_reasons = []
        if name in {"pectoral_length_sl", "pelvic_length_sl", "caudal_span_sl"} and annotation.quality.fins == "unknown":
            pose_reasons = ["fin_pose_unknown"]
        add(name, set(a + b + ("snout", "caudal_base")), ratio, label, extra_reasons=pose_reasons)
    for lm in FIN_LANDMARKS + ["eye_center", "tail_tip", "tail_fork"]:
        name = {"tail_tip": "total_length_photo_projected_sl", "tail_fork": "fork_length_photo_projected_sl"}.get(lm, lm + "_u")
        def projection(p, lm=lm):
            return Frame.from_points(p, require_dorsal=False).normalize(p[lm])[0]
        add(name, ["snout", "caudal_base", lm], projection, f"Axial position of {lm} / SL")
    for fin in ("dorsal", "anal"):
        names = [fin + "_origin", fin + "_insertion", fin + "_apex", "snout", "caudal_base"]
        def height(p, fin=fin):
            base = p[fin + "_insertion"] - p[fin + "_origin"]
            tip = p[fin + "_apex"] - p[fin + "_origin"]
            if np.linalg.norm(base) < 1e-8:
                raise ValueError("Fin base has zero length")
            cross = base[0] * tip[1] - base[1] * tip[0]
            return abs(cross) / (np.linalg.norm(base) * distance(p, ("snout", "caudal_base")))
        add(fin + "_height_sl", names, height, f"{fin.title()} apex height above its base chord / SL",
            extra_reasons=["fin_pose_unknown"] if annotation.quality.fins == "unknown" else [])
    profile = None
    profile_error = None
    normalized = {}
    if not common_reasons:
        try:
            frame = Frame.from_points(points)
            normalized = {name: frame.normalize(p).tolist() for name, p in points.items()}
            profile, _ = body_profile(annotation, image_shape, cfg.profile_samples, cfg.boundary_guard_px)
        except ValueError as exc:
            profile_error = str(exc)
    else:
        profile_error = "; ".join(common_reasons)
    for fraction in (.25, .50, .75):
        name = f"body_depth_{int(fraction * 100):02d}_sl"
        val = None
        if profile is not None:
            val = float(interpolate_supported(np.array([fraction]), profile["u"],
                        profile["ventral"] - profile["dorsal"])[0])
            if not np.isfinite(val):
                val = None
        results[name] = {"value": val, "unit": "ratio", "label": f"Fins-excluded body depth at {fraction:.0%} SL",
                         "status": "observed" if val is not None else "unavailable", "sd_annotation": None,
                         "reasons": [] if val is not None else [profile_error or "unsupported_or_occluded_cross_section"]}
    depths = np.array([]) if profile is None else profile["ventral"] - profile["dorsal"]
    good = np.isfinite(depths)
    # Only claim a maximum when most of the longitudinal support is present.
    complete = len(depths) > 0 and good.mean() >= .95
    results["max_sampled_body_depth_sl"] = {
        "value": float(np.max(depths[good])) if complete else None, "unit": "ratio",
        "label": "Maximum sampled fins-excluded depth / SL (not a continuous true maximum)",
        "status": "observed" if complete else "unavailable", "sd_annotation": None,
        "reasons": [] if complete else ["less_than_95_percent_body_profile_support"],
    }
    absolute = {}
    if standard_length_mm is not None:
        for name, row in results.items():
            if name.endswith("_sl") and row["value"] is not None:
                absolute[name.removesuffix("_sl") + "_mm"] = {
                    "value": row["value"] * standard_length_mm, "unit": "mm",
                    "basis": "image ratio times independently supplied SL; scale error not propagated"}
    return clean({"measurements": results, "normalized_landmarks": normalized, "body_profile": profile,
                  "body_profile_error": profile_error, "absolute_estimates": absolute,
                  "structures": annotation.structures, "structure_evidence": annotation.structure_evidence,
                  "quality": annotation.quality.model_dump(),
                  "uncertainty_limits": ["Photographic projection, boundary placement, systematic bias and scale uncertainty are not quantified.",
                                         "Raster boundary sampling is not landmark homology."]})
