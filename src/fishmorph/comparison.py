"""Separate descriptive diagnostics. No composite 'biological accuracy' score."""
from __future__ import annotations
import numpy as np
from .geometry import interpolate_supported
from .color import regional_delta_e
from .statistics import predict_allometry
from .util import clean


def compare(profile, candidate, exploratory_color=False):
    if profile.get("measurement_protocol") != candidate["provenance"].get("measurement_protocol"):
        raise ValueError("Measurement protocol definitions differ; rebuild with a compatible schema before comparison.")
    specimen = candidate["provenance"]["specimen_snapshot"]["specimen"]
    if candidate["specimen_id"] in {i["specimen_id"] for i in profile["individuals"]}:
        raise ValueError("Candidate specimen contributed to this profile. Use an independent reference or rebuild leave-one-specimen-out.")
    if specimen["taxon_id"] != profile["taxon"]["id"]:
        raise ValueError("Candidate and reference taxonomic concept differ; explicit re-identification is required.")
    q = candidate["morphology"]["quality"]
    mismatches = [key for key in ("view", "mouth", "fins") if q[key] != profile["group"][key]]
    if mismatches:
        raise ValueError("Incompatible projection/articulation: " + ", ".join(mismatches))
    for key in ("sex", "stage", "population", "locality", "season", "reproductive_condition"):
        if specimen[key] != "unknown" and profile["group"][key] != "unknown" and specimen[key] != profile["group"][key]:
            raise ValueError(f"Candidate {key} differs from reference stratum.")
    findings = []
    for name, reference in profile["measurements"].items():
        row = candidate["morphology"]["measurements"].get(name)
        if not row or row["value"] is None:
            findings.append({"measurement": name, "status": "not_comparable", "reasons": row.get("reasons", []) if row else ["measurement_missing"]})
            continue
        value, mean = row["value"], reference["mean"]
        outside = value < reference["min"] or value > reference["max"]
        status = "outside_observed_sample" if outside else "within_observed_sample"
        if reference["n_individuals"] < 5:
            status += "_small_n"
        entry = {"measurement": name, "label": row["label"], "candidate": value,
                 "reference_mean": mean, "observed_range": [reference["min"], reference["max"]],
                 "n_reference_individuals": reference["n_individuals"], "status": status,
                 "difference": value - mean, "relative_difference": (value - mean) / mean if mean else None,
                 "sd_annotation_candidate": row.get("sd_annotation"),
                 "descriptive_z": (value - mean) / reference["sd"] if reference.get("sd") and reference["n_individuals"] >= 5 else None}
        model = profile["allometry"].get(name, {})
        if model.get("status") == "fitted_exploratory" and specimen["standard_length_mm"] is not None:
            try:
                entry["size_conditioned"] = predict_allometry(model, specimen["standard_length_mm"])
            except ValueError as exc:
                entry["size_conditioned"] = {"status": "unavailable", "reason": str(exc)}
        findings.append(entry)
    comparable = [f for f in findings if f.get("relative_difference") is not None]
    comparable.sort(key=lambda f: abs(f["relative_difference"]), reverse=True)
    missing = [f for f in findings if f not in comparable]
    landmarks = []
    for name, point in candidate["morphology"]["normalized_landmarks"].items():
        reference = profile["mean_sl_landmarks"].get(name)
        if reference and name != "dorsal_reference":
            landmarks.append({"name": name, "candidate": point, "reference": reference["mean"],
                              "displacement_sl": float(np.linalg.norm(np.asarray(point) - reference["mean"]))})
    contour_metrics = {"status": "unavailable"}
    c = candidate["morphology"]["body_profile"]
    r = profile["body_profile"]
    if c is not None and r is not None:
        u = np.asarray(r["u"], float)
        ct = interpolate_supported(u, c["u"], c["dorsal"])
        cb = interpolate_supported(u, c["u"], c["ventral"])
        rt, rb = np.asarray(r["dorsal"]["mean"], float), np.asarray(r["ventral"]["mean"], float)
        valid = np.isfinite(ct + cb + rt + rb)
        if valid.sum() >= 5:
            intersection = np.maximum(0, np.minimum(cb, rb) - np.maximum(ct, rt))
            union = cb - ct + rb - rt - intersection
            contour_metrics = {"status": "partial_support_descriptive", "coverage_fraction": float(valid.mean()),
                               "body_envelope_iou_on_shared_support": float(intersection[valid].sum() / union[valid].sum()),
                               "mean_boundary_displacement_sl": float(np.mean(np.abs(ct[valid] - rt[valid]) + np.abs(cb[valid] - rb[valid])) / 2),
                               "u": u, "reference_dorsal": rt, "reference_ventral": rb,
                               "candidate_dorsal": ct, "candidate_ventral": cb, "shared_valid": valid,
                               "note": "Fins-excluded body envelope at matching SL positions; missing columns are excluded, not treated as anatomy."}
    covariance = profile["covariance"]
    multivariate = {"status": "unavailable", "reason": covariance["status"]}
    if covariance["status"] == "usable_descriptive":
        measurements = candidate["morphology"]["measurements"]
        names = covariance["names"]
        if all(n in measurements and measurements[n]["value"] is not None for n in names):
            delta = np.array([measurements[n]["value"] for n in names]) - covariance["mean"]
            d2 = float(delta @ np.linalg.solve(np.asarray(covariance["matrix"]), delta))
            multivariate = {"status": "descriptive_only", "mahalanobis_distance": float(np.sqrt(max(0., d2))),
                            "dimensions": len(names), "names": names,
                            "note": "No probability, p-value, threshold or biological-accuracy interpretation is assigned."}
    im = candidate["image_metadata"]
    color = {"status": "abstained", "reason": "No photometrically compatible color group. Raw field photos and neutral mesh renders are not directly color-validated."}
    for group in profile["color_groups"]:
        matched = (group["photometric_status"] == im["photometric_status"] and
                   group["photometric_protocol"] == im["photometric_protocol"] and
                   im["photometric_status"] in {"externally_calibrated", "synthetic"})
        if matched or exploratory_color:
            delta = regional_delta_e(group["regions"], candidate["color"]["regions"])
            if delta:
                color = {"status": "calibration_protocol_matched_descriptive" if matched else "uncalibrated_exploratory_only",
                         "regional_delta_E00": delta, "reference_protocol": group["photometric_protocol"],
                         "note": "Not a validated pigmentation pass/fail result. Matched labels do not independently prove calibration quality."}
                break
    structures = []
    for name, counts in profile.get("structures", {}).items():
        observed = candidate["morphology"].get("structures", {}).get(name, "not_visible")
        opposite = "absent_observed" if observed == "present" else "present"
        decisive = observed in {"present", "absent_observed"}
        structures.append({"structure": name, "candidate": observed, "reference_individual_counts": counts,
                           "status": "not_comparable" if not decisive else
                           "differs_from_observed_reference" if counts.get(opposite, 0) and not counts.get(observed, 0)
                           else "observed_state_recorded",
                           "note": "Manual observed states only; a hidden structure is not an absence. No universal species rule inferred."})
    return clean({"profile_id": profile["id"], "candidate_analysis_id": candidate["id"],
                  "candidate_specimen_id": candidate["specimen_id"], "group": profile["group"],
                  "reference_n_individuals": profile["n_individuals"], "measurements": comparable + missing,
                  "landmark_displacements": landmarks, "contour": contour_metrics,
                  "multivariate": multivariate, "color": color, "structures": structures,
                  "conclusion": "Descriptive deviations only. A valid individual may lie outside the observed sample; missing evidence is not failure.",
                  "limits": ["This workflow has software tests, not completed biological validation.",
                             "Unknown candidate demographics remain unknown; the selected stratum is a requested comparator.",
                             "Mesh texture is not treated as geometric anatomy."]})


def generation_constraints(profile):
    return {"profile_id": profile["id"], "taxon": profile["taxon"], "group": profile["group"],
            "n_individuals": profile["n_individuals"], "landmarks": profile["mean_sl_landmarks"],
            "body_profile": profile["body_profile"],
            "measurement_targets": {name: {"median": m["median"], "observed_min": m["min"], "observed_max": m["max"],
                                           "n_individuals": m["n_individuals"], "unit": m["unit"]}
                                    for name, m in profile["measurements"].items()},
            "color_variants": [{"protocol": g["photometric_protocol"], "status": g["photometric_status"],
                                "individuals": g["individuals"]} for g in profile["color_groups"]],
            "constraint_interpretation": "Soft empirical guides, not mandatory biological tolerances. Do not invent missing anatomy or clone an averaged texture.",
            "unmeasured": ["true 3D width", "cross sections", "unseen surfaces", "roughness", "metallicity", "subsurface scattering"]}
