"""Specimen-level aggregation, deterministic bootstrap, GPA and guarded allometry."""
from __future__ import annotations
from collections import defaultdict, Counter
import warnings
import numpy as np
from scipy.linalg import svd
from scipy.stats import t
import statsmodels.api as sm
from .models import ProfileConfig, AnalysisConfig
from .anatomy import LANDMARKS
from . import __version__, SCHEMA_VERSION
from .util import clean, digest


def quiet_nanmedian(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(a, float), axis=axis)


def scalar_summary(values, cfg: ProfileConfig, seed_offset=0):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if not n:
        return {"n_individuals": 0, "mean": None, "sd": None, "mean_ci95": None}
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    ci = None
    if n >= cfg.minimum_interval_n:
        rng = np.random.default_rng(cfg.seed + seed_offset)
        means = np.mean(x[rng.integers(0, n, size=(cfg.bootstrap_samples, n))], axis=1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return clean({"n_individuals": n, "mean": float(x.mean()), "median": median,
                  "sd": float(x.std(ddof=1)) if n > 1 else None,
                  "min": float(x.min()), "max": float(x.max()),
                  "q025": float(np.quantile(x, .025)), "q25": float(np.quantile(x, .25)),
                  "q75": float(np.quantile(x, .75)), "q975": float(np.quantile(x, .975)),
                  "mad": mad, "mean_ci95": ci,
                  "interval_definition": "percentile bootstrap of individual summaries; uncertainty in mean, NOT individual prediction limits",
                  "outlier_values_mad_3_5": x[np.abs(x - median) / (1.4826 * mad) > 3.5].tolist() if mad > 0 and n >= 5 else [],
                  "small_sample": n < cfg.minimum_interval_n})


def array_summary(arrays, minimum=1):
    """Missing-aware fields. Each first-axis item must represent one specimen."""
    a = np.asarray(arrays, float)
    finite = np.isfinite(a)
    n = finite.sum(axis=0)
    mean = np.divide(np.nansum(a, axis=0), n, out=np.full(a.shape[1:], np.nan), where=n > 0)
    ss = np.nansum((a - mean)**2, axis=0)
    sd = np.sqrt(np.divide(ss, n - 1, out=np.full_like(mean, np.nan), where=n > 1))
    mean[n < minimum] = np.nan
    return clean({"mean": mean, "sd": sd, "n_individuals": n})


def generalized_procrustes(shapes, tolerance=1e-10, max_iterations=200):
    """Translation/centroid-size normalization and proper rotations (NO reflection).

    Inputs must have the same homologous landmarks in the same order. Missing
    data are not filled. GPA is distinct from SL-normalized comparison geometry.
    """
    x = np.asarray(shapes, float)
    if x.ndim != 3 or x.shape[1] < 3 or x.shape[2] != 2 or not np.isfinite(x).all():
        raise ValueError("GPA needs finite [specimen, >=3 homologous landmarks, 2] data.")
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=(1, 2))
    if np.any(norms < 1e-12):
        raise ValueError("Degenerate landmark configuration.")
    x = x / norms[:, None, None]
    reference = x[0].copy()
    aligned = x.copy()
    for iteration in range(max_iterations):
        for i, shape in enumerate(x):
            u, _, vt = svd(shape.T @ reference)
            corr = np.diag([1., float(np.linalg.det(u @ vt))])
            aligned[i] = shape @ (u @ corr @ vt)
        mean = aligned.mean(axis=0)
        mean /= np.linalg.norm(mean)
        if np.linalg.norm(mean - reference) < tolerance:
            reference = mean
            break
        reference = mean
    else:
        raise ValueError("GPA did not converge; no shape result emitted.")
    flat = aligned.reshape(len(aligned), -1)
    centered = flat - flat.mean(axis=0)
    _, singular, vt = svd(centered, full_matrices=False)
    k = min(len(aligned) - 1, x.shape[1] * 2 - 4)
    variances = singular[:k]**2 / max(1, len(aligned) - 1)
    total = float(np.sum(variances))
    return clean({"mean_centroid_size_shape": reference, "aligned": aligned,
                  "principal_components": vt[:k], "variances": variances,
                  "explained_variance_ratio": variances / total if total > 1e-16 else np.zeros(k),
                  "iterations": iteration + 1, "reflections_allowed": False,
                  "note": "Exploratory tangent-coordinate PCA; neither a species classifier nor a full population model."})


def fit_allometry(lengths_mm, measurements_mm, minimum_n=8):
    x, y = np.asarray(lengths_mm, float), np.asarray(measurements_mm, float)
    if len(x) != len(y) or np.any(x <= 0) or np.any(y <= 0) or not np.isfinite(x + y).all():
        raise ValueError("Allometry requires paired, positive, finite, independently scaled measurements.")
    if len(x) < minimum_n or len(np.unique(x)) < 3:
        return {"status": "insufficient_data", "n_individuals": len(x), "minimum_n": minimum_n}
    xx, yy = np.log(x), np.log(y)
    if np.ptp(xx) < .05:
        return {"status": "insufficient_size_range", "n_individuals": len(x)}
    fit = sm.OLS(yy, sm.add_constant(xx)).fit(cov_type="HC3")
    return clean({"status": "fitted_exploratory", "n_individuals": len(x),
                  "equation": "log(measurement_mm)=intercept+slope*log(SL_mm)",
                  "intercept": fit.params[0], "slope": fit.params[1],
                  "coefficient_covariance_HC3": fit.cov_params(), "coefficient_ci95": fit.conf_int(),
                  "residual_variance_log": float(np.sum(fit.resid**2) / fit.df_resid),
                  "df_residual": float(fit.df_resid), "r_squared": fit.rsquared,
                  "size_domain_mm": [float(x.min()), float(x.max())],
                  "limitations": "OLS assumes negligible SL error; scale errors can affect both axes. No causal or cross-stage interpretation. HC3 uncertainty is approximate at small N."})


def predict_allometry(model, length_mm):
    if model.get("status") != "fitted_exploratory":
        raise ValueError("No fitted allometric model.")
    lo, hi = model["size_domain_mm"]
    if not lo <= length_mm <= hi:
        raise ValueError("Requested length is outside the observed size domain; extrapolation refused.")
    design = np.array([1., np.log(length_mm)])
    center = float(design @ [model["intercept"], model["slope"]])
    var = float(design @ np.asarray(model["coefficient_covariance_HC3"]) @ design)
    critical = float(t.ppf(.975, model["df_residual"]))
    margin = critical * np.sqrt(max(0., var) + model["residual_variance_log"])
    return {"conditional_median_mm": float(np.exp(center)),
            "approximate_prediction95_mm": [float(np.exp(center - margin)), float(np.exp(center + margin))],
            "note": "Back-transformed log model; central value is not the arithmetic mean. Prediction assumes Gaussian log residuals."}


STRATA = ("population", "locality", "sex", "stage", "season", "reproductive_condition", "material")


def group_key(row):
    specimen = row["provenance"]["specimen_snapshot"]["specimen"]
    quality = row["morphology"]["quality"]
    return tuple(specimen[key] for key in STRATA) + (quality["view"], quality["mouth"], quality["fins"])


def summarize_individual(rows):
    names = sorted(set().union(*(r["morphology"]["measurements"] for r in rows)))
    measurements = {}
    for name in names:
        good = [r["morphology"]["measurements"][name] for r in rows
                if name in r["morphology"]["measurements"] and r["morphology"]["measurements"][name]["value"] is not None]
        if not good or good[0]["unit"] == "px":
            continue
        values = [r["value"] for r in good]
        measurements[name] = {"value": float(np.median(values)), "n_images": len(values),
                              "within_individual_min": float(min(values)), "within_individual_max": float(max(values)),
                              "within_individual_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                              "annotation_sd_by_image": [r.get("sd_annotation") for r in good],
                              "unit": good[0]["unit"], "label": good[0]["label"]}
    landmarks = {}
    names = sorted(set().union(*(r["morphology"]["normalized_landmarks"] for r in rows)))
    for name in names:
        coords = [r["morphology"]["normalized_landmarks"][name] for r in rows if name in r["morphology"]["normalized_landmarks"]]
        landmarks[name] = np.median(np.asarray(coords, float), axis=0).tolist()
    contours = [r["morphology"]["body_profile"] for r in rows if r["morphology"]["body_profile"] is not None]
    contour = None
    if contours:
        contour = {"u": contours[0]["u"], "dorsal": quiet_nanmedian([c["dorsal"] for c in contours]),
                   "ventral": quiet_nanmedian([c["ventral"] for c in contours])}
    structures = {}
    structure_names = sorted(set().union(*(r["morphology"].get("structures", {}) for r in rows)))
    for name in structure_names:
        states = {r["morphology"].get("structures", {}).get(name, "not_visible") for r in rows}
        decisive = states & {"present", "absent_observed"}
        structures[name] = ("conflicting_observations" if len(decisive) > 1 else
                            next(iter(decisive)) if decisive else
                            "not_applicable" if states == {"not_applicable"} else "not_visible")
    return clean({"specimen_id": rows[0]["specimen_id"], "n_images": len(rows),
                  "specimen": rows[0]["provenance"]["specimen_snapshot"]["specimen"],
                  "measurements": measurements, "landmarks": landmarks, "body_profile": contour,
                  "analysis_ids": [r["id"] for r in rows], "structures": structures})


def color_groups(rows, cfg):
    groups = defaultdict(list)
    for row in rows:
        if row["color"]["status"] == "unavailable":
            continue
        im = row["image_metadata"]
        groups[(im["photometric_status"], im["photometric_protocol"])].append(row)
    result = []
    for key, data in sorted(groups.items()):
        by_individual = defaultdict(list)
        for row in data:
            by_individual[row["specimen_id"]].append(row)
        individuals, fields = [], []
        for sid, observations in sorted(by_individual.items()):
            region_values = {}
            region_names = sorted(set().union(*(r["color"]["regions"] for r in observations)))
            for name in region_names:
                arrays = [r["color"]["regions"][name]["mean_lab"] for r in observations
                          if name in r["color"]["regions"] and r["color"]["regions"][name].get("mean_lab") is not None]
                if arrays:
                    region_values[name] = np.median(np.asarray(arrays), axis=0).tolist()
            atlases = [r["color"]["atlas"] for r in observations if r["color"]["atlas"] is not None]
            if atlases:
                fields.append(quiet_nanmedian([a["nonmarking_lab"] for a in atlases]))
            # One observed image's patterns per individual, chosen by usable source coverage.
            representative = max(observations, key=lambda r: (r["color"]["usable_source_pixels"], r["id"]))
            individuals.append({"specimen_id": sid, "regions": region_values,
                                "representative_analysis_id": representative["id"],
                                "candidate_density_bins": representative["color"]["density_bins"],
                                "image_ids": [r["image_id"] for r in observations]})
        regions = {}
        region_names = sorted(set().union(*(i["regions"] for i in individuals)))
        for name in region_names:
            colors = np.array([i["regions"][name] for i in individuals if name in i["regions"]])
            regions[name] = {"mean_lab": colors.mean(axis=0), "n_individuals": len(colors),
                             "L": scalar_summary(colors[:, 0], cfg),
                             "a": scalar_summary(colors[:, 1], cfg), "b": scalar_summary(colors[:, 2], cfg)}
        result.append(clean({"photometric_status": key[0], "photometric_protocol": key[1],
                             "n_individuals": len(individuals), "regions": regions,
                             "individuals": individuals,
                             "nonmarking_field_diagnostic": array_summary(fields) if fields else None,
                             "interpretation": "Observed image color only. Field average excludes heuristic marking candidates and missing cells; NOT a synthesized texture or absolute pigmentation estimate.",
                             "confounding": "Camera, source, lighting, season, handling and registration effects are not statistically removed."}))
    return result


def build_profiles(store, taxon_id, cfg: ProfileConfig | None = None, analysis_config=None):
    cfg = cfg or ProfileConfig()
    acfg = analysis_config or AnalysisConfig()
    taxon = store.get_taxon(taxon_id)
    if not taxon.profile_allowed:
        raise ValueError("Taxonomic concept is unresolved; add a separately sourced, identified concept before profiling.")
    rows, excluded = store.analyses_for_taxon(taxon_id, acfg)
    groups = defaultdict(list)
    for row in rows:
        sp = row["provenance"]["specimen_snapshot"]["specimen"]
        if sp["material"] == "model":
            excluded.append({"image_id": row["image_id"], "reason": "model_render_not_empirical_reference"})
            continue
        if sp["identity_status"] == "unresolved" and not cfg.include_unresolved_identity:
            excluded.append({"image_id": row["image_id"], "reason": "unresolved_individual_identity"})
            continue
        any_measurement = any(m["value"] is not None for m in row["morphology"]["measurements"].values())
        if not any_measurement and row["color"]["status"] == "unavailable":
            excluded.append({"image_id": row["image_id"], "reason": "no_eligible_measurements_or_color"})
            continue
        groups[group_key(row)].append(row)
    profiles = []
    for key, observations in sorted(groups.items()):
        by_individual = defaultdict(list)
        for row in observations:
            by_individual[row["specimen_id"]].append(row)
        individuals = [summarize_individual(v) for _, v in sorted(by_individual.items())]
        names = sorted(set().union(*(i["measurements"] for i in individuals)))
        measurements, allometry = {}, {}
        for index, name in enumerate(names):
            present = [i for i in individuals if name in i["measurements"]]
            measurements[name] = scalar_summary([i["measurements"][name]["value"] for i in present], cfg, index)
            measurements[name].update(unit=present[0]["measurements"][name]["unit"], label=present[0]["measurements"][name]["label"])
            if name.endswith("_sl"):
                measured = [i for i in present if i["specimen"]["standard_length_mm"] is not None and i["measurements"][name]["value"] > 0]
                lengths = [i["specimen"]["standard_length_mm"] for i in measured]
                absolute = [i["specimen"]["standard_length_mm"] * i["measurements"][name]["value"] for i in measured]
                allometry[name] = fit_allometry(lengths, absolute) if lengths else {"status": "no_independently_measured_sizes"}
        shape = None
        usable = [i for i in individuals if len(i["landmarks"]) >= 3]
        if len(usable) >= 2:
            common = sorted(set.intersection(*(set(i["landmarks"]) for i in usable)))
            # dorsal_reference is an orientation aid, not a homologous biological landmark.
            common = [n for n in common if n != "dorsal_reference" and n in LANDMARKS]
            if len(common) >= 3:
                shape = generalized_procrustes([[i["landmarks"][n] for n in common] for i in usable])
                shape["landmark_names"] = common
                shape["specimen_ids"] = [i["specimen_id"] for i in usable]
        mean_landmarks = {}
        landmark_names = sorted(set().union(*(i["landmarks"] for i in individuals)))
        for name in landmark_names:
            coords = [i["landmarks"][name] for i in individuals if name in i["landmarks"]]
            mean_landmarks[name] = array_summary(coords)
        contours = [i["body_profile"] for i in individuals if i["body_profile"] is not None]
        contour = None if not contours else {"u": contours[0]["u"],
                    "dorsal": array_summary([c["dorsal"] for c in contours]),
                    "ventral": array_summary([c["ventral"] for c in contours]),
                    "interpretation": "Mean and between-specimen SD at observed positions; not simultaneous confidence bounds."}
        # Complete-case covariance only, never pairwise stitched or pseudoinverted.
        covariance = {"status": "insufficient_complete_individuals"}
        informative = [n for n in names if measurements[n].get("sd") is not None and measurements[n]["sd"] > 1e-10]
        complete = [i for i in individuals if all(n in i["measurements"] for n in informative)]
        if informative and len(complete) >= max(20, 3 * len(informative)):
            matrix = np.array([[i["measurements"][n]["value"] for n in informative] for i in complete])
            cov = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
            condition = np.linalg.cond(cov)
            if np.isfinite(condition) and condition < 1e8:
                covariance = {"status": "usable_descriptive", "names": informative, "mean": matrix.mean(axis=0),
                              "matrix": cov, "n_complete_individuals": len(complete), "condition_number": condition,
                              "note": "Engineering sufficiency guard, not proof of normality or validated species likelihood."}
            else:
                covariance = {"status": "singular_or_ill_conditioned", "condition_number": condition}
        categorical = {}
        for name in sorted(set().union(*(i["structures"] for i in individuals))):
            categorical[name] = dict(Counter(i["structures"].get(name, "not_visible") for i in individuals))
        group = dict(zip(STRATA + ("view", "mouth", "fins"), key))
        source_counts = Counter(r["image_metadata"]["source"] for r in observations)
        payload = clean({"taxon": taxon.model_dump(), "group": group,
                         "n_individuals": len(individuals), "n_images": len(observations),
                         "material_warning": "SYNTHETIC SOFTWARE TEST DATA, NOT BIOLOGY" if group["material"] == "synthetic" else None,
                         "measurements": measurements, "individuals": individuals,
                         "landmark_definitions": LANDMARKS, "mean_sl_landmarks": mean_landmarks,
                         "body_profile": contour, "shape_analysis": shape,
                         "covariance": covariance, "allometry": allometry, "structures": categorical, "color_groups": color_groups(observations, cfg),
                         "source_image_counts": dict(source_counts),
                         "input_analysis_ids": sorted(r["id"] for r in observations),
                         "input_snapshots": [{"analysis_id": r["id"], "image_id": r["image_id"],
                                               "specimen": r["provenance"]["specimen_snapshot"],
                                               "annotation_id": r["annotation_id"],
                                               "sha256": r["provenance"]["image_sha256"]} for r in observations],
                         "config": cfg.model_dump(), "analysis_config": acfg.model_dump(),
                         "schema_version": SCHEMA_VERSION, "software_version": __version__,
                         "measurement_protocol": observations[0]["provenance"]["measurement_protocol"],
                         "limitations": ["Reference distributions describe the sampled individuals, not an entire wild population.",
                                         "No population interval is a validated biological pass/fail tolerance.",
                                         "No imputation of obscured landmarks, inferred contours or unseen pigmentation.",
                                         "Bootstrap conditions on the curated observations and does not propagate acquisition bias.",
                                         "Individual medians give equal specimen weight; disagreements remain in individual records."]})
        profiles.append(store.save_document("profile", payload))
    return {"profiles": profiles, "excluded": excluded, "n_profiles": len(profiles)}
