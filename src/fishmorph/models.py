"""Validated public data contracts. Unknown is not a negative observation."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)

class Taxon(Model):
    id: str = Field(min_length=1, max_length=160)
    scientific_name: str = Field(min_length=1, max_length=240)
    common_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    status: Literal["verified_name", "name_usage_conflict", "unresolved", "synthetic"] = "unresolved"
    sources: list[str] = Field(default_factory=list)
    note: str = ""
    profile_allowed: bool = False

class Specimen(Model):
    id: str = Field(min_length=1, max_length=160)
    taxon_id: str
    identity_status: Literal["documented", "unresolved"] = "unresolved"
    identity_evidence: str = ""
    identification_evidence: str = ""
    taxonomic_confidence: float | None = Field(default=None, ge=0, le=1)
    sex: Literal["male", "female", "unknown", "indeterminate"] = "unknown"
    stage: Literal["juvenile", "subadult", "adult", "unknown"] = "unknown"
    demographic_evidence: str = ""
    population: str = "unknown"
    locality: str = "unknown"
    season: str = "unknown"
    reproductive_condition: str = "unknown"
    material: Literal["living_wild", "living_captive", "preserved", "illustration", "synthetic", "model"] = "living_wild"
    standard_length_mm: float | None = Field(default=None, gt=0)
    length_evidence: str = ""
    metadata_note: str = ""

    @model_validator(mode="after")
    def check_evidence(self):
        if self.identity_status == "documented" and not self.identity_evidence.strip():
            raise ValueError("Documented identity requires identity_evidence; filenames alone are not identity.")
        if self.standard_length_mm is not None and not self.length_evidence.strip():
            raise ValueError("An independently measured SL requires length_evidence.")
        if (self.sex in {"male", "female"} or self.stage != "unknown") and not self.demographic_evidence.strip():
            raise ValueError("Known sex or stage requires demographic_evidence.")
        return self

class ImageMetadata(Model):
    source: str = "local file"
    photographer: str = "unknown"
    license: str = "unknown; not cleared for redistribution"
    acquired: str | None = None
    camera: str = "unknown"
    session: str = "unknown"
    photometric_status: Literal["uncalibrated", "externally_calibrated", "synthetic", "render"] = "uncalibrated"
    photometric_protocol: str = "uncontrolled"
    calibration_evidence: str = ""
    note: str = ""

    @model_validator(mode="after")
    def check_calibration(self):
        if self.photometric_status == "externally_calibrated" and not self.calibration_evidence.strip():
            raise ValueError("External calibration requires documented protocol/evidence.")
        return self

class Landmark(Model):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    visibility: Literal["visible", "occluded", "uncertain", "not_applicable"] = "visible"
    basis: Literal["observed", "inferred"] = "observed"
    sigma_px: float | None = Field(default=None, ge=0)
    method: str = "manual"
    note: str = ""

class Quality(Model):
    view: Literal["lateral_left", "lateral_right", "oblique", "dorsal", "ventral", "frontal", "unknown"] = "unknown"
    geometry_reviewed: bool = False
    color_reviewed: bool = False
    curved: bool = False
    refraction: bool = False
    severe_perspective: bool = False
    mouth: Literal["closed", "open", "unknown"] = "unknown"
    fins: Literal["natural", "spread", "folded", "unknown"] = "unknown"
    blur: Literal["low", "moderate", "severe", "unknown"] = "unknown"
    resolution_note: str = ""
    note: str = ""

Point = tuple[float, float]
Polygon = list[Point]

class Annotation(Model):
    image_id: str
    author: str = Field(min_length=1)
    note: str = ""
    quality: Quality = Field(default_factory=Quality)
    landmarks: dict[str, Landmark] = Field(default_factory=dict)
    # body: fins-excluded anatomical envelope, NOT the visible fish segmentation.
    # Gaps in its boundary must be overpainted in occlusion or uncertain masks.
    masks: dict[str, list[Polygon]] = Field(default_factory=dict)
    regions: dict[str, list[Polygon]] = Field(default_factory=dict)
    body_basis: Literal["observed_with_masked_gaps", "inferred"] = "observed_with_masked_gaps"
    structures: dict[str, Literal["present", "absent_observed", "not_visible", "not_applicable"]] = Field(default_factory=dict)
    structure_evidence: str = ""

    @model_validator(mode="after")
    def check_polygons(self):
        if set(self.masks) - {"visible", "body", "occlusion", "uncertain", "artifact"}:
            raise ValueError("Unknown mask layer.")
        for polygons in [*self.masks.values(), *self.regions.values()]:
            for poly in polygons:
                if len(poly) < 3:
                    raise ValueError("A polygon needs at least three vertices.")
                if any(x < 0 or y < 0 for x, y in poly):
                    raise ValueError("Polygon coordinates cannot be negative.")
        if any(v == "absent_observed" for v in self.structures.values()) and not self.structure_evidence.strip():
            raise ValueError("Observed anatomical absence requires evidence, not just a hidden fin.")
        return self

class AnalysisConfig(Model):
    profile_samples: int = Field(default=129, ge=17, le=513)
    atlas_width: int = Field(default=128, ge=16, le=512)
    atlas_height: int = Field(default=40, ge=8, le=256)
    boundary_guard_px: int = Field(default=2, ge=0, le=20)
    mask_clipped_colors: bool = True
    dark_spot_contrast: float = Field(default=8.0, gt=0)
    light_spot_contrast: float = Field(default=8.0, gt=0)
    spot_background_sigma_sl: float = Field(default=0.018, gt=0, le=0.2)
    spot_min_area_sl2: float = Field(default=0.000015, gt=0)
    spot_max_area_sl2: float = Field(default=0.008, gt=0, title="Maximum spot area (SL2)")
    seed: int = 2026

class ProfileConfig(Model):
    bootstrap_samples: int = Field(default=1000, ge=100, le=20000)
    minimum_interval_n: int = Field(default=5, ge=3)
    seed: int = 2026
    include_unresolved_identity: bool = False
