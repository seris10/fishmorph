"""End-to-end operations shared by the command line and local application."""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2
from .models import ImageMetadata, Annotation, AnalysisConfig
from .mesh import render_mesh
from .comparison import compare
from .exports import export_document
from .util import digest


def import_model(store, path, specimen_id, longitudinal="+x", dorsal="+z", view="lateral_a", resolution=768):
    specimen = store.get_specimen(specimen_id)["specimen"]
    if specimen["material"] != "model":
        raise ValueError("Mesh renders must be assigned to a specimen with material='model'.")
    target = store.root / "outputs" / ("render_" + digest({"path": str(Path(path).resolve()),
              "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "longitudinal": longitudinal, "dorsal": dorsal, "view": view, "resolution": resolution})[:24])
    rendering = render_mesh(path, target, longitudinal, dorsal, view, resolution)
    image = store.ingest(rendering["image"], specimen_id, ImageMetadata(
        source=f"local generated mesh sha256:{rendering['sha256']}", license="user-supplied mesh; confirm license",
        photometric_status="render", photometric_protocol="geometry-only-neutral-v1",
        note=json.dumps(rendering)))
    mask = cv2.imread(rendering["silhouette"], cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [c.reshape(-1, 2).tolist() for c in contours if len(c) >= 3]
    draft = Annotation(image_id=image["id"], author="mesh_import", masks={"visible": polygons},
                       note="Unreviewed geometry render. Set anatomical orientation, pose, body envelope and visible landmarks manually.")
    if not store.annotation(image["id"]):
        store.save_annotation(draft)
    return {"image": image, "rendering": rendering}


def validate_image(store, profile_id, image_id, exploratory_color=False):
    profile = store.document(profile_id)
    if "individuals" not in profile:
        raise ValueError("The selected record is not a morphology profile.")
    analysis = store.analyze(image_id, AnalysisConfig.model_validate(profile["analysis_config"]))
    validation = store.save_document("validation", compare(profile, analysis, exploratory_color))
    export_document(validation, store.root / "outputs" / validation["id"])
    return validation
