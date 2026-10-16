"""Append-only SQLite records plus content-addressed originals and derivatives."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import uuid
import numpy as np
from PIL import Image, ImageCms
from . import __version__, SCHEMA_VERSION
from .models import Taxon, Specimen, ImageMetadata, Annotation, AnalysisConfig
from .util import dumps, digest, clean, environment, code_fingerprint, protocol_fingerprint

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS taxa(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS specimen_revisions(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, specimen_id TEXT NOT NULL, revision INTEGER NOT NULL,
 payload TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(specimen_id, revision));
CREATE TABLE IF NOT EXISTS images(
 id TEXT PRIMARY KEY, specimen_id TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
 raster_sha256 TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS annotations(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, image_id TEXT NOT NULL REFERENCES images(id),
 payload TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analyses(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, annotation_id TEXT NOT NULL REFERENCES annotations(id),
 image_id TEXT NOT NULL REFERENCES images(id), payload TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
 payload TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
PRAGMA user_version=1;
"""


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root="./fishmorph-data"):
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / "fishmorph.sqlite3"

    @contextmanager
    def connection(self):
        if not self.db_path.exists():
            raise ValueError(f"No project database at {self.root}. Run fishmorph init first.")
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def init(self, seed=True):
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("originals", "images", "outputs"):
            (self.root / folder).mkdir(exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"Unsupported database schema {version}; no downgrade or destructive migration attempted.")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA_SQL)
        if seed:
            from .taxonomy import SEED_TAXA
            with self.connection() as db:
                for taxon in SEED_TAXA:
                    db.execute("INSERT OR IGNORE INTO taxa VALUES (?,?)", (taxon.id, taxon.model_dump_json()))
        return self.root

    def add_taxon(self, taxon: Taxon):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM taxa WHERE id=?", (taxon.id,)).fetchone()
            if row:
                if Taxon.model_validate_json(row[0]) != taxon:
                    raise ValueError("Taxon IDs are immutable. Add a new, sourced taxonomic concept instead of overwriting.")
            else:
                db.execute("INSERT INTO taxa VALUES (?,?)", (taxon.id, taxon.model_dump_json()))
        return taxon.model_dump()

    def taxa(self):
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM taxa ORDER BY id")]

    def get_taxon(self, taxon_id):
        with self.connection() as db:
            r = db.execute("SELECT payload FROM taxa WHERE id=?", (taxon_id,)).fetchone()
        if not r:
            raise ValueError("Unknown taxon ID; add a sourced taxon first.")
        return Taxon.model_validate_json(r[0])

    def save_specimen(self, specimen: Specimen):
        self.get_taxon(specimen.taxon_id)
        with self.connection() as db:
            row = db.execute("SELECT payload, revision FROM specimen_revisions WHERE specimen_id=? ORDER BY seq DESC LIMIT 1", (specimen.id,)).fetchone()
            if row and Specimen.model_validate_json(row[0]) == specimen:
                return {"specimen": specimen.model_dump(), "revision": row[1]}
            revision = 1 if not row else row[1] + 1
            db.execute("INSERT INTO specimen_revisions(specimen_id,revision,payload,created) VALUES (?,?,?,?)",
                       (specimen.id, revision, specimen.model_dump_json(), now()))
        return {"specimen": specimen.model_dump(), "revision": revision}

    def get_specimen(self, specimen_id):
        with self.connection() as db:
            row = db.execute("SELECT payload,revision FROM specimen_revisions WHERE specimen_id=? ORDER BY seq DESC LIMIT 1", (specimen_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown specimen: {specimen_id}")
        return {"specimen": json.loads(row[0]), "revision": row[1]}

    def specimens(self):
        with self.connection() as db:
            rows = db.execute("SELECT payload,revision FROM specimen_revisions WHERE seq IN (SELECT MAX(seq) FROM specimen_revisions GROUP BY specimen_id) ORDER BY specimen_id").fetchall()
        return [{"specimen": json.loads(r[0]), "revision": r[1]} for r in rows]

    def ingest(self, path, specimen_id, metadata: ImageMetadata | None = None):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Image does not exist: {path}")
        return self.ingest_bytes(path.read_bytes(), path.name, specimen_id, metadata)

    def ingest_bytes(self, raw: bytes, filename, specimen_id, metadata: ImageMetadata | None = None):
        self.get_specimen(specimen_id)
        meta = metadata or ImageMetadata()
        if len(raw) > 100_000_000:
            raise ValueError("Image exceeds the 100 MB local ingestion limit.")
        sha = hashlib.sha256(raw).hexdigest()
        try:
            with Image.open(io.BytesIO(raw)) as im:
                if im.width * im.height > 24_000_000:
                    raise ValueError("Use a derivative below 24 megapixels and retain the original externally.")
                if getattr(im, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images must be split explicitly before ingestion.")
                im.load()
                image_format = im.format
                exif_orientation = im.getexif().get(274, 1)
                icc = im.info.get("icc_profile")
                # Do not EXIF-transpose: raw raster coordinates remain original-image coordinates.
                display = im.convert("RGB")
                icc_status = "no embedded ICC; sRGB assumption for descriptive color"
                if icc:
                    try:
                        display = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
                        icc_status = "embedded ICC converted to sRGB; not illumination calibration"
                    except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                        raise ValueError("Embedded ICC profile cannot be converted safely; supply a documented sRGB derivative.") from exc
                arr = np.array(display)
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Unreadable or unsupported raster image.") from exc
        raster_sha = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
        with self.connection() as db:
            row = db.execute("SELECT payload FROM images WHERE sha256=? OR raster_sha256=?", (sha, raster_sha)).fetchone()
            if row:
                existing = json.loads(row[0])
                if existing["specimen_id"] != specimen_id:
                    raise ValueError("Duplicate original or decoded raster is already assigned to a different specimen. Resolve identity instead of inflating N.")
                db.execute("INSERT INTO events(kind,payload,created) VALUES (?,?,?)",
                           ("duplicate_import", dumps({"image_id": existing["id"], "metadata": meta.model_dump()}), now()))
                return existing
            image_id = "im_" + sha[:24]
            original_relative = f"originals/{sha}.bin"
            derivative_relative = f"images/{sha}.png"
            (self.root / original_relative).write_bytes(raw)
            display.save(self.root / derivative_relative)
            payload = {"id": image_id, "specimen_id": specimen_id, "sha256": sha, "raster_sha256": raster_sha,
                       "filename": Path(filename).name, "original": original_relative, "display": derivative_relative,
                       "width": int(arr.shape[1]), "height": int(arr.shape[0]), "format": image_format,
                       "coordinate_space": "original decoded raster, top-left origin; EXIF orientation intentionally not applied",
                       "original_exif_orientation": exif_orientation, "icc_status": icc_status,
                       "metadata": meta.model_dump()}
            db.execute("INSERT INTO images VALUES (?,?,?,?,?,?)", (image_id, specimen_id, sha, raster_sha, dumps(payload), now()))
        return payload

    def images(self):
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM images ORDER BY created,id").fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_image(self, image_id):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown image: {image_id}")
        return json.loads(row[0])

    def rgb(self, image_id):
        image = self.get_image(image_id)
        return np.array(Image.open(self.root / image["display"]).convert("RGB"))

    def save_annotation(self, annotation: Annotation):
        # Validate nested dictionary mutations as well as construction-time input.
        annotation = Annotation.model_validate(annotation.model_dump(warnings=False))
        image = self.get_image(annotation.image_id)
        for lm in annotation.landmarks.values():
            if lm.x > image["width"] - 1 or lm.y > image["height"] - 1:
                raise ValueError("Landmark outside original raster coordinates.")
        for polygons in [*annotation.masks.values(), *annotation.regions.values()]:
            if any(x > image["width"] - 1 or y > image["height"] - 1 for poly in polygons for x, y in poly):
                raise ValueError("Polygon vertex outside original raster coordinates.")
        payload = annotation.model_dump()
        # Identical saves remain explicit events; identity is a revision, not a content hash.
        aid = "ann_" + uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO annotations(id,image_id,payload,created) VALUES (?,?,?,?)",
                       (aid, annotation.image_id, dumps(payload), now()))
        return {"id": aid, "content_hash": digest(payload), "annotation": payload}

    def annotation(self, image_id, annotation_id=None):
        self.get_image(image_id)
        with self.connection() as db:
            if annotation_id:
                row = db.execute("SELECT id,payload FROM annotations WHERE id=? AND image_id=?", (annotation_id, image_id)).fetchone()
            else:
                row = db.execute("SELECT id,payload FROM annotations WHERE image_id=? ORDER BY seq DESC LIMIT 1", (image_id,)).fetchone()
        return None if not row else {"id": row[0], "annotation": json.loads(row[1])}

    def annotation_history(self, image_id):
        with self.connection() as db:
            rows = db.execute("SELECT id,payload,created FROM annotations WHERE image_id=? ORDER BY seq DESC", (image_id,)).fetchall()
        return [{"id": r[0], "annotation": json.loads(r[1]), "created": r[2]} for r in rows]

    def analyze(self, image_id, config: AnalysisConfig | None = None, annotation_id=None):
        from .measurements import measure
        from .color import color_analysis
        from .vision import image_indicators
        cfg = config or AnalysisConfig()
        img = self.get_image(image_id)
        snapshot = self.get_specimen(img["specimen_id"])
        ann = self.annotation(image_id, annotation_id)
        if ann is None:
            raise ValueError("Annotate this image before measuring it.")
        annotation = Annotation.model_validate(ann["annotation"])
        env = environment()
        identity = {"image_sha256": img["sha256"], "annotation_id": ann["id"], "specimen_snapshot": snapshot,
                    "config": cfg.model_dump(), "software": __version__, "schema": SCHEMA_VERSION, "environment": env, "code_fingerprint": code_fingerprint(), "measurement_protocol": protocol_fingerprint()}
        run_id = "a_" + digest(identity)[:32]
        with self.connection() as db:
            row = db.execute("SELECT payload FROM analyses WHERE id=?", (run_id,)).fetchone()
        if row:
            return json.loads(row[0])
        rgb = self.rgb(image_id)
        payload = clean({"id": run_id, "image_id": image_id, "annotation_id": ann["id"],
                         "specimen_id": img["specimen_id"], "provenance": identity,
                         "image_metadata": img["metadata"],
                         "morphology": measure(annotation, rgb.shape[:2], cfg, snapshot["specimen"]["standard_length_mm"]),
                         "color": color_analysis(rgb, annotation, cfg), "image_indicators": image_indicators(rgb)})
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO analyses(id,annotation_id,image_id,payload,created) VALUES (?,?,?,?,?)",
                       (run_id, ann["id"], image_id, dumps(payload), now()))
        return payload

    def analyses_for_taxon(self, taxon_id, config=None):
        """Reanalyze current revisions deterministically; no stale annotations mixed into a build."""
        rows, excluded = [], []
        for image in sorted(self.images(), key=lambda i: i["id"]):
            sp = self.get_specimen(image["specimen_id"])["specimen"]
            if sp["taxon_id"] != taxon_id:
                continue
            if not self.annotation(image["id"]):
                excluded.append({"image_id": image["id"], "reason": "not_annotated"})
                continue
            rows.append(self.analyze(image["id"], config))
        return rows, excluded

    def save_document(self, kind, payload):
        payload = clean(payload)
        did = kind + "_" + digest(payload)[:32]
        result = {"id": did, **payload}
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO documents(id,kind,payload,created) VALUES (?,?,?,?)", (did, kind, dumps(result), now()))
        return result

    def document(self, document_id):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            raise ValueError("Unknown profile or validation document ID.")
        return json.loads(row[0])

    def documents(self, kind):
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM documents WHERE kind=? ORDER BY seq DESC", (kind,)).fetchall()
        return [json.loads(r[0]) for r in rows]
