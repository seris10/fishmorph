"""Anatomical coordinate transforms and conservative, fins-excluded profiles."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image, ImageDraw
from .models import Annotation


def polygon_mask(polygons, shape):
    h, w = shape
    im = Image.new("1", (w, h))
    draw = ImageDraw.Draw(im)
    for poly in polygons:
        draw.polygon([tuple(p) for p in poly], fill=1)
    return np.asarray(im, dtype=bool).copy()


def layers(annotation: Annotation, shape):
    return {key: polygon_mask(annotation.masks.get(key, []), shape)
            for key in ("visible", "body", "occlusion", "uncertain", "artifact")}


def at(mask, points, outside=True):
    """Nearest pixel membership, with conservative handling outside the raster."""
    p = np.atleast_2d(points)
    ij = np.rint(p).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[0])
    out = np.full(len(ij), outside, bool)
    out[inside] = mask[ij[inside, 1], ij[inside, 0]]
    return out


def visible_points(annotation: Annotation, image_shape=None):
    mask = None
    if image_shape is not None:
        mm = layers(annotation, image_shape)
        mask = mm["occlusion"] | mm["uncertain"]
    result = {}
    for name, lm in annotation.landmarks.items():
        if lm.visibility != "visible" or lm.basis != "observed":
            continue
        p = np.array([lm.x, lm.y], float)
        if mask is not None and at(mask, p)[0]:
            continue
        result[name] = p
    return result


@dataclass(frozen=True)
class Frame:
    origin: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    length: float

    @classmethod
    def from_points(cls, points, require_dorsal=True):
        if not {"snout", "caudal_base"} <= points.keys():
            raise ValueError("Visible, observed snout and anatomical caudal_base are required.")
        origin = np.asarray(points["snout"], float)
        d = np.asarray(points["caudal_base"], float) - origin
        length = float(np.linalg.norm(d))
        if length < 1e-8:
            raise ValueError("Snout and caudal base coincide.")
        ex = d / length
        ey = np.array([-ex[1], ex[0]])
        if require_dorsal:
            if "dorsal_reference" not in points:
                raise ValueError("A visible dorsal_reference is required to resolve anatomical orientation.")
            side = float(np.dot(points["dorsal_reference"] - origin, ey))
            if abs(side) / length < 1e-5:
                raise ValueError("Dorsal reference lies on the standard-length axis.")
            if side > 0:
                ey = -ey
        return cls(origin, ex, ey, length)

    def normalize(self, points):
        d = np.asarray(points, float) - self.origin
        return np.stack([d @ self.ex, d @ self.ey], axis=-1) / self.length

    def inverse(self, xy):
        xy = np.asarray(xy, float)
        return self.origin + self.length * (xy[..., 0, None] * self.ex + xy[..., 1, None] * self.ey)

    def export(self):
        return {"origin_px": self.origin.tolist(), "longitudinal": self.ex.tolist(),
                "ventral": self.ey.tolist(), "sl_px": self.length,
                "definition": "u=0 snout; u=1 caudal base; y is ventral-positive, in SL units"}


def geometry_reasons(annotation: Annotation):
    q = annotation.quality
    reasons = []
    if not q.geometry_reviewed:
        reasons.append("geometry_not_reviewed")
    if not q.view.startswith("lateral_"):
        reasons.append("non_lateral_or_unknown_view")
    if q.curved:
        reasons.append("curved_body: no straightening assumed")
    if q.refraction:
        reasons.append("unresolved_refraction")
    if q.severe_perspective:
        reasons.append("unresolved_perspective")
    if q.mouth != "closed":
        reasons.append("closed_mouth_SL_protocol_not_established")
    if q.blur == "severe":
        reasons.append("severe_blur")
    return reasons


def scan_polygon(poly, u):
    """Intersect a simple polygon with vertical lines; reject disjoint cross-sections.

    Half-open edge treatment avoids counting a shared vertex twice. An anatomical
    envelope with >2 crossings is not silently reduced to a bounding box.
    """
    poly = np.asarray(poly, float)
    a, b = poly, np.roll(poly, -1, axis=0)
    ymin = np.full(len(u), np.nan)
    ymax = ymin.copy()
    for i, x in enumerate(u):
        hit = ((a[:, 0] <= x) & (x < b[:, 0])) | ((b[:, 0] <= x) & (x < a[:, 0]))
        aa, bb = a[hit], b[hit]
        if len(aa) != 2:
            continue
        ys = aa[:, 1] + (x - aa[:, 0]) * (bb[:, 1] - aa[:, 1]) / (bb[:, 0] - aa[:, 0])
        ymin[i], ymax[i] = np.sort(ys)
    return ymin, ymax


def body_profile(annotation, shape, samples=129, guard=2):
    points = visible_points(annotation, shape)
    frame = Frame.from_points(points)
    polygons = annotation.masks.get("body", [])
    if len(polygons) != 1:
        raise ValueError("One simple, fins-excluded body envelope is required for a body profile.")
    if annotation.body_basis == "inferred":
        raise ValueError("Inferred body outlines cannot supply empirical body profiles or registered color.")
    u = (np.arange(samples) + 0.5) / samples
    top, bottom = scan_polygon(frame.normalize(polygons[0]), u)
    valid = np.isfinite(top) & np.isfinite(bottom) & ((bottom - top) > 1e-7)
    mm = layers(annotation, shape)
    # Interior occlusion does not invalidate an observed pair of body boundaries.
    boundary_block = mm["occlusion"] | mm["uncertain"]
    if guard:
        boundary_block = binary_dilation(boundary_block, iterations=guard)
    ids = np.flatnonzero(valid)
    if len(ids):
        t = frame.inverse(np.column_stack([u[ids], top[ids]]))
        b = frame.inverse(np.column_stack([u[ids], bottom[ids]]))
        valid[ids] &= ~(at(boundary_block, t) | at(boundary_block, b))
    top[~valid] = np.nan
    bottom[~valid] = np.nan
    return {"u": u, "dorsal": top, "ventral": bottom, "valid": valid,
            "coverage": float(valid.mean()), "frame": frame.export()}, frame


def interpolate_supported(x, xp, fp):
    """Linear interpolation only between adjacent valid samples. Never bridge a gap."""
    x = np.asarray(x, float)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    inds = np.searchsorted(xp, x)
    j = np.clip(inds, 1, len(xp) - 1)
    ok = (x >= xp[0]) & (x <= xp[-1]) & np.isfinite(fp[j - 1]) & np.isfinite(fp[j])
    out = np.full(x.shape, np.nan)
    z = (x - xp[j - 1]) / (xp[j] - xp[j - 1])
    out[ok] = fp[j - 1][ok] * (1 - z[ok]) + fp[j][ok] * z[ok]
    exact_index = np.clip(inds, 0, len(xp) - 1)
    exact = (x == xp[exact_index]) & np.isfinite(fp[exact_index])
    out[exact] = fp[exact_index][exact]
    return out
