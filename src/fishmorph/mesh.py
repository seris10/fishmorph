"""CPU-only orthographic geometry rendering. No GL context or Blender installation."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from .util import clean, write_json


def axis_vector(name):
    if name not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError("Axis must be one of +x, -x, +y, -y, +z, -z.")
    vector = np.zeros(3)
    vector["xyz".index(name[1])] = 1 if name[0] == "+" else -1
    return vector


def load_mesh(path):
    path = Path(path)
    if path.suffix.lower() not in {".glb", ".gltf", ".obj", ".ply"}:
        raise ValueError("Supported meshes: GLB, local glTF, OBJ, PLY.")
    if not path.is_file():
        raise ValueError("Mesh file not found.")
    try:
        scene = trimesh.load_scene(path, process=False, allow_remote=False)
        mesh = scene.to_mesh()
    except Exception as exc:
        raise ValueError(f"Mesh could not be loaded: {exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("No triangle geometry in the scene.")
    if len(mesh.faces) > 400_000:
        raise ValueError("CPU preview limit is 400,000 triangles; create a geometry-preserving preview mesh.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains non-finite vertices.")
    return mesh, {"source": str(path.name), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
                  "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
                  "bounds_source_units": mesh.bounds.tolist(),
                  "degenerate_triangles": int((mesh.area_faces < 1e-15).sum()),
                  "scene_transforms_applied": True,
                  "warnings": ["Scene geometry is not automatically recognized as fish anatomy.",
                               "No automatic axis, true physical scale, anatomical correspondence or material-property inference.",
                               "Non-watertight fins can be intentional; integrity findings are not anatomy failures."]}


def render_mesh(path, output_dir, longitudinal="+x", dorsal="+z", view="lateral_a", resolution=768):
    if not 128 <= resolution <= 2048:
        raise ValueError("Resolution must be between 128 and 2048.")
    mesh, metadata = load_mesh(path)
    ex, up = axis_vector(longitudinal), axis_vector(dorsal)
    if abs(float(ex @ up)) > 1e-8:
        raise ValueError("Longitudinal and dorsal axes must be perpendicular.")
    side = np.cross(ex, up)
    frames = {"lateral_a": (ex, up, side), "lateral_b": (-ex, up, -side),
              "dorsal": (ex, side, up), "ventral": (ex, -side, -up),
              "frontal": (side, up, -ex)}
    if view not in frames:
        raise ValueError("Unknown orthographic view.")
    right, upward, depth = frames[view]
    basis = np.column_stack([right, upward, depth])
    xyz = mesh.vertices @ basis
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = hi[:2] - lo[:2]
    if np.max(span) < 1e-12:
        raise ValueError("Degenerate projection.")
    pad = 16
    scale = (resolution - 2 * pad) / float(np.max(span))
    width = max(64, int(np.ceil(span[0] * scale)) + 2 * pad + 1)
    height = max(64, int(np.ceil(span[1] * scale)) + 2 * pad + 1)
    xy = np.column_stack([(xyz[:, 0] - lo[0]) * scale + pad,
                          (hi[1] - xyz[:, 1]) * scale + pad])
    zbuffer = np.full((height, width), -np.inf)
    shade = np.zeros((height, width), np.uint8)
    light = np.array([-.25, .45, .85]); light /= np.linalg.norm(light)
    normals = mesh.face_normals @ basis
    for face, normal in zip(mesh.faces, normals):
        p = xy[face]
        x0, y0 = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        x0, x1 = max(0, x0), min(width - 1, x1)
        y0, y1 = max(0, y0), min(height - 1, y1)
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = p[0]; bx, by = p[1]; cx, cy = p[2]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        w0 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denominator
        w1 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denominator
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        z = w0 * xyz[face[0], 2] + w1 * xyz[face[1], 2] + w2 * xyz[face[2], 2]
        target_z = zbuffer[y0:y1 + 1, x0:x1 + 1]
        update = inside & (z > target_z)
        target_z[update] = z[update]
        # Two-sided neutral geometry shading. Texture cannot masquerade as geometry.
        if normal[2] < 0:
            normal = -normal
        intensity = int(np.clip(65 + 180 * max(0., float(normal @ light)), 1, 250))
        shade[y0:y1 + 1, x0:x1 + 1][update] = intensity
    mask = np.isfinite(zbuffer)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rgb = np.full((height, width, 3), 22, np.uint8)
    rgb[mask] = np.repeat(shade[mask, None], 3, axis=1)
    image_path = out / f"{view}.png"
    mask_path = out / f"{view}_silhouette.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    projection = np.vstack([right * scale, -upward * scale, depth])
    offset = [pad - lo[0] * scale, pad + hi[1] * scale, 0.]
    metadata.update(view=view, longitudinal_axis=longitudinal, dorsal_axis=dorsal,
                    raster_width=width, raster_height=height, projection_matrix=projection.tolist(),
                    projection_offset=offset, projection_equation="[pixel_x,pixel_y,depth] = matrix @ source_xyz + offset",
                    image=str(image_path.resolve()), silhouette=str(mask_path.resolve()),
                    color_status="geometry-only neutral shading; not a base-color texture render")
    write_json(out / f"{view}_projection.json", metadata)
    return clean(metadata)
