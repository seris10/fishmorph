"""Portable JSON/CSV, source overlays, atlas files and standalone HTML reports."""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .models import Annotation
from .util import write_json


def overlay(rgb, annotation: Annotation, output):
    base = Image.fromarray(rgb).convert("RGBA")
    shade = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(shade)
    palette = {"visible": (42, 190, 165, 28), "body": (120, 190, 250, 38),
               "occlusion": (255, 130, 100, 105), "uncertain": (244, 206, 89, 90), "artifact": (183, 137, 230, 90)}
    for name, polygons in annotation.masks.items():
        for p in polygons:
            draw.polygon([tuple(x) for x in p], fill=palette[name], outline=palette[name][:3] + (230,), width=2)
    base = Image.alpha_composite(base, shade)
    draw = ImageDraw.Draw(base)
    for name, lm in annotation.landmarks.items():
        x, y = lm.x, lm.y
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="white", outline="black", width=1)
        draw.text((x + 6, y - 7), name, fill="white", stroke_width=2, stroke_fill="black")
    if {"snout", "caudal_base"} <= annotation.landmarks.keys():
        a, b = annotation.landmarks["snout"], annotation.landmarks["caudal_base"]
        draw.line([(a.x, a.y), (b.x, b.y)], fill="white", width=2)
    base.convert("RGB").save(output)


def atlas_image(atlas, output):
    rgb = np.array(atlas["rgb"], float)
    valid = np.asarray(atlas["valid"], bool)
    rgba = np.zeros((*valid.shape, 4), np.uint8)
    rgba[..., :3] = np.clip(np.nan_to_num(rgb) * 255, 0, 255).astype(np.uint8)
    rgba[..., 3] = valid * 255
    Image.fromarray(rgba).save(output)


def export_analysis(store, analysis, directory):
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "analysis.json", analysis)
    with (out / "measurements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["measurement", "value", "unit", "status", "annotation_sd", "rejection_reasons"])
        for name, row in analysis["morphology"]["measurements"].items():
            writer.writerow([name, row["value"], row["unit"], row["status"], row.get("sd_annotation"), "; ".join(row["reasons"])])
    ann = store.annotation(analysis["image_id"], analysis["annotation_id"])
    overlay(store.rgb(analysis["image_id"]), Annotation.model_validate(ann["annotation"]), out / "annotated.png")
    atlas = analysis["color"]["atlas"]
    if atlas is not None:
        atlas_image(atlas, out / "atlas.png")
        np.savez_compressed(out / "atlas.npz", rgb=np.asarray(atlas["rgb"], float),
                            lab=np.asarray(atlas["lab"], float), valid=np.asarray(atlas["valid"], bool),
                            source_xy=np.asarray(atlas["source_xy"], int), u=np.asarray(atlas["u"]), v=np.asarray(atlas["v"]))
    return out


def number(value):
    return "Not available" if value is None else f"{value:.5g}" if isinstance(value, (int, float)) else html.escape(str(value))


def contour_svg(contour):
    if not contour or contour.get("status") == "unavailable":
        return "<p>No compatible body-envelope comparison.</p>"
    u = contour["u"]
    arrays = [contour[k] for k in ("reference_dorsal", "reference_ventral", "candidate_dorsal", "candidate_ventral")]
    values = [v for row in arrays for v in row if v is not None]
    if not values:
        return ""
    low, high = min(values), max(values)
    scale = 210 / max(.1, high - low)
    paths = []
    for index, row in enumerate(arrays):
        segments, current = [], []
        for x, y in zip(u, row):
            if y is None:
                if current:
                    segments.append(current); current = []
            else:
                current.append(f"{40 + 700*x:.2f},{25 + (y-low)*scale:.2f}")
        if current:
            segments.append(current)
        for points in segments:
            dash = 'stroke-dasharray="5 4"' if index >= 2 else ""
            stroke = "#275e54" if index < 2 else "#a24e35"
            vertices = " ".join(points)
            paths.append(f'<polyline points="{vertices}" fill="none" stroke="{stroke}" stroke-width="2.2" {dash}/>')
    return '<svg viewBox="0 0 800 290" role="img" aria-label="SL-aligned body envelope comparison">' + ''.join(paths) + '<text x="40" y="275" font-size="12">u = 0: snout</text><text x="610" y="275" font-size="12">u = 1: caudal base</text></svg><p class="muted">Solid: reference mean. Dashed: candidate. Gaps remain missing.</p>'


def report_html(document):
    is_profile = "measurements" in document and isinstance(document["measurements"], dict)
    title = "Morphology profile" if is_profile else "Geometry & color comparison"
    group = document.get("group", {})
    if is_profile:
        table = '<tr><th>Measurement</th><th>N specimens</th><th>Mean</th><th>SD</th><th>Observed range</th><th>95% CI of mean</th></tr>'
        for name, r in document["measurements"].items():
            ci = r["mean_ci95"]
            table += f'<tr><td>{html.escape(name)}</td><td>{r["n_individuals"]}</td><td>{number(r["mean"])}</td><td>{number(r["sd"])}</td><td>{number(r["min"])} to {number(r["max"])}</td><td>{"Not estimated" if ci is None else number(ci[0])+" to "+number(ci[1])}</td></tr>'
        extra = '<h2>Interpretation</h2><p>One specimen contributes one summary, even when it has multiple photographs. Confidence intervals describe uncertainty in the mean, not a valid range for individual fish. Observed ranges are not biological tolerances.</p>'
        count = f'{document["n_individuals"]} specimens · {document["n_images"]} images'
        if document.get("material_warning"):
            extra = '<div class="warning">'+html.escape(document["material_warning"])+"</div>"+extra
    else:
        table = '<tr><th>Measurement</th><th>Candidate</th><th>Reference mean</th><th>Observed range</th><th>Finding</th></tr>'
        for r in document["measurements"]:
            interval = r.get("observed_range")
            table += f'<tr><td>{html.escape(r["measurement"])}</td><td>{number(r.get("candidate"))}</td><td>{number(r.get("reference_mean"))}</td><td>{"Not available" if not interval else number(interval[0])+" to "+number(interval[1])}</td><td>{html.escape(r["status"].replace("_", " "))}</td></tr>'
        extra = '<h2>Body envelope</h2>' + contour_svg(document.get("contour"))
        extra += '<h2>Color assessment</h2><pre>' + html.escape(json.dumps(document["color"], indent=2)) + '</pre>'
        extra += '<p>' + html.escape(document["conclusion"]) + '</p>'
        count = f'{document["reference_n_individuals"]} reference specimens'
    limitations = document.get("limitations", document.get("limits", []))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FishMorph | {title}</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;background:#f5f4ef;color:#24352f;margin:0}}main{{max-width:1100px;margin:auto;padding:48px 30px}}.eyebrow{{letter-spacing:.18em;text-transform:uppercase;font-size:12px;color:#557266}}h1{{font-size:38px;line-height:1.15;font-weight:500}}h2{{font-size:23px;margin-top:36px;font-weight:500}}.muted{{color:#69766f}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #e3e6df}}th{{background:#eaf0e9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px}}.warning{{padding:16px;border:1px solid #c89951;background:#fbf1df}}svg{{width:100%;background:white}}code{{overflow-wrap:anywhere}}footer{{margin-top:38px;border-top:1px solid #ccd5cd;padding-top:18px;font-size:12px}}</style>
<main><div class="eyebrow">FishMorph / Evidence record</div><h1>{title}</h1><p>{html.escape(count)}</p><p class="muted">{html.escape(' · '.join(str(v) for v in group.values()))}</p><table>{table}</table>{extra}<h2>Limits</h2>{''.join('<p>'+html.escape(l)+'</p>' for l in limitations)}<footer>Record <code>{html.escape(document.get('id','unsaved'))}</code><br>Measurements and descriptive diagnostics, not a biological-accuracy percentage.</footer></main></html>'''


def export_document(document, directory):
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "record.json", document)
    (out / "report.html").write_text(report_html(document), encoding="utf-8")
    return out


def plot_profile(profile, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    contour = profile.get("body_profile")
    if contour:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        u = np.asarray(contour["u"])
        for name in ("dorsal", "ventral"):
            y = np.asarray(contour[name]["mean"], float)
            sd = np.asarray(contour[name]["sd"], float)
            ax.plot(u, y, label=name + " mean")
            ax.fill_between(u, y - sd, y + sd, alpha=.15)
        ax.invert_yaxis(); ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="Longitudinal position / SL", ylabel="Ventral-positive position / SL",
               title="Mean body outline ± between-specimen SD (not confidence bounds)")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "body_profile.svg"); plt.close(fig)
    names = [n for n in profile["measurements"] if n in {"head_length_sl", "body_depth_50_sl", "eye_diameter_head"}]
    if names:
        fig, ax = plt.subplots(figsize=(9, 4))
        for j, name in enumerate(names):
            values = [i["measurements"][name]["value"] for i in profile["individuals"] if name in i["measurements"]]
            ax.scatter(np.full(len(values), j), values, alpha=.7)
        ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names])
        ax.set(ylabel="Ratio (denominator indicated in label)", title="Individual specimen summaries")
        fig.tight_layout(); fig.savefig(out / "measurement_distributions.svg"); plt.close(fig)
