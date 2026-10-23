# FishMorph

FishMorph is a local Python application for annotating fish photographs, calculating image measurements, summarizing observations by specimen, and comparing annotated mesh renders with reference profiles.

The demo uses synthetic images and geometry. Bring your own photographs, annotations and scale measurements for a real collection; the package does not include a biological reference dataset.

## Installation

Requires Python 3.11 or newer. Create an isolated environment from this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate with `.venv\Scripts\activate`. Installation downloads Python dependencies. The installed analysis and web interface run locally without an API key or model download.

## Run

Generate a synthetic demonstration project, then start the web interface:

```sh
fishmorph --project ./demo-data demo
fishmorph --project ./demo-data serve
```

Open `http://127.0.0.1:8765`. The server binds to loopback. Demo images, SQLite records, profiles, plots, and mesh comparisons are generated inside `demo-data/`; they are not a biological reference dataset. Re-running `demo` can add annotation and profile revisions.

For a new collection:

```sh
fishmorph --project ./my-fish init
fishmorph --project ./my-fish serve
```

1. Create a specimen and record identity and provenance. Record uncertain identifications as unknown and group repeated photographs under the same specimen.
2. Import photographs and mark the snout, caudal base, and dorsal reference. Additional anatomical landmarks enable additional measurements.
3. Draw the visible-fish mask and a separate fins-excluded body envelope. Mark occlusion, uncertain boundaries, and photometric artifacts separately.
4. Record view, articulation, and quality flags. Review geometry and color before saving an analysis.
5. Build profiles from compatible, reviewed observations. Unresolved identities are excluded by default.
6. Import a mesh as a specimen with `material: "model"`, declare its axes, and annotate the rendered view before comparing it with an independent compatible profile.

GrabCut produces a foreground proposal. Landmark placement, anatomical masks, specimen identification, and review remain manual.

## CLI and records

The global `--project` argument precedes the subcommand. Files in `examples/` are metadata templates and require actual source and identity evidence before use.

```sh
fishmorph --project ./my-fish taxa
fishmorph --project ./my-fish specimens --save-json examples/specimen.json
fishmorph --project ./my-fish import photo.jpg --specimen reference_001 --metadata-json examples/image_metadata.json
fishmorph --project ./my-fish annotate IMAGE_ID annotation.json
fishmorph --project ./my-fish measure IMAGE_ID --out ./measurement-output
fishmorph --project ./my-fish profile build --taxon yellowstone_cutthroat
fishmorph --project ./my-fish profile export PROFILE_ID --out profile.json
fishmorph --project ./my-fish mesh import fish.glb --specimen model_001 --longitudinal +x --dorsal +z
fishmorph --project ./my-fish validate ANNOTATED_RENDER_IMAGE_ID --profile PROFILE_ID --out ./comparison
fishmorph schema annotation --out annotation.schema.json
```

Create `model_001` with the model template before importing its mesh. `validate` takes an annotated image ID, including a mesh render; it cannot infer anatomical landmarks from an unannotated asset.

Original file bytes and decoded-raster hashes are retained. SQLite stores append-only specimen, annotation, analysis, and profile records. Analysis provenance includes source hashes, revisions, configuration, dependency versions, and implementation and measurement fingerprints. Exports include JSON records, CSV measurements, annotated images, PNG/NPZ color atlases, SVG plots, and HTML reports. JSON contracts are in `schemas/`.

## Measurement limits

Lengths are ratios relative to the annotated snout-to-caudal-base chord. Millimeters require an independently obtained standard length with evidence. Lateral images cannot establish unseen body width or correct arbitrary perspective, refraction, curvature, or occlusion. Point-annotation uncertainty excludes systematic landmark bias, lens error, boundary placement, and physical-scale uncertainty.

Compatible images are summarized within specimens before equal-specimen aggregation. Bootstrap intervals describe uncertainty in the sampled mean, not a range of biologically acceptable individuals. Unknown demographics remain explicit. Procrustes/PCA and log allometry are exploratory; allometry refuses extrapolation outside observed sizes.

Color outputs describe recorded image values. ICC conversion and sRGB linearization do not recover reflectance under unknown lighting. Pattern candidates can include shadows and texture artifacts. Neutral mesh renders normally lack comparable color evidence, so color comparison abstains.

Seeded taxonomic records contain names and sources only. The cutthroat seed records conflicting name usage, and the unresolved sculpin seed cannot build profiles. Locality, identification, licensing, and observation quality still require evidence. No automatic species identification, demographic inference, landmark detector, or biological validation is implemented.

Measurement definitions are in `src/fishmorph/anatomy.py`. Method references include [FAO measurement definitions](https://www.fao.org/4/f0752e/F0752E03.htm), [measurement error in morphometrics](https://doi.org/10.1007/s00427-016-0537-4), [image calibration](https://doi.org/10.1111/2041-210X.12439), and [pattern registration](https://doi.org/10.1111/2041-210X.12853). These references provide methodological context; their validation results do not validate this implementation.

## Tests

```sh
python -m pytest -q
```

The tests use synthetic fixtures to check geometry, occlusion handling, color arithmetic, specimen aggregation, revision storage, schema exports, mesh projections, and local API behavior.

Optional browser component checks use Playwright and an in-process FastAPI adapter:

```sh
python -m pip install -e '.[dev,browser]'
python -m playwright install chromium
python tests/browser_components.py
```

These checks generate their own synthetic collection and artifacts under `.browser-work/`. They exercise browser components, not a live browser-to-server network connection.

## License

The existing MIT license covers this implementation and its authored synthetic fixtures. User-supplied images, meshes, and datasets retain their respective rights; no third-party biological dataset is distributed here.
