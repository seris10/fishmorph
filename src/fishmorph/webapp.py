"""Local-only web interface. No telemetry, cloud calls, account or API key required."""
from __future__ import annotations
import hashlib
import secrets
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .storage import Store
from .models import (Model, Specimen, Taxon, Annotation, ImageMetadata,
                     AnalysisConfig, ProfileConfig)
from .anatomy import LANDMARKS
from .statistics import build_profiles
from .exports import report_html, export_document
from .workflows import import_model, validate_image
from .vision import grabcut

SESSION_COOKIE = "fishmorph_session"


class BuildRequest(Model):
    taxon_id: str
    config: ProfileConfig = ProfileConfig()

class CompareRequest(Model):
    profile_id: str
    image_id: str
    exploratory_color: bool = False

class SegmentRequest(Model):
    image_id: str
    rectangle: list[int]


def create_app(project="./fishmorph-data"):
    store = Store(project)
    if not store.db_path.exists():
        store.init()
    token = secrets.token_urlsafe(32)
    app = FastAPI(title="FishMorph local", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = token
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        path = request.url.path
        # Image and report URLs are loaded by <img> and <iframe>, which cannot send a header,
        # so those two read routes accept the session cookie set when the app page is opened.
        supplied = request.headers.get("x-fishmorph-token", "")
        if not supplied and path.startswith(("/image/", "/report/")):
            supplied = request.cookies.get(SESSION_COOKIE, "")
        if path.startswith(("/api/", "/image/", "/report/")) and not secrets.compare_digest(supplied, token):
            return JSONResponse({"detail": "Local session token required. Open the app from its local URL."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = HTMLResponse((static / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token))
        page.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict", path="/")
        return page

    @app.get("/api/state")
    def state():
        profiles = [{k: p[k] for k in ("id", "taxon", "group", "n_individuals", "n_images")}
                    for p in store.documents("profile")]
        return {"project": str(store.root), "taxa": store.taxa(), "specimens": store.specimens(),
                "images": store.images(), "profiles": profiles, "landmarks": LANDMARKS}

    @app.post("/api/specimens")
    def specimen(body: Specimen):
        return store.save_specimen(body)

    @app.post("/api/taxa")
    def taxon(body: Taxon):
        return store.add_taxon(body)

    @app.get("/api/images/{image_id}")
    def image_detail(image_id: str):
        image = store.get_image(image_id)
        return {"image": image, "specimen": store.get_specimen(image["specimen_id"]),
                "annotation": store.annotation(image_id), "history": store.annotation_history(image_id)}

    @app.get("/image/{image_id}")
    def image_file(image_id: str):
        image = store.get_image(image_id)
        return FileResponse(store.root / image["display"], media_type="image/png")

    async def limited_body(request):
        chunks, total = [], 0
        async for part in request.stream():
            total += len(part)
            if total > 100_000_000:
                raise HTTPException(413, "Upload limit: 100 MB")
            chunks.append(part)
        return b"".join(chunks)

    @app.post("/api/import")
    async def import_image(request: Request, specimen_id: str, filename: str, source: str = "local file", license: str = "unknown"):
        raw = await limited_body(request)
        return store.ingest_bytes(raw, filename, specimen_id, ImageMetadata(source=source, license=license))

    @app.post("/api/mesh")
    async def mesh(request: Request, specimen_id: str, longitudinal: str = "+x", dorsal: str = "+z", view: str = "lateral_a"):
        raw = await limited_body(request)
        if raw[:4] != b"glTF":
            raise ValueError("Web upload requires a self-contained GLB. Use the CLI for local glTF or OBJ.")
        path = store.root / "outputs" / (hashlib.sha256(raw).hexdigest() + ".glb")
        path.write_bytes(raw)
        # A local user explicitly requested this bounded CPU operation.
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(import_model, store, path, specimen_id, longitudinal, dorsal, view)

    @app.post("/api/annotations")
    def annotation(body: Annotation):
        return store.save_annotation(body)

    @app.post("/api/analyze/{image_id}")
    def analyze(image_id: str, body: AnalysisConfig | None = None):
        return store.analyze(image_id, body)

    @app.post("/api/segment")
    def segment(body: SegmentRequest):
        if len(body.rectangle) != 4:
            raise ValueError("Rectangle must contain x,y,width,height.")
        return grabcut(store.rgb(body.image_id), body.rectangle)

    @app.post("/api/profiles")
    def build(body: BuildRequest):
        result = build_profiles(store, body.taxon_id, body.config)
        for profile in result["profiles"]:
            export_document(profile, store.root / "outputs" / profile["id"])
        return {"profile_ids": [p["id"] for p in result["profiles"]], "excluded": result["excluded"]}

    @app.get("/api/documents/{document_id}")
    def document(document_id: str):
        return store.document(document_id)

    @app.post("/api/compare")
    def comparison(body: CompareRequest):
        return validate_image(store, body.profile_id, body.image_id, body.exploratory_color)

    @app.get("/report/{document_id}", response_class=HTMLResponse)
    def report(document_id: str):
        return report_html(store.document(document_id))

    return app
