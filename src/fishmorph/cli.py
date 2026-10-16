"""Command line entry point. Every scientific operation also has a Python API."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .storage import Store
from .models import Taxon, Specimen, ImageMetadata, Annotation, AnalysisConfig, ProfileConfig
from .util import dumps, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parser():
    p=argparse.ArgumentParser(prog="fishmorph",description="Local fish morphometrics, specimen profiles and mesh comparison.")
    p.add_argument("--project",default="./fishmorph-data",help="Project directory containing SQLite and source imagery.")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("init",help="Create a project; seed names only, not biological measurements.")
    sub.add_parser("demo",help="Build synthetic fixtures, profiles and a mesh comparison.")
    serve=sub.add_parser("serve",help="Open the local annotation workbench.");serve.add_argument("--port",type=int,default=8765)
    taxa=sub.add_parser("taxa");taxa.add_argument("--add-json",help="Sourced Taxon JSON file")
    sp=sub.add_parser("specimens");sp.add_argument("--save-json",help="Specimen JSON file; edits append metadata revisions")
    ingest=sub.add_parser("import",help="Import photos of one known or unresolved individual.")
    ingest.add_argument("paths",nargs="+");ingest.add_argument("--specimen",required=True);ingest.add_argument("--metadata-json")
    annotate=sub.add_parser("annotate",help="Import a manual annotation JSON; use serve for graphical editing.")
    annotate.add_argument("image_id");annotate.add_argument("annotation_json")
    analyze=sub.add_parser("measure");analyze.add_argument("image_id");analyze.add_argument("--out");analyze.add_argument("--config-json")
    profiles=sub.add_parser("profile");prof=profiles.add_subparsers(dest="action",required=True)
    build=prof.add_parser("build");build.add_argument("--taxon",required=True);build.add_argument("--config-json");build.add_argument("--out")
    export=prof.add_parser("export");export.add_argument("profile_id");export.add_argument("--out",required=True);export.add_argument("--generation-constraints",action="store_true")
    mesh=sub.add_parser("mesh");meshes=mesh.add_subparsers(dest="action",required=True)
    for cmd in ("render","import"):
        m=meshes.add_parser(cmd);m.add_argument("path");m.add_argument("--longitudinal",default="+x");m.add_argument("--dorsal",default="+z")
        m.add_argument("--view",default="lateral_a",choices=["lateral_a","lateral_b","dorsal","ventral","frontal"])
        m.add_argument("--resolution",type=int,default=768)
        if cmd=="import":m.add_argument("--specimen",required=True)
        else:m.add_argument("--out",required=True)
    validate=sub.add_parser("validate",help="Compare an annotated candidate IMAGE, including a mesh render, with a profile.")
    validate.add_argument("image_id");validate.add_argument("--profile",required=True);validate.add_argument("--out");validate.add_argument("--exploratory-color",action="store_true")
    report=sub.add_parser("report");report.add_argument("document_id");report.add_argument("--out",required=True)
    schema=sub.add_parser("schema");schema.add_argument("kind",choices=["annotation","specimen","image","taxon","analysis-config","profile-config"]);schema.add_argument("--out")
    return p


def run(args):
    store=Store(args.project)
    if args.command=="init":return {"project":str(store.init())}
    if args.command=="demo":
        from .demo import build_demo
        return build_demo(store)
    if args.command=="serve":
        import uvicorn
        from .webapp import create_app
        uvicorn.run(create_app(args.project),host="127.0.0.1",port=args.port,log_level="info")
        return None
    if args.command=="schema":
        classes={"annotation":Annotation,"specimen":Specimen,"image":ImageMetadata,"taxon":Taxon,"analysis-config":AnalysisConfig,"profile-config":ProfileConfig}
        result=classes[args.kind].model_json_schema()
        if args.out:write_json(args.out,result);return {"written":args.out}
        return result
    if args.command=="taxa":return store.add_taxon(Taxon.model_validate(read_json(args.add_json))) if args.add_json else store.taxa()
    if args.command=="specimens":return store.save_specimen(Specimen.model_validate(read_json(args.save_json))) if args.save_json else store.specimens()
    if args.command=="import":
        metadata=ImageMetadata.model_validate(read_json(args.metadata_json)) if args.metadata_json else ImageMetadata()
        return [store.ingest(path,args.specimen,metadata) for path in args.paths]
    if args.command=="annotate":
        data=read_json(args.annotation_json)
        if data.get("image_id") not in (None,args.image_id):raise ValueError("Annotation image_id differs from the selected image.")
        data["image_id"]=args.image_id
        return store.save_annotation(Annotation.model_validate(data))
    if args.command=="measure":
        from .exports import export_analysis
        cfg=AnalysisConfig.model_validate(read_json(args.config_json)) if args.config_json else AnalysisConfig()
        result=store.analyze(args.image_id,cfg)
        if args.out:export_analysis(store,result,args.out);return {"analysis_id":result["id"],"output":str(Path(args.out).resolve())}
        return result
    if args.command=="profile":
        from .statistics import build_profiles
        from .exports import export_document, plot_profile
        if args.action=="build":
            cfg=ProfileConfig.model_validate(read_json(args.config_json)) if args.config_json else ProfileConfig()
            result=build_profiles(store,args.taxon,cfg)
            for profile in result["profiles"]:
                out=(Path(args.out) if args.out else store.root/"outputs")/profile["id"]
                export_document(profile,out);plot_profile(profile,out)
            return {"profile_ids":[p["id"] for p in result["profiles"]],"excluded":result["excluded"]}
        result=store.document(args.profile_id)
        if args.generation_constraints:
            from .comparison import generation_constraints
            write_json(args.out,generation_constraints(result))
        else:write_json(args.out,result)
        return {"written":args.out}
    if args.command=="mesh":
        if args.action=="render":
            from .mesh import render_mesh
            return render_mesh(args.path,args.out,args.longitudinal,args.dorsal,args.view,args.resolution)
        from .workflows import import_model
        return import_model(store,args.path,args.specimen,args.longitudinal,args.dorsal,args.view,args.resolution)
    if args.command=="validate":
        from .workflows import validate_image
        from .exports import export_document
        result=validate_image(store,args.profile,args.image_id,args.exploratory_color)
        if args.out:export_document(result,args.out)
        return result
    if args.command=="report":
        from .exports import export_document
        result=store.document(args.document_id);export_document(result,args.out)
        return {"output":str(Path(args.out).resolve())}
    raise ValueError("Unknown command")


def main():
    try:
        result=run(parser().parse_args())
        if result is not None:print(dumps(result,indent=2))
    except (ValueError,OSError,json.JSONDecodeError) as exc:
        print(f"fishmorph: {exc}",file=sys.stderr)
        raise SystemExit(2)

if __name__=="__main__":main()
