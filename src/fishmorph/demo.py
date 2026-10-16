"""Deterministic synthetic fixtures. Nothing here represents a measured fish species."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from .models import Taxon, Specimen, ImageMetadata, Annotation, Landmark, Quality
from .geometry import polygon_mask
from .storage import Store
from .statistics import build_profiles
from .workflows import import_model, validate_image
from .exports import export_analysis, export_document, plot_profile
from .util import write_json


def height_at(u, depth=.13):
    return depth * np.maximum(0, np.sin(np.pi * np.asarray(u)))**.75 + .024 * np.asarray(u)


def fixture(index=0, depth=.13, scale=1., mirrored=False, occluded=False):
    width, height = 820, 310
    snout = np.array([90., 164.]); sl=560.
    def xy(u, y):
        return [float(snout[0]+sl*u), float(snout[1]+sl*y)]
    u=np.linspace(0,1,95); rad=height_at(u,depth)
    body=[xy(x,-r) for x,r in zip(u,rad)] + [xy(x,r) for x,r in zip(u[::-1],rad[::-1])]
    fins=[ [xy(.42,-height_at(.42,depth)),xy(.50,-height_at(.50,depth)-.10),xy(.60,-height_at(.60,depth))],
           [xy(.70,height_at(.70,depth)),xy(.78,height_at(.78,depth)+.08),xy(.85,height_at(.85,depth))],
           [xy(.22,.03),xy(.36,.14),xy(.28,.04)],
           [xy(.54,.08),xy(.65,.18),xy(.59,.08)],
           [xy(.83,-height_at(.83,depth)),xy(.86,-height_at(.86,depth)-.028),xy(.89,-height_at(.89,depth))],
           [xy(1,-.024),xy(1.22,-.105),xy(1.14,0),xy(1.22,.105),xy(1,.024)] ]
    img=np.full((height,width,3),[231,233,223],np.uint8)
    bm=polygon_mask([body],(height,width)); vm=polygon_mask([body,*fins],(height,width))
    yy,xx=np.indices((height,width)); uu=(xx-snout[0])/sl
    rr=height_at(np.clip(uu,0,1),depth)*sl
    vv=np.clip((yy-snout[1]+rr)/(2*np.maximum(rr,.1)),0,1)
    top=np.array([65+index,89,68]);bottom=np.array([204,198-index,170])
    pigment=top[None,None,:]*(1-vv[...,None])+bottom[None,None,:]*vv[...,None]
    band=np.exp(-((vv-.52)/.12)**2)[...,None]
    pigment+=band*np.array([26.,-8.,-6.])
    img[vm]=[153,155,123];img[bm]=np.clip(pigment[bm],1,253).astype(np.uint8)
    image=Image.fromarray(img);draw=ImageDraw.Draw(image)
    rng=np.random.default_rng(800+index)
    for _ in range(18):
        x=float(rng.uniform(.34,.96)); r=float(height_at(x,depth));y=float(rng.uniform(-.72,.6)*r)
        p=xy(x,y);radius=float(rng.uniform(2.3,4.2));draw.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),fill=(39,46,37))
    eye=xy(.12,-.025);draw.ellipse((eye[0]-6,eye[1]-6,eye[0]+6,eye[1]+6),fill=(32,35,30))
    draw.text((20,18),f"SYNTHETIC SOFTWARE FIXTURE {index:02d} / NOT A BIOLOGICAL REFERENCE",fill=(85,102,83))
    points={"snout":xy(0,0),"caudal_base":xy(1,0),"dorsal_reference":xy(.28,-height_at(.28,depth)),
            "operculum":xy(.235+index*.0015,0),"eye_anterior":[eye[0]-6,eye[1]],
            "eye_posterior":[eye[0]+6,eye[1]],"eye_center":eye,
            "dorsal_origin":fins[0][0],"dorsal_apex":fins[0][1],"dorsal_insertion":fins[0][2],
            "adipose_origin":fins[4][0],"adipose_apex":fins[4][1],
            "anal_origin":fins[1][0],"anal_apex":fins[1][1],"anal_insertion":fins[1][2],
            "pectoral_origin":fins[2][0],"pectoral_tip":fins[2][1],
            "pelvic_origin":fins[3][0],"pelvic_tip":fins[3][1],
            "peduncle_anterior":xy(.88,0),"peduncle_dorsal":xy(.98,-height_at(.98,depth)),
            "peduncle_ventral":xy(.98,height_at(.98,depth)),
            "tail_tip":xy(1.22,-.105),"tail_upper":xy(1.22,-.105),"tail_lower":xy(1.22,.105),"tail_fork":xy(1.14,0)}
    masks={"visible":[body,*fins],"body":[body]}
    if occluded:
        p=[xy(.57,-.015),xy(.66,-.015),xy(.66,.23),xy(.57,.23)]
        draw.polygon([tuple(v) for v in p],fill=(195,148,115));masks["occlusion"]=[p]
        # Additional interior-only occlusion leaves observed boundaries usable.
        interior=[xy(.35,-.015),xy(.40,-.015),xy(.40,.025),xy(.35,.025)]
        draw.polygon([tuple(v) for v in interior],fill=(195,148,115));masks["occlusion"].append(interior)
    regions={"head":[xy(0,-.14),xy(.27,-.14),xy(.27,.14),xy(0,.14)],
             "operculum":[xy(.19,-.11),xy(.27,-.11),xy(.27,.11),xy(.19,.11)],
             "throat":[xy(.05,.025),xy(.22,.025),xy(.22,.12),xy(.05,.12)]}
    regions={k:[v] for k,v in regions.items()}
    if mirrored:
        image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if scale != 1:
        image=image.resize((int(round(width*scale)),int(round(height*scale))),Image.Resampling.BICUBIC)
    def transform(p):
        return [(width-1-p[0] if mirrored else p[0])*scale,p[1]*scale]
    transformed_masks={k:[[transform(p) for p in poly] for poly in ps] for k,ps in masks.items()}
    transformed_regions={k:[[transform(p) for p in poly] for poly in ps] for k,ps in regions.items()}
    annotation=Annotation(image_id="fixture",author="synthetic_fixture_generator",note="Analytical test geometry, not fish anatomy.",
                          quality=Quality(view="lateral_left",geometry_reviewed=True,color_reviewed=True,mouth="closed",fins="natural",blur="low"),
                          landmarks={k:Landmark(x=transform(v)[0],y=transform(v)[1],sigma_px=.4*scale,method="known_synthetic_fixture") for k,v in points.items()},
                          masks=transformed_masks,regions=transformed_regions,
                          structures={"adipose_fin":"present"})
    return image,annotation


def synthetic_mesh(path, depth=.18):
    xs=np.linspace(0,1,50);angles=np.linspace(0,2*np.pi,40,endpoint=False)
    verts=[]
    for x in xs:
        r=float(height_at(x,depth))
        for angle in angles:
            verts.append([x,.45*r*np.cos(angle),r*np.sin(angle)])
    faces=[];n=len(angles)
    for j in range(len(xs)-1):
        for k in range(n):
            a=j*n+k;b=j*n+(k+1)%n;c=(j+1)*n+k;d=(j+1)*n+(k+1)%n
            faces.extend([[a,b,c],[b,d,c]])
    def triangle(points):
        start=len(verts);verts.extend(points);faces.append([start,start+1,start+2])
    triangle([[.42,0,height_at(.42,depth)],[.50,0,height_at(.50,depth)+.10],[.60,0,height_at(.60,depth)]])
    triangle([[.70,0,-height_at(.70,depth)],[.78,0,-height_at(.78,depth)-.08],[.85,0,-height_at(.85,depth)]])
    triangle([[1,0,.024],[1.22,0,.105],[1.14,0,0]])
    triangle([[1,0,-.024],[1.22,0,-.105],[1.14,0,0]])
    mesh=trimesh.Trimesh(vertices=np.asarray(verts),faces=np.asarray(faces),process=False)
    mesh.export(path)
    return mesh


def build_demo(store: Store, count=8):
    store.init()
    taxon=Taxon(id="synthetic_test_fish",scientific_name="Synthetic geometry fixture (not a taxon)",
                common_name="Synthetic test fish",status="synthetic",profile_allowed=True,
                note="Every measurement is an authored test quantity. Never use this profile to validate a biological species.")
    store.add_taxon(taxon)
    source=store.root/"demo_sources";source.mkdir(exist_ok=True)
    images=[]
    for i in range(count):
        sid=f"demo_{i:02d}"
        specimen=Specimen(id=sid,taxon_id=taxon.id,identity_status="documented",identity_evidence="Distinct synthetic fixture seed.",
                          material="synthetic",standard_length_mm=150+i*20,length_evidence="Authored test scale, not measured biology.")
        store.save_specimen(specimen)
        for repeat in range(2 if i==0 else 1):
            image,annotation=fixture(i,depth=.125+.0015*i,scale=1 if repeat==0 else .8,
                                     mirrored=bool(repeat),occluded=(i==2))
            path=source/f"synthetic_{i:02d}_{repeat}.png";image.save(path)
            record=store.ingest(path,sid,ImageMetadata(source="FishMorph synthetic fixture generator",license="MIT synthetic test image",
                                                     photometric_status="synthetic",photometric_protocol="synthetic_srgb_v1"))
            annotation.image_id=record["id"]
            previous=store.annotation(record["id"])
            if previous is None or previous["annotation"] != annotation.model_dump():store.save_annotation(annotation)
            images.append(record)
    built=build_profiles(store,taxon.id)
    for profile in built["profiles"]:
        out=store.root/"outputs"/profile["id"];export_document(profile,out);plot_profile(profile,out)
    export_analysis(store,store.analyze(images[0]["id"]),store.root/"outputs"/"example_analysis")
    model_id="demo_generated_model"
    store.save_specimen(Specimen(id=model_id,taxon_id=taxon.id,identity_status="documented",identity_evidence="Generated analytic mesh fixture.",material="model"))
    mesh_path=source/"synthetic_candidate.glb";synthetic_mesh(mesh_path)
    model=import_model(store,mesh_path,model_id,resolution=768)
    meta=model["rendering"];matrix=np.array(meta["projection_matrix"]);offset=np.array(meta["projection_offset"])
    def px(p):return (matrix@np.asarray(p)+offset)[:2].tolist()
    u=np.linspace(0,1,95);rad=height_at(u,.18)
    body=[px([x,0,r]) for x,r in zip(u,rad)]+[px([x,0,-r]) for x,r in zip(u[::-1],rad[::-1])]
    world={"snout":[0,0,0],"caudal_base":[1,0,0],"dorsal_reference":[.28,0,height_at(.28,.18)],
           "operculum":[.33,0,0],"dorsal_origin":[.42,0,height_at(.42,.18)],
           "dorsal_insertion":[.60,0,height_at(.60,.18)],"dorsal_apex":[.50,0,height_at(.50,.18)+.10],
           "anal_origin":[.70,0,-height_at(.70,.18)],"anal_insertion":[.85,0,-height_at(.85,.18)],
           "anal_apex":[.78,0,-height_at(.78,.18)-.08],"tail_tip":[1.22,0,.105],
           "tail_upper":[1.22,0,.105],"tail_lower":[1.22,0,-.105],"tail_fork":[1.14,0,0]}
    draft=Annotation.model_validate(store.annotation(model["image"]["id"])["annotation"])
    draft.author="synthetic_fixture_generator";draft.quality=Quality(view="lateral_left",geometry_reviewed=True,mouth="closed",fins="natural",blur="low")
    draft.masks["body"]=[body]
    draft.landmarks={k:Landmark(x=px(p)[0],y=px(p)[1],method="known_synthetic_anchor") for k,p in world.items()}
    draft.note="Synthetic test anchors from known geometry. The opercular anchor is authored, NOT anatomical identification on a real fish mesh. Deliberately deeper body and shifted head endpoint."
    store.save_annotation(draft)
    result=validate_image(store,built["profiles"][0]["id"],model["image"]["id"])
    manifest={"synthetic_only":True,"reference_profile_id":built["profiles"][0]["id"],
              "comparison_id":result["id"],"candidate_image_id":model["image"]["id"],
              "reference_image_ids":[i["id"] for i in images]}
    write_json(store.root/"demo_manifest.json",manifest)
    return manifest
