import copy
import io
import json
import subprocess
import sys
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from fishmorph.mesh import render_mesh, load_mesh
from fishmorph.models import Specimen, Annotation
from fishmorph.statistics import build_profiles
from fishmorph.comparison import compare
from fishmorph.demo import fixture
from fishmorph.exports import export_analysis, report_html
from fishmorph.webapp import create_app
from fishmorph.vision import grabcut
from conftest import add_observation

@pytest.mark.parametrize('view',['lateral_a','lateral_b','dorsal','ventral','frontal'])
def test_mesh_scene_transform_and_orthographic_render(tmp_path,view):
    box=trimesh.creation.box(extents=[2,.4,.8]);scene=trimesh.Scene();transform=np.eye(4);transform[:3,3]=[4,3,2]
    scene.add_geometry(box,transform=transform);path=tmp_path/'box.glb';scene.export(path)
    mesh,meta=load_mesh(path);assert mesh.centroid==pytest.approx([4,3,2],abs=1e-6)
    out=render_mesh(path,tmp_path/view,view=view,resolution=160)
    mask=np.asarray(Image.open(out['silhouette']));assert (mask>0).sum()>100
    rgb=np.array(Image.open(out['image']));assert (rgb[:,:,0]==rgb[:,:,1]).all()
    matrix=np.array(out['projection_matrix']);offset=np.array(out['projection_offset'])
    pixels=mesh.vertices@matrix.T+offset
    assert pixels[:,0].min()>=15.99 and pixels[:,1].min()>=15.99

def test_invalid_axes_refused(tmp_path):
    path=tmp_path/'cube.glb';trimesh.creation.box().export(path)
    with pytest.raises(ValueError,match='perpendicular'):render_mesh(path,tmp_path/'out',longitudinal='+x',dorsal='-x')

def test_comparison_is_descriptive_and_independent(store,config):
    original=add_observation(store,'reference',0)
    profile=build_profiles(store,'test',analysis_config=config)['profiles'][0]
    image,ann=fixture(9,depth=.18);ann.landmarks['operculum'].x=90+.34*560
    candidate=add_observation(store,'candidate',9,annotation=ann,image=image)
    result=compare(profile,store.analyze(candidate['id'],config))
    head=next(x for x in result['measurements'] if x['measurement']=='head_length_sl')
    assert head['status']=='outside_observed_sample_small_n'
    assert result['contour']['body_envelope_iou_on_shared_support']<.9
    assert 'accuracy' not in result and result['multivariate']['status']=='unavailable'
    with pytest.raises(ValueError,match='contributed'):compare(profile,store.analyze(original['id'],config))
    analysis=store.analyze(candidate['id'],config)
    analysis['morphology']['quality']['mouth']='open'
    with pytest.raises(ValueError,match='articulation'):compare(profile,analysis)

def test_uncalibrated_color_abstains_and_hidden_fin_not_absence(store,config):
    add_observation(store,'reference',0);p=build_profiles(store,'test',analysis_config=config)['profiles'][0]
    im=add_observation(store,'candidate',1);a=store.analyze(im['id'],config)
    a['image_metadata']['photometric_status']='uncalibrated'
    a['morphology']['structures']={}
    r=compare(p,a)
    assert r['color']['status']=='abstained' and r['structures'][0]['status']=='not_comparable'
    a['morphology']['structures']={'adipose_fin':'absent_observed'}
    assert compare(p,a)['structures'][0]['status']=='differs_from_observed_reference'

def test_export_json_csv_png_npz_and_escape(store,config,tmp_path):
    im=add_observation(store);a=store.analyze(im['id'],config)
    out=tmp_path/'exports';export_analysis(store,a,out)
    assert list(out.glob('*.csv')) and list(out.glob('*.png')) and list(out.glob('*.json'))
    for f in out.glob('*.json'):json.loads(f.read_text())
    for f in out.glob('*.npz'):
        with np.load(f,allow_pickle=False) as data:assert data.files
    profile=build_profiles(store,'test',analysis_config=config)['profiles'][0]
    profile['group']['population']='<script>alert(1)</script>'
    html=report_html(profile);assert '<script>alert(1)' not in html and '&lt;script&gt;' in html

def test_local_api_token_catalog_save_and_measure(store,config):
    im=add_observation(store);app=create_app(store.root)
    with TestClient(app) as client:
        assert client.get('/api/state').status_code==403
        assert client.get('/').status_code==200
        assert client.get('/static/app.js').status_code==200
        client.headers['x-fishmorph-token']=app.state.token
        assert len(client.get('/api/state').json()['images'])==1
        assert client.post('/api/analyze/'+im['id'],json=config.model_dump()).status_code==200
        assert client.get('/api/images/missing').status_code==400
        assert client.get('/api/state',headers={'host':'evil.example'}).status_code==400
        annotation=store.annotation(im['id'])['annotation'];annotation['author']='api test'
        assert client.post('/api/annotations',json=annotation).status_code==200
        assert len(store.annotation_history(im['id']))==2

def test_grabcut_manual_assist_and_invalid_box():
    rgb=np.full((80,120,3),220,np.uint8);rgb[20:60,25:95]=[45,75,55]
    result=grabcut(rgb,[15,10,90,60],iterations=2)
    assert result['polygons'] and 'unreviewed' in result['method']
    with pytest.raises(ValueError):grabcut(rgb,[0,0,120,80])

def test_cli_init_and_help(tmp_path):
    command=[sys.executable,'-m','fishmorph','--project',str(tmp_path/'cli')]
    assert subprocess.run(command+['init'],capture_output=True).returncode==0
    out=subprocess.run(command+['--help'],capture_output=True,text=True)
    assert out.returncode==0 and 'profile' in out.stdout

def test_changed_measurement_protocol_refused(store,config):
    add_observation(store,'reference',0);p=build_profiles(store,'test',analysis_config=config)['profiles'][0]
    im=add_observation(store,'candidate',1);a=store.analyze(im['id'],config)
    a['provenance']['measurement_protocol']='a different anatomical definition'
    with pytest.raises(ValueError,match='protocol'):compare(p,a)
