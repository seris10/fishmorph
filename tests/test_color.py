import numpy as np
import pytest
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from fishmorph.color import color_analysis, linearize_srgb, candidate_markings, regional_delta_e
from fishmorph.demo import fixture
from fishmorph.geometry import Frame, layers, visible_points
from fishmorph.models import AnalysisConfig


def test_srgb_transfer_known_endpoints():
    assert linearize_srgb(np.array([0.,.04045,1.])) == pytest.approx([0.,.04045/12.92,1.])

def test_lab_white_and_equal_color_distance():
    assert rgb2lab(np.ones((1,1,3)))[0,0] == pytest.approx([100,0,0], abs=.01)
    x={'region':{'mean_lab':[52,14,-12]}}
    assert regional_delta_e(x,x)['region'] == 0

def test_source_pixels_traceable_and_hands_missing(config):
    im,a=fixture(occluded=True);rgb=np.asarray(im);out=color_analysis(rgb,a,config)
    atlas=out['atlas'];xy=np.asarray(atlas['source_xy']);ok=np.asarray(atlas['valid'])
    mm=layers(a,rgb.shape[:2]);source=xy[ok]
    assert not mm['occlusion'][source[:,1],source[:,0]].any()
    assert np.asarray(atlas['rgb'],float)[ok] == pytest.approx(rgb[source[:,1],source[:,0]]/255)
    assert (xy[~ok]==-1).all() and 0<atlas['coverage']<1

def test_head_direction_color_registration(config):
    im,a=fixture();mirror,b=fixture(mirrored=True)
    x=color_analysis(np.asarray(im),a,config)['atlas'];y=color_analysis(np.asarray(mirror),b,config)['atlas']
    ok=np.asarray(x['valid']) & np.asarray(y['valid'])
    assert np.asarray(x['rgb'],float)[ok] == pytest.approx(np.asarray(y['rgb'],float)[ok],abs=1/255)

def test_raw_color_survives_unsuitable_morphometrics(example,config):
    im,a=example;a.quality.curved=True
    out=color_analysis(np.asarray(im),a,config)
    assert out['status']=='observed_image_color' and out['atlas'] is None
    assert out['regions']['head']['n_source_pixels']>0

def test_clipped_colors_excluded(example,config):
    im,a=example;rgb=np.array(im);rgb[150:180,220:250]=255
    out=color_analysis(rgb,a,config)
    assert out['clipped_visible_fraction']>0
    assert out['usable_source_pixels']<out['visible_source_pixels']

def test_color_review_required(example,config):
    im,a=example;a.quality.color_reviewed=False
    assert color_analysis(np.asarray(im),a,config)['status']=='unavailable'

def test_lighting_change_not_falsely_recovered(example,config):
    im,a=example;rgb=np.array(im);a.quality.curved=True
    x=color_analysis(rgb,a,config)['regions']['whole_visible_fish']['mean_lab']
    changed=np.clip(rgb.astype(float)*[.75,.95,1.02],1,254).astype(np.uint8)
    y=color_analysis(changed,a,config)['regions']['whole_visible_fish']['mean_lab']
    assert np.linalg.norm(np.asarray(x)-y)>5

def test_three_synthetic_dark_spots_counted_source_space():
    rgb=np.full((120,260,3),.65)
    valid=np.zeros((120,260),bool);valid[20:100,15:245]=True
    yy,xx=np.indices(valid.shape)
    for x,y in [(60,45),(130,75),(195,45)]:rgb[(xx-x)**2+(yy-y)**2<=4**2]=.12
    frame=Frame.from_points({'snout':np.array([15,60]),'caudal_base':np.array([245,60]),'dorsal_reference':np.array([60,20])})
    marks,mask,density=candidate_markings(rgb,valid,frame,AnalysisConfig(light_spot_contrast=30))
    dark=[m for m in marks if m['kind']=='dark']
    assert len(dark)==3
    assert sum(d['dark_count'] for d in density)==3
    assert mask.sum()==3*49

def test_missing_color_is_null_not_black(example,config):
    im,a=example;out=color_analysis(np.asarray(im),a,config)['atlas']
    valid=np.asarray(out['valid']);rgb=np.asarray(out['rgb'],float)
    assert np.isnan(rgb[~valid]).all()
