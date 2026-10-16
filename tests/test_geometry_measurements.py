import numpy as np
import pytest
from fishmorph.demo import fixture, height_at
from fishmorph.geometry import Frame, visible_points, body_profile, interpolate_supported, scan_polygon
from fishmorph.measurements import measure
from fishmorph.models import Annotation


def test_known_lengths_and_fin_position(example):
    im, ann = example; m = measure(ann, (im.height, im.width))['measurements']
    assert m['standard_length_px']['value'] == pytest.approx(560)
    assert m['head_length_sl']['value'] == pytest.approx(.235)
    assert m['eye_diameter_head']['value'] == pytest.approx(12 / (560 * .235))
    assert m['dorsal_origin_u']['value'] == pytest.approx(.42)
    assert m['body_depth_50_sl']['value'] == pytest.approx(2 * height_at(.5, .13), abs=.0001)
    assert m['total_length_photo_projected_sl']['value'] == pytest.approx(1.22)
    assert 'total_length_mm' not in m

@pytest.mark.parametrize('scale,mirror', [(0.5,False),(2,False),(1,True),(.8,True)])
def test_scaling_and_head_right_invariance(scale, mirror):
    im, a = fixture(scale=scale, mirrored=mirror)
    m = measure(a, (im.height,im.width))
    assert m['measurements']['head_length_sl']['value'] == pytest.approx(.235)
    assert m['normalized_landmarks']['dorsal_reference'][1] < 0
    assert m['measurements']['head_length_sl']['sd_annotation'] == pytest.approx(
        measure(fixture()[1],(310,820))['measurements']['head_length_sl']['sd_annotation'], rel=1e-6)

@pytest.mark.parametrize('angle', [0, .6, 1.57, 3.14])
def test_frame_rotation_translation_and_inverse(angle):
    p = {'snout':np.array([0.,0.]), 'caudal_base':np.array([3.,0.]), 'dorsal_reference':np.array([1.,-1.])}
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    q={n: v@rot.T*5+[45,62] for n,v in p.items()}
    f=Frame.from_points(q)
    assert f.normalize(q['caudal_base']) == pytest.approx([1,0])
    assert f.normalize(q['dorsal_reference']) == pytest.approx([1/3,-1/3])
    points=np.array([[.3,-.1],[.8,.2]])
    assert f.normalize(f.inverse(points)) == pytest.approx(points)

@pytest.mark.parametrize('flag,value', [('view','oblique'),('view','dorsal'),('curved',True),('refraction',True),('severe_perspective',True),('mouth','open'),('geometry_reviewed',False),('blur','severe')])
def test_geometry_quality_refusals(example, flag, value):
    im, a=example; setattr(a.quality,flag,value)
    m=measure(a,(im.height,im.width))
    assert m['measurements']['head_length_sl']['value'] is None
    assert m['body_profile'] is None

@pytest.mark.parametrize('field,value', [('visibility','occluded'),('visibility','uncertain'),('basis','inferred')])
def test_unobserved_landmark_not_measured(example, field, value):
    im,a=example;setattr(a.landmarks['operculum'],field,value)
    m=measure(a,(im.height,im.width))['measurements']
    assert m['head_length_sl']['value'] is None
    assert m['dorsal_origin_u']['value'] == pytest.approx(.42)

def test_hand_mask_overrides_visible_label(example):
    im,a=example;p=a.landmarks['operculum'];a.masks['occlusion']=[[(p.x-4,p.y-4),(p.x+4,p.y-4),(p.x+4,p.y+4),(p.x-4,p.y+4)]]
    assert 'operculum' not in visible_points(a,(im.height,im.width))

def test_uncertainty_is_conditional_and_optional(example):
    im,a=example;m=measure(a,(im.height,im.width))['measurements']
    assert m['standard_length_px']['sd_annotation'] == pytest.approx(np.sqrt(2)*.4)
    a.landmarks['operculum'].sigma_px=None
    assert measure(a,(im.height,im.width))['measurements']['head_length_sl']['sd_annotation'] is None

def test_absolute_scale_never_invented(example):
    im,a=example
    assert measure(a,(im.height,im.width))['absolute_estimates'] == {}
    result=measure(a,(im.height,im.width),standard_length_mm=200)
    assert result['absolute_estimates']['head_length_mm']['value'] == pytest.approx(47)

def test_occluded_boundary_missing_but_interior_not_erased():
    im,a=fixture(occluded=True);p,_=body_profile(a,(im.height,im.width))
    u=p['u']; assert not p['valid'][(u>.59)&(u<.64)].any()
    assert p['valid'][(u>.36)&(u<.39)].all()

def test_inferred_body_rejected(example):
    im,a=example;a.body_basis='inferred'
    assert measure(a,(im.height,im.width))['body_profile'] is None

def test_no_interpolation_across_missing_sections():
    out=interpolate_supported([0,.5,1,1.5,2,3], [0,1,2,3], [0,np.nan,4,6])
    assert out[0] == 0 and out[4] == 4 and out[5] == 6
    assert np.isnan(out[1:4]).all()

def test_disjoint_polygon_sections_are_not_bounding_boxes():
    poly=[[0,0],[3,0],[3,1],[1,1],[1,2],[3,2],[3,3],[0,3]]
    top,bottom=scan_polygon(poly,[.5,2])
    assert top[0]==0 and bottom[0]==3 and np.isnan(top[1])

def test_dorsal_reference_required_for_atlas_not_basic_ratio(example):
    im,a=example;del a.landmarks['dorsal_reference']
    out=measure(a,(im.height,im.width))
    assert out['measurements']['head_length_sl']['value'] == pytest.approx(.235)
    assert out['body_profile'] is None

def test_absence_requires_documented_evidence():
    with pytest.raises(ValueError,match='absence'):
        Annotation(image_id='x',author='a',structures={'adipose':'absent_observed'})
