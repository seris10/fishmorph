import copy
import io
import numpy as np
import pytest
from fishmorph.statistics import scalar_summary, array_summary, generalized_procrustes, fit_allometry, predict_allometry, build_profiles
from fishmorph.models import ProfileConfig, Specimen, Annotation, ImageMetadata
from fishmorph.demo import fixture
from fishmorph.util import dumps
from conftest import add_observation


def test_bootstrap_reproducible_and_is_mean_interval():
    cfg=ProfileConfig(bootstrap_samples=400)
    a=scalar_summary([1,2,3,4,5,6,7,8],cfg);b=scalar_summary([1,2,3,4,5,6,7,8],cfg)
    assert a==b and a['n_individuals']==8
    assert a['min'] < a['mean_ci95'][0] < a['mean'] < a['mean_ci95'][1] < a['max']

def test_small_sample_not_precise_distribution():
    out=scalar_summary([2],ProfileConfig())
    assert out['sd'] is None and out['mean_ci95'] is None

def test_array_missingness_uses_cell_specific_individual_counts():
    out=array_summary([[1,None,3],[3,None,None]])
    assert out['mean']==[2,None,3] and out['n_individuals']==[2,0,1]
    assert out['sd'][2] is None

def test_gpa_removes_translation_scale_rotation_not_reflection():
    x=np.array([[0.,0],[2,0],[.6,1.1],[1.4,.4]])
    angle=.5;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    result=generalized_procrustes([x, 3*x@r + 9])
    aligned=np.asarray(result['aligned']);assert aligned[0]==pytest.approx(aligned[1],abs=1e-9)
    reflected=generalized_procrustes([x,x*[-1,1]])
    assert np.linalg.norm(np.diff(np.asarray(reflected['aligned']),axis=0))>.1
    assert reflected['reflections_allowed'] is False

def test_gpa_rejects_missing_or_degenerate():
    with pytest.raises(ValueError):generalized_procrustes(np.zeros((2,4,2)))
    with pytest.raises(ValueError):generalized_procrustes(np.full((2,4,2),np.nan))

def test_allometric_exponent_recovery_and_no_extrapolation():
    x=np.linspace(100,350,12);y=.07*x**1.2
    model=fit_allometry(x,y)
    assert model['slope']==pytest.approx(1.2)
    assert predict_allometry(model,200)['conditional_median_mm']==pytest.approx(.07*200**1.2)
    with pytest.raises(ValueError,match='extrapolation'):predict_allometry(model,500)

def test_allometry_sufficiency_guards():
    assert fit_allometry([100]*3,[20]*3)['status']=='insufficient_data'
    assert fit_allometry(np.linspace(100,101,10),np.linspace(20,21,10))['status']=='insufficient_size_range'
    with pytest.raises(ValueError):fit_allometry([0]*10,[20]*10)

def test_original_preservation_duplicates_and_conflicts(store):
    im=add_observation(store);original=store.root/im['original']
    repeat=store.ingest(original,'one')
    assert repeat['id']==im['id'] and len(store.images())==1
    import hashlib
    assert hashlib.sha256(original.read_bytes()).hexdigest()==im['sha256']
    store.save_specimen(Specimen(id='different',taxon_id='test'))
    with pytest.raises(ValueError,match='specimen'):store.ingest(original,'different')

def test_annotation_history_and_cache_invalidation(store,config):
    im=add_observation(store);first=store.analyze(im['id'],config)
    assert first==store.analyze(im['id'],config)
    old=store.annotation(im['id']);a=Annotation.model_validate(old['annotation']);a.landmarks['operculum'].x+=10
    store.save_annotation(a);second=store.analyze(im['id'],config)
    assert first['id'] != second['id'] and len(store.annotation_history(im['id']))==2
    assert store.annotation(im['id'],old['id'])==old
    assert store.analyze(im['id'],config,old['id'])==first
    assert 'code_fingerprint' in first['provenance']

def test_documented_metadata_requires_evidence():
    with pytest.raises(ValueError):Specimen(id='a',taxon_id='test',sex='male')
    with pytest.raises(ValueError):Specimen(id='a',taxon_id='test',standard_length_mm=12)
    with pytest.raises(ValueError):ImageMetadata(photometric_status='externally_calibrated')

def test_profile_specimen_weight_strata_identity_and_rebuild(store,config):
    one=add_observation(store,'one',0)
    image,ann=fixture(0,mirrored=True)
    add_observation(store,'one',0,annotation=ann,image=image)
    add_observation(store,'two',1)
    # Different individual with unresolved identity must not inflate empirical N.
    extra=add_observation(store,'unknown',2)
    sp=Specimen.model_validate(store.get_specimen('unknown')['specimen']);sp.identity_status='unresolved';store.save_specimen(sp)
    add_observation(store,'preserved',3)
    sp=Specimen.model_validate(store.get_specimen('preserved')['specimen']);sp.material='preserved';store.save_specimen(sp)
    out=build_profiles(store,'test',analysis_config=config)
    assert len(out['profiles'])==2
    synth=next(p for p in out['profiles'] if p['group']['material']=='synthetic')
    assert synth['n_individuals']==2 and synth['n_images']==3
    assert synth['measurements']['head_length_sl']['n_individuals']==2
    assert synth['color_groups'][0]['n_individuals']==2
    assert synth['structures']['adipose_fin']['present']==2
    assert any(r['reason']=='unresolved_individual_identity' for r in out['excluded'])
    again=build_profiles(store,'test',analysis_config=config)
    assert {p['id'] for p in out['profiles']}=={p['id'] for p in again['profiles']}
    a=Annotation.model_validate(store.annotation(one['id'])['annotation']);a.landmarks['operculum'].x+=20;store.save_annotation(a)
    rebuilt=build_profiles(store,'test',analysis_config=config)
    assert synth['id'] not in {p['id'] for p in rebuilt['profiles']}
    assert store.document(synth['id'])==synth

def test_out_of_bounds_annotation_rejected(store):
    im=add_observation(store);a=Annotation.model_validate(store.annotation(im['id'])['annotation']);a.landmarks['snout'].x=10000
    with pytest.raises(ValueError,match='outside'):store.save_annotation(a)

def test_unresolved_taxon_cannot_make_reference(store,config):
    from fishmorph.models import Taxon
    store.add_taxon(Taxon(id='unresolved',scientific_name='Cottus sp.',status='unresolved'))
    with pytest.raises(ValueError,match='Taxonomic'):build_profiles(store,'unresolved',analysis_config=config)

def test_strict_json_uses_null():
    assert dumps({'a':np.nan,'b':np.float64(2)})=='{"a": null, "b": 2.0}'

def test_exported_profile_matches_portable_json_schema(store,config):
    import json
    from pathlib import Path
    import jsonschema
    add_observation(store)
    profile=build_profiles(store,'test',analysis_config=config)['profiles'][0]
    schema=json.loads((Path(__file__).resolve().parents[1]/'schemas/morphology_profile.schema.json').read_text())
    jsonschema.Draft202012Validator(schema).validate(profile)
