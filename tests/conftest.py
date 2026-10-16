import io
import pytest
from fishmorph.demo import fixture
from fishmorph.models import Specimen, Taxon, ImageMetadata, AnalysisConfig
from fishmorph.storage import Store

@pytest.fixture
def example():
    return fixture()

@pytest.fixture
def config():
    return AnalysisConfig(atlas_width=32, atlas_height=12, profile_samples=65)

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'project'); s.init(seed=False)
    s.add_taxon(Taxon(id='test', scientific_name='Synthetic test, not biology', status='synthetic', profile_allowed=True))
    return s

def add_observation(store, sid='one', index=0, annotation=None, image=None, **kwargs):
    sp = Specimen(id=sid, taxon_id='test', material='synthetic', identity_status='documented',
                  identity_evidence='Authored independent test fixture', **kwargs)
    store.save_specimen(sp)
    if image is None:
        image, default = fixture(index)
        annotation = annotation or default
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    im = store.ingest_bytes(buffer.getvalue(), f'{sid}-{index}.png', sid,
                           ImageMetadata(photometric_status='synthetic', photometric_protocol='synthetic-test'))
    annotation.image_id = im['id']; store.save_annotation(annotation)
    return im
