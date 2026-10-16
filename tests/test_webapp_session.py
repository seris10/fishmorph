from fastapi.testclient import TestClient
from fishmorph.statistics import build_profiles
from fishmorph.webapp import create_app
from conftest import add_observation


def test_image_and_report_routes_require_the_session_token(store, config):
    first = add_observation(store, 'one', 0)
    add_observation(store, 'two', 1)
    profile_id = build_profiles(store, 'test', analysis_config=config)['profiles'][0]['id']
    app = create_app(store.root)
    image_url, report_url = '/image/' + first['id'], '/report/' + profile_id

    with TestClient(app) as stranger:
        assert stranger.get(image_url).status_code == 403
        assert stranger.get(report_url).status_code == 403
        stranger.cookies.set('fishmorph_session', 'wrong')
        assert stranger.get(image_url).status_code == 403
        assert stranger.get(report_url).status_code == 403

    with TestClient(app) as browser:
        assert browser.get('/').status_code == 200
        assert 'httponly' in browser.get('/').headers['set-cookie'].lower()
        assert browser.get(image_url).status_code == 200
        assert browser.get(image_url).headers['content-type'] == 'image/png'
        assert browser.get(report_url).status_code == 200

    with TestClient(app) as native:
        native.headers['x-fishmorph-token'] = app.state.token
        assert native.get(image_url).status_code == 200
        assert native.get(report_url).status_code == 200
