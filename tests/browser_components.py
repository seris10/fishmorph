"""Optional browser component checks against real FastAPI routes via an in-process adapter.

Runs without browser URL navigation. Requires pip install -e '.[browser,dev]' and
an installed Playwright Chromium or FISHMORPH_CHROMIUM executable. The adapter
mounts reports as srcdoc; this is not live-browser HTTP integration coverage.
"""
import json
from pathlib import Path
import re
import os
import shutil
from urllib.parse import urlsplit
from fastapi.testclient import TestClient
from fishmorph.webapp import create_app
from fishmorph.demo import build_demo
from fishmorph.storage import Store
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT / '.browser-work'
if WORK.exists():
    shutil.rmtree(WORK)
manifest=build_demo(Store(WORK))
out=WORK/'screenshots';out.mkdir(parents=True,exist_ok=True)
checks=[];errors=[]
app=create_app(WORK)
with TestClient(app) as client, sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('FISHMORPH_CHROMIUM') or shutil.which('chromium'),headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':1100},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    def dispatch(route):
        request=route.request; u=urlsplit(request.url)
        if request.method=='OPTIONS':
            route.fulfill(status=204,headers={'access-control-allow-origin':'*','access-control-allow-headers':'*','access-control-allow-methods':'*'});return
        response=client.request(request.method,u.path+('?' + u.query if u.query else ''),
                                content=request.post_data_buffer,headers=request.headers)
        headers=dict(response.headers);headers.update({'access-control-allow-origin':'*','access-control-allow-headers':'*'})
        route.fulfill(status=response.status_code,headers=headers,body=response.content)
    page.route('**/*',dispatch)
    static=ROOT/'src/fishmorph/static'
    html=client.get('/').text.replace('<head>','<head><base href="http://127.0.0.1:8765/">')
    html=re.sub(r'<link[^>]+href="/static/style.css"[^>]*>',lambda m:'<style>'+(static/'style.css').read_text()+'</style>',html)
    html=html.replace('<script src="/static/app.js"></script>','<script>'+(static/'app.js').read_text()+'</script>')
    page.set_content(html,wait_until='networkidle')
    page.wait_for_function('analysis !== null && raster !== null')
    assert page.locator('.image-card').count()==10
    checks.append('Catalog loads ten synthetic source/render images, annotations and analysis')
    page.locator('[data-image="'+manifest['reference_image_ids'][3]+'"]').click()
    page.wait_for_function('analysis?.image_id === '+json.dumps(manifest['reference_image_ids'][3]))
    page.screenshot(path=str(out/'annotation-workbench.png'),full_page=True)
    # Drag an existing anatomical landmark through real pointer events.
    page.locator('#autoAdvance').uncheck()
    loc=page.evaluate('''() => {const r=editor.getBoundingClientRect(),lm=ann.landmarks.operculum;
        return {x:r.x+viewport.x+lm.x*viewport.s,y:r.y+viewport.y+lm.y*viewport.s,old:lm.x};}''')
    page.mouse.move(loc['x'],loc['y']);page.mouse.down();page.mouse.move(loc['x']+7,loc['y'],steps=5);page.mouse.up()
    assert page.evaluate('ann.landmarks.operculum.x')>loc['old']
    checks.append('Canvas pointer events drag an existing landmark and mark the draft dirty')
    # Place an occlusion mask using UI controls and finish it.
    page.locator('#maskLayer').select_option('occlusion')
    before=page.evaluate('(ann.masks.occlusion||[]).length')
    coords=page.evaluate('''() => {const r=editor.getBoundingClientRect();return [[300,151],[320,151],[320,172],[300,172]].map(([x,y])=>[r.x+viewport.x+x*viewport.s,r.y+viewport.y+y*viewport.s]);}''')
    for x,y in coords:page.mouse.click(x,y)
    page.locator('#finishPoly').click()
    assert page.evaluate('ann.masks.occlusion.length')==before+1
    checks.append('Manual polygon workflow adds a hand/occlusion mask')
    page.locator('#saveBtn').click();page.wait_for_function('!dirty && analysis!==null && !document.getElementById("saveBtn").disabled')
    assert page.locator('#historySelect option').count()>=3
    checks.append('Save and analyze persists a new annotation revision')
    page.locator('[data-tab="measurements"]').click()
    assert page.locator('#measurementTable').inner_text().find('head length sl')>=0
    checks.append('Measurement table contains structured available and unavailable results')
    page.locator('[data-tab="color"]').click()
    atlas=page.locator('#atlasCanvas').bounding_box()
    page.mouse.move(atlas['x']+atlas['width']*.45,atlas['y']+atlas['height']*.45)
    assert 'source pixel' in page.locator('#atlasReadout').inner_text()
    checks.append('Color atlas hover maps to a source pixel')
    page.evaluate("document.getElementById('toast').classList.remove('show')")
    page.screenshot(path=str(out/'spatial-color-atlas.png'),full_page=True)
    page.locator('[data-tab="profiles"]').click()
    page.locator('#profileTaxon').select_option('synthetic_test_fish')
    page.locator('#buildProfile').click()
    page.wait_for_function('document.getElementById("profileNotice").textContent.includes("profiles built")')
    page.locator('.profile-card').first.click()
    page.wait_for_function("document.getElementById('profileReport').getAttribute('src')!==null")
    src=page.locator('#profileReport').get_attribute('src')
    page.locator('#profileReport').evaluate('(el,html)=>{el.removeAttribute("src");el.srcdoc=html;}',client.get(src).text)
    page.frame_locator('#profileReport').locator('h1').wait_for()
    checks.append('Profile build and standalone embedded report work')
    page.locator('[data-image="'+manifest['candidate_image_id']+'"]').click()
    page.wait_for_function('analysis !== null && current.image.id === '+json.dumps(manifest['candidate_image_id']))
    page.locator('[data-tab="compare"]').click()
    page.locator('#compareProfile').select_option(manifest['reference_profile_id'])
    page.locator('#compareBtn').click()
    page.wait_for_function("document.getElementById('comparisonReport').getAttribute('src')!==null")
    src=page.locator('#comparisonReport').get_attribute('src')
    page.locator('#comparisonReport').evaluate('(el,html)=>{el.removeAttribute("src");el.srcdoc=html;}',client.get(src).text)
    page.frame_locator('#comparisonReport').locator('h1').wait_for()
    assert 'abstained' in page.frame_locator('#comparisonReport').locator('body').inner_text()
    checks.append('Independent geometry candidate comparison loads; color assessment abstains')
    page.screenshot(path=str(out/'geometry-comparison.png'),full_page=True)
    # Smaller viewport: verify normal layout instead of forcing desktop horizontal scroll.
    page.set_viewport_size({'width':1050,'height':900})
    page.locator('[data-tab="annotate"]').click()
    page.locator('#zoomFit').click()
    assert page.locator('#saveBtn').is_visible()
    checks.append('Annotation controls remain available at a 1050 px viewport')
    browser.close()
(WORK/'browser-test-results.json').write_text(json.dumps({'browser':'Chromium via Playwright', 'mode':'In-process FastAPI TestClient adapter; browser navigation prohibited by runtime policy. Report HTML mounted as srcdoc. Not live-network end-to-end coverage.','checks':checks,'page_errors':errors},indent=2)+'\n')
print(json.dumps({'checks':checks,'page_errors':errors},indent=2))
assert not errors
