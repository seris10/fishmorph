/* FishMorph local workbench. No external scripts, fonts, network services or telemetry. */
'use strict';
const $ = id => document.getElementById(id);
const esc = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone = x => JSON.parse(JSON.stringify(x));
let catalog = null, current = null, ann = null, analysis = null, raster = null, rasterPixels = null;
let activeTab = 'annotate', tool = 'landmark', pending = [], history = [], dirty = false;
let viewport = {s:1,x:0,y:0}, drag = null, box = null, toastTimer = null;
const editor = $('editor'), ctx = editor.getContext('2d');
const layerColors = {visible:'#32b995',body:'#639fc8',occlusion:'#da8063',uncertain:'#d7b94f',artifact:'#ad80c5',region:'#df71a1'};
async function api(path, method='GET', body) {
  const headers = {'X-FishMorph-Token':window.FISHMORPH_TOKEN};
  if(body !== undefined) headers['Content-Type'] = body instanceof ArrayBuffer ? 'application/octet-stream' : 'application/json';
  const res = await fetch('/api/'+path, {method,headers,body:body === undefined ? undefined : body instanceof ArrayBuffer ? body : JSON.stringify(body)});
  const result = await res.json();
  if(!res.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
  return result;
}
function toast(message, error=false) {
  $('toast').textContent = message; $('toast').className = 'toast show'+(error?' error':'');
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').classList.remove('show'),error?11000:5000);
}
async function busy(button, fn) {
  const text=button.textContent; button.disabled=true;button.textContent='Working…';
  try {return await fn();} catch(e){toast(e.message,true);} finally{button.disabled=false;button.textContent=text;}
}
function markDirty(){dirty=true;$('dirty').textContent='Unsaved changes';}
function snapshot(){if(ann){history.push(clone(ann));if(history.length>50)history.shift();}}
function options(select, items, oldValue) {
  select.replaceChildren(...items.map(([value,label])=>{const o=document.createElement('option');o.value=value;o.textContent=label;return o;}));
  if([...select.options].some(o=>o.value===oldValue))select.value=oldValue;
}
async function refresh() {
  catalog=await api('state');
  $('projectPath').textContent=catalog.project;
  $('imageCount').textContent=catalog.images.length;
  const taxa=catalog.taxa.map(t=>[t.id,t.common_name||t.scientific_name]);
  options($('taxonFilter'),[['','All taxa'],...taxa],$('taxonFilter').value);
  options($('profileTaxon'),taxa,$('profileTaxon').value || current?.specimen?.specimen?.taxon_id);
  const specimens=catalog.specimens.map(s=>[s.specimen.id,`${s.specimen.id} · ${s.specimen.material}`]);
  options($('importSpecimen'),specimens,$('importSpecimen').value);
  options($('editSpecimen'),[['','Create new'],...specimens],$('editSpecimen').value);
  options($('meshSpecimen'),catalog.specimens.filter(s=>s.specimen.material==='model').map(s=>[s.specimen.id,s.specimen.id]),$('meshSpecimen').value);
  const landmarkOld=$('landmarkSelect').value;
  options($('landmarkSelect'),Object.keys(catalog.landmarks).map(n=>[n,n.replaceAll('_',' ')]),landmarkOld);
  landmarkInfo(); drawCatalog(); drawProfiles();
}
function drawCatalog(){
  const filter=$('taxonFilter').value;
  const taxonBySpecimen=new Map(catalog.specimens.map(s=>[s.specimen.id,s.specimen.taxon_id]));
  const images=catalog.images.filter(i=>!filter || taxonBySpecimen.get(i.specimen_id)===filter);
  $('imageList').innerHTML=images.map(i=>`<button class="image-card ${current?.image.id===i.id?'active':''}" data-image="${esc(i.id)}"><img src="/image/${encodeURIComponent(i.id)}" alt=""><span><strong>${esc(i.filename)}</strong><small>${esc(i.specimen_id)}</small></span></button>`).join('') || '<p class="hint">No images in this collection.</p>';
  $('imageList').querySelectorAll('[data-image]').forEach(b=>b.onclick=()=>selectImage(b.dataset.image).catch(e=>toast(e.message,true)));
}
async function selectImage(id){
  if(dirty && !confirm('Switch images and discard the unsaved annotation draft?'))return;
  current=await api('images/'+encodeURIComponent(id));
  ann=current.annotation?.annotation || {image_id:id,author:'local_annotator',note:'',quality:{view:'unknown',geometry_reviewed:false,color_reviewed:false,curved:false,refraction:false,severe_perspective:false,mouth:'unknown',fins:'unknown',blur:'unknown',resolution_note:'',note:''},landmarks:{},masks:{},regions:{},body_basis:'observed_with_masked_gaps',structures:{},structure_evidence:''};
  dirty=false;history=[];pending=[];analysis=null;$('dirty').textContent='';$('saveBtn').disabled=false;
  $('imageTitle').textContent=current.image.filename;
  $('specimenEyebrow').textContent=`${current.image.specimen_id} / ${current.specimen.specimen.material.replaceAll('_',' ').toUpperCase()}`;
  $('imageDimensions').textContent=`${current.image.width} × ${current.image.height} PX`;
  $('provenanceInfo').textContent=`Source: ${current.image.metadata.source}\nLicense: ${current.image.metadata.license}\nSHA256: ${current.image.sha256}\n${current.image.icc_status}`;
  $('provenanceInfo').style.whiteSpace='pre-wrap';
  options($('historySelect'),[['','Current draft'],...current.history.map(h=>[h.id,`${h.annotation.author} · ${h.created.slice(0,19)}`])],'');
  syncFields(); drawCatalog(); switchTab(activeTab);
  raster=new Image(); raster.crossOrigin='anonymous'; raster.src='/image/'+encodeURIComponent(id); await raster.decode();
  const off=document.createElement('canvas');off.width=raster.width;off.height=raster.height;
  off.getContext('2d').drawImage(raster,0,0);rasterPixels=off.getContext('2d').getImageData(0,0,raster.width,raster.height);
  fit(); renderAnalysis();
  if(current.annotation){
    const loaded=await api('analyze/'+id,'POST',{});
    if(current.image.id===id){analysis=loaded;renderAnalysis();}
  }
}
function syncFields(){
  if(!ann)return;
  for(const name of ['view','mouth','fins','blur'])$(name).value=ann.quality[name];
  for(const name of ['curved','refraction','severe_perspective','geometry_reviewed','color_reviewed'])$(name).checked=ann.quality[name];
  $('author').value=ann.author;$('annotationNote').value=ann.note;landmarkInfo();
}
function landmarkInfo(){
  const name=$('landmarkSelect').value;
  $('landmarkDefinition').textContent=catalog?.landmarks[name]||'';
  const lm=ann?.landmarks[name];
  $('visibility').value=lm?.visibility||'visible';$('pointSigma').value=lm?.sigma_px??'';
}
function switchTab(tab){
  activeTab=tab;
  document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  document.querySelectorAll('.tab-content').forEach(p=>p.classList.toggle('active',p.id==='tab-'+tab));
  $('emptyState').hidden=!!current || !['annotate','measurements','color'].includes(tab);
  if(!current && ['annotate','measurements','color'].includes(tab))$('tab-'+tab).classList.remove('active');
  if(tab==='annotate')requestAnimationFrame(()=>{resizeCanvas();draw();});
  if(tab==='color')renderColor();
}
function setTool(value){tool=value;pending=[];box=null;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('selected',b.dataset.tool===value));editor.style.cursor=value==='pan'?'grab':'crosshair';draw();}
function resizeCanvas(){const rect=$('canvasWrap').getBoundingClientRect();if(rect.width<1)return;editor.width=Math.round(rect.width);editor.height=Math.round(rect.height);}
function fit(){if(!raster)return;resizeCanvas();viewport.s=Math.min((editor.width-35)/raster.width,(editor.height-35)/raster.height);viewport.x=(editor.width-raster.width*viewport.s)/2;viewport.y=(editor.height-raster.height*viewport.s)/2;draw();}
function canvasPoint(event){const r=editor.getBoundingClientRect();return [(event.clientX-r.left)*(editor.width/r.width),(event.clientY-r.top)*(editor.height/r.height)];}
function imagePoint(event){const p=canvasPoint(event);return [(p[0]-viewport.x)/viewport.s,(p[1]-viewport.y)/viewport.s];}
function inImage(p){return raster && p[0]>=0 && p[1]>=0 && p[0]<raster.width && p[1]<raster.height;}
function clamped(p){return [Math.max(0,Math.min(raster.width-1,p[0])),Math.max(0,Math.min(raster.height-1,p[1]))];}
function polyTarget(){const layer=$('maskLayer').value;const name=$('regionName').value.trim()||'unnamed_region';return layer==='region'?[ann.regions,name]:[ann.masks,layer];}
function draw(){
  ctx.clearRect(0,0,editor.width,editor.height);if(!raster||!ann)return;
  ctx.save();ctx.translate(viewport.x,viewport.y);ctx.scale(viewport.s,viewport.s);ctx.drawImage(raster,0,0);
  const width=1.5/viewport.s;
  function polygon(poly,color,closed=true){if(!poly.length)return;ctx.beginPath();ctx.moveTo(...poly[0]);poly.slice(1).forEach(p=>ctx.lineTo(...p));if(closed)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();if(closed){ctx.fillStyle=color+'27';ctx.fill();}}
  if($('showMasks').checked){
    for(const [layer,polys] of Object.entries(ann.masks))for(const p of polys)polygon(p,layerColors[layer]);
    for(const polys of Object.values(ann.regions))for(const p of polys)polygon(p,layerColors.region);
  }
  if(tool==='vertex')for(const target of [ann.masks,ann.regions])for(const polys of Object.values(target))for(const poly of polys)for(const p of poly){ctx.fillStyle='white';ctx.fillRect(p[0]-2.5/viewport.s,p[1]-2.5/viewport.s,5/viewport.s,5/viewport.s);}
  if(pending.length){polygon(pending,layerColors[$('maskLayer').value],false);for(const p of pending){ctx.beginPath();ctx.arc(...p,3/viewport.s,0,Math.PI*2);ctx.fillStyle='white';ctx.fill();}}
  if(box){ctx.strokeStyle='white';ctx.lineWidth=width;ctx.setLineDash([5/viewport.s,3/viewport.s]);ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);ctx.setLineDash([]);}
  if($('showLandmarks').checked){
    const selected=$('landmarkSelect').value;
    for(const [name,p] of Object.entries(ann.landmarks)){
      ctx.beginPath();ctx.arc(p.x,p.y,(name===selected?5:3.5)/viewport.s,0,Math.PI*2);ctx.fillStyle=p.visibility==='visible'?'#fff9db':'#e49372';ctx.fill();ctx.strokeStyle='#253e32';ctx.lineWidth=1/viewport.s;ctx.stroke();
      ctx.font=`${10/viewport.s}px system-ui`;ctx.lineWidth=2.5/viewport.s;ctx.strokeStyle='#172b22';ctx.strokeText(name,p.x+7/viewport.s,p.y-6/viewport.s);ctx.fillStyle='#fffef0';ctx.fillText(name,p.x+7/viewport.s,p.y-6/viewport.s);
    }
  }
  ctx.restore();
}
function finishPolygon(){if(!ann)return;if(pending.length<3){toast('Place at least three polygon vertices.',true);return;}snapshot();const [target,name]=polyTarget();if(name==='body')target[name]=[pending.map(p=>clamped(p))];else (target[name]??=[]).push(pending.map(p=>clamped(p)));pending=[];markDirty();draw();}
editor.addEventListener('pointerdown',event=>{
  if(!ann||!raster)return;const p=imagePoint(event), cp=canvasPoint(event);editor.setPointerCapture(event.pointerId);
  if(tool==='pan'||event.button===1){drag={kind:'pan',start:cp,x:viewport.x,y:viewport.y};return;}
  if(!inImage(p))return;
  if(tool==='box'){box=[...p,...p];drag={kind:'box'};return;}
  if(tool==='polygon'){pending.push(clamped(p));draw();return;}
  if(tool==='vertex'){
    let best=null,dist=12/viewport.s;
    for(const target of [ann.masks,ann.regions])for(const polys of Object.values(target))for(const poly of polys)poly.forEach((pt,index)=>{const d=Math.hypot(pt[0]-p[0],pt[1]-p[1]);if(d<dist){best={poly,index};dist=d;}});
    if(best){snapshot();drag={kind:'vertex',...best};}return;
  }
  if(tool==='landmark'){
    let near=null,dist=10/viewport.s;
    for(const [name,lm] of Object.entries(ann.landmarks)){const d=Math.hypot(lm.x-p[0],lm.y-p[1]);if(d<dist){near=name;dist=d;}}
    snapshot();
    if(near){$('landmarkSelect').value=near;landmarkInfo();drag={kind:'landmark',name:near};}
    else{
      const name=$('landmarkSelect').value;const sigma=$('pointSigma').value;
      ann.landmarks[name]={x:p[0],y:p[1],visibility:$('visibility').value,basis:'observed',sigma_px:sigma===''?null:Number(sigma),method:'manual',note:''};
      markDirty();if($('autoAdvance').checked){const s=$('landmarkSelect');s.selectedIndex=(s.selectedIndex+1)%s.options.length;landmarkInfo();}
    }draw();
  }
});
editor.addEventListener('pointermove',event=>{
  if(!raster)return;const p=imagePoint(event),cp=canvasPoint(event);
  if(inImage(p)&&rasterPixels){const x=Math.round(p[0]),y=Math.round(p[1]);const i=(Math.min(raster.height-1,y)*raster.width+Math.min(raster.width-1,x))*4;const rgb=Array.from(rasterPixels.data.slice(i,i+3));$('pixelReadout').textContent=`x ${x} · y ${y} | RGB ${rgb.join(', ')} | ${Math.round(viewport.s*100)}% zoom`;}
  if(!drag)return;
  if(drag.kind==='pan'){viewport.x=drag.x+cp[0]-drag.start[0];viewport.y=drag.y+cp[1]-drag.start[1];}
  if(drag.kind==='landmark'){const c=clamped(p);ann.landmarks[drag.name].x=c[0];ann.landmarks[drag.name].y=c[1];markDirty();}
  if(drag.kind==='vertex'){drag.poly[drag.index]=clamped(p);markDirty();}
  if(drag.kind==='box'){const c=clamped(p);box[2]=c[0];box[3]=c[1];}
  draw();
});
editor.addEventListener('pointerup',async()=>{
  if(drag?.kind==='box'&&box){
    const x=Math.floor(Math.min(box[0],box[2])),y=Math.floor(Math.min(box[1],box[3]));const w=Math.ceil(Math.abs(box[2]-box[0])),h=Math.ceil(Math.abs(box[3]-box[1]));box=null;drag=null;
    await busy($('grabcutBtn'),async()=>{const result=await api('segment','POST',{image_id:ann.image_id,rectangle:[x,y,w,h]});snapshot();ann.masks.visible=result.polygons;(ann.masks.uncertain??=[]).push(...result.holes);markDirty();toast('Foreground proposal loaded. Review it, and outline the fins-excluded body separately.');});setTool('polygon');draw();
  }drag=null;
});
function zoom(factor,point=[editor.width/2,editor.height/2]){const old=viewport.s;viewport.s=Math.max(.03,Math.min(20,old*factor));viewport.x=point[0]-(point[0]-viewport.x)*(viewport.s/old);viewport.y=point[1]-(point[1]-viewport.y)*(viewport.s/old);draw();}
editor.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0015),canvasPoint(e));},{passive:false});
function renderAnalysis(){
  if(!analysis){$('measurementTable').innerHTML='<p class="empty-caption">Save an annotation to calculate eligible measurements.</p>';$('measurementSummary').replaceChildren();renderColor();return;}
  const rows=Object.entries(analysis.morphology.measurements),good=rows.filter(([,r])=>r.value!==null).length;
  $('measurementSummary').innerHTML=`<div class="stat"><strong>${good}</strong><small>ELIGIBLE MEASUREMENTS</small></div><div class="stat"><strong>${rows.length-good}</strong><small>UNAVAILABLE / EXPLAINED</small></div><div class="stat"><strong>${analysis.morphology.body_profile?Math.round(analysis.morphology.body_profile.coverage*100)+'%':'—'}</strong><small>BODY PROFILE SUPPORT</small></div>`;
  $('measurementTable').innerHTML='<table><thead><tr><th>Measurement</th><th>Value</th><th>Unit</th><th>Annotation SD</th><th>Status / reason</th></tr></thead><tbody>'+rows.map(([name,r])=>`<tr><td title="${esc(r.label)}">${esc(name.replaceAll('_',' '))}</td><td>${r.value===null?'—':r.value.toPrecision(5)}</td><td>${esc(r.unit)}</td><td>${r.sd_annotation==null?'Not quantified':r.sd_annotation.toPrecision(3)}</td><td class="reason">${r.value===null?`<span class="tag missing">Unavailable</span> ${esc(r.reasons.join('; '))}`:'<span class="tag good">Observed</span>'}</td></tr>`).join('')+'</tbody></table>';
  renderColor();
}
function renderColor(){
  const canvas=$('atlasCanvas'),context=canvas.getContext('2d');
  const atlas=analysis?.color?.atlas;canvas.width=atlas?atlas.u.length:128;canvas.height=atlas?atlas.v.length:40;context.clearRect(0,0,canvas.width,canvas.height);
  if(atlas){const pixels=context.createImageData(canvas.width,canvas.height);for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){const i=(y*canvas.width+x)*4;if(atlas.valid[y][x]){for(let c=0;c<3;c++)pixels.data[i+c]=Math.round(atlas.rgb[y][x][c]*255);pixels.data[i+3]=255;}}context.putImageData(pixels,0,0);}
  $('colorStatus').innerHTML=analysis?`<p class="hint">${esc(analysis.color.status)} · ${analysis.color.usable_source_pixels} usable source pixels<br>${esc(analysis.color.atlas_reason||'Every atlas sample retains a source-image pixel coordinate.')}<br>${analysis.color.markings.length} heuristic marking candidates</p>`:'<p class="hint">No color analysis loaded.</p>';
  const regions=analysis?.color?.regions||{};
  $('regionalColors').innerHTML='<table><tr><th>Observed region</th><th>Source pixels</th><th>Mean Lab (D65)</th><th>Mean RGB</th></tr>'+Object.entries(regions).map(([name,r])=>`<tr><td>${esc(name.replaceAll('_',' '))}</td><td>${r.n_source_pixels}</td><td>${r.mean_lab?r.mean_lab.map(v=>v.toFixed(2)).join(', '):'—'}</td><td>${r.mean_rgb?'<span class="color-swatch" style="background:rgb('+r.mean_rgb.map(v=>Math.round(v*255)).join(',')+')"></span>'+r.mean_rgb.map(v=>Math.round(v*255)).join(', '):'—'}</td></tr>`).join('')+'</table>';
  colorSource();
}
function colorSource(pointer){const c=$('colorSource'),cx=c.getContext('2d');if(!raster){c.width=128;c.height=40;return;}const scale=Math.min(1,760/raster.width);c.width=Math.round(raster.width*scale);c.height=Math.round(raster.height*scale);cx.drawImage(raster,0,0,c.width,c.height);cx.strokeStyle='#f2c160';cx.lineWidth=1.3;for(const mark of analysis?.color?.markings||[]){const [x,y]=mark.source_centroid_xy;cx.beginPath();cx.arc(x*scale,y*scale,Math.max(3,Math.sqrt(mark.area_px/Math.PI)*scale),0,Math.PI*2);cx.stroke();}if(pointer){cx.strokeStyle='#ffffff';cx.beginPath();cx.arc(pointer[0]*scale,pointer[1]*scale,7,0,Math.PI*2);cx.stroke();}}
$('atlasCanvas').addEventListener('mousemove',e=>{const atlas=analysis?.color?.atlas;if(!atlas)return;const r=e.currentTarget.getBoundingClientRect(),x=Math.min(atlas.u.length-1,Math.floor((e.clientX-r.left)/r.width*atlas.u.length)),y=Math.min(atlas.v.length-1,Math.floor((e.clientY-r.top)/r.height*atlas.v.length));if(!atlas.valid[y][x]){$('atlasReadout').textContent=`u ${atlas.u[x].toFixed(3)} · v ${atlas.v[y].toFixed(3)} | MISSING observation`;colorSource();return;}const p=atlas.source_xy[y][x];$('atlasReadout').textContent=`u ${atlas.u[x].toFixed(3)} · v ${atlas.v[y].toFixed(3)} | source pixel (${p[0]}, ${p[1]})`;colorSource(p);});
function drawProfiles(){
  const profiles=catalog?.profiles||[];
  options($('compareProfile'),profiles.map(p=>[p.id,`${p.taxon.common_name||p.taxon.id} · ${p.group.stage} / ${p.group.sex} · N=${p.n_individuals} · ${p.group.material}`]),$('compareProfile').value);
  $('profileCards').innerHTML=profiles.map(p=>`<button class="profile-card" data-profile="${esc(p.id)}"><span class="section-label">${p.n_individuals} SPECIMENS / ${p.n_images} IMAGES</span><h3>${esc(p.taxon.common_name||p.taxon.scientific_name)}</h3><small>${esc([p.group.stage,p.group.sex,p.group.population,p.group.material,p.group.view].join(' · '))}</small></button>`).join('')||'<p class="empty-caption">No profiles yet. Build from reviewed, identified individuals, not unlinked image files.</p>';
  $('profileCards').querySelectorAll('[data-profile]').forEach(b=>b.onclick=()=>{$('profileReport').src='/report/'+encodeURIComponent(b.dataset.profile);$('profileReport').hidden=false;});
}
function downloadJSON(value,filename){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
// Interface events.
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
$('landmarkSelect').onchange=()=>{landmarkInfo();draw();};$('taxonFilter').onchange=drawCatalog;
$('maskLayer').onchange=()=>{$('regionName').hidden=$('maskLayer').value!=='region';setTool('polygon');};
$('finishPoly').onclick=finishPolygon;$('undoBtn').onclick=()=>{if(pending.length){pending.pop();draw();return;}if(history.length){ann=history.pop();markDirty();syncFields();draw();}};
$('clearLayer').onclick=()=>{if(!ann)return;snapshot();const [target,key]=polyTarget();target[key]?.pop();markDirty();draw();};
$('grabcutBtn').onclick=()=>{if(!ann)return;setTool('box');toast('Drag a box around the fish, leaving background outside the box.');};
$('zoomFit').onclick=fit;$('zoomIn').onclick=()=>zoom(1.3);$('zoomOut').onclick=()=>zoom(1/1.3);
$('showMasks').onchange=draw;$('showLandmarks').onchange=draw;
for(const name of ['view','mouth','fins','blur','curved','refraction','severe_perspective','geometry_reviewed','color_reviewed'])$(name).onchange=()=>{if(!ann)return;snapshot();ann.quality[name]=$(name).type==='checkbox'?$(name).checked:$(name).value;markDirty();};
$('author').onchange=()=>{if(ann){snapshot();ann.author=$('author').value;markDirty();}};
$('annotationNote').onchange=()=>{if(ann){snapshot();ann.note=$('annotationNote').value;markDirty();}};
for(const name of ['visibility','pointSigma'])$(name).onchange=()=>{const lm=ann?.landmarks[$('landmarkSelect').value];if(lm){snapshot();lm.visibility=$('visibility').value;lm.sigma_px=$('pointSigma').value===''?null:Number($('pointSigma').value);markDirty();draw();}};
$('historySelect').onchange=()=>{const h=current?.history.find(h=>h.id===$('historySelect').value);if(h){snapshot();ann=clone(h.annotation);markDirty();syncFields();draw();toast('Historical annotation loaded as a new editable draft.');}};
$('saveBtn').onclick=()=>busy($('saveBtn'),async()=>{if(!ann)return;await api('annotations','POST',ann);analysis=await api('analyze/'+ann.image_id,'POST',{});dirty=false;$('dirty').textContent='Saved';current=await api('images/'+ann.image_id);options($('historySelect'),[['','Current draft'],...current.history.map(h=>[h.id,`${h.annotation.author} · ${h.created.slice(0,19)}`])],'');renderAnalysis();toast('Saved as a new revision. Measurements and color analysis updated.');});
$('analysisDownload').onclick=()=>{if(analysis)downloadJSON(analysis,'fishmorph-analysis.json');};
$('buildProfile').onclick=()=>busy($('buildProfile'),async()=>{const result=await api('profiles','POST',{taxon_id:$('profileTaxon').value});$('profileNotice').innerHTML=`<div class="warning">${result.profile_ids.length} profiles built. ${result.excluded.length} observations excluded.${result.excluded.length?' '+esc(result.excluded.map(x=>x.reason).join('; ')):''}</div>`;await refresh();toast('Profile records saved with contributing specimens and annotation revisions.');});
$('compareBtn').onclick=()=>busy($('compareBtn'),async()=>{if(!current)throw new Error('Select a candidate image first.');if(dirty)throw new Error('Save the candidate annotation before comparing.');if(!$('compareProfile').value)throw new Error('Build or select a reference profile.');const result=await api('compare','POST',{profile_id:$('compareProfile').value,image_id:current.image.id,exploratory_color:$('exploratoryColor').checked});$('comparisonReport').src='/report/'+result.id;$('comparisonReport').hidden=false;});
$('importBtn').onclick=()=>{$('importDialog').showModal();};
$('doImport').onclick=()=>busy($('doImport'),async()=>{if(!$('importSpecimen').value)throw new Error('Create a specimen first.');const files=[...$('imageFiles').files];if(!files.length)throw new Error('Select one or more image files.');let last;for(const file of files){const query=new URLSearchParams({specimen_id:$('importSpecimen').value,filename:file.name,source:$('importSource').value,license:$('importLicense').value||'unknown'});last=await api('import?'+query,'POST',await file.arrayBuffer());}await refresh();$('importDialog').close();await selectImage(last.id);toast(`${files.length} import requests processed. Exact duplicates do not create new observations.`);});
function specimenEditor(){const id=$('editSpecimen').value;const existing=catalog.specimens.find(s=>s.specimen.id===id)?.specimen;const template=existing||{id:'specimen_001',taxon_id:catalog.taxa.find(t=>t.profile_allowed)?.id||'yellowstone_cutthroat',identity_status:'unresolved',identity_evidence:'',identification_evidence:'',sex:'unknown',stage:'unknown',demographic_evidence:'',population:'unknown',locality:'unknown',season:'unknown',reproductive_condition:'unknown',material:'living_wild',standard_length_mm:null,length_evidence:'',metadata_note:''};$('specimenJson').value=JSON.stringify(template,null,2);}
$('specimenBtn').onclick=()=>{specimenEditor();$('specimenDialog').showModal();};$('editSpecimen').onchange=specimenEditor;
$('saveSpecimen').onclick=()=>busy($('saveSpecimen'),async()=>{await api('specimens','POST',JSON.parse($('specimenJson').value));await refresh();$('specimenDialog').close();toast('Specimen revision saved. Rebuild profiles to incorporate changed metadata.');});
$('meshBtn').onclick=()=>{$('meshDialog').showModal();};
$('doMesh').onclick=()=>busy($('doMesh'),async()=>{if(!$('meshSpecimen').value)throw new Error('Create a specimen with material="model" first.');const file=$('meshFile').files[0];if(!file)throw new Error('Select a GLB mesh.');const query=new URLSearchParams({specimen_id:$('meshSpecimen').value,longitudinal:$('meshLong').value,dorsal:$('meshUp').value,view:$('meshView').value});const result=await api('mesh?'+query,'POST',await file.arrayBuffer());await refresh();$('meshDialog').close();await selectImage(result.image.id);switchTab('annotate');toast('Geometry rendered. Review side, pose, body outline and visible landmarks before comparison.');});
window.addEventListener('resize',()=>{resizeCanvas();draw();});
window.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='Enter'&&pending.length){e.preventDefault();finishPolygon();}if(e.key==='Escape'){pending=[];box=null;drag=null;draw();}if(e.key==='Backspace'&&pending.length){e.preventDefault();pending.pop();draw();}});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
refresh().then(async()=>{switchTab('annotate');if(catalog.images.length)await selectImage(catalog.images[0].id);}).catch(e=>toast(e.message,true));
