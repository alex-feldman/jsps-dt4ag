"""Self-contained Canvas 2D inspection viewer; no reconstruction or inference."""
import json
import numpy as np


def write_viewer(out, bundle, candidates=None, plant_asset=None, labels_path=None):
    """Write out/viewer.html. ``plant_asset`` embeds per-leaf export records for selection/export."""
    with np.load(labels_path or out/'leaf_ids2.npz') as arrays:
        points = {}
        for leaf in bundle['leaves']:
            selected = arrays['X'][arrays['label']==leaf['id']]
            # Deterministic display sample; full data remain in the NPZ.
            positions = np.linspace(0,len(selected)-1,min(300,len(selected)),dtype=int)
            points[str(leaf['id'])] = selected[positions].tolist()
    graph = bundle['source_graph'] or {'skeleton_nodes':[],'skeleton_edges':[],'leaves':[]}
    data = {'capture':bundle['capture_id'],'points':points,'graph':graph,'leaves':bundle['leaves'],
            'branchCandidates':candidates,'plantAsset':plant_asset}
    payload = json.dumps(data,allow_nan=False).replace('<','\\u003c')
    page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Organ analysis viewer</title><style>
html,body{height:100%;margin:0;background:#10151b;color:#edf2f7;font:14px system-ui}
body{display:grid;grid-template-columns:minmax(0,1fr) 290px}canvas{width:100%;height:100%;touch-action:none}
aside{padding:18px;overflow:auto;background:#19212b;overflow-wrap:anywhere}p{line-height:1.5}button{padding:8px}
.review-toolbar{position:sticky;top:-18px;background:#19212b;padding:10px 0;z-index:2;border-bottom:1px solid #526171}
.endpoint-row{padding:10px 0;border-bottom:1px solid #34404c}.endpoint-row label{display:block;margin:4px 0}
.endpoint-row select,.endpoint-row input{width:100%;box-sizing:border-box;padding:6px;background:#10151b;color:#edf2f7;border:1px solid #526171}
#leafChoice{display:block;width:100%;box-sizing:border-box;margin-top:4px}
@media(max-width:700px){body{grid-template-columns:1fr;grid-template-rows:65% 35%}}
</style><canvas id="view" aria-label="Provisional organ points and skeleton"></canvas><aside>
<h1 id="title"></h1><div class="review-toolbar" id="reviewToolbar" hidden><button id="exportReview">Download endpoint review CSV</button><p id="saveStatus" role="status"></p></div>
<p>Drag to rotate; wheel to zoom. Reset uses dataparser z as approximate up, not gravity.</p>
<button id="reset">Reset view</button><p><label><input id="points" type="checkbox" checked>Leaf points</label></p>
<p><label><input id="edges" type="checkbox" checked>Skeleton</label></p>
<section id="leafSelection" hidden><h2>Select one leaf</h2>
<p>Click a leaf's points or choose it below. The download holds that leaf's full point coordinates with their original row IDs, its petiole polyline and status, and its attachment record, in the plant-asset leaf-export format. Cyan dashed lines are inferred-petiole connections: synthetic, not recovered tissue. Nothing else is changed.</p>
<p><label>Leaf <select id="leafChoice"><option value="">No leaf selected</option></select></label></p>
<p id="leafSummary" role="status"></p><p><button id="exportLeaf" disabled>Download selected leaf (JSON)</button></p></section>
<details id="endpointReviewPanel" hidden open><summary>Review every endpoint</summary>
<p>Choose a category for each endpoint. Your earlier confirmed decisions are prefilled. Notes are optional. This records annotations only; it does not delete geometry. Download CSV to send the review back.</p>
<div id="endpointReviewRows"></div></details>
<section id="candidates" hidden><h2>Leaf-free branch candidates</h2>
<p><label><input id="highlightCandidates" type="checkbox" checked>Highlight candidates</label></p>
<p><label><input id="hideCandidates" type="checkbox">Preview without candidates</label></p>
<p><label><input id="hideConfirmed" type="checkbox">Preview confirmed cleanup only</label></p>
<p><label><input id="endpointLabels" type="checkbox" checked>Endpoint IDs</label></p>
<p>Amber paths have no saved leaf-tip association. Dashed portions have nearby-leaf or uncertain-base evidence and need extra review. Any candidate may still be a tendril or missing tissue. The preview changes only this display; original data remain intact.</p>
<p><a href="branch-candidates.csv" download style="color:#ffd166">Candidate table (CSV)</a></p><div id="candidateList"></div></section><div id="legend"></div>
<p>Red: candidate main axis. Gray: unresolved other paths. Colored dots: at most 300 centers per leaf cluster.
These are Gaussian centers and graph lines, not rendered splats or measured organs. Zero-length fallback paths do not mean absent petioles.
No tag/string removal, gravity alignment, metric scale, or biological acceptance is implied.</p></aside>
<script>const D=__DATA__;
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const colors=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7'];
document.getElementById('title').textContent=D.capture+' organ evidence';
const nodes=new Map(D.graph.skeleton_nodes.map(n=>[n.id,n.xyz]));
const reviewCategories=['not reviewed','unclear','leaf','false-branch','tip/apex','tendril','stem/base','string/support','other'];
const endpointRows=D.branchCandidates?.endpoint_records||[];
const reviewKey='organ-endpoint-review-v1:'+D.branchCandidates?.source_graph_sha256;
let savedAnswers={},storageAvailable=true;
try{const stored=JSON.parse(localStorage.getItem(reviewKey)||'null');if(stored?.version===1&&stored.graph_sha256===D.branchCandidates?.source_graph_sha256)savedAnswers=stored.answers||{}}
catch(e){storageAvailable=false}
const answers=Object.create(null);
function saveReview(){try{localStorage.setItem(reviewKey,JSON.stringify({version:1,graph_sha256:D.branchCandidates.source_graph_sha256,answers}));storageAvailable=true}
catch(e){storageAvailable=false}
document.getElementById('saveStatus').textContent=storageAvailable?'Saved in this browser. Download CSV for a durable copy.':'Browser saving unavailable. Download CSV before closing.'}
if(endpointRows.length){document.getElementById('reviewToolbar').hidden=false;document.getElementById('endpointReviewPanel').hidden=false;
for(const endpoint of endpointRows){const id=endpoint.endpoint_id,prior=endpoint.human_disposition;
const initial=prior==='confirmed_false'?'false-branch':prior==='retain_leaf_connected'?'leaf':'not reviewed';
const stored=savedAnswers[id]?.source_node_id===endpoint.source_node_id?savedAnswers[id]:null;answers[id]={source_node_id:endpoint.source_node_id,category:reviewCategories.includes(stored?.category)?stored.category:initial,notes:typeof stored?.notes==='string'?stored.notes.slice(0,2000):''};
const row=document.createElement('div');row.className='endpoint-row';row.dataset.endpointId=id;
const label=document.createElement('label');label.textContent=id+' (node '+endpoint.source_node_id+')';
const select=document.createElement('select');select.setAttribute('aria-label',id+' category');select.dataset.endpointId=id;
for(const value of reviewCategories){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option)}select.value=answers[id].category;
select.onchange=()=>{answers[id].category=select.value;saveReview()};label.append(select);row.append(label);
const notesLabel=document.createElement('label');notesLabel.textContent='Notes';
const notes=document.createElement('input');notes.type='text';notes.maxLength=2000;notes.setAttribute('aria-label',id+' notes');notes.value=answers[id].notes;
notes.oninput=()=>{answers[id].notes=notes.value;saveReview()};notesLabel.append(notes);row.append(notesLabel);document.getElementById('endpointReviewRows').append(row)}saveReview()}
function endpointReviewCSV(){const headers=['capture_id','source_graph_sha256','endpoint_id','source_node_id','x','y','z','review_group','prior_disposition','review_status','human_category','human_notes'];
const rows=endpointRows.map(e=>{const a=answers[e.endpoint_id],xyz=nodes.get(e.source_node_id);return[D.capture,D.branchCandidates.source_graph_sha256,e.endpoint_id,e.source_node_id,...xyz,e.review_group||'',e.human_disposition||'',a.category==='not reviewed'?'not reviewed':a.category==='unclear'?'unclear':'reviewed',a.category,a.notes]});
const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';return[headers,...rows].map(row=>row.map(quote).join(',')).join('\\r\\n')}
document.getElementById('exportReview').onclick=()=>{const blob=new Blob(['\\ufeff',endpointReviewCSV()],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');
link.href=url;link.download=D.capture.replace(/[^a-zA-Z0-9_-]/g,'_')+'-endpoint-review.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
const candidateEdges=new Set(),candidateTerminals=[],guardedEdges=new Set(D.branchCandidates?.uncertainty_guard_edge_ids||[]);
const extraEdges=new Set(),confirmedEdges=new Set();
if(D.branchCandidates){document.getElementById('candidates').hidden=false;
for(const c of D.branchCandidates.candidates){for(const e of c.edge_ids){candidateEdges.add(e);if(c.human_disposition==='confirmed_false')confirmedEdges.add(e)}
for(const n of c.terminal_node_ids)candidateTerminals.push([n,c.review_label]);
const row=document.createElement('p');row.textContent=c.review_label+': '+c.edge_ids.length+' edges, '+c.terminal_node_ids.length+' terminal(s)'+(c.human_disposition==='confirmed_false'?' · confirmed false':'')+(c.eligible_for_pruning?'':' · guarded: near a leaf or uncertain base');document.getElementById('candidateList').append(row)}
for(const c of D.branchCandidates.retained_branches||[]){const row=document.createElement('p');row.textContent=c.review_label+': retained, ends in a leaf';row.style.color='#6fdfa6';document.getElementById('candidateList').append(row)}
for(const path of D.branchCandidates.extra_review_paths||[]){for(const e of path.edge_ids)extraEdges.add(e);const row=document.createElement('p');row.textContent=path.review_label+': incoming path flagged for review; shares saved leaf route; kept in preview';row.style.color='#d4a1ff';document.getElementById('candidateList').append(row)}}
for(const leaf of D.leaves){const row=document.createElement('p');row.style.color=colors[leaf.id%colors.length];
const status=leaf.geometry_status==='inferred-petiole'?'inferred-petiole (synthetic)':leaf.geometry_status==='human_reviewed_existing_route'?'human-reviewed extracted route':leaf.association_support==='small_unreviewed_cluster'?'small unreviewed cluster':leaf.geometry_status==='unresolved_no_path'?'petiole path unresolved':'candidate geometry';
row.textContent='Cluster '+leaf.id+' · '+leaf.point_count+' centers · '+status+(leaf.annotation?' · '+JSON.stringify(leaf.annotation,null,1):'');document.getElementById('legend').append(row)}
const PA=D.plantAsset,leafChoice=document.getElementById('leafChoice'),leafSummary=document.getElementById('leafSummary'),exportLeaf=document.getElementById('exportLeaf');
let selectedLeaf=null,press=null;
function describeLeaf(doc){const p=doc.petiole,l=doc.leaf;return l.id+' · '+l.role+' · '+l.point_geometry.point_count+' centers · petiole: '+(doc.petiole_status==='inferred-petiole'?'inferred-petiole (synthetic connection, not measured)':doc.petiole_status==='extracted'?'extracted ('+p.geometry_status+')':'none'+(l.petiole_unresolved_reason?' ('+l.petiole_unresolved_reason+')':l.role==='combined-apex-region'?' (combined apex region)':''))}
function selectLeaf(id){const key=id==null||id===''?null:String(id);selectedLeaf=PA&&key!==null&&PA.exports[key]?key:null;
if(PA){leafChoice.value=selectedLeaf??'';leafSummary.textContent=selectedLeaf?describeLeaf(PA.exports[selectedLeaf]):'No leaf selected.';exportLeaf.disabled=!selectedLeaf}draw();return selectedLeaf}
function leafExportText(id){return JSON.stringify(PA.exports[String(id)])+'\\n'}
if(PA){document.getElementById('leafSelection').hidden=false;leafSummary.textContent='No leaf selected.';
for(const[id,doc]of Object.entries(PA.exports)){const option=document.createElement('option');option.value=id;option.textContent=doc.leaf.id+' ('+doc.leaf.role+(doc.petiole_status==='inferred-petiole'?', inferred petiole':'')+')';leafChoice.append(option)}
leafChoice.onchange=()=>selectLeaf(leafChoice.value);
exportLeaf.onclick=()=>{if(!selectedLeaf)return;const blob=new Blob([leafExportText(selectedLeaf)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
link.href=url;link.download=(PA.plant_id+'-leaf-'+selectedLeaf).replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
window.plantAssetViewer={selectLeaf,leafExportText,selected:()=>selectedLeaf}}
const all=Object.values(D.points).flat().concat([...nodes.values()]);
const lo=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...all.map(p=>p[k])));
const center=lo.map((v,k)=>(v+hi[k])/2),span=Math.max(1e-9,...lo.map((v,k)=>hi[k]-v));
let yaw=-.65,pitch=.28,zoom=1,drag=null;
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
const X=Math.cos(yaw)*x-Math.sin(yaw)*y,Y=Math.sin(yaw)*x+Math.cos(yaw)*y;
const up=Math.cos(pitch)*z-Math.sin(pitch)*Y,depth=Math.cos(pitch)*Y+Math.sin(pitch)*z;
const s=Math.min(canvas.clientWidth,canvas.clientHeight)*.78/span*zoom;
return[canvas.clientWidth/2+X*s,canvas.clientHeight/2-up*s,depth]}
function draw(){ctx.font='12px system-ui';ctx.lineWidth=1;ctx.setLineDash([]);ctx.fillStyle='#10151b';ctx.fillRect(0,0,canvas.clientWidth,canvas.clientHeight);
if(document.getElementById('edges').checked)for(const e of D.graph.skeleton_edges){
const candidate=candidateEdges.has([e.a,e.b].sort((a,b)=>a-b).join('-'));
if(candidate&&document.getElementById('hideCandidates').checked)continue;
if(confirmedEdges.has([e.a,e.b].sort((a,b)=>a-b).join('-'))&&document.getElementById('hideConfirmed').checked)continue;
const highlighted=candidate&&document.getElementById('highlightCandidates').checked;
const extra=extraEdges.has([e.a,e.b].sort((a,b)=>a-b).join('-'))&&document.getElementById('highlightCandidates').checked;
const a=project(nodes.get(e.a)),b=project(nodes.get(e.b));
ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineWidth=e.main?3:highlighted||extra?2.5:1.5;ctx.strokeStyle=e.main?'#ff514b':extra?'#d4a1ff':highlighted?'#ffd166':'#8796a3';ctx.setLineDash(extra||highlighted&&guardedEdges.has([e.a,e.b].sort((a,b)=>a-b).join('-'))?[5,3]:[]);
if(e.classification==='inferred-petiole'){ctx.strokeStyle='#55e6ed';ctx.lineWidth=2.5;ctx.setLineDash([7,5])}ctx.stroke();ctx.setLineDash([])}
if(document.getElementById('points').checked){const pts=[];for(const[id,arr]of Object.entries(D.points))for(const p of arr)pts.push([...project(p),Number(id)]);
pts.sort((a,b)=>a[2]-b[2]);for(const p of pts){const dim=selectedLeaf!==null&&String(p[3])!==selectedLeaf;ctx.globalAlpha=dim?.18:1;ctx.beginPath();ctx.arc(p[0],p[1],selectedLeaf!==null&&!dim?3.2:2.2,0,Math.PI*2);ctx.fillStyle=colors[p[3]%colors.length];ctx.fill()}ctx.globalAlpha=1}
if(selectedLeaf!==null){const doc=PA.exports[selectedLeaf];if(doc.petiole){const line=doc.petiole.polyline.map(project),inferred=doc.petiole_status==='inferred-petiole';ctx.beginPath();line.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
ctx.lineWidth=4;ctx.strokeStyle=inferred?'#55e6ed':'#ffe066';ctx.setLineDash(inferred?[7,5]:[]);ctx.stroke();ctx.setLineDash([]);const a=project(doc.attachment.xyz);ctx.beginPath();ctx.arc(a[0],a[1],7,0,Math.PI*2);ctx.lineWidth=2;ctx.strokeStyle='#ffe066';ctx.stroke();ctx.lineWidth=1}}
for(const l of D.graph.leaves){const p=project(l.insertion_xyz);ctx.beginPath();ctx.arc(p[0],p[1],4.5,0,Math.PI*2);ctx.fillStyle=colors[l.leaf_id%colors.length];ctx.fill();ctx.strokeStyle='white';ctx.stroke();ctx.fillStyle='white';ctx.fillText(String(l.leaf_id)+(l.classification==='inferred-petiole'?' [inferred]':l.source_endpoint_id?' ['+l.source_endpoint_id+']':''),p[0]+7,p[1]-6)}
if(document.getElementById('edges').checked&&document.getElementById('endpointLabels').checked){
const labels=D.branchCandidates?.endpoint_records;
if(labels)for(const e of labels){const p=project(nodes.get(e.source_node_id));ctx.fillStyle=e.status==='retained_leaf_connected'?'#6fdfa6':e.review_group?'#ffd166':'#e2e8ef';ctx.font='bold 13px system-ui';ctx.fillText(e.endpoint_id,p[0]+6,p[1]-5)}
else if(document.getElementById('highlightCandidates').checked&&!document.getElementById('hideCandidates').checked)
for(const[n,label]of candidateTerminals){const p=project(nodes.get(n));ctx.fillStyle='#ffd166';ctx.font='bold 13px system-ui';ctx.fillText(label,p[0]+6,p[1]-5)}}}
function fit(){const d=devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*d);canvas.height=Math.round(canvas.clientHeight*d);ctx.setTransform(d,0,0,d,0,0);draw()}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];press=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
function pickLeaf(e){if(!document.getElementById('points').checked)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,front=-Infinity;
for(const[id,arr]of Object.entries(D.points)){if(!PA.exports[id])continue;for(const p of arr){const q=project(p);if((q[0]-x)**2+(q[1]-y)**2<=64&&q[2]>front){front=q[2];best=id}}}if(best!==null)selectLeaf(best)}
canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};
canvas.onpointerup=e=>{const click=press&&Math.hypot(e.clientX-press[0],e.clientY-press[1])<4;drag=null;press=null;if(click&&PA)pickLeaf(e)};
canvas.onpointercancel=()=>{drag=null;press=null};
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.getElementById('reset').onclick=()=>{yaw=-.65;pitch=.28;zoom=1;draw()};
for(const id of['points','edges','highlightCandidates','hideCandidates','hideConfirmed','endpointLabels'])document.getElementById(id).onchange=draw;
addEventListener('resize',fit);fit();
</script></html>'''
    with (out/'viewer.html').open('x') as f:
        f.write(page.replace('__DATA__',payload))
