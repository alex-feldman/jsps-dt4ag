"""Self-contained Canvas 2D inspection viewer; no reconstruction or inference."""
import json
import numpy as np


def write_viewer(out, bundle):
    with np.load(out/'leaf_ids2.npz') as arrays:
        points = {}
        for leaf in bundle['leaves']:
            selected = arrays['X'][arrays['label']==leaf['id']]
            # Deterministic display sample; full data remain in the NPZ.
            positions = np.linspace(0,len(selected)-1,min(300,len(selected)),dtype=int)
            points[str(leaf['id'])] = selected[positions].tolist()
    graph = bundle['source_graph'] or {'skeleton_nodes':[],'skeleton_edges':[],'leaves':[]}
    data = {'capture':bundle['capture_id'],'points':points,'graph':graph,'leaves':bundle['leaves']}
    payload = json.dumps(data,allow_nan=False).replace('<','\\u003c')
    page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Organ analysis viewer</title><style>
html,body{height:100%;margin:0;background:#10151b;color:#edf2f7;font:14px system-ui}
body{display:grid;grid-template-columns:minmax(0,1fr) 290px}canvas{width:100%;height:100%;touch-action:none}
aside{padding:18px;overflow:auto;background:#19212b;overflow-wrap:anywhere}p{line-height:1.5}button{padding:8px}
@media(max-width:700px){body{grid-template-columns:1fr;grid-template-rows:65% 35%}}
</style><canvas id="view" aria-label="Provisional organ points and skeleton"></canvas><aside>
<h1 id="title"></h1><p>Drag to rotate; wheel to zoom. Reset uses dataparser z as approximate up, not gravity.</p>
<button id="reset">Reset view</button><p><label><input id="points" type="checkbox" checked>Leaf points</label></p>
<p><label><input id="edges" type="checkbox" checked>Skeleton</label></p><div id="legend"></div>
<p>Red: candidate main axis. Gray: unresolved other paths. Colored dots: at most 300 centers per leaf cluster.
These are Gaussian centers and graph lines, not rendered splats or measured organs. Zero-length fallback paths do not mean absent petioles.
No tag/string removal, gravity alignment, metric scale, or biological acceptance is implied.</p></aside>
<script>const D=__DATA__;
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const colors=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7'];
document.getElementById('title').textContent=D.capture+' organ evidence';
const nodes=new Map(D.graph.skeleton_nodes.map(n=>[n.id,n.xyz]));
for(const leaf of D.leaves){const row=document.createElement('p');row.style.color=colors[leaf.id%colors.length];
const status=leaf.association_support==='small_unreviewed_cluster'?'small unreviewed cluster':leaf.geometry_status==='unresolved_no_path'?'petiole path unresolved':'candidate geometry';
row.textContent='Cluster '+leaf.id+' · '+leaf.point_count+' centers · '+status+(leaf.annotation?' · '+JSON.stringify(leaf.annotation,null,1):'');document.getElementById('legend').append(row)}
const all=Object.values(D.points).flat().concat([...nodes.values()]);
const lo=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...all.map(p=>p[k])));
const center=lo.map((v,k)=>(v+hi[k])/2),span=Math.max(1e-9,...lo.map((v,k)=>hi[k]-v));
let yaw=-.65,pitch=.28,zoom=1,drag=null;
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
const X=Math.cos(yaw)*x-Math.sin(yaw)*y,Y=Math.sin(yaw)*x+Math.cos(yaw)*y;
const up=Math.cos(pitch)*z-Math.sin(pitch)*Y,depth=Math.cos(pitch)*Y+Math.sin(pitch)*z;
const s=Math.min(canvas.clientWidth,canvas.clientHeight)*.78/span*zoom;
return[canvas.clientWidth/2+X*s,canvas.clientHeight/2-up*s,depth]}
function draw(){ctx.fillStyle='#10151b';ctx.fillRect(0,0,canvas.clientWidth,canvas.clientHeight);
if(document.getElementById('edges').checked)for(const e of D.graph.skeleton_edges){const a=project(nodes.get(e.a)),b=project(nodes.get(e.b));
ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineWidth=e.main?3:1.5;ctx.strokeStyle=e.main?'#ff514b':'#8796a3';ctx.stroke()}
if(document.getElementById('points').checked){const pts=[];for(const[id,arr]of Object.entries(D.points))for(const p of arr)pts.push([...project(p),Number(id)]);
pts.sort((a,b)=>a[2]-b[2]);for(const p of pts){ctx.beginPath();ctx.arc(p[0],p[1],2.2,0,Math.PI*2);ctx.fillStyle=colors[p[3]%colors.length];ctx.fill()}}
for(const l of D.graph.leaves){const p=project(l.insertion_xyz);ctx.beginPath();ctx.arc(p[0],p[1],4.5,0,Math.PI*2);ctx.fillStyle=colors[l.leaf_id%colors.length];ctx.fill();ctx.strokeStyle='white';ctx.stroke();ctx.fillStyle='white';ctx.fillText(String(l.leaf_id),p[0]+7,p[1]-6)}}
function fit(){const d=devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*d);canvas.height=Math.round(canvas.clientHeight*d);ctx.setTransform(d,0,0,d,0,0);draw()}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.getElementById('reset').onclick=()=>{yaw=-.65;pitch=.28;zoom=1;draw()};
for(const id of['points','edges'])document.getElementById(id).onchange=draw;
addEventListener('resize',fit);fit();
</script></html>'''
    with (out/'viewer.html').open('x') as f:
        f.write(page.replace('__DATA__',payload))
