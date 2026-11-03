"""Compare shared verified matches against two saved camera models. See README.md."""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import struct

def read(f,fmt):
 s=struct.Struct('<'+fmt); x=f.read(s.size)
 if len(x)!=s.size:raise ValueError('truncated binary')
 return s.unpack(x)
def norm(q):
 n=math.sqrt(sum(x*x for x in q))
 if not math.isfinite(n) or n==0:raise ValueError('invalid quaternion')
 return tuple(x/n for x in q)
def mul(a,b):
 w,x,y,z=a; W,X,Y,Z=b
 return (w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W)
def conj(q):return (q[0],-q[1],-q[2],-q[3])
def rotate(q,v):return mul(mul(q,(0,*v)),conj(q))[1:]
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def undistort(p,cam):
 f,cx,cy,k=cam; xd=(p[0]-cx)/f; yd=(p[1]-cy)/f; rd=math.hypot(xd,yd)
 if not rd:return (0.,0.,1.)
 r=rd
 for _ in range(30):
  deriv=1+3*k*r*r
  if deriv<=0:raise ValueError('noninvertible radial mapping')
  delta=(r*(1+k*r*r)-rd)/deriv; r-=delta
  if r<0:raise ValueError('negative undistorted radius')
  if abs(delta)<1e-13:break
 else:raise ValueError('radial inversion did not converge')
 return (xd*r/rd,yd*r/rd,1.)
def essential(a,b):
 qa,ta,ca=a; qb,tb,cb=b
 qr=norm(mul(qb,conj(qa))); rt=rotate(qr,ta); t=tuple(y-x for x,y in zip(rt,tb))
 length=math.sqrt(dot(t,t))
 if length<1e-12:raise ValueError('degenerate baseline')
 return qr,tuple(x/length for x in t)
def constraint(p1,p2,a,b,ep):
 q,t=ep; x=undistort(p1,a[2]); y=undistort(p2,b[2])
 return dot(y,cross(t,rotate(q,x)))
def residual(p1,p2,a,b,ep):
 # Pixel-domain first-order Sampson error, with distortion inversion in the Jacobian.
 g=constraint(p1,p2,a,b,ep); h=.01; grad=[]
 for i in range(4):
  v=list(p1+p2);v[i]+=h; plus=constraint(tuple(v[:2]),tuple(v[2:]),a,b,ep)
  v[i]-=2*h; minus=constraint(tuple(v[:2]),tuple(v[2:]),a,b,ep);grad.append((plus-minus)/(2*h))
 den=math.sqrt(dot(grad,grad))
 if den<1e-15:raise ValueError('degenerate epipolar Jacobian')
 return abs(g)/den
def project(point,pose):
 q,t,cam=pose; v=rotate(q,point); v=tuple(x+y for x,y in zip(v,t)); assert v[2]>0
 x,y=v[0]/v[2],v[1]/v[2]; f,cx,cy,k=cam; d=1+k*(x*x+y*y)
 return (cx+f*x*d,cy+f*y*d)
def load(path):
 with (path/'cameras.bin').open('rb') as f:
  n,=read(f,'Q'); cams={}
  for _ in range(n):
   cid,model,w,h=read(f,'iiQQ')
   if model!=2:raise ValueError('only SIMPLE_RADIAL supported')
   params=read(f,'dddd')
   if cid in cams or not w or not h or params[0]<=0 or not all(math.isfinite(x) for x in params):raise ValueError('invalid camera')
   cams[cid]=(w,h,params)
 with (path/'images.bin').open('rb') as f:
  n,=read(f,'Q'); result={}; info={}
  for _ in range(n):
   v=read(f,'i7di'); name=bytearray()
   while True:
    c=f.read(1)
    if c==b'\0':break
    if not c:raise ValueError('truncated image name')
    name+=c
   name=name.decode(); count,=read(f,'Q')
   if len(f.read(count*24))!=count*24:raise ValueError('truncated image observations')
   if not all(math.isfinite(x) for x in v[5:8]):raise ValueError('invalid translation')
   if name in result:raise ValueError('duplicate image')
   result[name]=(norm(v[1:5]),v[5:8],cams[v[8]][2]); info[name]=(v[8],cams[v[8]][0:2])
 return result,info
def percentile(v,p):
 v=sorted(v); x=(len(v)-1)*p; i=int(x)
 return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)

def verified_pairs(conn,id1,id2):
 low,high=sorted((id1,id2))
 if low==high:raise ValueError('pair names must differ')
 row=conn.execute('select rows,cols,data,config,length(F) from two_view_geometries where pair_id=?',(low*2147483647+high,)).fetchone()
 if row is None:raise ValueError('missing verified geometry')
 n,cols,data,config,flen=row
 # Only uncalibrated fundamental-matrix verification, explicitly supported here.
 if config!=3 or flen!=72 or cols!=2 or data is None or len(data)!=n*8 or n==0:
  raise ValueError('unsupported or unverified geometry; require nonempty UNCALIBRATED F inliers')
 return [(i,j) if id1==low else (j,i) for i,j in struct.iter_unpack('<II',data)]

def keypoints(conn,image_id):
 row=conn.execute('select rows,cols,data from keypoints where image_id=?',(image_id,)).fetchone()
 if row is None:raise ValueError('missing keypoints')
 n,cols,data=row
 if cols<2 or data is None or len(data)!=n*cols*4:raise ValueError('malformed keypoints')
 return [struct.unpack_from('<ff',data,i*cols*4) for i in range(n)]

def compare_matches(matches,k1,k2,combinations):
 values={k:[] for k in combinations}; refused=0
 for i,j in matches:
  if i>=len(k1) or j>=len(k2):raise ValueError('out-of-range correspondence index')
  if not all(math.isfinite(v) for v in (*k1[i],*k2[j])):
   refused+=1; continue
  try:
   r={k:residual(k1[i],k2[j],*v) for k,v in combinations.items()}
   if not all(math.isfinite(x) for x in r.values()):raise ValueError('nonfinite residual')
  except (ValueError,OverflowError,ZeroDivisionError):
   refused+=1; continue
  for k in values:values[k].append(r[k])
 if not next(iter(values.values())):raise ValueError('no jointly valid correspondences')
 return values,refused

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--database',type=Path,required=True)
 parser.add_argument('--workspace-a',type=Path,required=True)
 parser.add_argument('--workspace-b',type=Path,required=True)
 parser.add_argument('--model-a',default='0');parser.add_argument('--model-b',default='0')
 parser.add_argument('--pair',nargs=2,action='append',required=True,metavar=('IMAGE_A','IMAGE_B'))
 parser.add_argument('--cross',action='store_true',help='also cross pose and intrinsic choices')
 parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args(argv)
 if args.out.exists():raise ValueError('output must be a fresh directory')
 if not args.database.is_file():raise ValueError('database does not exist')
 loaded=[load(w/'sparse'/m) for w,m in ((args.workspace_a,args.model_a),(args.workspace_b,args.model_b))]
 poses=[x[0] for x in loaded];info=[x[1] for x in loaded]
 uri=args.database.resolve().as_uri()+'?mode=ro'
 conn=sqlite3.connect(uri,uri=True);conn.execute('pragma query_only=ON')
 rows=[]
 for n1,n2 in args.pair:
  if n1==n2:raise ValueError('pair names must differ')
  if any(n not in model for n in (n1,n2) for model in poses):raise ValueError('model image name missing')
  records=[]
  for name in (n1,n2):
   found=conn.execute('select i.image_id,c.width,c.height from images i join cameras c on c.camera_id=i.camera_id where i.name=?',(name,)).fetchall()
   if len(found)!=1:raise ValueError('database image missing or ambiguous: '+name)
   image_id,width,height=found[0]
   if any(info[i][name][1]!=(width,height) for i in (0,1)):raise ValueError('image domain dimensions disagree: '+name)
   records.append(image_id)
  if (info[0][n1][0]==info[0][n2][0]) != (info[1][n1][0]==info[1][n2][0]):raise ValueError('camera association mismatch')
  matches=verified_pairs(conn,*records); k1,k2=(keypoints(conn,i) for i in records)
  combinations={}
  for p,i in ([(0,0),(1,1),(0,1),(1,0)] if args.cross else [(0,0),(1,1)]):
   a=poses[p][n1][:2]+(poses[i][n1][2],);b=poses[p][n2][:2]+(poses[i][n2][2],)
   combinations[(p,i)]=(a,b,essential(a,b))
  values,refused=compare_matches(matches,k1,k2,combinations)
  for (p,i),v in values.items():
   rows.append({'image_a':n1,'image_b':n2,'poses':'AB'[p],'intrinsics':'AB'[i],
                'verified':len(matches),'joint_used':len(v),'joint_refused':refused,
                'median_px':percentile(v,.5),'p90_px':percentile(v,.9)})
 conn.close()
 args.out.mkdir(parents=True,exist_ok=False)
 with (args.out/'epipolar.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 meta={'database':str(args.database),'workspace_a':str(args.workspace_a),'workspace_b':str(args.workspace_b),
       'model_a':args.model_a,'model_b':args.model_b,'cross':args.cross,
       'metric':'first-order distortion-aware Sampson approximation in pixels, not exact distance',
       'limits':'Matching names and dimensions do not prove identical pixel content or preprocessing; verified matches can contain outliers and are not independent ground truth; large errors are approximate; no acceptance threshold.'}
 (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
 print(f'{len(args.pair)} pairs; {len(rows)} comparison rows: {args.out}')

if __name__=='__main__':main()
