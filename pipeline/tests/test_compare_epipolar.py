import csv
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/camera-review/compare-epipolar.py'
spec=importlib.util.spec_from_file_location('epipolar',path)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class EpipolarTests(unittest.TestCase):
 def setUp(self):
  self.cam=(900.,500.,400.,.08)
  self.a=((1.,0.,0.,0.),(0.,0.,0.),self.cam)
  self.b=((math.cos(.2),0.,math.sin(.2),0.),(-.7,.1,.05),self.cam)
  self.point=(1.,.2,5.)
  self.p1=e.project(self.point,self.a);self.p2=e.project(self.point,self.b)
 def test_distortion_rotation_and_nonzero(self):
  x=e.undistort(self.p1,self.cam)
  self.assertAlmostEqual(x[0],.2);self.assertAlmostEqual(x[1],.04)
  ep=e.essential(self.a,self.b)
  self.assertAlmostEqual(e.residual(self.p1,self.p2,self.a,self.b,ep),0,places=7)
  shifted=(self.p2[0],self.p2[1]+10)
  self.assertGreater(e.residual(self.p1,shifted,self.a,self.b,ep),1)
  self.assertAlmostEqual(e.residual(self.p1,shifted,self.a,self.b,ep),e.residual(shifted,self.p1,self.b,self.a,e.essential(self.b,self.a)),places=6)
  with self.assertRaises(ValueError):e.essential(self.a,self.a)
 def test_reversal_and_joint_refusal(self):
  c=sqlite3.connect(':memory:');c.execute('create table two_view_geometries (pair_id,rows,cols,data,config,F)')
  c.execute('insert into two_view_geometries values (?,?,?,?,?,?)',(2147483647+2,1,2,struct.pack('<II',4,7),3,bytes(72)))
  self.assertEqual(e.verified_pairs(c,1,2),[(4,7)]);self.assertEqual(e.verified_pairs(c,2,1),[(7,4)])
  c.execute('update two_view_geometries set config=1')
  with self.assertRaises(ValueError):e.verified_pairs(c,1,2)
  matches=[(0,0),(1,1)];k1=[self.p1,(float('nan'),0)];k2=[self.p2,self.p2]
  ep=e.essential(self.a,self.b); values,refused=e.compare_matches(matches,k1,k2,{'A':(self.a,self.b,ep),'B':(self.a,self.b,ep)})
  self.assertEqual(refused,1);self.assertEqual([len(v) for v in values.values()],[1,1])
 def test_cli_binary_database(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); db=root/'matches.db'; c=sqlite3.connect(db)
   c.executescript('create table cameras(camera_id integer,width integer,height integer); create table images(image_id integer,name text,camera_id integer); create table keypoints(image_id integer,rows integer,cols integer,data blob); create table two_view_geometries(pair_id integer,rows integer,cols integer,data blob,config integer,F blob);')
   c.execute('insert into cameras values (1,1000,800)')
   for i,(name,pt) in enumerate((('006/a.png',self.p1),('006/b.png',self.p2)),1):
    c.execute('insert into images values (?,?,1)',(i,name));c.execute('insert into keypoints values (?,1,2,?)',(i,struct.pack('<ff',*pt)))
   c.execute('insert into two_view_geometries values (?,?,?,?,?,?)',(2147483649,1,2,struct.pack('<II',0,0),3,bytes(72)));c.commit();c.close()
   ws=[]
   for label in ('a','b'):
    w=root/label;folder=w/'sparse'/'0';folder.mkdir(parents=True);ws.append(w)
    with (folder/'cameras.bin').open('wb') as f:f.write(struct.pack('<QiiQQdddd',1,1,2,1000,800,*self.cam))
    with (folder/'images.bin').open('wb') as f:
     f.write(struct.pack('<Q',2))
     for i,(name,pose) in enumerate((('006/a.png',self.a),('006/b.png',self.b)),1):
      f.write(struct.pack('<i7di',i,*pose[0],*pose[1],1));f.write(name.encode()+b'\0');f.write(struct.pack('<Q',0))
   out=root/'out'
   cmd=[sys.executable,str(path),'--database',str(db),'--workspace-a',str(ws[0]),'--workspace-b',str(ws[1]),'--pair','006/b.png','006/a.png','--cross','--out',str(out)]
   subprocess.run(cmd,check=True,capture_output=True,text=True)
   with (out/'epipolar.csv').open(newline='') as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),4);self.assertTrue(all(r['joint_used']=='1' for r in rows));self.assertTrue(all(float(r['median_px'])<.001 for r in rows))
   self.assertIn('not exact',json.loads((out/'metadata.json').read_text())['metric'])
   self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

if __name__=='__main__':unittest.main()
