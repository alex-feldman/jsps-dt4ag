"""Explicit PLY/camera inputs for the organ-analysis prototype; see DESIGN.md."""
import json
from pathlib import Path
import numpy as np


def load_ply(path):
    """Read scalar little-endian splat vertex properties, refusing other layouts."""
    types = {'float':'<f4','double':'<f8','uchar':'u1','uint8':'u1','int':'<i4'}
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        props, count, vertex, fmt = [], None, False, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            parts = line.decode('ascii').strip().split()
            if not parts:
                continue
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[:2] == ['element','vertex']:
                count, vertex = int(parts[2]), True
            elif parts[0] == 'element':
                raise ValueError('Only vertex-only splat PLY files are supported')
            elif parts[0] == 'property' and vertex:
                if len(parts) != 3 or parts[1] not in types:
                    raise ValueError('Unsupported PLY property')
                props.append((parts[2], types[parts[1]]))
            elif parts[0] == 'end_header':
                break
        if fmt != 'binary_little_endian' or not count or not props:
            raise ValueError('Expected nonempty binary_little_endian splat vertices')
        names = [p[0] for p in props]
        required = ['x','y','z','opacity','scale_0','scale_1','scale_2','f_dc_0','f_dc_1','f_dc_2']
        if len(set(names)) != len(names) or not set(required).issubset(names):
            raise ValueError('Missing or duplicate splat properties')
        data = np.fromfile(f, dtype=np.dtype(props), count=count)
        if len(data) != count or any(not np.isfinite(data[k]).all() for k in required):
            raise ValueError('Truncated or nonfinite splat data')
        return data


class Geometry:
    def __init__(self, inputs):
        self.inputs = inputs
        self.PLANT = str(inputs['plant_masks'])
        self.LEAF = str(inputs['leaf_instances'])
        self.IMAGES = str(inputs['images'])

    def image_path(self, name):
        return Path(self.IMAGES)/(name + self.inputs['image_extension'])

    def plant_path(self, name):
        return Path(self.PLANT)/(name + self.inputs['mask_extension'])

    def load_ply(self):
        return load_ply(self.inputs['ply'])

    def cameras(self, frame_map):
        with open(self.inputs['transforms']) as f:
            t = json.load(f)
        with open(self.inputs['dataparser_transforms']) as f:
            d = json.load(f)
        if t.get('camera_model','OPENCV') not in ['OPENCV','PINHOLE','SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL']:
            raise ValueError('Unsupported camera model')
        # The verified camera model supports only the first radial coefficient.
        for key in ['k2','k3','k4','p1','p2']:
            if t.get(key, 0) or any(fr.get(key, 0) for fr in t['frames']):
                raise ValueError('Unsupported nonzero distortion: ' + key)
        T, A = np.eye(4), np.eye(4)
        T[:3, :] = np.array(d['transform'])
        A[:3, :] = np.array(t.get('applied_transform', np.eye(4)[:3]))
        T = T @ np.linalg.inv(A)
        scale = float(d['scale'])
        if not np.isfinite(T).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid camera transform/scale')
        out = []
        for fr in t['frames']:
            for key in ['fl_x','fl_y','cx','cy','w','h','k1']:
                if key in fr and fr[key] != t.get(key, 0):
                    raise ValueError('Per-frame intrinsics/distortion are not supported')
            c2w = T @ np.array(fr['transform_matrix'])
            c2w[:3, 3] *= scale
            c2w[:3, 1:3] *= -1
            stem = Path(fr['file_path']).stem
            K = np.array([[t['fl_x'],0,t['cx']],[0,t['fl_y'],t['cy']],[0,0,1.0]])
            if not np.isfinite(c2w).all() or not np.isfinite(K).all() or t['fl_x'] <= 0 or t['fl_y'] <= 0:
                raise ValueError('Invalid camera pose/intrinsics')
            out.append({'frame':stem,'name':frame_map[stem],'K':K,'w2c':np.linalg.inv(c2w),
                        'k1':t.get('k1',0.0),'w':t['w'],'h':t['h']})
        return sorted(out, key=lambda c:c['name'])

    @staticmethod
    def project(cam, X):
        Xc = X @ cam['w2c'][:3,:3].T + cam['w2c'][:3,3]
        z = Xc[:,2]
        x, y = Xc[:,0]/z, Xc[:,1]/z
        r2 = x*x + y*y
        f = 1 + cam['k1']*r2
        K = cam['K']
        return K[0,0]*x*f + K[0,2], K[1,1]*y*f + K[1,2], z
