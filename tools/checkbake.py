"""校验 CAD 烘焙：源模型每个 body 的视觉网格（STL+geom 位姿）AABB vs duck_cad.bin 里烘焙的 AABB"""
import re, struct, io, os
import numpy as np

RL = 'C:/Users/guang/Desktop/microduck_rl'
MESH_DIR = os.path.join(RL, 'src/mjlab_microduck/robot/microduck/assets')
xml = io.open(os.path.join(RL, 'mjmodel.xml'), encoding='utf-8').read()

# ── 1) 解析 body 树 + visual geom（mesh,pos,quat）──
bodies = {}   # name -> (pos, quat, parent)
geoms = {}    # body -> [(mesh, pos, quat)]
for m in re.finditer(r'<body name="([^"]+)"([^>]*)>(.*?)(?=<body |</worldbody>)', xml, re.S):
    name, attrs, inner = m.group(1), m.group(2), m.group(3)
    pos = [float(x) for x in re.search(r'pos="([^"]+)"', attrs).group(1).split()] if 'pos=' in attrs else [0,0,0]
    quat = [float(x) for x in re.search(r'quat="([^"]+)"', attrs).group(1).split()] if 'quat=' in attrs else [1,0,0,0]
    bodies[name] = (pos, quat)
    gs = []
    for g in re.finditer(r'<geom ([^>]*)/>', inner):
        ga = g.group(1)
        if 'class="visual"' not in ga or 'mesh=' not in ga: continue
        mesh = re.search(r'mesh="([^"]+)"', ga).group(1)
        gp = [float(x) for x in re.search(r'pos="([^"]+)"', ga).group(1).split()] if 'pos=' in ga else [0,0,0]
        gq = [float(x) for x in re.search(r'quat="([^"]+)"', ga).group(1).split()] if 'quat=' in ga else [1,0,0,0]
        gs.append((mesh, gp, gq))
    geoms[name] = gs

def quat_mat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def stl_verts(path):
    with open(path,'rb') as f:
        f.read(80); n = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n*50), dtype=np.uint8).reshape(n,50)
        v = data[:,12:48].copy().view('<f4').reshape(n,3,3)
    return v.reshape(-1,3)

memo = {}
def load(mesh):
    if mesh not in memo:
        p = os.path.join(MESH_DIR, mesh + '.stl')
        memo[mesh] = stl_verts(p) if os.path.exists(p) else None
    return memo[mesh]

# ── 2) 读 duck_cad.bin 的每 body AABB ──
raw = io.open('assets/duck_cad.bin','rb').read()
assert raw[:8] == b'DUCKCAD3', raw[:8]
off = 8
nbody = struct.unpack_from('<I', raw, off)[0]; off += 4
bin_bodies = {}
for _ in range(nbody):
    # meta: nameLen(2) triN(4) min(12) max(12)；随后才是 name，再是量化三角
    nl = struct.unpack_from('<H', raw, off)[0]
    tri = struct.unpack_from('<I', raw, off + 2)[0]
    lo = np.array(struct.unpack_from('<3f', raw, off + 6))
    hi = np.array(struct.unpack_from('<3f', raw, off + 18))
    off += 30
    name = raw[off:off+nl].decode('latin-1'); off += nl
    off += tri * (9*2 + 3)
    bin_bodies[name] = (lo, hi, tri)

# ── 3) 对比 ──
print(f'bin bodies={len(bin_bodies)}  xml bodies={len(bodies)}')
print(f'{"body":26} {"Δmin(mm)":>10} {"Δmax(mm)":>10}  三角形')
worst = []
ALIAS = {'upper_leg_left': 'left_upper_leg', 'upper_leg_right': 'right_upper_leg',
         'jaw_soft': 'bottom_head_shell', 'neck_pitch': 'neck_pitch_body'}
JAW_SET = {'jaw', 'jaw_soft'}
def src_geoms_for(key):
    is_jaw = key.endswith('#jaw')
    page = key[:-4] if is_jaw else key
    out = []
    for bname, gs in geoms.items():
        if (ALIAS.get(bname, bname)) != page:
            continue
        for mesh, gp, gq in gs:
            if (mesh in JAW_SET) == is_jaw:
                out.append((mesh, gp, gq))
    return out

for name, (lo, hi, tri) in bin_bodies.items():
    gs = src_geoms_for(name)
    if not gs: continue
    pts = []
    for mesh, gp, gq in gs:
        v = load(mesh)
        if v is None: continue
        pts.append(v @ quat_mat(gq).T + np.array(gp))
    if not pts: continue
    P = np.vstack(pts)
    slo, shi = P.min(axis=0), P.max(axis=0)
    d = np.abs(np.array([slo, shi]) - np.array([lo, hi])) * 1000
    worst.append((max(d.max(axis=1)), name, d[0], d[1], tri))
worst.sort(reverse=True)
print('偏差最大的 8 个 body（Δ = |源AABB − 烘焙AABB|）:')
for w, name, dlo, dhi, tri in worst[:8]:
    print(f'{name:26} {dlo.max():10.2f} {dhi.max():10.2f}  {tri}')
print(f'\n全部 {len(worst)} 个可比 body 中，最大偏差 {worst[0][0]:.2f} mm')
