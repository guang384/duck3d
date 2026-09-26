"""站立姿态下各 body 视觉网格的世界 AABB 两两相交检测。

数据来源：源模型 microduck_rl/mjmodel.xml（body 树 + visual geom + STL）
          + 我们的 BodyDef 表（duck3d/app/src/main.rs）
          + 站立关节角（duck3d/assets/mj/scene.xml 的 STAND keyframe）
FK：world = parent_world · T(body.pos, body.quat) · R(joint.axis, θ)
"""
import re, io, os, struct
import numpy as np

RL = 'C:/Users/guang/Desktop/microduck_rl'
MD = os.path.join(RL, 'src/mjlab_microduck/robot/microduck/assets')
xml = io.open(os.path.join(RL, 'mjmodel.xml'), encoding='utf-8').read()
main = io.open('app/src/main.rs', encoding='utf-8').read()

# ── 1) 源模型：栈式解析 body / joint / visual geom ──
ALIAS = {'upper_leg_left': 'left_upper_leg', 'upper_leg_right': 'right_upper_leg',
         'neck_pitch': 'neck_pitch_body'}
axes, geoms = {}, {}
stack = []
for line in xml.split('\n'):
    bm = re.search(r'<body name="([^"]+)"', line)
    if bm:
        stack.append(bm.group(1))
        continue
    if '</body>' in line:
        if stack:
            stack.pop()
        continue
    if not stack:
        continue
    body = ALIAS.get(stack[-1], stack[-1])
    jm = re.search(r'<joint [^>]*axis="([^"]+)"', line)
    if jm:
        axes[body] = [float(v) for v in jm.group(1).split()]
    gm = re.search(r'<geom[^>]*class="visual"[^>]*>', line)
    if gm:
        tag = gm.group(0)
        mesh = re.search(r'mesh="([^"]+)"', tag)
        if not mesh:
            continue
        gp = re.search(r'pos="([^"]+)"', tag)
        gq = re.search(r'quat="([^"]+)"', tag)
        geoms.setdefault(body, []).append((
            mesh.group(1),
            [float(v) for v in gp.group(1).split()] if gp else [0, 0, 0],
            [float(v) for v in gq.group(1).split()] if gq else [1, 0, 0, 0]))

# ── 2) 我们的 BodyDef 表（顺序即渲染顺序）──
defs = []
for m in re.finditer(r'BodyDef \{ name: "([^"]+)", parent: (\w+|\d+), pos: \[([^\]]+)\], quat: \[([^\]]+)\], joint: ([^}]+)\}', main):
    defs.append(dict(name=m.group(1), parent=m.group(2),
                     pos=[float(x) for x in m.group(3).split(',')],
                     quat=[float(x) for x in m.group(4).split(',')],
                     joint=m.group(5).strip()))

def qmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def T(p, q):
    M = np.eye(4); M[:3, :3] = qmat(q); M[:3, 3] = p; return M

def rot_axis(ax, th):
    ax = np.array(ax, float); ax /= (np.linalg.norm(ax) or 1.0)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(4); R[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return R

# ── 3) 站立关节角（scene.xml 的 STAND keyframe）──
scene = io.open('assets/mj/scene.xml', encoding='utf-8').read()
scene = re.sub(r'<!--.*?-->', '', scene, flags=re.S)   # 去掉注释里的旧 keyframe
qpos = [float(v) for v in re.search(r'<key name="STAND"\s+qpos="([^"]+)"', scene, re.S).group(1).split()]
angles = qpos[7:]                       # 前 7 位是 trunk 自由关节；训练模型里嘴是刚性件 → 只有 14 个关节
def ang(proto_idx):                     # 我们的 BodyDef 用协议关节序（含嘴 idx 9）→ 映射到 MuJoCo 序
    return angles[proto_idx if proto_idx < 9 else proto_idx - 1]

# ── 4) FK ──
world = {}
for d in defs:
    if d['joint'] == 'None':
        M = T(qpos[:3], qpos[3:7]) @ T(d['pos'], d['quat'])
    else:
        k = int(re.search(r'(\d+)', d['joint']).group(1))
        pn = d['parent']
        par = defs[0] if pn in ('TRUNK', '0') else (defs[int(pn)] if pn.isdigit()
             else next(x for x in defs if x['name'] == pn))
        M = world[par['name']] @ T(d['pos'], d['quat']) @ rot_axis(axes.get(d['name'], [0, 0, 1]), ang(k))
    world[d['name']] = M

memo = {}
def verts(mesh):
    if mesh in memo:
        return memo[mesh]
    p = os.path.join(MD, mesh + '.stl')
    if not os.path.exists(p):
        memo[mesh] = None; return None
    f = open(p, 'rb'); f.read(80)
    n = struct.unpack('<I', f.read(4))[0]
    d = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    v = d[:, 12:48].copy().view('<f4').reshape(-1, 3)
    f.close(); memo[mesh] = v; return v

A = {}
for name, M in world.items():
    pts = []
    for mesh, gp, gq in geoms.get(name, []):
        v = verts(mesh)
        if v is None:
            continue
        pts.append(v @ qmat(gq).T + np.array(gp))
    if not pts:
        continue
    P = np.vstack(pts) @ M[:3, :3].T + M[:3, 3]
    A[name] = (P.min(axis=0), P.max(axis=0))

# ── 5) 两两 AABB 相交 ──
names = sorted(A)
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        lo = np.maximum(A[names[i]][0], A[names[j]][0])
        hi = np.minimum(A[names[i]][1], A[names[j]][1])
        ov = hi - lo
        if ov.min() > 0.0005:
            pairs.append((float(ov.min() * 1000), names[i], names[j], ov * 1000))
pairs.sort(reverse=True)
print('站立姿态下视觉包围盒相交的 body 对（> 0.5mm）：')
for ov, n1, n2, o in pairs:
    print(f'  {n1:20} × {n2:20} 最小重叠 {ov:6.2f} mm   (x{o[0]:5.1f} y{o[1]:5.1f} z{o[2]:5.1f})')
print(f'共 {len(pairs)} 对')
