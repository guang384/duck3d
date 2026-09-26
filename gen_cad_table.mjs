// 从 microduck_rl 的 mjmodel.xml 生成 duck3d 用的资产（一次性工具，跑完可删）：
//   1. 解析 body 树 + visual geom（网格、位姿、材质）
//   2. 逐 geom 施加变换（含 ankle_right 坐标系补偿），材质色烘焙为逐三角颜色
//   3. 输出 duck_cad.bin（全精度，~73.6 万三角面，原生应用用）
//      格式 DUCKCAD3：'DUCKCAD3' | uint32 bodyCount | 每 body 交错记录 {
//        nameLen(2) name triN(4) min(12) max(12) | triN × (9×uint16 量化坐标 + 3×uint8 色) }
import { readFileSync, writeFileSync, mkdirSync, readdirSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

const RL = 'C:/Users/guang/Desktop/microduck_rl';
const HERE = new URL('.', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');   // 本文件所在目录 = 项目根
const xml = readFileSync(join(RL, 'mjmodel.xml'), 'utf8');

// ── 材质表（名称 → rgb 0..1）──
const materials = {};
for (const m of xml.matchAll(/<material name="([^"]+)_material" rgba="([^"]+)"/g)) {
  materials[m[1]] = m[2].split(' ').slice(0, 3).map(Number);
}

// ── 只解析第一个机器人（worldbody → d1_trunk_base 之前）──
const body = xml.slice(xml.indexOf('<worldbody>'), xml.indexOf('d1_trunk_base'));
const SKIP_MESHES = new Set(['np_f970', 'elec_rpi_robot_hat_pcb', 'pcb__raspberry_pi_zero_2_w', 'speaker']);
const geoms = [];
const stack = [];
for (const line of body.split('\n')) {
  const bm = line.match(/<body name="([^"]+)"/);
  if (bm) { stack.push(bm[1]); continue; }
  if (/<\/body>/.test(line)) { stack.pop(); continue; }
  const gm = line.match(/<geom[^>]*class="visual"[^>]*>/);
  if (!gm) continue;
  const tag = gm[0];
  const mesh = tag.match(/mesh="([^"]+)"/)?.[1];
  if (!mesh || SKIP_MESHES.has(mesh)) continue;
  const pos = (tag.match(/pos="([^"]+)"/)?.[1] ?? '0 0 0').split(' ').map(Number);
  const quat = (tag.match(/quat="([^"]+)"/)?.[1] ?? '1 0 0 0').split(' ').map(Number); // MJCF [w,x,y,z]
  const mat = tag.match(/material="([^"]+)_material"/)?.[1] ?? '';
  geoms.push({ body: stack[stack.length - 1], mesh, pos, quat, mat });
}

// ── CAD body 名 → 页面 pivot 名（两份 XML 命名差异）──
const ALIAS = {
  upper_leg_left: 'left_upper_leg',
  upper_leg_right: 'right_upper_leg',
  jaw_soft: 'bottom_head_shell',
  neck_pitch: 'neck_pitch_body',
};
const JAW_SET = new Set(['jaw', 'jaw_soft']);
const FRAME_FIX = new Set(['ankle_right']);

function qRot(v, q) { // MJCF quat [w,x,y,z] 旋转向量
  const w = q[0], qx = q[1], qy = q[2], qz = q[3];
  const uv = [qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0]];
  const uuv = [qy * uv[2] - qz * uv[1], qz * uv[0] - qx * uv[2], qx * uv[1] - qy * uv[0]];
  return [
    v[0] + 2 * (w * uv[0] + uuv[0]),
    v[1] + 2 * (w * uv[1] + uuv[1]),
    v[2] + 2 * (w * uv[2] + uuv[2]),
  ];
}

function stlTris(data) {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const n = dv.getUint32(80, true);
  const out = new Float32Array(n * 9);
  let o = 84;
  for (let i = 0; i < n; i++) {
    o += 12; // 法线，渲染时重算
    for (let k = 0; k < 9; k++) { out[i * 9 + k] = dv.getFloat32(o, true); o += 4; }
    o += 2;
  }
  return out;
}

// 原始 STL 只读一次，两个精度档共用
const rawCache = new Map();
const rawOf = (mesh) => {
  if (!rawCache.has(mesh)) {
    rawCache.set(mesh, stlTris(readFileSync(join(RL, 'src/mjlab_microduck/robot/microduck/assets', mesh + '.stl'))));
  }
  return rawCache.get(mesh);
};

// geom 变换也只做一次：v' = Δ( R·v + p )，Δ 只在 ankle_right 上是 RotZ(180°)
const movedCache = new Map();
const movedOf = (g) => {
  const k = g.mesh + '@' + g.body + '@' + g.pos + '@' + g.quat;
  if (movedCache.has(k)) return movedCache.get(k);
  const raw = rawOf(g.mesh);
  const q = g.quat, p = g.pos, fix = FRAME_FIX.has(g.body);
  const moved = new Float32Array(raw.length);
  for (let v = 0; v < raw.length / 3; v++) {
    let x = raw[v * 3], y = raw[v * 3 + 1], z = raw[v * 3 + 2];
    [x, y, z] = qRot([x, y, z], q);
    x += p[0]; y += p[1]; z += p[2];
    if (fix) { x = -x; y = -y; }
    moved[v * 3] = x; moved[v * 3 + 1] = y; moved[v * 3 + 2] = z;
  }
  movedCache.set(k, moved);
  return moved;
};

// 顶点聚类抽稀：格子从 1.5mm 起步逐步加粗，直到面数 ≤ target（Infinity = 不抽）
function decimate(tris, target) {
  if (tris.length / 9 <= target) return tris;
  let cell = 0.0015;
  let cur = tris;
  while (true) {
    const key = new Map();
    const reps = [];
    const ids = new Int32Array(cur.length / 3);
    for (let v = 0; v < cur.length / 3; v++) {
      const k = Math.round(cur[v * 3] / cell) + ',' + Math.round(cur[v * 3 + 1] / cell) + ',' + Math.round(cur[v * 3 + 2] / cell);
      let id = key.get(k);
      if (id === undefined) {
        id = reps.length;
        key.set(k, id);
        reps.push([0, 0, 0, 0]);
      }
      ids[v] = id;
      reps[id][0] += cur[v * 3]; reps[id][1] += cur[v * 3 + 1]; reps[id][2] += cur[v * 3 + 2]; reps[id][3]++;
    }
    const out = [];
    for (let t = 0; t < cur.length / 9; t++) {
      const a = ids[t * 3], b = ids[t * 3 + 1], c = ids[t * 3 + 2];
      if (a === b || b === c || a === c) continue;      // 退化三角丢弃
      for (const id of [a, b, c]) {
        const r = reps[id];
        out.push(r[0] / r[3], r[1] / r[3], r[2] / r[3]);
      }
    }
    cur = Float32Array.from(out);
    if (cur.length / 9 <= target || cell > 0.03) return cur;
    cell *= 1.5;
  }
}

// ── 打包一个精度档 ──
function emit(target, filename) {
  const bodies = new Map();   // pageBodyName[#jaw] → { tris: number[], col: number[] }
  let totalIn = 0, totalOut = 0;
  for (const g of geoms) {
    const page = ALIAS[g.body] || g.body;
    const isJaw = g.body === 'jaw_soft' && JAW_SET.has(g.mesh);
    const key = page + (isJaw ? '#jaw' : '');
    if (!bodies.has(key)) bodies.set(key, { tris: [], col: [] });
    const dst = bodies.get(key);
    const rgb = materials[g.mat] || [0.7, 0.7, 0.7];
    const moved = movedOf(g);
    totalIn += moved.length / 9;
    const dec = decimate(moved, target);
    totalOut += dec.length / 9;
    for (let k = 0; k < dec.length / 3; k++) {
      dst.tris.push(dec[k * 3], dec[k * 3 + 1], dec[k * 3 + 2]);
      dst.col.push(Math.round(rgb[0] * 255), Math.round(rgb[1] * 255), Math.round(rgb[2] * 255));
    }
  }
  const out = [Buffer.from('DUCKCAD3', 'ascii')];
  const cb = Buffer.alloc(4);
  cb.writeUInt32LE(bodies.size, 0);
  out.push(cb);
  let bytes = 0;
  for (const [name, { tris, col }] of bodies) {
    const triN = tris.length / 9;            // 三角数（tris 是 9 浮点/三角的平铺数组）
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < tris.length; k++) {
      const a = k % 3;
      if (tris[k] < min[a]) min[a] = tris[k];
      if (tris[k] > max[a]) max[a] = tris[k];
    }
    const nameBuf = Buffer.from(name, 'ascii');
    const meta = Buffer.alloc(2 + 4 + 24);
    meta.writeUInt16LE(nameBuf.length, 0);
    meta.writeUInt32LE(triN, 2);
    for (let a = 0; a < 3; a++) meta.writeFloatLE(min[a], 6 + a * 4);
    for (let a = 0; a < 3; a++) meta.writeFloatLE(max[a], 18 + a * 4);
    const data = Buffer.alloc(triN * (9 * 2 + 3));
    let o = 0;
    for (let k = 0; k < triN; k++) {
      for (let v = 0; v < 3; v++) {
        for (let a = 0; a < 3; a++) {
          data.writeUInt16LE(Math.round((tris[k * 9 + v * 3 + a] - min[a]) / (max[a] - min[a] || 1) * 65535), o); o += 2;
        }
      }
      data[o] = col[k * 9]; data[o + 1] = col[k * 9 + 1]; data[o + 2] = col[k * 9 + 2]; o += 3;
    }
    out.push(meta, nameBuf, data);
    bytes += meta.length + nameBuf.length + data.length;
  }
  writeFileSync(join(HERE, 'assets', filename), Buffer.concat(out));
  console.log(`${filename}: tris ${Math.round(totalIn)} -> ${Math.round(totalOut)} | ${(bytes / 1048576).toFixed(2)} MB`);
}

mkdirSync(join(HERE, 'assets'), { recursive: true });
emit(Infinity, 'duck_cad.bin');        // 全精度：原生应用（硬件渲染）

// 只清本工具旧产物（.stl 是老格式的散件）；assets/ 下其他内容（如 mujoco/）不动
for (const f of readdirSync(join(HERE, 'assets'))) {
  if (f.endsWith('.stl')) unlinkSync(join(HERE, 'assets', f));
}
