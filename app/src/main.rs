// Windows：release 版不弹终端窗口（GUI 子系统）；debug 版保留控制台看日志
#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

//! duck3d 原生应用 —— 网页版的重构：同样的鸭子、同样的策略、同样的控制调度，
//! 但渲染走 wgpu(DX12/Vulkan) 硬件管线，仿真循环是原生线程的 50Hz 定时器，
//! 没有浏览器定时器节流、没有软件光栅化。
//!
//! 数据契约与网页版完全一致：
//!   - 鸭子外观：assets/duck_cad.bin（DUCKCAD3，gen_cad_table.mjs 生成）
//!   - 策略：仓库 policies/*.onnx，obs[1,61] → actions[1,14]（duck-control/obs.rs 布局）
//!   - 调度：robotd/src/control.rs 的移植（优先级链、技能窗口、缩放/增益、训练低通）
//!
//! 运行：cargo run --release（在 duck3d/app 下）。--selftest 无窗口跑仿真自检。

mod hud;

use hud::{apply_cmd, build_egui_ctx, build_hud, DlgPick, UiCmd};
mod render;

use render::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use std::sync::Arc;
use glam::{Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/* ════════════════════════════════════════════════════════════════════════
   1 · 常量 —— duck-control/src/model.rs、obs.rs、robotd 一字对齐
   ════════════════════════════════════════════════════════════════════════ */

const NUM_JOINTS: usize = 15;
pub(crate) const MOUTH_INDEX: usize = 9;
const ACTION_LEN: usize = 14;
const OBS_LEN: usize = 61;
const TICK_DT: f64 = 1.0 / 50.0;

const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
];

// model.rs DEFAULT_POSITION（= 训练环境 HOME_FRAME）
pub(crate) const DEFAULT_POSITION: [f64; NUM_JOINTS] = [
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
];

// control.rs Tuning::default() —— alpha 出厂值，滤波是训练值
const ACTION_SCALE: f64 = 0.9;
const STANDING_ACTION_SCALE: f64 = 1.0;
const STANDING_GAIN_RATIO: f64 = 0.8;
const GAIN: u16 = 200;
const HEAD_LOWPASS: f64 = 0.5;
const LEGS_LOWPASS: f64 = 0.7;

const GROUND_PICK_PERIOD: f64 = 4.0;
const GROUND_PICK_END_PHASE: f64 = 0.7;
pub(crate) const KICK_DURATION: f64 = 3.0;   // 官方 infer_policy.py 的 kick_duration 默认值
pub(crate) const ROULADE_DURATION: f64 = 2.0;   // 官方 infer_policy.py 默认值；到点没立起会自适应续窗
const ROULADE_CHAIN_WINDOW: f64 = 0.15;
pub(crate) const RISE_SECS: f64 = 1.0;
const STANDING_THRESHOLD: f64 = 0.05;

const CMD_ALPHA: f64 = 0.2;
const HEAD_ALPHA: f64 = 0.2;
/// body-pose 平滑系数（robotd 的 pose glide）：0.3 ≈ 0.2s 到位。
/// 比 twist/head 慢一档是刻意的（站姿指令直接改伺服目标），但 0.1 太肉（~0.9s），
/// 键鼠调姿都嫌迟滞。
const BODY_ALPHA: f64 = 0.3;
pub(crate) const MAX_LINEAR: f64 = 0.3;
pub(crate) const MAX_ANGULAR: f64 = 1.5;
pub(crate) const MAX_HEAD: f64 = 2.5;
/// 站姿指令行程：z 蹲/升（m，非对称）、roll/pitch 滑条行程（rad）。
/// 三处共用这一份：tick 的限幅、键盘满偏、HUD 的条量程与滑条范围。
pub(crate) const BODY_Z_RANGE: [f64; 2] = [-0.040, 0.030];   // 训练课程最终档（standup env）
pub(crate) const BODY_TILT_RANGE: f64 = 0.50;               // 滑条允许到 ±0.5rad ≈ ±29°（超出训练域，实验用）
pub(crate) const BODY_KEY_TILT: f64 = 0.26;                 // 键盘满偏 = 训练行程 ±0.26rad ≈ ±15°
const DEADZONE: f64 = 0.1;
const SERVO_TAU: f64 = 0.05;

const MOUTH_CLOSED: f64 = -5.0 * std::f64::consts::PI / 180.0;

// MJCF 各关节行程（rad），物理铰链限位
pub(crate) const JOINT_RANGE: [[f32; 2]; 14] = [
    [-0.436332, 0.523599],   // left_hip_yaw（观测槽序，无嘴）
    [-0.383972, 0.383972],   // left_hip_roll
    [-1.570796, 1.570796],   // left_hip_pitch
    [-1.570796, 1.570796],   // left_knee
    [-1.570796, 1.570796],   // left_ankle
    [-1.570796, 1.047197],   // neck_pitch
    [-1.570796, 1.570796],   // head_pitch
    [-2.967060, 2.967060],   // head_yaw
    [-0.436332, 0.436332],   // head_roll
    [-0.523599, 0.436332],   // right_hip_yaw
    [-0.383972, 0.383972],   // right_hip_roll
    [-1.570796, 1.570796],   // right_hip_pitch
    [-1.570796, 1.570796],   // right_knee
    [-1.570796, 1.570796],   // right_ankle
];
const MOUTH_OPEN: f64 = 30.0 * std::f64::consts::PI / 180.0;


// 相机拖拽灵敏度（rad/px、m/px）。0.006 起步时转得晃眼，调慢到 1/3；
// 俯仰再慢一点，因为它同时受上下两个极限约束。

// 观测里的"策略关节序"：跳过嘴（obs.rs joint_of 的镜像）
/// 环境开关：热路径（50Hz tick、每帧 render）里反复 `std::env::var` 会带来
/// String 分配 + 环境查找，而开关在一次运行内不变 —— 每个开关一个 OnceLock，
/// 初始化后只剩一次原子读。
macro_rules! env_flag {
    ($fn_name:ident, $var:literal) => {
        pub(crate) fn $fn_name() -> bool {
            static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *V.get_or_init(|| std::env::var($var).is_ok())
        }
    };
}
env_flag!(keylog, "DUCK3D_KEYLOG");       // 按键/tick 日志
env_flag!(norender, "DUCK3D_NORENDER");   // 隔离实验：只跳渲染
env_flag!(nohud, "DUCK3D_NOHUD");         // 关掉 HUD
env_flag!(shot_hud, "DUCK3D_SHOT_HUD");   // 离屏帧叠 HUD

fn joint_of(slot: usize) -> usize { if slot < MOUTH_INDEX { slot } else { slot + 1 } }
fn policy_joints(v: &[f64; NUM_JOINTS]) -> [f64; 14] {
    let mut out = [0.0; 14];
    for s in 0..14 { out[s] = v[joint_of(s)]; }
    out
}
fn scatter_action(a: &[f32; ACTION_LEN]) -> [f64; NUM_JOINTS] {
    let mut out = [0.0; NUM_JOINTS];
    for (s, v) in a.iter().enumerate() { out[joint_of(s)] = *v as f64; }
    out
}

/* ════════════════════════════════════════════════════════════════════════
   2 · 运动学树 —— kinematics/assets/alpha/robot_walk.xml
   ════════════════════════════════════════════════════════════════════════ */

struct BodyDef {
    name: &'static str,
    parent: usize,
    pos: [f32; 3],
    quat: [f32; 4],         // MJCF [w,x,y,z]
    joint: Option<usize>,   // DEFAULT_POSITION 索引
}

const TRUNK: usize = 0;
const HEAD: usize = 9;
const JAW_SLOT: usize = 15; // BODIES.len()，fk 输出的第 16 格

const BODIES: &[BodyDef] = &[
    BodyDef { name: "trunk_base", parent: 0, pos: [0.0, 0.0, 0.12], quat: [1.0, 0.0, 0.0, 0.0], joint: None },
    BodyDef { name: "yaw2roll", parent: TRUNK, pos: [0.006, 0.0175, -0.005], quat: [0.0, -0.707107, -0.707107, 0.0], joint: Some(0) },
    BodyDef { name: "hip_l", parent: 1, pos: [0.0, 0.0165, 0.0125], quat: [0.707107, -0.707107, 0.0, 0.0], joint: Some(1) },
    BodyDef { name: "left_upper_leg", parent: 2, pos: [0.025, 0.0, -0.0185], quat: [0.5, -0.5, 0.5, -0.5], joint: Some(2) },
    BodyDef { name: "leg", parent: 3, pos: [0.022, 0.0357771, -0.004], quat: [0.0, 0.707107, 0.707107, 0.0], joint: Some(3) },
    BodyDef { name: "ankle_left", parent: 4, pos: [0.0, 0.042, -0.026], quat: [0.0, 1.0, 0.0, 0.0], joint: Some(4) },
    BodyDef { name: "neck", parent: TRUNK, pos: [0.026, 0.0145011, 0.0324424], quat: [0.0, 0.0, -0.707107, 0.707107], joint: Some(5) },
    BodyDef { name: "neck_pitch_body", parent: 6, pos: [0.0, -0.05, 0.0], quat: [0.0, 1.0, 0.0, 0.0], joint: Some(6) },
    BodyDef { name: "yaw_roll_motion", parent: 7, pos: [0.0, 0.0186931, -0.0145], quat: [0.0, 0.0, -0.707107, -0.707107], joint: Some(7) },
    BodyDef { name: "bottom_head_shell", parent: 8, pos: [-0.0179, 0.0, 0.0145], quat: [0.707107, 0.0, -0.707107, 0.0], joint: Some(8) },
    BodyDef { name: "bearing_roll", parent: TRUNK, pos: [0.006, -0.0175, -0.005], quat: [0.0, -0.707107, -0.707107, 0.0], joint: Some(10) },
    BodyDef { name: "hip_l_2", parent: 10, pos: [0.0, 0.0165, 0.0125], quat: [0.0, 0.0, 0.707107, 0.707107], joint: Some(11) },
    BodyDef { name: "right_upper_leg", parent: 11, pos: [0.025, 0.0, -0.0185], quat: [0.5, -0.5, 0.5, -0.5], joint: Some(12) },
    BodyDef { name: "leg_2", parent: 12, pos: [-0.022, 0.0357771, -0.004], quat: [0.0, 1.0, 0.0, 0.0], joint: Some(13) },
    BodyDef { name: "ankle_right", parent: 13, pos: [-0.042, 0.0, -0.026], quat: [0.0, -0.707107, 0.707107, 0.0], joint: Some(14) },
];

const FOOT_SITES: [(usize, [f32; 3]); 2] = [
    (5, [0.0, -0.0237879, -0.0140852]),
    (14, [0.0, 0.0237879, -0.0140852]),
];

const JAW_HINGE: [f32; 3] = [-0.0045, 0.0, -0.018];

/// 各 body 的世界矩阵（y-up 世界；MJCF z-up 经根部的 Rx(-90°) 转换）。
/// 末尾第 16 格是下颚（铰链系顶点 + 绕壳系 y 开合）。
/// 支架托举量：把最低脚点顶到 y=0.004（render 的平滑推进与离屏的一次性取值共用）
fn rig_lift_target(models: &[glam::Mat4]) -> f32 {
    let mut min_foot_y = f32::INFINITY;
    for (i, site) in FOOT_SITES {
        min_foot_y = min_foot_y.min(models[i].transform_point3(Vec3::from_array(site)).y);
    }
    if min_foot_y < 0.004 { 0.004 - min_foot_y } else { 0.0 }
}

fn fk(q: &[f64; NUM_JOINTS]) -> Vec<glam::Mat4> {
    let mut m = vec![glam::Mat4::IDENTITY; BODIES.len() + 1];
    let root = glam::Mat4::from_quat(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));
    m[TRUNK] = root * glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 0.12));
    for (i, b) in BODIES.iter().enumerate().skip(1) {
        let rot = Quat::from_xyzw(b.quat[1], b.quat[2], b.quat[3], b.quat[0])
            * b.joint.map(|j| Quat::from_rotation_z(q[j] as f32)).unwrap_or(Quat::IDENTITY);
        let local = glam::Mat4::from_translation(Vec3::from_array(b.pos)) * glam::Mat4::from_quat(rot);
        m[i] = m[b.parent] * local;
    }
    m[JAW_SLOT] = m[HEAD]
        * glam::Mat4::from_translation(Vec3::from_array(JAW_HINGE))
        // 下颚：CAD 按"闭口"打包，而 q 是绝对目标角（闭口 = MOUTH_CLOSED）—— 减去它即相对铰链角
        * glam::Mat4::from_quat(Quat::from_rotation_y((q[MOUTH_INDEX] - MOUTH_CLOSED) as f32));
    m
}

/* ════════════════════════════════════════════════════════════════════════
   3 · CAD 包解码（DUCKCAD3，与 gen_cad_table.mjs / 网页版一致）
   ════════════════════════════════════════════════════════════════════════ */


struct CadBody {
    slot: usize,   // fk 输出的槽位
    start: u32,
    count: u32,
}

struct Cad {
    vertices: Vec<Vertex>,
    bodies: Vec<CadBody>,
}

/// CAD 没有眼睛网格 —— 生成两颗 UV 球（头壳局部系，位置与网页版一致）。
fn add_eyes(cad: &mut Cad) {
    const EYE_POS: [f32; 3] = [0.013, 0.0185, -0.048];   // (上, 左, 后)
    const R: f32 = 0.0065;
    const SEG_U: u32 = 16;
    const SEG_V: u32 = 12;
    let start = cad.vertices.len() as u32;
    for side in [-1.0f32, 1.0f32] {
        let c = [EYE_POS[0], side * EYE_POS[1], EYE_POS[2]];
        let push_tri = |a: [f32; 3], b: [f32; 3], d: [f32; 3], v: &mut Vec<Vertex>| {
            let e1 = Vec3::from_array(b) - Vec3::from_array(a);
            let e2 = Vec3::from_array(d) - Vec3::from_array(a);
            let n = e1.cross(e2).normalize_or_zero();
            for p in [a, b, d] {
                v.push(Vertex { pos: p, nrm: [n.x, n.y, n.z], col: [0.06, 0.07, 0.09] });
            }
        };
        let pt = |i: u32, j: u32| -> [f32; 3] {
            let u = i as f32 / SEG_U as f32 * std::f32::consts::TAU;
            let v = j as f32 / SEG_V as f32 * std::f32::consts::PI;
            [
                c[0] + R * v.sin() * u.cos(),
                c[1] + R * v.sin() * u.sin(),
                c[2] + R * v.cos(),
            ]
        };
        for i in 0..SEG_U {
            for j in 0..SEG_V {
                let a = pt(i, j);
                let b = pt(i + 1, j);
                let d = pt(i, j + 1);
                let e = pt(i + 1, j + 1);
                push_tri(a, b, d, &mut cad.vertices);
                push_tri(b, e, d, &mut cad.vertices);
            }
        }
    }
    let count = cad.vertices.len() as u32 - start;
    cad.bodies.push(CadBody { slot: HEAD, start, count });
}

fn load_cad(path: &Path) -> Result<Cad> {
    let buf = std::fs::read(path).context("读 duck_cad.bin")?;
    if buf.len() < 12 || &buf[0..8] != b"DUCKCAD3" {
        bail!("duck_cad.bin 魔数不对（旧格式？重跑 gen_cad_table.mjs 生成）");
    }
    let rd_u16 = |o: usize| u16::from_le_bytes([buf[o], buf[o + 1]]);
    let rd_u32 = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    let rd_f32 = |o: usize| f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);

    let mut off = 8;
    let body_count = rd_u32(off) as usize;
    off += 4;
    let mut vertices = Vec::new();
    let mut bodies = Vec::new();
    let slot_of = |name: &str| -> usize {
        if let Some(i) = BODIES.iter().position(|b| b.name == name) { i }
        else if name == "bottom_head_shell#jaw" { JAW_SLOT }
        else { usize::MAX }
    };
    for _ in 0..body_count {
        let name_len = rd_u16(off) as usize; off += 2;
        let tri_n = rd_u32(off) as usize; off += 4;
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for a in 0..3 {
            min[a] = rd_f32(off + a * 4);
            max[a] = rd_f32(off + 12 + a * 4);
        }
        off += 24;
        let name = String::from_utf8_lossy(&buf[off..off + name_len]).to_string();
        off += name_len;
        let slot = slot_of(&name);
        if slot == usize::MAX {
            off += tri_n * (9 * 2 + 3);
            continue;
        }
        let span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let start = vertices.len() as u32;
        for _ in 0..tri_n {
            let mut p = [[0.0f32; 3]; 3];
            for v in 0..3 {
                for a in 0..3 {
                    let q16 = rd_u16(off); off += 2;
                    p[v][a] = min[a] + q16 as f32 / 65535.0 * if span[a] == 0.0 { 1.0 } else { span[a] };
                }
            }
            let r = buf[off] as f32 / 255.0;
            let g = buf[off + 1] as f32 / 255.0;
            let b = buf[off + 2] as f32 / 255.0;
            off += 3;
            let e1 = Vec3::from_array(p[1]) - Vec3::from_array(p[0]);
            let e2 = Vec3::from_array(p[2]) - Vec3::from_array(p[0]);
            let n = e1.cross(e2).normalize_or_zero();
            let shift = if slot == JAW_SLOT { JAW_HINGE } else { [0.0f32; 3] };
            for v in 0..3 {
                vertices.push(Vertex {
                    pos: [p[v][0] - shift[0], p[v][1] - shift[1], p[v][2] - shift[2]],
                    nrm: [n.x, n.y, n.z],
                    col: [r, g, b],
                });
            }
        }
        bodies.push(CadBody { slot, start, count: (tri_n * 3) as u32 });
    }
    Ok(Cad { vertices, bodies })
}

/* ════════════════════════════════════════════════════════════════════════
   4 · 策略运行时（ort；robotd 同一引擎）
   ════════════════════════════════════════════════════════════════════════ */

const ROLES: [(&str, &str); 7] = [
    ("walk", "alpha_walking.onnx"),
    ("stand", "alpha_stand.onnx"),
    ("sitstand", "alpha_sitstand.onnx"),
    ("ground_pick", "alpha_ground_pick.onnx"),
    ("kick_left", "ball_kick_left.onnx"),
    ("kick_right", "ball_kick_right.onnx"),
    ("roulade", "roulade.onnx"),
];

/// 一个策略槽位：角色 + 当前文件 + 会话（或失败原因）。
/// 运行时可替换（界面加载），失败的槽位不影响其它槽位。
pub(crate) struct PolicySlot {
    pub(crate) role: &'static str,
    pub(crate) file: String,
    pub(crate) path: PathBuf,
    pub(crate) session: Option<ort::session::Session>,
    pub(crate) err: Option<String>,
}

pub(crate) struct Policies {
    pub(crate) slots: Vec<PolicySlot>,
}

/// 建会话 + 形状契约校验 + 零观测热身（robotd 的 Policy::load 同款）。
/// 块作用域先结束 run 的借用，才能把 session 移出去。
fn open_policy(path: &Path) -> Result<ort::session::Session> {
    use ort::session::builder::GraphOptimizationLevel;
    let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let mut s = ort::session::Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(path)
        .with_context(|| format!("加载 {name}"))?;
    {
        let zero = ort::value::Tensor::from_array(([1usize, OBS_LEN], vec![0.0f32; OBS_LEN]))?;
        let out = s.run(ort::inputs!["obs" => zero])?;
        let (shape, data) = out["actions"].try_extract_tensor::<f32>()?;
        if data.len() != ACTION_LEN {
            bail!(
                "{name}: action count is {} (dims {shape:?}), expected {ACTION_LEN}",
                data.len()
            );
        }
    }
    Ok(s)
}

impl Policies {
    /// 启动时按槽位表加载；单个失败只记录原因，不中断（walk 会在别处单独校验）
    fn load(policies_dir: &Path) -> Self {
        let mut slots = Vec::new();
        for (role, file) in ROLES {
            let path = policies_dir.join(file);
            let (session, err) = if path.is_file() {
                match open_policy(&path) {
                    Ok(s) => (Some(s), None),
                    Err(e) => (None, Some(format!("{e}"))),
                }
            } else {
                (None, Some(format!("找不到文件 {}", path.display())))
            };
            match (&session, &err) {
                (Some(_), _) => println!("  policy {role:12} ✓ {file}"),
                (None, Some(m)) => println!("  policy {role:12} ✗ {m}"),
                _ => {}
            }
            slots.push(PolicySlot { role, file: file.to_string(), path, session, err });
        }
        Policies { slots }
    }

    fn has(&self, role: &str) -> bool {
        self.slots.iter().any(|s| s.role == role && s.session.is_some())
    }

    /// 运行时把某个槽位换成新文件（界面加载走这里）：先建好新会话，成功才替换
    fn load_one(&mut self, role: &str, path: &Path) -> Result<()> {
        let session = open_policy(path)?;
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.role == role)
            .with_context(|| format!("未知槽位 {role}"))?;
        slot.session = Some(session);
        slot.file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        slot.path = path.to_path_buf();
        slot.err = None;
        Ok(())
    }

    fn infer(&mut self, role: &str, obs: &[f32; OBS_LEN]) -> Result<[f32; ACTION_LEN]> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.role == role && s.session.is_some())
            .or_else(|| self.slots.iter().position(|s| s.role == "walk" && s.session.is_some()))
            .context("没有可用的策略（walk 槽为空）")?;
        let s = self.slots[idx].session.as_mut().unwrap();
        let t = ort::value::Tensor::from_array(([1usize, OBS_LEN], obs.to_vec()))?;
        let out = s.run(ort::inputs!["obs" => t])?;
        let (_, data) = out["actions"].try_extract_tensor::<f32>()?;
        let mut a = [0.0f32; ACTION_LEN];
        for (i, x) in data.iter().take(ACTION_LEN).enumerate() { a[i] = *x; }
        Ok(a)
    }
}

/* ════════════════════════════════════════════════════════════════════════
   5 · 控制器 + 仿真 —— robotd/src/control.rs 的移植
   ════════════════════════════════════════════════════════════════════════ */

#[derive(Clone, Copy, Default)]
struct Command {
    twist: [f64; 3],
    head: [f64; 4],
    /// 站姿姿态：z（蹲/升, m）、roll、pitch（rad）。body-pose 模式下生效，
    /// 行走中必须为零（walk 策略训练时这些槽是零）
    body: [f64; 3],
}

pub(crate) struct Controller {
    last_action: [f32; ACTION_LEN],
    previous: Option<[f64; NUM_JOINTS]>,
    pub(crate) ground_pick: Option<f64>,
    pub(crate) kick: Option<(bool, f64)>,
    pub(crate) roulade: Option<f64>,
    /// 滚翻续窗已追加的秒数：到点没站直就续（半滚交还 = 主策略永远趴着）
    pub(crate) roulade_ext: f64,
    roulade_chain: f64,
    pub(crate) sit: Sit,
    pub(crate) rise_remaining: f64,
    pub(crate) label: &'static str,
    pub(crate) gain: u16,
}

#[derive(PartialEq, Clone, Copy)]
enum Sit { Up, Sitting, Rising }

pub(crate) struct Sim {
    pub(crate) q: [f64; NUM_JOINTS],
    qd: [f64; NUM_JOINTS],
    pub(crate) targets: [f64; NUM_JOINTS],
    pub(crate) enabled: bool,
    pub(crate) ema_twist: [f64; 3],
    pub(crate) ema_head: [f64; 4],
    head_cmd: [f64; 4],
    mouth_open: f64,
    quack_until: Option<Instant>,
    treadmill: [f32; 2],
    ride_lift: f32,
    /// 动作缩放倍率（HUD 滑条；1.0 = 训练值，同 robotd 的 scale_mult）
    scale_mult: f64,
    /// body-pose 平滑值（robotd 的 pose glide）—— HUD 用它显示"生效中"的站姿
    pub(crate) ema_body: [f64; 3],
    /// 姿态模式的键盘弹簧偏移：叠加在拖拽基值 body_cmd 上，松开归零
    pub(crate) body_key: [f64; 3],
    /// 拖拽设定的行进基值（vx/vy m/s，ω rad/s）—— 键盘在其上瞬时叠加，松开回到基值
    pub(crate) twist_base: [f64; 3],
    /// 拖拽设定的头颈持久基值（原始 −1..1），与 head_cmd 同序：[0]颈 [1]头p [2]头y [3]头r
    pub(crate) head_base: [f64; 4],
}

impl Sim {
    fn new() -> Self {
        Sim {
            q: DEFAULT_POSITION,
            qd: [0.0; NUM_JOINTS],
            targets: DEFAULT_POSITION,
            enabled: true,
            ema_twist: [0.0; 3],
            ema_head: [0.0; 4],
            head_cmd: [0.0; 4],
            mouth_open: 0.0,
            quack_until: None,
            treadmill: [0.0, 0.0],
            ride_lift: 0.0,
            scale_mult: 1.0,
            ema_body: [0.0; 3],
            body_key: [0.0; 3],
            twist_base: [0.0; 3],
            head_base: [0.0; 4],
        }
    }

    fn servo_step(&mut self, dt: f64) {
        let k = 1.0 - (-dt / SERVO_TAU).exp();
        let prev = self.q;
        for i in 0..NUM_JOINTS {
            self.q[i] += (self.targets[i] - self.q[i]) * k;
        }
        for i in 0..NUM_JOINTS {
            let v = (self.q[i] - prev[i]) / dt;
            self.qd[i] = self.qd[i] * 0.6 + v * 0.4;
        }
    }
}

impl Controller {
    fn new() -> Self {
        Self::with_ground(2)
    }

    fn with_ground(_level: u8) -> Self {
        Controller {
            last_action: [0.0; ACTION_LEN],
            previous: None,
            ground_pick: None,
            kick: None,
            roulade: None,
            roulade_ext: 0.0,
            roulade_chain: 0.0,
            sit: Sit::Up,
            rise_remaining: 0.0,
            label: "idle",
            gain: GAIN,
        }
    }

    fn reset(&mut self) {
        self.last_action = [0.0; ACTION_LEN];
        self.previous = None;
        // 技能窗口一并清：卡死的 roulade/kick 窗口不该活过复位
        self.roulade = None;
        self.roulade_ext = 0.0;
        self.roulade_chain = 0.0;
        self.kick = None;
        self.ground_pick = None;
        self.sit = Sit::Up;
        self.rise_remaining = 0.0;
    }

    /// 一拍前半：过期窗口 → 优先级链 → 有效指令。返回 (net, eff, label)。
    fn begin(&mut self, cmd: &Command, body_active: bool) -> (String, Command, &'static str) {
        if let Some((_, r)) = self.kick {
            if r <= 0.0 {
                self.kick = None;
                self.last_action = [0.0; ACTION_LEN];
                self.previous = None;
            }
        }
        if let Some(r) = self.roulade {
            if r <= 0.0 {
                let chained = self.roulade_chain > 0.0;
                self.roulade = if chained { Some(ROULADE_DURATION) } else { None };
                if chained {
                    // 链滚 = 重新触发：清动作状态，让下一滚和首次触发完全一致
                    // （带着上一滚的动作历史，策略会输出"原地不动"的哑滚）
                    self.last_action = [0.0; ACTION_LEN];
                    self.previous = None;
                    self.roulade_ext = 0.0;
                }
                if self.roulade.is_none() {
                    // 官方 _end_behavior：行为结束换回主策略且清零指令；观测里的
                    // last_action 是滚翻末尾的输出，主策略不该看到它，一并清掉
                    self.last_action = [0.0; ACTION_LEN];
                    self.previous = None;
                }
            }
        }
        if self.sit == Sit::Rising && self.rise_remaining <= 0.0 { self.sit = Sit::Up; }

        let zero = Command::default();
        if self.roulade.is_some() {
            ("roulade".into(), zero, "roulade")
        } else if let Some((left, _)) = self.kick {
            let n = if left { "kick_left" } else { "kick_right" };
            (n.into(), zero, n)
        } else if let Some(phase) = self.ground_pick {
            let a = std::f64::consts::TAU * phase;
            let mut c = Command::default();
            c.twist = [a.cos(), a.sin(), 0.0];
            ("ground_pick".into(), c, "ground_pick")
        } else {
            let mut eff = *cmd;
            match self.sit {
                Sit::Sitting => { eff.twist = [1.0, 0.0, 0.0]; ("sitstand".into(), eff, "sit") }
                Sit::Rising => { eff.twist = [0.0; 3]; ("sitstand".into(), eff, "rise") }
                Sit::Up => {
                    if body_active { eff.twist = [0.0; 3]; }
                    let mag = eff.twist.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if mag <= STANDING_THRESHOLD || body_active { ("stand".into(), eff, "stand") }
                    else { ("walk".into(), eff, "walk") }
                }
            }
        }
    }

    /// 一拍后半：动作 → 缩放/增益 → home+scale×action → 低通 → 推进窗口。
    fn complete(&mut self, action: &[f32; ACTION_LEN], eff: &Command, label: &'static str, sim: &mut Sim) {
        self.last_action = *action;
        let mag = eff.twist.iter().map(|v| v * v).sum::<f64>().sqrt();
        let standing_tuned = label == "stand"
            || (matches!(label, "kick_left" | "kick_right" | "sit" | "rise") && mag <= STANDING_THRESHOLD);
        let (scale, gain) = match label {
            "roulade" => (1.0, GAIN),
            "ground_pick" => (1.0, GAIN),
            "sitstand" | "sit" | "rise" =>
                (1.0, if standing_tuned { (GAIN as f64 * STANDING_GAIN_RATIO).round() as u16 } else { GAIN }),
            _ if standing_tuned => (STANDING_ACTION_SCALE, (GAIN as f64 * STANDING_GAIN_RATIO).round() as u16),
            _ => (ACTION_SCALE, GAIN),
        };
        let offsets = scatter_action(action);
        let mut targets = [0.0; NUM_JOINTS];
        for j in 0..NUM_JOINTS {
            targets[j] = DEFAULT_POSITION[j] + scale * sim.scale_mult * offsets[j];
        }
        if let Some(prev) = self.previous {
            for j in 5..9 { targets[j] = HEAD_LOWPASS * targets[j] + (1.0 - HEAD_LOWPASS) * prev[j]; }
            for j in 0..NUM_JOINTS {
                if (5..9).contains(&j) || j == MOUTH_INDEX { continue; }
                targets[j] = LEGS_LOWPASS * targets[j] + (1.0 - LEGS_LOWPASS) * prev[j];
            }
        }
        self.previous = Some(targets);
        sim.targets = targets;

        if let Some(p) = self.ground_pick.as_mut() {
            *p += TICK_DT / GROUND_PICK_PERIOD;
            if *p >= GROUND_PICK_END_PHASE { self.ground_pick = None; }
        }
        if let Some((_, r)) = self.kick.as_mut() { *r -= TICK_DT; }
        if let Some(r) = self.roulade.as_mut() {
            *r -= TICK_DT;
            self.roulade_chain = (self.roulade_chain - TICK_DT).max(0.0);
        }
        if self.sit == Sit::Rising { self.rise_remaining -= TICK_DT; }

        self.label = label;
        self.gain = gain;
    }
}

/* ════════════════════════════════════════════════════════════════════════
   6 · 世界（输入 + tick）
   ════════════════════════════════════════════════════════════════════════ */

/// 视觉地面样式（纯外观）：平面网格 / 棋盘格。
/// 坡体不属于这里 —— 它是物理场景文件自带的（xml 里有坡体碰撞就画楔形）
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Scene {
    Flat,
    Checker,
}


impl Scene {
    fn label(self) -> &'static str {
        match self {
            Scene::Flat => "平面",
            Scene::Checker => "棋盘",
        }
    }
}

/// 一个可加载的场景：MuJoCo XML（物理）+ 视觉地面样式。
/// 启动时扫描 assets/mj/ 自动发现；拖 .xml 入窗口可追加自定义场景。
#[derive(Clone, Debug)]
pub struct SceneFile {
    /// 显示名（文件名去扩展名）
    pub(crate) name: String,
    /// MuJoCo XML 路径
    pub(crate) path: PathBuf,
    /// 视觉地面：文件名含 ramp → 画坡体楔形，否则用当前选的平面/棋盘样式
    pub(crate) has_ramp: bool,
}

/// 扫描场景目录：assets/mj/*.xml，排除被 include 的机器人模型和备份文件。
/// 排序保证 scene.xml / scene_ramp.xml 在前（键盘 1/2/3 对应的默认三档）。
fn discover_scenes(root: &Path) -> Vec<SceneFile> {
    let dir = root.join("assets/mj");
    let mut out: Vec<SceneFile> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x.eq_ignore_ascii_case("xml")) != Some(true) { continue; }
            let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            // robot_groundcontact.xml 是被 include 的机器人定义，不是场景
            if name.starts_with("robot_") { continue; }
            let has_ramp = name.to_lowercase().contains("ramp");
            out.push(SceneFile { name, path: p, has_ramp });
        }
    }
    // 默认场景排最前，其余按名字排在后面
    out.sort_by_key(|s| {
        let rank = match s.name.as_str() {
            "scene" => 0,
            "scene_ramp" => 1,
            _ => 2,
        };
        (rank, s.name.clone())
    });
    for s in &out {
        println!("  scene {:<16} ✓ {}{}", s.name,
            s.path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(),
            if s.has_ramp { "  (坡体)" } else { "" });
    }
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Mode {
    /// 固定支架（gait viewer）：躯干悬空看步态，不摔
    Rig,
    /// 物理环境（Rapier）：真实动力学，会摔、滚翻是真的滚
    Physics,
}

pub(crate) struct World {
    root: PathBuf,
    sim: Sim,
    ctl: Controller,
    pol: Policies,
    keys: HashSet<KeyCode>,
    body_active: bool,
    infer_us: f64,
    tick_count: u64,
    pub mode: Mode,
    pub phys: Option<Phys>,
    /// 最近一次 IMU（HUD 显示）：gravity + gyro
    pub imu: ([f32; 3], [f32; 3]),
    pub gpu_name: String,
    /// 界面加载的目标槽位（拖拽/浏览的目标）
    pub load_target: usize,
    /// body-pose 目标（z, roll, pitch）—— B 模式下生效
    pub body_cmd: [f64; 3],
    /// 当前场景（物理 XML 与视觉地面同步切换）
    pub scene: Scene,
    /// 自动发现的场景列表（assets/mj/*.xml；拖 .xml 入窗口可追加）
    pub scenes: Vec<SceneFile>,
    /// 当前选中的场景下标（scenes 内的物理 XML）
    pub scene_sel: usize,
    /// 最近一拍观测里的 body 段（自检用）
    pub body_obs: [f32; 3],
    /// 动作缩放倍率（HUD 滑条）：action_scale × mult，同 robotd 的 scale_mult
    pub scale_mult: f64,
    /// 拖拽悬停提示（松手前的状态行）
    pub drop_hint: Option<String>,
    /// 最近一次界面加载的结果提示
    pub load_msg: Option<(bool, String, Instant)>,
    /// 游戏式 HUD：详细控制台抽屉（TAB 开关），默认展开
    pub console_open: bool,
    /// 开机时刻（3 秒倒计时后自动进物理模式）
    pub boot: Instant,
    pub boot_switched: bool,
    /// 全屏公告（技能发动 / 模式切换）：文本 + 时刻，1.4s 渐隐
    pub announce: Option<(String, Instant)>,
}

impl World {
    fn say(&mut self, text: &str) {
        self.announce = Some((text.to_string(), Instant::now()));
    }

    pub(crate) fn toggle_mode(&mut self) {
        if keylog() {
            println!("[mode] toggle from {:?}", self.mode);
        }
        // 任何模式切换（含手动 P）都取消开机倒计时
        self.boot_switched = true;
        self.say(if self.mode == Mode::Rig { "物理环境 PHYSICS" } else { "支架模式 RIG" });
        if self.mode == Mode::Physics {
            self.mode = Mode::Rig;
        } else {
            if self.phys.is_none() {
                let xml = self.scene_path();
                match Phys::new(&xml) {
                    Ok(ph) => self.phys = Some(ph),
                    Err(e) => {
                        println!("MuJoCo 载入失败: {e}");
                        return;
                    }
                }
            }
            self.ctl.reset();
            self.mode = Mode::Physics;
        }
    }

    /// 界面加载：把路径挂到某个槽位（文件对话框与拖拽都走这里）
    fn load_policy_path(&mut self, slot: usize, path: &Path) {
        let role = ROLES.get(slot).map(|(r, _)| *r).unwrap_or("walk");
        let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        if path.extension().map(|e| e.eq_ignore_ascii_case("onnx")) != Some(true) {
            let msg = format!("{name} 不是 .onnx");
            println!("[load] ✗ {msg}");
            self.load_msg = Some((false, msg, Instant::now()));
            return;
        }
        match self.pol.load_one(role, path) {
            Ok(()) => {
                let msg = format!("{role} ← {name}");
                println!("[load] ✓ {msg}");
                self.load_msg = Some((true, msg, Instant::now()));
                self.ctl.reset();
            }
            Err(e) => {
                // 形状不符等：保留原会话，只报原因（66 维/51 维一眼可见）
                let msg = format!("{role} ← {name} 失败：{e}");
                println!("[load] ✗ {msg}");
                self.load_msg = Some((false, msg, Instant::now()));
            }
        }
    }

    /// 当前选中场景的 MuJoCo XML 路径（列表为空时回退默认 scene.xml）
    pub(crate) fn scene_path(&self) -> PathBuf {
        self.scenes
            .get(self.scene_sel)
            .map(|s| s.path.clone())
            .unwrap_or_else(|| self.root.join("assets/mj/scene.xml"))
    }

    /// 当前场景是否带坡体（决定是否画楔形视觉）
    pub(crate) fn scene_has_ramp(&self) -> bool {
        self.scenes.get(self.scene_sel).map(|s| s.has_ramp).unwrap_or(false)
    }

    /// 地面重建判据（视觉样式 + 坡体）
    pub(crate) fn ground_kind(&self) -> GroundKind {
        GroundKind { vis: self.scene, ramp: self.scene_has_ramp() }
    }

    /// 按名字查场景下标
    fn scene_idx_by_name(&self, name: &str) -> Option<usize> {
        self.scenes.iter().position(|s| s.name == name)
    }

    /// 视觉地面样式（键盘 1/2 与 HUD 圆点）：纯外观，不碰物理世界
    pub(crate) fn set_scene(&mut self, vis: Scene) {
        self.scene = vis;
    }

    /// 键 3 / HUD：坡体场景开↔关（切到带坡的场景文件，再按切回无坡的第一个）
    fn toggle_ramp_scene(&mut self) {
        let target = if self.scene_has_ramp() {
            self.scenes.iter().position(|s| !s.has_ramp).unwrap_or(0)
        } else {
            match self.scenes.iter().position(|s| s.has_ramp) {
                Some(i) => i,
                None => return,   // 没有带坡的场景文件：不动
            }
        };
        self.select_scene(target);
    }

    /// 选场景文件：物理 XML 换成 scenes[idx]（坡体视觉随之自动切换）。
    /// 物理模式立即重建 MuJoCo 世界（保留 PD 增益），支架模式只记选择。
    pub(crate) fn select_scene(&mut self, idx: usize) {
        let Some(sf) = self.scenes.get(idx).cloned() else { return };
        if self.scene_sel == idx { return; }
        let (kp, kd) = self.phys.as_ref().map(|p| (p.kp, p.kd)).unwrap_or((0.55, 0.0));
        self.scene_sel = idx;
        self.reset_all();
        if self.mode == Mode::Physics {
            match Phys::new(&sf.path) {
                Ok(mut ph) => {
                    ph.kp = kp;
                    ph.kd = kd;
                    ph.apply_gains();
                    self.phys = Some(ph);
                }
                Err(e) => println!("场景 {} 载入失败: {e}", sf.name),
            }
        }
    }

    /// 拖入 .xml：注册为自定义场景并选中（与拖 .onnx 加载策略对称）
    fn load_scene_path(&mut self, path: &Path) {
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if path.extension().map(|e| e.eq_ignore_ascii_case("xml")) != Some(true) {
            self.load_msg = Some((false, format!("{name} 不是 .xml 场景"), Instant::now()));
            return;
        }
        let has_ramp = name.to_lowercase().contains("ramp");
        // 同名场景替换（重载），否则追加
        let idx = match self.scenes.iter().position(|s| s.name == name) {
            Some(i) => { self.scenes[i].path = path.to_path_buf(); i }
            None => {
                self.scenes.push(SceneFile { name: name.clone(), path: path.to_path_buf(), has_ramp });
                self.scenes.len() - 1
            }
        };
        self.select_scene(idx);
        // 校验载入是否真的成功：物理模式下 Phys::new 失败会保持旧世界
        let ok = self.phys.is_some() || self.mode != Mode::Physics;
        self.load_msg = Some((ok, format!("场景 ← {name}"), Instant::now()));
        println!("[scene] ✓ {name} ← {}", path.display());
    }

    pub(crate) fn reset_all(&mut self) {
        self.sim.q = DEFAULT_POSITION;
        self.sim.qd = [0.0; NUM_JOINTS];
        self.sim.ema_twist = [0.0; 3];
        self.sim.twist_base = [0.0; 3];
        self.sim.head_base = [0.0; 4];
        self.sim.body_key = [0.0; 3];
        self.ctl.reset();
        if let Some(ph) = self.phys.as_mut() {
            ph.reset();
        }
    }
}

impl World {
    fn tick(&mut self, dt_real: f64) {
        // 指令（键盘 → twist/head，平滑同 robotd cmd_alpha/head_alpha）
        let k = |c: KeyCode| self.keys.contains(&c) as i32 as f64;
        let dz = |v: f64| if v.abs() < DEADZONE { 0.0 } else { v };
        let vx = (k(KeyCode::KeyW) - k(KeyCode::KeyS)).clamp(-1.0, 1.0);
        let vy = (k(KeyCode::KeyA) - k(KeyCode::KeyD)).clamp(-1.0, 1.0);
        let vyaw = (k(KeyCode::KeyQ) - k(KeyCode::KeyE)).clamp(-1.0, 1.0);
        if self.body_active {
            // B 姿态模式：行走键 = 姿态弹簧（与 twist/head 同一约定：按住偏离、松开回基值）。
            // 按住即给满偏，过渡交给 ema_body 的一阶平滑（~0.3s 到位，不再"爬"）；
            // 满偏量取训练行程（z −4/+3cm、倾角 ±0.26rad ≈ ±15°），再大策略没见过会摔。
            // twist 平滑归零（行走暂停）。
            self.sim.body_key = [
                if vx > 0.5 { BODY_Z_RANGE[1] } else if vx < -0.5 { BODY_Z_RANGE[0] } else { 0.0 },
                vy.clamp(-1.0, 1.0) * BODY_KEY_TILT,
                vyaw.clamp(-1.0, 1.0) * BODY_KEY_TILT,
            ];
            for i in 0..3 { self.sim.ema_twist[i] += CMD_ALPHA * (0.0 - self.sim.ema_twist[i]); }
        } else {
            self.sim.body_key = [0.0; 3];   // 退出姿态模式：清键盘偏移，别留残余
            // 键盘叠加在拖拽基值上：松开回到基值（拖 vx=0.2 即巡航 0.2），按住在基值上加速
            let raw = [
                (dz(vx) * MAX_LINEAR + self.sim.twist_base[0]).clamp(-MAX_LINEAR, MAX_LINEAR),
                (dz(vy) * MAX_LINEAR + self.sim.twist_base[1]).clamp(-MAX_LINEAR, MAX_LINEAR),
                (dz(vyaw) * MAX_ANGULAR + self.sim.twist_base[2]).clamp(-MAX_ANGULAR, MAX_ANGULAR),
            ];
            for i in 0..3 { self.sim.ema_twist[i] += CMD_ALPHA * (raw[i] - self.sim.ema_twist[i]); }
        }
        // 头部四轴同一手感：方向键 = 弹簧满偏（EMA 平滑），松手回基值；Ctrl 切换两对轴
        // （不按 = 颈/头y；按住 = 头p/头r，此时颈/头y 保持基值）
        let ctrl = self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::ControlRight);
        let (head_ud, head_lr, head_p, head_r) = if ctrl {
            (0.0, 0.0, k(KeyCode::ArrowUp) - k(KeyCode::ArrowDown), k(KeyCode::ArrowLeft) - k(KeyCode::ArrowRight))
        } else {
            (k(KeyCode::ArrowUp) - k(KeyCode::ArrowDown), k(KeyCode::ArrowLeft) - k(KeyCode::ArrowRight), 0.0, 0.0)
        };
        self.sim.head_cmd[0] = (self.sim.head_base[0] + head_ud).clamp(-1.0, 1.0);
        self.sim.head_cmd[1] = (self.sim.head_base[1] + head_p).clamp(-1.0, 1.0);
        self.sim.head_cmd[2] = (self.sim.head_base[2] + head_lr).clamp(-1.0, 1.0);
        self.sim.head_cmd[3] = (self.sim.head_base[3] + head_r).clamp(-1.0, 1.0);
        let head_raw: [f64; 4] = self.sim.head_cmd.map(|v| v * MAX_HEAD);
        for i in 0..4 { self.sim.ema_head[i] += HEAD_ALPHA * (head_raw[i] - self.sim.ema_head[i]); }

        // 嘴（嘎一声 420ms 包络）
        if let Some(t) = self.sim.quack_until {
            let ph = 1.0 - t.elapsed().as_millis() as f64 / 420.0;
            if ph <= 0.0 { self.sim.quack_until = None; self.sim.mouth_open = 0.0; }
            else { self.sim.mouth_open = if ph < 0.5 { ph * 2.0 } else { (1.0 - ph) * 2.0 }; }
        }
        self.sim.targets[MOUTH_INDEX] = MOUTH_CLOSED
            + self.sim.mouth_open.clamp(0.0, 1.0) * (MOUTH_OPEN - MOUTH_CLOSED);

        // 滚翻链：X 按住 = 每拍续窗（同 padd 的按住连滚）
        if self.keys.contains(&KeyCode::KeyX) && self.ctl.roulade.is_some() {
            self.ctl.roulade_chain = ROULADE_CHAIN_WINDOW;
        }

        // 滚翻自适应交还：窗口到点但躯干还没立起来 → 续 0.5s（累计 +2s）。
        // 必须在 ctl.begin() 之前：begin 会处理过期并关闭窗口，之后再续就是死代码。
        if self.mode == Mode::Physics {
            if let Some(ph) = self.phys.as_ref() {
                if let Some(r) = self.ctl.roulade {
                    if r <= 0.0 {
                        let (_, z, up) = ph.trunk_pose();
                        if (up < 0.75 || z < 0.085) && self.ctl.roulade_ext < 2.0 {
                            self.ctl.roulade = Some(0.5);
                            self.ctl.roulade_ext += 0.5;
                            println!("[roulade] 续窗 +0.5s（未立起: z={z:.3} up={up:+.2}）");
                        }
                    }
                }
            }
        }

        if self.sim.enabled {
            self.sim.scale_mult = self.scale_mult;
            // body 指令：B 模式下经一阶平滑滑向「拖拽基值 + 键盘弹簧偏移」（robotd 的
            // pose glide），退出瞬间直接归零（同原型 B 键退出）
            if self.body_active {
                const LIM: [[f64; 2]; 3] = [BODY_Z_RANGE,
                    [-BODY_TILT_RANGE, BODY_TILT_RANGE], [-BODY_TILT_RANGE, BODY_TILT_RANGE]];
                for i in 0..3 {
                    let t = (self.body_cmd[i] + self.sim.body_key[i]).clamp(LIM[i][0], LIM[i][1]);
                    self.sim.ema_body[i] += BODY_ALPHA * (t - self.sim.ema_body[i]);
                }
            } else {
                self.sim.ema_body = [0.0; 3];
            }
            let cmd = Command {
                twist: self.sim.ema_twist,
                head: self.sim.ema_head,
                body: if self.body_active { self.sim.ema_body } else { [0.0; 3] },
            };
            let (net, eff, label) = self.ctl.begin(&cmd, self.body_active);
            // 观测（obs.rs 布局，61 维）。物理模式用真实 IMU/关节反馈
            let mut obs = [0.0f32; OBS_LEN];
            match self.mode {
                Mode::Rig => {
                    obs[5] = -1.0;                               // gravity 直立
                    let q = policy_joints(&self.sim.q);
                    let home = policy_joints(&DEFAULT_POSITION);
                    let qd = policy_joints(&self.sim.qd);
                    for i in 0..14 {
                        obs[6 + i] = (q[i] - home[i]) as f32;
                        obs[20 + i] = qd[i] as f32;
                        obs[34 + i] = self.ctl.last_action[i];
                    }
                }
                Mode::Physics => {
                    if let Some(ph) = self.phys.as_mut() {
                        let (q, qd, grav, gyro) = ph.read(TICK_DT);
                        self.sim.q = q;
                        self.sim.qd = qd;
                        self.imu = (grav, gyro);
                        for a in 0..3 {
                            obs[a] = gyro[a];
                            obs[3 + a] = grav[a];
                        }
                        let qp = policy_joints(&self.sim.q);
                        let home = policy_joints(&DEFAULT_POSITION);
                        let qdp = policy_joints(&self.sim.qd);
                        for i in 0..14 {
                            obs[6 + i] = (qp[i] - home[i]) as f32;
                            obs[20 + i] = qdp[i] as f32;
                            obs[34 + i] = self.ctl.last_action[i];
                        }
                    }
                }
            }
            obs[48] = eff.twist[0] as f32;
            obs[49] = eff.twist[1] as f32;
            obs[50] = eff.twist[2] as f32;
            obs[51] = eff.head[0] as f32;
            obs[52] = eff.head[1] as f32;
            obs[53] = eff.head[2] as f32;
            obs[54] = eff.head[3] as f32;
            // body 段顺序 z, roll, pitch（不是 z, pitch, roll！）；yaw 未绑定恒零
            obs[57] = eff.body[0] as f32;
            obs[58] = eff.body[1] as f32;
            obs[59] = eff.body[2] as f32;
            self.body_obs = [obs[57], obs[58], obs[59]];
            let t0 = Instant::now();
            if let Ok(a) = self.pol.infer(&net, &obs) {
                self.infer_us = t0.elapsed().as_secs_f64() * 1e6;
                self.ctl.complete(&a, &eff, label, &mut self.sim);
            }
        } else {
            self.ctl.label = "idle";
        }

        // 动作早退：固定窗口只是"保底上限"，动作实际完成（直立 + 关节速度
        // 沉降）就立即交还主策略 —— 否则踢腿干等 2.5s、滚翻干等 0.5s。
        // 阈值依据实测（--selftest 踢腿测量）：动作中 qdmax≈3，结束<0.3。
        if self.sim.enabled {
            let qd_max = (0..NUM_JOINTS).filter(|&i| i != MOUTH_INDEX)
                .map(|i| self.sim.qd[i].abs()).fold(0.0f64, f64::max);
            let upright = if self.mode == Mode::Physics {
                let (_, z, up) = self.phys.as_ref().map(|p| p.trunk_pose()).unwrap_or((0.0, 0.0, 1.0));
                up > 0.9 && z > 0.095
            } else {
                true   // 支架模式躯干固定
            };
            if qd_max < 1.5 && upright {
                if let Some(r) = self.ctl.roulade {
                    let total = ROULADE_DURATION + self.ctl.roulade_ext;
                    if total - r > 1.2 {
                        self.ctl.roulade = None;
                        self.ctl.roulade_ext = 0.0;
                        self.ctl.last_action = [0.0; ACTION_LEN];
                        self.ctl.previous = None;
                    }
                }
                if let Some((_, r)) = self.ctl.kick {
                    if KICK_DURATION - r > 0.8 {
                        self.ctl.kick = None;
                        self.ctl.last_action = [0.0; ACTION_LEN];
                        self.ctl.previous = None;
                    }
                }
            }
        }

        match self.mode {
            Mode::Rig => {
                for _ in 0..4 { self.sim.servo_step(TICK_DT / 4.0); }
                self.sim.treadmill[0] += (self.sim.ema_twist[0] * dt_real) as f32;
                self.sim.treadmill[1] += (self.sim.ema_twist[1] * dt_real) as f32;
                for i in 0..2 {
                    // 回绕周期须同时整除两种视觉图案：网格线距 0.25m、棋盘周期 1.0m
                    let m = 1.0f32;
                    if self.sim.treadmill[i].abs() > m {
                        self.sim.treadmill[i] = self.sim.treadmill[i].rem_euclid(m) - m;
                    }
                }
            }
            Mode::Physics => {
                if let Some(ph) = self.phys.as_mut() {
                    // HUD 改了 kp/kd 就写回 MuJoCo 参数
                    if ph.gains_dirty() {
                        ph.apply_gains();
                    }
                    ph.sync_motor_targets(&self.sim.targets);
                    ph.step();
                }
            }
        }
        if keylog() && self.tick_count % 25 == 0 {
            let (x, z) = match self.phys.as_ref() {
                Some(ph) => {
                    let (x, z, _) = ph.trunk_pose();
                    (x as f64, z as f64)
                }
                None => (0.0, 0.0),
            };
            println!(
                "[tick] mode={:?} label={} cmd=({:+.2},{:+.2},{:+.2}) enabled={} x={x:+.3} z={z:.3}",
                self.mode, self.ctl.label, self.sim.ema_twist[0], self.sim.ema_twist[1],
                self.sim.ema_twist[2], self.sim.enabled
            );
        }
        self.tick_count += 1;
    }

    /// 技能触发（边沿）。**不写按键集合**：键盘路径的"按住"由 window_event
    /// 的 press/release 维护；HUD 按钮走这里 —— 若按钮也写集合，集合里永远
    /// 不会有对应的 release，"按住连滚"的判定就会永久刷新滚翻窗口
    /// （现象：滚完 label 永远停在 roulade/BUSY）。
    pub(crate) fn on_key(&mut self, code: KeyCode) {
        if keylog() {
            println!("[key] {code:?}");
        }
        match code {
            KeyCode::KeyG => {
                if self.ctl.ground_pick.is_none() { self.ctl.ground_pick = Some(0.0); self.say("捡地 GROUND PICK"); }
            }
            KeyCode::KeyZ => {
                if self.ctl.kick.is_none() { self.ctl.kick = Some((true, KICK_DURATION)); self.say("左踢 KICK L"); }
            }
            KeyCode::KeyC => {
                if self.ctl.kick.is_none() { self.ctl.kick = Some((false, KICK_DURATION)); self.say("右踢 KICK R"); }
            }
            KeyCode::KeyV => match self.ctl.sit {
                Sit::Up => { self.ctl.sit = Sit::Sitting; self.say("坐下"); }
                Sit::Sitting => { self.ctl.sit = Sit::Rising; self.ctl.rise_remaining = RISE_SECS; self.say("起身"); }
                Sit::Rising => {}
            },
            KeyCode::KeyX => {
                if self.ctl.roulade.is_none() && self.ctl.ground_pick.is_none() {
                    self.ctl.roulade_ext = 0.0;
                    self.ctl.roulade = Some(ROULADE_DURATION);
                    self.say("前滚翻 ROULADE");
                }
            }
            KeyCode::KeyM => self.sim.quack_until = Some(Instant::now()),
            KeyCode::KeyB => {
                self.body_active = !self.body_active;
                self.say(if self.body_active {
                    "姿态模式：WS 升降 · AD 侧倾 · QE 俯仰（松开回中）"
                } else {
                    "退出姿态"
                });
            }
            _ => {}
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════
   6.5 · 物理（MuJoCo）—— 训练环境同源，策略才站得住
   · 模型 assets/mj/scene.xml（robot_groundcontact + 地面），与
     microduck_rl 训练/官方 CPU 部署脚本 scripts/infer_policy.py 同一份
   · 执行器：XML 的 chosen_actuator 位置伺服（kp 0.55 N·m/rad、τmax ±0.96）。
     这是"软"伺服：无策略时 home 位会塌 —— 站立全靠策略闭环（官方脚本的
     standby 模式要临时把 kp 调到 2.0 才能保持姿势，即此原因）。
   · 时序：timestep 0.002（500Hz），每 20ms 控制拍 10 个子步
   ════════════════════════════════════════════════════════════════════════ */

use mujoco_rs::prelude::*;

/// 渲染槽（BODIES 序）→ MuJoCo body 名（两份模型的命名差异）
const MJ_BODY_ALIAS: [(&str, &str); 4] = [
    ("left_upper_leg", "upper_leg_left"),
    ("right_upper_leg", "upper_leg_right"),
    ("neck_pitch_body", "neck_pitch"),
    ("bottom_head_shell", "jaw_soft"),
];

/// 从裸指针数组读第 i 个 mjtNum（MuJoCo C API 的数组都是裸指针）
#[inline]
unsafe fn f64_at(p: *const f64, i: usize) -> f64 {
    unsafe { *p.add(i) }
}
#[inline]
unsafe fn i32_at(p: *const i32, i: usize) -> i32 {
    unsafe { *p.add(i) }
}

pub(crate) struct Phys {
    data: MjData<Box<MjModel>>,
    act_slot: [usize; 14],
    qpos_adr: [usize; 14],
    qvel_adr: [usize; 14],
    mj_body: [usize; BODIES.len() + 1],
    /// trunk 自由关节的角速度段起点（qvel[adr..adr+3]）
    trunk_qvel_adr: usize,
    substeps: u32,
    pub kp: f64,
    pub kd: f64,
    pub max_force: f64,
    /// MuJoCo 的 dof_damping 原值（仅诊断参考；kd 通过 biasprm 施加）
    base_damping: Vec<f64>,
    prev_q: [f64; NUM_JOINTS],
    applied: (f64, f64),
}

impl Phys {
    /// HUD 改过 kp/kd？（只在变化时写回 MuJoCo 参数）
    fn gains_dirty(&self) -> bool {
        (self.kp - self.applied.0).abs() > 1e-9 || (self.kd - self.applied.1).abs() > 1e-9
    }
}

impl Phys {
    fn new(scene_xml: &Path) -> Result<Self> {
        let xml = scene_xml.to_path_buf();
        let model = Box::new(
            MjModel::from_xml(&xml).map_err(|e| anyhow::anyhow!("加载 {}: {e}", xml.display()))?,
        );
        let data = MjData::new(model);
        {
            let m = data.model().ffi();
            let dt = m.opt.timestep;
            println!(
                "MuJoCo: nq={} nv={} nu={} njnt={} timestep={dt:.4}s",
                m.nq, m.nv, m.nu, m.njnt,
            );
        }
        let mut me = Phys {
            data,
            act_slot: [0; 14],
            qpos_adr: [0; 14],
            qvel_adr: [0; 14],
            mj_body: [0; BODIES.len() + 1],
            trunk_qvel_adr: 0,
            substeps: 10,
            kp: 0.55,
            // kd = 附加执行器速度阻尼（XML 的 biasprm[2] = 0，默认就是不加）。
            // 训练时的动作传递特性依赖这一项为 0 —— 随手加一点就会从"挺直走"
            // 变成"蹲着挪"（实测）。
            kd: 0.0,
            max_force: 0.96,
            base_damping: Vec::new(),
            prev_q: DEFAULT_POSITION,
            applied: (0.55, 0.0),
        };
        me.build()?;
        Ok(me)
    }

    fn build(&mut self) -> Result<()> {
        // 关节/执行器映射（按名字；策略槽 = JOINT_NAMES 去掉嘴的顺序）
        for slot in 0..14 {
            let name = JOINT_NAMES[joint_of(slot)];
            let model = self.data.model();
            let jid = model
                .name_to_id(MjtObj::mjOBJ_JOINT, name)
                .ok_or_else(|| anyhow::anyhow!("模型里没有关节 {name}"))?;
            let aid = model
                .name_to_id(MjtObj::mjOBJ_ACTUATOR, name)
                .ok_or_else(|| anyhow::anyhow!("模型里没有执行器 {name}"))?;
            let m = model.ffi();
            self.act_slot[slot] = aid;
            self.qpos_adr[slot] = unsafe { i32_at(m.jnt_qposadr, jid) as usize };
            self.qvel_adr[slot] = unsafe { i32_at(m.jnt_dofadr, jid) as usize };
        }
        {
            let model = self.data.model();
            let free = model
                .name_to_id(MjtObj::mjOBJ_JOINT, "trunk_base_freejoint")
                .ok_or_else(|| anyhow::anyhow!("模型里没有 trunk_base_freejoint"))?;
            self.trunk_qvel_adr = unsafe { i32_at(model.ffi().jnt_dofadr, free) as usize } + 3;
            for (i, b) in BODIES.iter().enumerate() {
                let name = MJ_BODY_ALIAS
                    .iter()
                    .find(|(from, _)| *from == b.name)
                    .map(|(_, to)| *to)
                    .unwrap_or(b.name);
                self.mj_body[i] = model
                    .name_to_id(MjtObj::mjOBJ_BODY, name)
                    .ok_or_else(|| anyhow::anyhow!("模型里没有 body {name}"))?;
            }
            self.mj_body[BODIES.len()] = self.mj_body[HEAD];
            let dt = model.ffi().opt.timestep;
            self.substeps = (0.02 / dt).round().max(1.0) as u32;
            // 基线阻尼（kd 滑条在此之上叠加）
            let nv = model.ffi().nv as usize;
            let damp = model.ffi().dof_damping;
            self.base_damping = (0..nv).map(|i| unsafe { f64_at(damp, i) }).collect();
            // 执行器参数（增益/力矩上限）
            let nu = model.ffi().nu as usize;
            let gp = model.ffi().actuator_gainprm;
            let fr = model.ffi().actuator_forcerange;
            self.kp = if nu > 0 { unsafe { f64_at(gp, 0) } } else { 0.55 };
            // XML 的执行器速度反馈（biasprm[2]）基线，通常为 0
            let bp = model.ffi().actuator_biasprm;
            let base_kv = if nu > 0 { -(unsafe { f64_at(bp, 2) }) } else { 0.0 };
            self.kd = base_kv;
            self.max_force = (0..nu)
                .map(|a| unsafe { f64_at(fr, a * 2 + 1) })
                .fold(0.0f64, f64::max);
            println!(
                "MuJoCo: kp={:.3} kv={:.4}（XML 原值） τmax=±{:.2} N·m  子步={}",
                self.kp, self.kd, self.max_force, self.substeps
            );
        }
        self.reset();
        Ok(())
    }

    /// HUD 的 kp/kd 写回 MuJoCo 参数（位置伺服）。
    ///
    /// **stride 是 10，不是 9**（gainprm/biasprm 都是 nactuator×10）：按 9 写会把
    /// 增益写到别的执行器的槽里，静默毁掉大部分关节的位置反馈 —— 现象是"能站、
    /// 一走就蹲"。坐标系对齐同理，读写都必须用同一个 stride。
    ///   gainprm[a*10+0] = kp          （固定增益）
    ///   biasprm[a*10+1] = −kp         （位置反馈项）
    ///   biasprm[a*10+2] = −kd         （速度反馈项，XML 里为 0）
    fn apply_gains(&mut self) {
        const STRIDE: usize = 10;
        let kp = self.kp;
        let kd = self.kd;
        unsafe {
            let m = self.data.model_mut().ffi_mut();
            let nu = m.nu as usize;
            for a in 0..nu {
                for k in 0..STRIDE {
                    *m.actuator_gainprm.add(a * STRIDE + k) = 0.0;
                    *m.actuator_biasprm.add(a * STRIDE + k) = 0.0;
                }
                *m.actuator_gainprm.add(a * STRIDE) = kp;
                *m.actuator_biasprm.add(a * STRIDE + 1) = -kp;
                *m.actuator_biasprm.add(a * STRIDE + 2) = -kd;
            }
        }
        if keylog() {
            println!("[gains] kp={kp:.3} kd={kd:.4}");
        }
        self.applied = (self.kp, self.kd);
    }

    fn sync_motor_targets(&mut self, targets: &[f64; NUM_JOINTS]) {
        // 位置伺服：ctrl = 目标角（rad），与默认位形同一坐标系
        unsafe {
            let m = self.data.ffi_mut();
            for slot in 0..14 {
                *m.ctrl.add(self.act_slot[slot]) = targets[joint_of(slot)];
            }
        }
    }

    fn step(&mut self) {
        for _ in 0..self.substeps {
            self.data.step();
        }
    }

    fn reset(&mut self) {
        self.data.reset();
        unsafe {
            let m = self.data.ffi_mut();
            for slot in 0..14 {
                *m.ctrl.add(self.act_slot[slot]) = DEFAULT_POSITION[joint_of(slot)];
            }
        }
        self.prev_q = DEFAULT_POSITION;
    }

    /// 读回：关节角/角速度 + 真实 IMU（trunk 系，z-up 世界）
    fn read(&mut self, _dt: f64) -> ([f64; NUM_JOINTS], [f64; NUM_JOINTS], [f32; 3], [f32; 3]) {
        let mut q = self.prev_q;
        let mut qd = [0.0f64; NUM_JOINTS];
        let (g, w);
        unsafe {
            let m = self.data.ffi_mut();
            for slot in 0..14 {
                let j = joint_of(slot);
                q[j] = f64_at(m.qpos, self.qpos_adr[slot]);
                qd[j] = f64_at(m.qvel, self.qvel_adr[slot]);
            }
            let fb = self.mj_body[TRUNK];
            let qw = f64_at(m.xquat, fb * 4);
            let qx = f64_at(m.xquat, fb * 4 + 1);
            let qy = f64_at(m.xquat, fb * 4 + 2);
            let qz = f64_at(m.xquat, fb * 4 + 3);
            // gravity = R^T·(0,0,−1)
            g = [
                (-2.0 * (qx * qz - qw * qy)) as f32,
                (-2.0 * (qy * qz + qw * qx)) as f32,
                (-(1.0 - 2.0 * (qx * qx + qy * qy))) as f32,
            ];
            // 自由关节角速度就在 body frame
            w = [
                f64_at(m.qvel, self.trunk_qvel_adr) as f32,
                f64_at(m.qvel, self.trunk_qvel_adr + 1) as f32,
                f64_at(m.qvel, self.trunk_qvel_adr + 2) as f32,
            ];
        }
        q[MOUTH_INDEX] = self.prev_q[MOUTH_INDEX];
        self.prev_q = q;
        (q, qd, g, w)
    }

    /// 渲染矩阵（y-up 世界；MuJoCo z-up 经同一套根部 Rx(−90°)）
    fn world_matrices(&self, mouth: f64) -> Vec<glam::Mat4> {
        let root = glam::Mat4::from_quat(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));
        let m = self.data.ffi();
        let mut out = Vec::with_capacity(BODIES.len() + 1);
        for (i, _) in BODIES.iter().enumerate() {
            let b = self.mj_body[i];
            let pos = Vec3::new(
                unsafe { f64_at(m.xpos, b * 3) } as f32,
                unsafe { f64_at(m.xpos, b * 3 + 1) } as f32,
                unsafe { f64_at(m.xpos, b * 3 + 2) } as f32,
            );
            let q = Quat::from_xyzw(
                unsafe { f64_at(m.xquat, b * 4 + 1) } as f32,
                unsafe { f64_at(m.xquat, b * 4 + 2) } as f32,
                unsafe { f64_at(m.xquat, b * 4 + 3) } as f32,
                unsafe { f64_at(m.xquat, b * 4) } as f32,
            );
            let mut world = root * glam::Mat4::from_translation(pos) * glam::Mat4::from_quat(q);
            // 坐标系补偿：CAD 网格按 microduck 仓库的 body 系打包（打包时已把
            // ankle_right 的网格转过 180°）；而 MuJoCo 用的是训练模型的 body 系，
            // 两者在 ankle_right 上相差绕关节轴 180° —— 这里补回来，否则右脚朝后。
            if BODIES[i].name == "ankle_right" {
                world *= glam::Mat4::from_quat(Quat::from_rotation_z(std::f32::consts::PI));
            }
            out.push(world);
        }
        // 下颚（装饰；训练模型没有嘴关节）
        out.push(
            out[HEAD]
                * glam::Mat4::from_translation(Vec3::from_array(JAW_HINGE))
                * glam::Mat4::from_quat(Quat::from_rotation_y((mouth + 0.087) as f32)),
        );
        out
    }

    /// 诊断：打印 MuJoCo 自己算的左右脚 site 世界坐标（与渲染矩阵互验）
    fn print_sites(&self) {
        let m = self.data.ffi();
        for name in ["left_foot", "right_foot"] {
            if let Some(sid) = self.data.model().name_to_id(MjtObj::mjOBJ_SITE, name) {
                let x = unsafe { f64_at(m.site_xpos, sid * 3) };
                let y = unsafe { f64_at(m.site_xpos, sid * 3 + 1) };
                let z = unsafe { f64_at(m.site_xpos, sid * 3 + 2) };
                // MuJoCo 世界是 z-up：渲染世界 (x, y, z)_render = (x, z, −y)_mj
                println!("  mujoco {name}: world=({x:+.4},{z:+.4},{:+.4})", -y);
            }
        }
    }

    /// 诊断：躯干平面坐标（MuJoCo x/y，米）—— 横移/侧向碰撞验证用
    fn trunk_xy(&self) -> (f32, f32) {
        let b = self.mj_body[TRUNK];
        let m = self.data.ffi();
        (unsafe { f64_at(m.xpos, b * 3) } as f32,
         unsafe { f64_at(m.xpos, b * 3 + 1) } as f32)
    }

    /// 诊断：躯干位置（x 前进方向、z 高度，MuJoCo z-up 世界）与直立度
    fn trunk_pose(&self) -> (f32, f32, f32) {
        let b = self.mj_body[TRUNK];
        let m = self.data.ffi();
        let x = unsafe { f64_at(m.xpos, b * 3) } as f32;
        let z = unsafe { f64_at(m.xpos, b * 3 + 2) } as f32;
        let qw = unsafe { f64_at(m.xquat, b * 4) };
        let qx = unsafe { f64_at(m.xquat, b * 4 + 1) };
        let qy = unsafe { f64_at(m.xquat, b * 4 + 2) };
        let qz = unsafe { f64_at(m.xquat, b * 4 + 3) };
        // trunk +z 轴的世界 z 分量
        let up = (1.0 - 2.0 * (qx * qx + qy * qy)) as f32;
        let _ = qw;
        let _ = qz;
        (x, z, up)
    }
}

#[derive(Debug)]
struct TickEvent;

struct App {
    root: PathBuf,
    world: World,
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    cam: Camera,
    drag: Option<(MouseButton, f32, f32)>,
    cursor: Option<(f32, f32)>,
    last_title: Instant,
    // HUD（resumed 时初始化）
    egui: Option<HudState>,
    frame_dt: f32,
    last_tick_wall: Instant,
    /// 下一个应处理的 tick 截止时间（截止时间驱动，落后即重对齐）
    next_tick: Instant,
    auto_start: Instant,
    auto_stage: u8,
    auto_xpress: Option<Instant>,
    /// 文件对话框结果通道（对话框在后台线程弹，避免主线程/渲染冻结）
    dlg_tx: std::sync::mpsc::Sender<DlgPick>,
    dlg_rx: std::sync::mpsc::Receiver<DlgPick>,
    last_render: Instant,
    tick_meter: (u64, Instant),   // (计数, 窗口起点)
    pub tick_hz: f32,
}

struct HudState {
    ctx: egui::Context,
    winit: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    screen_desc: egui_wgpu::ScreenDescriptor,
}

impl App {
    fn render(&mut self) {
        // 对话框选完的文件：策略 → 当前目标槽位；场景 → 注册并选中
        while let Ok(pick) = self.dlg_rx.try_recv() {
            match pick {
                DlgPick::Policy(slot, path) => self.world.load_policy_path(slot, &path),
                DlgPick::Scene(path) => self.world.load_scene_path(&path),
            }
        }
        if norender() {
            return;   // 隔离实验：只跳渲染，其余（窗口/节拍/物理）不变
        }
        let t_render = Instant::now();
        let (gpu, window) = match (&mut self.gpu, &self.window) {
            (Some(g), Some(w)) => (g, w),
            _ => return,
        };
        self.frame_dt = self.frame_dt * 0.9 + t_render.duration_since(self.last_render).as_secs_f32() * 0.1;
        self.last_render = t_render;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 { return; }
        // DPI/尺寸每帧对齐：跨屏拖动会改 scale_factor（双屏缩放比例不同的典型场景），
        // tessellate 和 egui 输入必须用同一个当前值 —— screen_desc 若只在启动赋值，
        // 跨屏/调窗后 egui 内部比例已变而这里还是旧值，HUD 全部错位乱跑
        if let Some(hud) = self.egui.as_mut() {
            hud.screen_desc.size_in_pixels = [size.width, size.height];
            hud.screen_desc.pixels_per_point = window.scale_factor() as f32;
        }

        // 模型矩阵：支架 = FK + 托举；物理 = 刚体真实位姿
        let (models, physics_follow) = match self.world.mode {
            Mode::Rig => {
                let models = fk(&self.world.sim.q);
                let want = rig_lift_target(&models);
                self.world.sim.ride_lift += (want - self.world.sim.ride_lift) * 0.25;
                (models, None)   // 支架模式无刚体位姿可跟：相机原地
            }
            Mode::Physics => {
                let ph = self.world.phys.as_ref().unwrap();
                let models = ph.world_matrices(self.world.sim.targets[MOUTH_INDEX]);
                let t = models[TRUNK].transform_point3(Vec3::ZERO);
                (models, Some(Vec3::new(t.x, t.y * 0.6 + 0.02, t.z)))
            }
        };
        // 物理模式相机轻轻跟着鸭子
        if let Some(follow) = physics_follow {
            self.cam.target += (follow - self.cam.target) * 0.04;
        }
        // 只有滚轮缩放走平滑；拖拽 1:1 跟手（平滑会"飘"）
        self.cam.glide(self.frame_dt.min(0.05));

        // 尺寸变化 → 重建深度/MSAA 纹理 + 重配 surface
        if gpu.depth_size != (size.width, size.height) {
            gpu.config.width = size.width;
            gpu.config.height = size.height;
            gpu.surface.configure(&gpu.device, &gpu.config);
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            gpu.depth_view = Some(tex.create_view(&Default::default()));
            let msaa = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa"),
                size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format: gpu.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            gpu.msaa_view = Some(msaa.create_view(&Default::default()));
            gpu.depth_size = (size.width, size.height);
            if let Some(hud) = self.egui.as_mut() {
                hud.screen_desc.size_in_pixels = [size.width, size.height];
            }
        }

        let vp = self.cam.view_proj(size.width as f32 / size.height as f32);
        // —— 每帧 uniform：相机 + 模型矩阵（渲染素材全部由 Renderer 持有）——
        gpu.renderer.set_camera(vp);
        gpu.renderer.set_models(&models);
        // 场景切换 → 重建地面素材（视觉样式或坡体任一变化都要重画）
        gpu.renderer.ensure_ground(self.world.ground_kind());
        // 地面跟随：物理模式吸附到格点（视觉上无限延伸）；支架模式原地滚动。
        // 吸附步长必须整除视觉图案的"相同外观周期"（棋盘 1.0m / 网格 0.25m），
        // 由 ground_snap_pos 统一决定 —— 两条渲染路径共用，避免只改一边。
        let ground_mat = if self.world.mode == Mode::Physics {
            let tp = models[TRUNK].transform_point3(Vec3::ZERO);
            let (sx, sz) = ground_snap_pos(tp.x, tp.z, self.world.scene);
            glam::Mat4::from_translation(Vec3::new(sx, 0.0, sz))
        } else {
            // 支架模式：图案滚动（棋盘此前只有网格线滚、底板不动）
            glam::Mat4::from_translation(Vec3::new(
                -self.world.sim.treadmill[0], 0.0, -self.world.sim.treadmill[1]))
        };
        gpu.renderer.set_ground_mat(ground_mat);

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame.texture.create_view(&Default::default());
        // 深度/MSAA 视图由尺寸变化分支创建；此处守卫而非 unwrap（不变量不该靠隐式顺序维持）
        let (Some(msaa_view), Some(depth_view)) = (gpu.msaa_view.as_ref(), gpu.depth_view.as_ref())
        else {
            return;
        };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.055, g: 0.07, b: 0.086, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            gpu.renderer.draw(&mut rpass);
        }
        // —— HUD（egui）：面板事件 → 命令队列 → 应用 ——
        let mut user_cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
        let hud_off = nohud();
        if let Some(hud) = self.egui.as_mut().filter(|_| !hud_off) {
            hud.ctx.begin_pass(hud.winit.take_egui_input(&window));
            let mut cmds = Vec::new();
            let frame_dt = self.frame_dt;
            build_hud(&hud.ctx, &self.world, frame_dt, self.tick_hz, &mut cmds);
            let full = hud.ctx.end_pass();
            for c in cmds {
                if matches!(c, UiCmd::PickFile(_) | UiCmd::PickScene) {
                    // 后台弹对话框：主线程继续跑仿真与渲染，不冻结
                    let tx = self.dlg_tx.clone();
                    let (for_scene, slot) = match c { UiCmd::PickFile(i) => (false, i), _ => (true, 0) };
                    std::thread::spawn(move || {
                        let picked = if for_scene {
                            rfd::FileDialog::new()
                                .add_filter("MuJoCo 场景 XML", &["xml"])
                                .set_title("选择场景文件")
                                .pick_file()
                        } else {
                            rfd::FileDialog::new()
                                .add_filter("ONNX 策略", &["onnx"])
                                .set_title("选择策略文件")
                                .pick_file()
                        };
                        if let Some(path) = picked {
                            let _ = tx.send(if for_scene { DlgPick::Scene(path) } else { DlgPick::Policy(slot, path) });
                        }
                    });
                } else {
                    apply_cmd(&mut self.world, c);
                }
            }
            let jobs = hud.ctx.tessellate(full.shapes, hud.screen_desc.pixels_per_point);
            for (id, delta) in &full.textures_delta.set {
                hud.renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
            user_cmd_bufs = hud.renderer.update_buffers(
                &gpu.device, &gpu.queue, &mut enc, &jobs, &hud.screen_desc,
            );
            {
                let egui_pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hud"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,      // 叠在 3D 上
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                // forget_lifetime：egui 0.31 要求 'static pass；pass 不逃逸本块，安全（官方同款）
                hud.renderer.render(&mut egui_pass.forget_lifetime(), &jobs, &hud.screen_desc);
            }
        }
        gpu.queue.submit(
            std::iter::once(enc.finish()).chain(user_cmd_bufs.into_iter()),
        );
        frame.present();

        if self.last_title.elapsed() >= Duration::from_secs(1) {
            self.last_title = Instant::now();
            let w = &self.world;
            let mode = if w.mode == Mode::Physics { "PHYSICS" } else { "RIG" };
            window.set_title(&format!(
                "duck3d · {} · {} · gain {} · infer {:.2}ms · tick {:.0}Hz #{}",
                mode, w.ctl.label, w.ctl.gain, w.infer_us / 1000.0, self.tick_hz, w.tick_count
            ));
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════
   8.5 · HUD（egui）—— 科技感遥测面板
   ════════════════════════════════════════════════════════════════════════ */

impl ApplicationHandler<TickEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title("duck3d")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 800));
        let window = Arc::new(event_loop.create_window(attrs).expect("开窗口"));
        let mut cad = load_cad(&self.root.join("assets/duck_cad.bin")).expect("加载 CAD");
        add_eyes(&mut cad);
        let _ = &mut cad;
        println!("CAD: {} 顶点 / {} body 记录", cad.vertices.len(), cad.bodies.len());
        let gpu = init_gpu(&window, cad).expect("初始化 wgpu");
        self.world.gpu_name = gpu.name.clone();
        // DUCK3D_SCENE=<场景名>：启动即自动选中该场景（免手点；名字见启动日志的 scene 列表）
        if let Ok(name) = std::env::var("DUCK3D_SCENE") {
            if let Some(i) = self.world.scene_idx_by_name(&name) {
                self.world.select_scene(i);
                println!("[scene] 启动自动选中 {name}（#{i}）");
            } else {
                println!("[scene] 找不到场景 {name}，用默认");
            }
        }
        // HUD（暗底青高亮 + 中文）上下文，样式与离屏截图共用
        let ctx = build_egui_ctx();
        let ppp = window.scale_factor() as f32;
        let winit = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(ppp),
            None,
            None,
        );
        let size = window.inner_size();
        self.egui = Some(HudState {
            ctx,
            winit,
            renderer: egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, None, 1, false),
            screen_desc: egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: ppp,
            },
        });
        self.gpu = Some(gpu);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // HUD 先吃输入（面板上的指针/键盘）；CloseRequested/Resized 等不受影响
        if let Some(hud) = self.egui.as_mut() {
            // 键盘事件**不**转发给 egui：拖过滑条后 egui 持有键盘焦点会消费
            // keyup，技能键的松开事件丢失 → 按键集合永久含 X（面板没有文本
            // 输入，键盘对 HUD 无用）。
            // Resized/ScaleFactorChanged 必须转发：egui 的 screen_rect 和 DPI
            // 比例全靠它更新 —— 跨屏（双屏缩放不同）或调窗后不转发，锚定全错
            if matches!(
                event,
                WindowEvent::CursorMoved { .. }
                    | WindowEvent::MouseInput { .. }
                    | WindowEvent::MouseWheel { .. }
                    | WindowEvent::Touch { .. }
                    | WindowEvent::Resized { .. }
                    | WindowEvent::ScaleFactorChanged { .. }
            ) {
                if let Some(win) = self.window.clone() {
                    let resp = hud.winit.on_window_event(&win, &event);
                    if resp.repaint {
                        win.request_redraw();
                    }
                    if resp.consumed {
                        return;
                    }
                }
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state, repeat, .. },
                ..
            } => {
                if keylog() {
                    println!("[kbd] {code:?} {state:?} repeat={repeat}");
                }
                match state {
                ElementState::Pressed => match code {
                    KeyCode::Escape => event_loop.exit(),
                    KeyCode::Space => self.world.sim.enabled = !self.world.sim.enabled,
                    KeyCode::KeyP => self.world.toggle_mode(),
                    KeyCode::Tab => self.world.console_open = !self.world.console_open,
                    KeyCode::Digit1 => self.world.set_scene(Scene::Flat),
                    KeyCode::Digit2 => self.world.set_scene(Scene::Checker),
                    KeyCode::Digit3 => self.world.toggle_ramp_scene(),
                    KeyCode::KeyR => self.world.reset_all(),
                    // 平移/转向/头/连滚判定都读按键集合 —— 键盘按下时登记，
                    // HUD 按钮路径（on_key）不登记
                    _ => {
                        self.world.keys.insert(code);
                        self.world.on_key(code);
                    }
                },
                ElementState::Released => { self.world.keys.remove(&code); }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if let Some((btn, x0, y0)) = self.drag {
                    // 逻辑像素：双屏缩放比例不同，拖拽手感保持一致
                    let ppp = self.window.as_ref().map(|w| w.scale_factor() as f32).unwrap_or(1.0);
                    let (dx, dy) = ((x - x0) / ppp, (y - y0) / ppp);
                    match btn {
                        MouseButton::Left => {
                            self.cam.yaw += dx * DRAG_YAW;
                            self.cam.pitch = (self.cam.pitch + dy * DRAG_PITCH).clamp(0.02, 1.35);
                        }
                        MouseButton::Right | MouseButton::Middle => {
                            let right = Vec3::new(-self.cam.yaw.sin(), 0.0, self.cam.yaw.cos());
                            self.cam.target += right * dx * DRAG_PAN * self.cam.dist;
                            self.cam.target.y += dy * DRAG_PAN * self.cam.dist;
                        }
                        _ => {}
                    }
                }
                self.cursor = Some((x, y));
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    self.drag = Some((button, self.cursor.map(|c| c.0).unwrap_or(0.0), self.cursor.map(|c| c.1).unwrap_or(0.0)));
                }
                ElementState::Released => self.drag = None,
            },
            // 拖 .onnx 进窗口 = 加载到 HUD 里选中的槽位（无需文件对话框）
            WindowEvent::HoveredFile(ref path) => {
                let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                // .xml = 场景（MuJoCo XML），其余按策略槽处理
                let is_xml = path.extension().map(|e| e.eq_ignore_ascii_case("xml")) == Some(true);
                self.world.drop_hint = Some(if is_xml {
                    format!("松手加载场景 {name}")
                } else {
                    let role = ROLES.get(self.world.load_target).map(|(r, _)| *r).unwrap_or("walk");
                    format!("松手把 {name} 加载到 {role}")
                });
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            WindowEvent::HoveredFileCancelled => {
                self.world.drop_hint = None;
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            WindowEvent::DroppedFile(path) => {
                self.world.drop_hint = None;
                // 扩展名分流：.xml → 场景；其余 → 策略槽（.onnx 校验在 load_policy_path 内）
                if path.extension().map(|e| e.eq_ignore_ascii_case("xml")) == Some(true) {
                    self.world.load_scene_path(&path);
                } else {
                    let slot = self.world.load_target;
                    self.world.load_policy_path(slot, &path);
                }
                if let Some(w) = &self.window { w.request_redraw(); }
                return;   // path 已移出，不能再匹配 event
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.001,
                };
                self.cam.g_dist = (self.cam.g_dist * (1.0 - d)).clamp(0.18, 1.6);   // 平滑逼近
            }
            _ => {}
        }
        // 只在 tick（50Hz）和尺寸变化时重绘 —— 渲染帧率封顶 50Hz，
        // 鼠标移动不再触发额外重绘（相机更新在下一拍重绘里生效，依旧顺滑）
        if matches!(event, WindowEvent::Resized { .. } | WindowEvent::ScaleFactorChanged { .. }) {
            if let Some(w) = &self.window { w.request_redraw(); }
        }
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, _ev: TickEvent) {
        let now = Instant::now();
        // 开机 3 秒倒计时 → 自动进物理模式（用户在倒计时内手动切过模式就取消）
        if !self.world.boot_switched && self.world.mode == Mode::Rig {
            let elapsed = self.world.boot.elapsed().as_secs_f32();
            if elapsed >= 3.0 {
                self.world.boot_switched = true;
                self.world.toggle_mode();
            }
        }
        // 截止时间驱动的节拍：正常跟上时锁 50Hz；主线程忙（渲染 20ms+）时
        // 队列会积压 —— 落后超过 3 拍就重对齐时间轴并丢弃积压（不追帧），
        // 保证策略收到的状态滞后恒定在 ~1 拍内。否则闭环相位滞后会让姿态垮掉。
        let period = Duration::from_millis(20);
        if now < self.next_tick {
            return;                                   // 早到的排队事件：丢
        }
        let behind = now.duration_since(self.next_tick);
        self.next_tick = if behind > period * 3 {
            now + period                              // 落后太多：重对齐，丢积压
        } else {
            self.next_tick + period
        };
        self.last_tick_wall = now;
        // 自动驾驶脚本（验证用）：DUCK3D_AUTODRIVE=<切物理秒>；状态机避免重复触发
        if let Ok(v) = std::env::var("DUCK3D_AUTODRIVE") {
            let t0: u64 = v.parse().unwrap_or(3);
            let stage = self.auto_stage;
            if stage == 0 && self.auto_start.elapsed().as_secs() >= t0 {
                self.world.toggle_mode();
                println!("[auto] t={t0}s → 切物理模式");
                self.auto_stage = 1;
            } else if self.auto_stage == 1 && self.auto_start.elapsed().as_secs() >= t0 + 2 {
                self.world.keys.insert(KeyCode::KeyW);
                println!("[auto] → 按住 W");
                self.auto_stage = 2;
            } else if self.auto_stage == 2 && self.auto_start.elapsed().as_secs() >= t0 + 8 {
                self.world.keys.remove(&KeyCode::KeyW);
                println!("[auto] → 松开 W");
                self.auto_stage = 3;
            } else if self.auto_stage == 3 {
                if let Ok(xt) = std::env::var("DUCK3D_AUTODRIVE_X") {
                    let xt: u64 = xt.parse().unwrap_or(10);
                    if self.auto_start.elapsed().as_secs() >= xt {
                        if std::env::var("DUCK3D_AUTODRIVE_XMODE").map(|m| m != "stand").unwrap_or(true) {
                            self.world.keys.insert(KeyCode::KeyW);
                        }
                        // 单击 = 只调 on_key（不按住）；按住 = 进 keys 停 N 拍再松
                        let hold: u64 = std::env::var("DUCK3D_XHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                        self.world.on_key(KeyCode::KeyX);
                        if hold > 0 {
                            self.world.keys.insert(KeyCode::KeyX);
                        }
                        println!("[auto] → 触发滚翻（hold={hold}tick）");
                        self.auto_xpress = Some(Instant::now());
                        self.auto_stage = 4;
                    }
                }
            } else if self.auto_stage == 4 {
                let hold_s: f64 = std::env::var("DUCK3D_XHOLD").ok().and_then(|v| v.parse::<u64>().ok()).map(|t| t as f64 * 0.02).unwrap_or(0.0);
                if hold_s > 0.0 && self.world.keys.contains(&KeyCode::KeyX) {
                    // 按住 N 拍后松手（近似）
                    let pressed_for = self.auto_xpress.map(|t| t.elapsed().as_secs_f64()).unwrap_or(f64::MAX);
                    if pressed_for >= hold_s {
                        self.world.keys.remove(&KeyCode::KeyX);
                        self.world.keys.remove(&KeyCode::KeyW);
                        println!("[auto] → 松开 X/W");
                    }
                }
                if self.auto_start.elapsed().as_secs() >= 10 + 12 {
                let (_, z, up) = self.world
                    .phys
                    .as_ref()
                    .map(|ph| ph.trunk_pose())
                    .unwrap_or((0.0, 0.0, 0.0));
                    println!("[auto] 滚翻后 12s: z={z:.3} up={up:+.2} label={}", self.world.ctl.label);
                    self.auto_stage = 5;
                }
            }
        }
        self.world.tick(TICK_DT);
        // 节拍频率计量：1 秒窗口内的实际 tick 数（此前该字段从不更新，恒 0）
        {
            let (n, t0) = self.tick_meter;
            let n = n + 1;
            let el = t0.elapsed().as_secs_f32();
            if el >= 1.0 {
                self.tick_hz = n as f32 / el;
                self.tick_meter = (0, Instant::now());
            } else {
                self.tick_meter = (n, t0);
            }
        }
        if let Some(w) = &self.window { w.request_redraw(); }
    }
}

/* ════════════════════════════════════════════════════════════════════════
   9 · 路径发现 + main
   ════════════════════════════════════════════════════════════════════════ */

fn find_repo_root() -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = vec![];
    if let Ok(p) = std::env::var("DUCK3D_ROOT") { candidates.push(PathBuf::from(p)); }
    if let Ok(cwd) = std::env::current_dir() {
        let mut d: &Path = &cwd;
        for _ in 0..4 {
            candidates.push(d.to_path_buf());
            d = d.parent().unwrap_or(d);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        // 从 exe 往上找 4 层：app/target/release → … → 项目根（独立项目下至少 3 层）
        let mut d: &Path = exe.parent().unwrap_or(&exe);
        for _ in 0..4 {
            candidates.push(d.to_path_buf());
            d = d.parent().unwrap_or(d);
        }
    }
    for c in candidates {
        if c.join("policies/alpha_walking.onnx").is_file() && c.join("assets/duck_cad.bin").is_file() {
            return Ok(c);
        }
    }
    bail!("找不到 project 根目录（需要 policies/ 与 assets/ 同级）。用 DUCK3D_ROOT 环境变量指定。")
}

const HELP: &str = "\
duck3d 原生版 · 键位
  W/S 前后 · A/D 平移 · Q/E 转向
  ↑/↓ 摆颈 · ←/→ 头部偏航
  空格 使能开关 · R 复位 · P 支架/物理切换 · B body-pose 模式
  G 捡地 · X 前滚翻(按住连滚) · Z/C 左右踢 · V 坐/起 · M 嘴动画
  鼠标左键 旋转 · 滚轮 缩放 · 右键 平移 · Esc 退出
";

fn selftest(root: &Path) -> Result<()> {
    println!("== selftest：无窗口仿真 ==");
    let pol = Policies::load(&root.join("policies"));
    let mut world = World {
        root: root.to_path_buf(),
        load_target: 0,
        body_cmd: [0.0; 3],
        body_obs: [0.0; 3],
        scene: Scene::Flat,
        scenes: discover_scenes(&root),
        scene_sel: 0,
        scale_mult: 1.0,
        drop_hint: None,
        load_msg: None,
        console_open: true,
        boot: Instant::now(),
        boot_switched: false,
        announce: None,
        sim: Sim::new(),
        ctl: Controller::new(),
        pol,
        keys: HashSet::new(),
        body_active: false,
        infer_us: 0.0,
        tick_count: 0,
        mode: Mode::Rig,
        phys: None,
        imu: ([0.0, 0.0, -1.0], [0.0; 3]),
        gpu_name: String::new(),
    };
    let mut labels: Vec<&'static str> = Vec::new();
    for _ in 0..100 { world.tick(TICK_DT); labels.push(world.ctl.label); }       // 站 2s
    world.keys.insert(KeyCode::KeyW);
    for _ in 0..200 { world.tick(TICK_DT); labels.push(world.ctl.label); }       // 走 4s
    world.keys.remove(&KeyCode::KeyW);
    world.on_key(KeyCode::KeyV);
    for _ in 0..50 { world.tick(TICK_DT); labels.push(world.ctl.label); }        // 坐 1s
    world.on_key(KeyCode::KeyV);
    for _ in 0..100 { world.tick(TICK_DT); labels.push(world.ctl.label); }       // 起 2s
    let saw = |l: &str| labels.contains(&l);
    let q = world.sim.q;
    println!("labels: stand={} walk={} sit={} rise={}", saw("stand"), saw("walk"), saw("sit"), saw("rise"));
    println!("infer: {:.2} ms/tick", world.infer_us / 1000.0);
    println!("final hip_pitch L/R: {:.3}/{:.3}  knee L/R: {:.3}/{:.3}", q[2], q[12], q[3], q[13]);
    anyhow::ensure!(saw("walk"), "没有出现 walk 标签");
    anyhow::ensure!(q[2].abs() > 0.05 || q[12].abs() > 0.05, "关节几乎没动，策略没驱动？");

    // —— 物理环境（MuJoCo）：策略闭环站立 12s ——
    world.reset_all();
    world.toggle_mode();   // 载入训练模型（scene.xml）
    anyhow::ensure!(world.phys.is_some(), "MuJoCo 模型没载入");
    // 复现窗口时序：先 Rig 跑一会儿 → 切物理 → 站 2s → W 走（DUCK3D_REPRO=1）
    if std::env::var("DUCK3D_REPRO").is_ok() {
        println!("── 复现窗口时序：Rig 5s → 切物理 → 2s → W 3s ──");
        for _ in 0..250 { world.tick(TICK_DT); }
        world.toggle_mode();
        for _ in 0..100 { world.tick(TICK_DT); }
        world.keys.insert(KeyCode::KeyW);
        for t in 0..150 {
            world.tick(TICK_DT);
            if t % 25 == 0 {
                let (x, z, up) = world.phys.as_ref().unwrap().trunk_pose();
                println!("  repro t={:4.2}s x={x:+.3} z={z:.3} up={up:+.2} label={}", t as f32 * 0.02, world.ctl.label);
            }
        }
        world.keys.remove(&KeyCode::KeyW);
        let (x, z, up) = world.phys.as_ref().unwrap().trunk_pose();
        println!("  repro 结束: x={x:+.3} z={z:.3} up={up:+.2}");
        return Ok(());
    }

    let settle_s: u32 = std::env::var("DUCK3D_SETTLE").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    println!("── MuJoCo 站立测试（{settle_s}s）──");
    let mut min_up = 1.0f32;
    for t in 0..(settle_s as i32 * 50) {
        world.tick(TICK_DT);
        let (_, z, up) = world.phys.as_ref().unwrap().trunk_pose();
        if t >= 200 { min_up = min_up.min(up); }
        if t % 150 == 0 {
            println!("  t={:4.1}s z={z:.4} up={up:+.3} label={}", t as f32 * TICK_DT as f32, world.ctl.label);
        }
    }
    let (fx, fz, fup) = world.phys.as_ref().unwrap().trunk_pose();
    println!("MuJoCo: 站立 12s 末态 z={fz:.3} up={fup:+.3}  (2s后最低 up={min_up:+.3})");

    // 行走：按住 W 4 秒，看是否真的前进（x 位移 + 仍直立）
    world.keys.insert(KeyCode::KeyW);
    for _ in 0..200 { world.tick(TICK_DT); }
    world.keys.remove(&KeyCode::KeyW);
    let (wx, wz, wup) = world.phys.as_ref().unwrap().trunk_pose();
    println!("MuJoCo: 前进 4s  Δx={:+.3}m  z={wz:.3} up={wup:+.3} label={}", wx - fx, world.ctl.label);
    for _ in 0..100 { world.tick(TICK_DT); }   // 停下再站 2s
    let (_, rz, rup) = world.phys.as_ref().unwrap().trunk_pose();
    println!("MuJoCo: 停走 2s  z={rz:.3} up={rup:+.3}");
    anyhow::ensure!(wx - fx > 0.1, "行走没有前进（Δx={:+.3}m）", wx - fx);
    anyhow::ensure!(rup > 0.6, "走完站不住了（up={rup:+.3}）");

    let saw = |l: &str| labels.contains(&l);
    let q = world.sim.q;
    println!("labels: stand={} walk={} sit={} rise={}", saw("stand"), saw("walk"), saw("sit"), saw("rise"));
    println!("infer: {:.2} ms/tick", world.infer_us / 1000.0);
    println!("final hip_pitch L/R: {:.3}/{:.3}  knee L/R: {:.3}/{:.3}", q[2], q[12], q[3], q[13]);
    anyhow::ensure!(saw("walk"), "没有出现 walk 标签");
    anyhow::ensure!(q[2].abs() > 0.05 || q[12].abs() > 0.05, "关节几乎没动，策略没驱动？");

    anyhow::ensure!(fz > 0.06 && fz < 0.20, "MuJoCo 物理没站住（末态 z={fz:.3}）");
    let _ = min_up;

    // ── 滚翻（roulade）：X 触发 → 2s 窗口 → 应回到主策略并恢复站立 ──
    println!("── 滚翻测试 ──");
    world.on_key(KeyCode::KeyX);   // 触发（和窗口按 X 同一条路径）
    world.tick(TICK_DT);
    world.keys.remove(&KeyCode::KeyX);
    let mut roll_handback = None;
    for t in 0..300 {              // 6s：滚翻 + 恢复
        world.tick(TICK_DT);
        if roll_handback.is_none() && world.ctl.label != "roulade" {
            roll_handback = Some(t as f64 * TICK_DT);
        }
    }
    let roll_handback = roll_handback.unwrap_or(f64::MAX);
    println!("  滚翻交回控制权用时 {:.2}s（窗口 2.0s+续窗）", roll_handback);
    let (fz2, fup2) = { let (_, z, up) = world.phys.as_ref().unwrap().trunk_pose(); (z, up) };
    println!("  roulade 恢复后: z={fz2:.3} up={fup2:+.2} label={}", world.ctl.label);
    // 连滚（按住 X 不放：链式触发第二次滚翻，第二滚带着第一滚的动作历史）
    world.keys.insert(KeyCode::KeyX);           // 一直按住
    world.tick(TICK_DT);                        // 触发第一滚
    for _ in 0..150 { world.tick(TICK_DT); }    // 3s：第一滚 + 链第二滚
    world.keys.remove(&KeyCode::KeyX);          // 松手
    for _ in 0..250 { world.tick(TICK_DT); }    // 最多 5s 恢复
    let (cz, cup) = { let (_, z, up) = world.phys.as_ref().unwrap().trunk_pose(); (z, up) };
    println!("  连滚恢复后: z={cz:.3} up={cup:+.2}");
    anyhow::ensure!(cz > 0.06 && cup > 0.5, "连滚后没恢复（z={cz:.3} up={cup:+.2}）");

    // ── 测量：踢腿动作的自然时长（qd 沉降 = 动作结束）──
    println!("── 踢腿时长测量 ──");
    world.on_key(KeyCode::KeyZ);   // 触发（同窗口按键路径）
    world.tick(TICK_DT);
    let mut kick_done = None;
    for t in 1..=175 {
        world.tick(TICK_DT);
        let qd_max = (0..NUM_JOINTS).filter(|&i| i != MOUTH_INDEX)
            .map(|i| world.sim.qd[i].abs()).fold(0.0f64, f64::max);
        if t % 13 == 0 {
            println!("  kick t={:4.2}s qdmax={qd_max:5.2} label={}", t as f32 * 0.02, world.ctl.label);
        }
        if kick_done.is_none() && t as f64 * 0.02 > 0.8 && qd_max < 1.5 {
            kick_done = Some(t as f64 * 0.02);
        }
    }
    println!("  踢腿运动沉降于 {:.2}s（窗口 3.0s）", kick_done.unwrap_or(f64::MAX));
    // 交还时间 = 触发到 label 回 stand/walk 的秒数（早退应远小于窗口）
    let handback = {
        world.on_key(KeyCode::KeyZ);
        world.tick(TICK_DT);
        let t0 = world.tick_count;
        loop {
            world.tick(TICK_DT);
            let back = world.ctl.label != "kick_left" && world.ctl.label != "kick_right";
            let elapsed = (world.tick_count - t0) as f64 * TICK_DT;
            if back || elapsed > 3.2 { break elapsed; }
        }
    };
    println!("  踢腿交回控制权用时 {:.2}s（窗口 3.0s）", handback);
    anyhow::ensure!(handback < 1.6, "踢腿后死等太久（{handback:.2}s）");
    for _ in 0..150 { world.tick(TICK_DT); }

    // ── body-pose：B 模式 + 姿态指令进观测（蹲 2cm + pitch 15°）──
    // ── 测量：踢腿动作的自然时长（qd 沉降 = 动作结束）──
    println!("── 踢腿时长测量 ──");
    world.on_key(KeyCode::KeyZ);   // 触发（同窗口按键路径）
    world.tick(TICK_DT);
    let mut kick_done = None;
    for t in 1..=175 {
        world.tick(TICK_DT);
        let qd_max = (0..NUM_JOINTS).filter(|&i| i != MOUTH_INDEX)
            .map(|i| world.sim.qd[i].abs()).fold(0.0f64, f64::max);
        if t % 13 == 0 {
            println!("  kick t={:4.2}s qdmax={qd_max:5.2} label={}", t as f32 * 0.02, world.ctl.label);
        }
        if kick_done.is_none() && t as f64 * 0.02 > 0.8 && qd_max < 1.5 {
            kick_done = Some(t as f64 * 0.02);
        }
    }
    println!("  踢腿运动沉降于 {:.2}s（窗口 3.0s）", kick_done.unwrap_or(f64::MAX));
    // 交还时间 = 触发到 label 回 stand/walk 的秒数（早退应远小于窗口）
    let handback = {
        world.on_key(KeyCode::KeyZ);
        world.tick(TICK_DT);
        let t0 = world.tick_count;
        loop {
            world.tick(TICK_DT);
            let back = world.ctl.label != "kick_left" && world.ctl.label != "kick_right";
            let elapsed = (world.tick_count - t0) as f64 * TICK_DT;
            if back || elapsed > 3.2 { break elapsed; }
        }
    };
    println!("  踢腿交回控制权用时 {:.2}s（窗口 3.0s）", handback);
    anyhow::ensure!(handback < 1.6, "踢腿后死等太久（{handback:.2}s）");
    for _ in 0..150 { world.tick(TICK_DT); }

    // ── body-pose：B 模式 + 姿态指令进观测（蹲 2cm + pitch 15°）──
    println!("── body-pose 测试 ──");

    world.body_cmd = [0.0, 0.0, 0.35];   // 前倾 20°（超出训练 ±15° 一点，策略跟得住）
    world.on_key(KeyCode::KeyB);   // 进入姿态模式
    for t in 0..150 {              // 3s：滑向目标姿态并稳住
        world.tick(TICK_DT);
        if t % 50 == 0 {
            let (_, z, up) = world.phys.as_ref().unwrap().trunk_pose();
            println!("  body t={:4.1}s z={z:.3} up={up:+.2} obs=({:+.3},{:+.3},{:+.3}) label={}",
                t as f32 * 0.02, world.body_obs[0], world.body_obs[1], world.body_obs[2], world.ctl.label);
        }
    }
    println!("  body: obs=({:+.3},{:+.3},{:+.3}) label={}",
        world.body_obs[0], world.body_obs[1], world.body_obs[2], world.ctl.label);
    anyhow::ensure!(world.body_obs[2] > 0.30, "body pitch 没进观测");
    let (_, _, up_lean) = world.phys.as_ref().unwrap().trunk_pose();
    anyhow::ensure!(up_lean < 0.98, "pitch 姿态没有实际倾斜（up={up_lean:+.2}）—— 追踪失效");
    anyhow::ensure!(world.ctl.label == "stand", "姿态模式下应走站立策略");
    world.on_key(KeyCode::KeyB);   // 退出
    for _ in 0..100 { world.tick(TICK_DT); }
    anyhow::ensure!(world.body_obs[0].abs() < 1e-3 && world.body_obs[2].abs() < 1e-3, "退出后 body 指令没归零");
    println!("  body-pose PASS");

    // 行进中触发滚翻（用户真实操作：走路时按 X）
    world.keys.insert(KeyCode::KeyW);
    for _ in 0..50 { world.tick(TICK_DT); }        // 走 1s
    world.on_key(KeyCode::KeyX);
    world.tick(TICK_DT);
    world.keys.remove(&KeyCode::KeyX);             // 松开 W 和 X
    for _ in 0..200 { world.tick(TICK_DT); }       // 滚翻 + 最多续 2s
    for _ in 0..250 { world.tick(TICK_DT); }       // 恢复期
    let (wz, wup) = { let (_, z, up) = world.phys.as_ref().unwrap().trunk_pose(); (z, up) };
    println!("  走动中滚翻恢复后: z={wz:.3} up={wup:+.2}");
    anyhow::ensure!(fz2 > 0.06, "滚翻后没站起来（z={fz2:.3}）");
    anyhow::ensure!(fup2 > 0.5, "滚翻后姿态崩了（up={fup2:+.2}）");
    anyhow::ensure!(wz > 0.06 && wup > 0.5, "走动中滚翻没恢复（z={wz:.3} up={wup:+.2}）");
    // ── 界面加载自检：槽位替换 + 失败路径都要干净 ──
    println!("── 界面加载（拖拽/浏览同一条路径）──");
    let stand_path = root.join("policies/alpha_stand.onnx");
    world.load_policy_path(6, &stand_path);            // 6 = roulade 槽
    println!("  槽位替换: {:?}", world.load_msg.as_ref().map(|(ok, m, _)| (*ok, m.clone())));
    anyhow::ensure!(world.pol.has("roulade"), "槽位替换没生效");
    world.load_policy_path(6, &root.join("policies/README.md"));   // 非 onnx
    anyhow::ensure!(matches!(world.load_msg, Some((false, _, _))), "非 onnx 文件应被拒绝");
    world.load_policy_path(6, &root.join("policies/no_such.onnx")); // 不存在
    anyhow::ensure!(matches!(world.load_msg, Some((false, _, _))), "不存在的文件应报错");
    anyhow::ensure!(world.pol.has("roulade"), "失败的加载不该丢掉原会话");
    println!("  失败路径: 两种都干净报错且保留原会话 ✓");


    anyhow::ensure!(min_up > 0.5, "MuJoCo 物理姿态崩了（最低 up={min_up:+.2}）");

    // ── 姿态键弹簧验证：按住 A 倾斜 → 松开回中（body 段观测值应回落）──
        world.on_key(KeyCode::KeyB);               // 进姿态模式
    world.keys.insert(KeyCode::KeyA);          // 按住侧倾
    for _ in 0..10 { world.tick(TICK_DT); }    // 0.2s：应已基本到位（响应速度断言）
    let quick = world.body_obs;
    for _ in 0..40 { world.tick(TICK_DT); }    // 到 1s
    let lean = world.body_obs;
    world.keys.remove(&KeyCode::KeyA);         // 松开 → 弹簧应回零
    for _ in 0..75 { world.tick(TICK_DT); }    // 1.5s
    let back = world.body_obs;
    println!("姿态键: 按住 A 1s → body_obs roll={:+.3}；松开 1.5s → roll={:+.3}", lean[1], back[1]);
    anyhow::ensure!(lean[1] > 0.10, "按住 A 应产生侧倾（实测 {}）", lean[1]);
    anyhow::ensure!(quick[1] > 0.9 * 0.26, "姿态键响应太慢：0.2s 只到 {:.3}（应 ≥0.234）", quick[1]);
    anyhow::ensure!(back[1].abs() < 0.02, "松开后应回中（实测 {}）", back[1]);
    println!("姿态键弹簧 ✓");
    world.on_key(KeyCode::KeyB);               // 退出姿态模式

    println!("selftest PASS");
    Ok(())
}


/* ------------------------------------------------------------------
   10 · 离屏渲染（--shot out.bmp）：无窗口渲一帧存 BMP —— 自动化验证用
   ------------------------------------------------------------------ */

fn offscreen_shot(root: &Path, out: &Path) -> Result<()> {
    let pol = Policies::load(&root.join("policies"));
    let mut cad = load_cad(&root.join("assets/duck_cad.bin"))?;
    add_eyes(&mut cad);
    let mut world = World {
        root: root.to_path_buf(),
        load_target: 0,
        body_cmd: [0.0; 3],
        body_obs: [0.0; 3],
        scene: Scene::Flat,
        scenes: discover_scenes(&root),
        scene_sel: 0,
        scale_mult: 1.0,
        drop_hint: None,
        load_msg: None,
        console_open: true,
        boot: Instant::now(),
        boot_switched: false,
        announce: None,
        sim: Sim::new(),
        ctl: Controller::new(),
        pol,
        keys: HashSet::new(),
        body_active: false,
        infer_us: 0.0,
        tick_count: 0,
        mode: Mode::Rig,
        phys: None,
        imu: ([0.0, 0.0, -1.0], [0.0; 3]),
        gpu_name: String::new(),
    };
    // 场景选择必须在物理初始化之前（场景决定载入哪个 MuJoCo XML）
    match std::env::var("DUCK3D_SHOT_SCENE").as_deref() {
        Ok("checker") => world.scene = Scene::Checker,
        Ok("ramp") => world.toggle_ramp_scene(),
        _ => {}
    }
    if std::env::var("DUCK3D_SHOT_PHYSICS").is_ok() {
        world.toggle_mode();                    // MuJoCo 物理模式（含当前场景的坡体）
        for _ in 0..100 { world.tick(TICK_DT); }
    } else if std::env::var("DUCK3D_SHOT_IDLE").is_ok() {
        world.sim.enabled = false;                  // 不出策略：停在 home 位
        for _ in 0..260 { world.tick(TICK_DT); }
    } else {
        for _ in 0..260 { world.tick(TICK_DT); }   // 支架模式：站立稳定
    }
    // 行走验证（DUCK3D_WALKSEC=秒数）：按 W 走，打印轨迹 —— 验证坡体碰撞。
    // DUCK3D_STRAFE=1 同时按住 A（斜向走）—— 验证坡体侧面走来不穿模
    if let Ok(sec) = std::env::var("DUCK3D_WALKSEC").ok().and_then(|v| v.parse::<u64>().ok()).ok_or(0) {
        let strafe = std::env::var("DUCK3D_STRAFE").is_ok();
        world.keys.insert(KeyCode::KeyW);
        if strafe { world.keys.insert(KeyCode::KeyA); }
        println!("── 行走轨迹（{sec}s{}）──", if strafe { " · 侧向" } else { "" });
        for t in 0..(sec * 50) {
            world.tick(TICK_DT);
            if t % 25 == 0 {
                let (x, z, up) = world.phys.as_ref().unwrap().trunk_pose();
                let (_, y) = world.phys.as_ref().unwrap().trunk_xy();
                println!("  walk t={:4.1}s x={x:+.3} y={y:+.3} z={z:.3} up={up:+.2}", t as f32 * 0.02);
            }
        }
        world.keys.remove(&KeyCode::KeyW);
        world.keys.remove(&KeyCode::KeyA);
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow::anyhow!("没有可用的 GPU 适配器"))?;
    println!("GPU: {:?}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))?;
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;

    // 渲染素材（管线/缓冲/地面/坡体/CAD）全部取自共用 Renderer ——
    // 与窗口路径同一份代码，离屏只差 render target 与 HUD 叠加
    let mut renderer = Renderer::new(&device, &queue, format, 1, cad)?;   // 离屏：目标单采样

    // 截图尺寸：默认 1280x800，DUCK3D_SHOT_SIZE=宽x高 可覆盖
    //（控制台内容比 800px 高，验证下半部分时需要更高的画布）
    let (w, h) = std::env::var("DUCK3D_SHOT_SIZE")
        .ok()
        .and_then(|s| {
            let mut it = s.split('x');
            Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
        })
        .unwrap_or((1280u32, 800u32));
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());

    let models = match world.mode {
        Mode::Physics => world.phys.as_ref().unwrap().world_matrices(world.sim.targets[MOUTH_INDEX]),
        Mode::Rig => fk(&world.sim.q),
    };
    let lift_m = glam::Mat4::from_translation(Vec3::new(0.0, rig_lift_target(&models), 0.0));
    let cam = if let Ok(spec) = std::env::var("DUCK3D_SHOT_CAM") {
        let v: Vec<f32> = spec.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 6 {
            Camera::new(v[0], v[1], v[2], Vec3::new(v[3], v[4], v[5]))
        } else {
            Camera::new(-0.85, 0.28, 0.48, Vec3::new(0.01, 0.09, 0.0))
        }
    } else {
        Camera::new(-0.85, 0.28, 0.48, Vec3::new(0.01, 0.09, 0.0))
    };
    renderer.set_camera(cam.view_proj(w as f32 / h as f32));
    // 物理模式用真实刚体位姿，支架模式用 FK + 托举（同窗口路径）
    let posed: Vec<glam::Mat4> = models.iter().map(|m| lift_m * *m).collect();
    renderer.set_models(&posed);
    let tp = (lift_m * models[TRUNK]).transform_point3(Vec3::ZERO);
    let (sx, sz) = ground_snap_pos(tp.x, tp.z, world.scene);
    let grid = glam::Mat4::from_translation(Vec3::new(sx, 0.0, sz));
    renderer.set_ground_mat(grid);

    renderer.ensure_ground(world.ground_kind());
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.055, g: 0.07, b: 0.086, a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None, occlusion_query_set: None,
        });
        renderer.draw(&mut rpass);
    }
    // HUD 叠进离屏帧（DUCK3D_SHOT_HUD=1）：窗口里的面板长什么样，这里就什么样
    let mut hud_cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
    if shot_hud() {
        if std::env::var("DUCK3D_NOCONSOLE").is_ok() {
            world.console_open = false;
        }
        let ctx = build_egui_ctx();
        let mut renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: 1.0,
        };
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(w as f32, h as f32),
            )),
            ..Default::default()
        };
        // egui 的 anchored Area 首帧是 sizing pass（invisible → 全部 Noop），
        // 第二帧才真正渲染 —— 离屏跑两趟：第一趟学尺寸+产字体图集，
        // 第二趟出真实画面。两趟的纹理 delta 都要喂给 renderer，缺一趟字就没了。
        ctx.begin_pass(raw.clone());
        let mut cmds = Vec::new();
        build_hud(&ctx, &world, 1.0 / 60.0, 50.0, &mut cmds);
        let full1 = ctx.end_pass();
        for (id, delta) in &full1.textures_delta.set {
            renderer.update_texture(&device, &queue, *id, delta);
        }
        ctx.begin_pass(raw);
        let mut cmds = Vec::new();
        build_hud(&ctx, &world, 1.0 / 60.0, 50.0, &mut cmds);
        let full = ctx.end_pass();
        for (id, delta) in &full.textures_delta.set {
            renderer.update_texture(&device, &queue, *id, delta);
        }
        let jobs = ctx.tessellate(full.shapes, 1.0);
        println!("[hud-shot] prims={}", jobs.len());
        hud_cmd_bufs = renderer.update_buffers(&device, &queue, &mut enc, &jobs, &screen);
        let rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud-shot"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.render(&mut rpass.forget_lifetime(), &jobs, &screen);
    }

    let bytes_per_row = (w * 4).next_multiple_of(256);
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"), size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &read_buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(enc.finish()).chain(hud_cmd_bufs.into_iter()));
    let slice = read_buf.slice(..);
    let (sx, res) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sx.send(r); });
    device.poll(wgpu::Maintain::Wait);
    res.recv().unwrap()?;
    let data = slice.get_mapped_range().to_vec();

    // 写 24 位 BMP（无依赖）
    let row = (w * 3) as usize;
    let pad = (4 - row % 4) % 4;
    let img_size = (row + pad) * h as usize;
    let mut bmp = Vec::with_capacity(54 + img_size);
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&((54 + img_size) as u32).to_le_bytes());
    bmp.extend_from_slice(&[0u8; 4]);
    bmp.extend_from_slice(&54u32.to_le_bytes());
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(w as i32).to_le_bytes());
    bmp.extend_from_slice(&(h as i32).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&[0u8; 4]);
    bmp.extend_from_slice(&(img_size as u32).to_le_bytes());
    bmp.extend_from_slice(&[0u8; 16]);
    for y in (0..h).rev() {
        for x in 0..w {
            let o = (y * bytes_per_row + x * 4) as usize;
            let b = data[o]; let g = data[o + 1]; let r = data[o + 2];
            bmp.extend_from_slice(&[b, g, r]);
        }
        bmp.extend(std::iter::repeat(0u8).take(pad));
    }
    std::fs::write(out, &bmp)?;
    println!("shot -> {}", out.display());

    // 脚部坐标系交叉验证：渲染矩阵变换 FOOT_SITES ↔ MuJoCo 自己的 site 世界坐标
    for (name, slot, site) in [
        ("left_foot", 5usize, [0.0f32, -0.0237879, -0.0140852]),
        ("right_foot", 14, [0.0, 0.0237879, -0.0140852]),
    ] {
        let p = models[slot].transform_point3(Vec3::from_array(site));
        // 脚朝向：局部 x / y 轴的世界方向（左右脚应当同向：局部轴指向一致）
        let ax = (models[slot] * glam::Vec4::new(1.0, 0.0, 0.0, 0.0)).truncate();
        let ay = (models[slot] * glam::Vec4::new(0.0, 1.0, 0.0, 0.0)).truncate();
        println!(
            "  render {name}: world=({:+.4},{:+.4},{:+.4})  localX=({:+.2},{:+.2},{:+.2})  localY=({:+.2},{:+.2},{:+.2})",
            p.x, p.y, p.z, ax.x, ax.y, ax.z, ay.x, ay.y, ay.z
        );
    }
    if let Some(ph) = world.phys.as_ref() {
        ph.print_sites();
    }
    Ok(())
}

/// Windows 上动态链接的 mujoco.dll 必须与 exe 同目录：首次运行时从
/// mujoco-dist（构建期自动下载的解压目录）复制过去，用户零配置。
fn ensure_mujoco_dll(root: &Path) {
    let exe_dir = match std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        Some(d) => d,
        None => return,
    };
    let target = exe_dir.join("mujoco.dll");
    if target.exists() {
        return;
    }
    let dist = root.join("app/mujoco-dist");
    if let Ok(entries) = std::fs::read_dir(&dist) {
        for e in entries.flatten() {
            let candidate = e.path().join("bin").join("mujoco.dll");
            if candidate.exists() {
                match std::fs::copy(&candidate, &target) {
                    Ok(_) => println!("mujoco.dll → {}", target.display()),
                    Err(err) => println!("拷贝 mujoco.dll 失败: {err}"),
                }
                return;
            }
        }
    }
}

fn main() -> Result<()> {
    let root = find_repo_root()?;
    ensure_mujoco_dll(&root);
    println!("repo root: {}", root.display());
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--selftest") {
        return selftest(&root);
    }
    if let Some(i) = args.iter().position(|a| a == "--shot") {
        let out = args.get(i + 1).cloned().unwrap_or_else(|| "duck3d_shot.bmp".into());
        return offscreen_shot(&root, Path::new(&out));
    }
    println!("{HELP}");
    let pol = Policies::load(&root.join("policies"));

    let event_loop = EventLoop::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    // 50Hz 节拍线程：绝对时钟调度（每拍对齐 t0+n·20ms，无漂移）；
    // 落后超过 200ms（系统休眠/调试断点）就重对表，绝不追帧快进。
    std::thread::spawn(move || {
        let mut t0 = Instant::now();
        let mut n: u64 = 0;
        loop {
            if proxy.send_event(TickEvent).is_err() { return; }
            n += 1;
            let next = t0 + Duration::from_millis(n * 20);
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            } else if now.duration_since(next).as_millis() > 200 {
                t0 = now;
                n = 0;
            }
        }
    });

    let (dlg_tx, dlg_rx) = std::sync::mpsc::channel::<DlgPick>();
    let mut app = App {
        root: root.clone(),
        egui: None,
        frame_dt: 0.0,
        last_tick_wall: Instant::now(),
        next_tick: Instant::now(),
        auto_start: Instant::now(),
        auto_stage: 0,
        auto_xpress: None,
        dlg_tx: dlg_tx.clone(),
        dlg_rx,
        last_render: Instant::now(),
        tick_meter: (0, Instant::now()),
        tick_hz: 0.0,
        world: World {
            root: root.clone(),
            load_target: 0,
            body_cmd: [0.0; 3],
            body_obs: [0.0; 3],
        scene: Scene::Flat,
        scenes: discover_scenes(&root),
        scene_sel: 0,
            scale_mult: 1.0,
            drop_hint: None,
            load_msg: None,
        console_open: false,   // 默认收起：开局画面干净，TAB / 控制台按钮展开
        boot: Instant::now(),
        boot_switched: false,
        announce: None,
            sim: Sim::new(),
            ctl: Controller::new(),
            pol,
            keys: HashSet::new(),
            body_active: false,
            infer_us: 0.0,
            tick_count: 0,
            mode: Mode::Rig,
            phys: None,
            imu: ([0.0, 0.0, -1.0], [0.0; 3]),
            gpu_name: String::new(),
        },
        gpu: None,
        window: None,
        cam: Camera::new(-0.9, 0.3, 0.5, Vec3::new(0.0, 0.1, 0.0)),
        drag: None,
        cursor: None,
        last_title: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
