//! 渲染：wgpu 管线 + 场景网格 + 相机。
//!
//! 实时路径（surface）与离屏截图（texture）共用这里的全部绘制素材 ——
//! 地面吸附、坡体网格、管线布局都只此一份：**历史上两处各写一份，
//! 改了一边漏一边**（棋盘的图案周期修复就漏过离屏路径）。

use std::sync::Arc;

use anyhow::Result;
use glam::Vec3;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::{Cad, CadBody, Scene};

pub(crate) const GRID_CELL: f32 = 0.25;          // 地面网格间距
pub(crate) const GROUND_SPAN: f32 = 24.0;        // 地面总跨度（跟随鸭子 → 视觉上无限）
pub(crate) const SAMPLES: u32 = 4;   // MSAA 抗锯齿：精细 CAD 的硬边不再锯齿

pub(crate) const DRAG_YAW: f32 = 0.002;
pub(crate) const DRAG_PITCH: f32 = 0.0016;
pub(crate) const DRAG_PAN: f32 = 0.0004;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub(crate) pos: [f32; 3],
    pub(crate) nrm: [f32; 3],
    pub(crate) col: [f32; 3],
}

/// 地面重建的判据：视觉样式 + 当前场景是否带坡（二者任一变化都要重画）
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct GroundKind {
    pub(crate) vis: Scene,
    pub(crate) ramp: bool,
}

/* ════════════════════════════════════════════════════════════════════════
   7 · 渲染（wgpu）
   ════════════════════════════════════════════════════════════════════════ */

pub(crate) const SHADER: &str = r#"
struct Global { view_proj: mat4x4<f32>, light1: vec4<f32>, light2: vec4<f32> };
struct Model  { model: mat4x4<f32> };
@group(0) @binding(0) var<uniform> g: Global;
@group(1) @binding(0) var<uniform> m: Model;

struct VSIn { @location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>, @location(2) col: vec3<f32> };
struct VSOut { @builtin(position) clip: vec4<f32>, @location(0) col: vec3<f32> };

@vertex fn vs(i: VSIn) -> VSOut {
    let world = m.model * vec4<f32>(i.pos, 1.0);
    let n = normalize((m.model * vec4<f32>(i.nrm, 0.0)).xyz);
    // 主光（暖）+ 补光（冷，弱）+ 环境 —— CAD 精细件靠双光出立体感
    let l1 = max(dot(n, -normalize(g.light1.xyz)), 0.0);
    let l2 = max(dot(n, -normalize(g.light2.xyz)), 0.0);
    let lit = vec3<f32>(0.34, 0.36, 0.40)           // 环境（微冷）
        + vec3<f32>(1.02, 0.98, 0.90) * l1          // 主光
        + vec3<f32>(0.28, 0.33, 0.42) * l2;         // 补光
    var o: VSOut;
    o.clip = g.view_proj * world;
    o.col = i.col * lit;
    return o;
}
@fragment fn fs(i: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.col, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GlobalU {
    pub(crate) view_proj: [f32; 16],
    pub(crate) light1: [f32; 4],
    pub(crate) light2: [f32; 4],
}

pub(crate) const UNIFORM_ALIGN: u64 = 256;
pub(crate) const MAX_MODEL_SLOTS: usize = 24;  // 16 body + 下颚 + 网格 + 余量（17 号空位）
pub(crate) const SLOT_GRID: usize = 16;
pub(crate) const SLOT_GROUNDBASE: usize = 18;
pub(crate) const SLOT_IDENTITY: usize = 19;

pub(crate) struct GroundMesh {
    pub(crate) base_buf: wgpu::Buffer,   // 底板（平面 6 顶点 / 棋盘 48×48 格）
    pub(crate) base_count: u32,
    pub(crate) line_buf: wgpu::Buffer,   // 网格线
    pub(crate) line_count: u32,
}

/// 地面：实体底板 + 0.25m 网格线，跟随鸭子（吸附到格点）→ 视觉上无限延伸，
/// 与 MuJoCo 场景里的无限平面物理一致。
/// 地面吸附步长：必须整除视觉图案的「相同外观周期」，否则走路时图案跳相 ——
/// 棋盘是 0.5m 格 + 双色交替（周期 1.0m），网格线距 0.25m。
/// 实时渲染与离屏截图两条路径共用这一个函数：历史上两处各写一份，修了一边漏一边。
pub(crate) fn ground_snap_step(vis: Scene) -> f32 {
    match vis {
        Scene::Checker => 1.0,
        Scene::Flat => GRID_CELL,
    }
}

/// 地面跟随位置：按图案周期吸附到格点（视觉上无限延伸）
pub(crate) fn ground_snap_pos(x: f32, z: f32, vis: Scene) -> (f32, f32) {
    let step = ground_snap_step(vis);
    ((x / step).round() * step, (z / step).round() * step)
}

pub(crate) fn make_ground(device: &wgpu::Device, scene: Scene) -> GroundMesh {
    use wgpu::util::DeviceExt;
    let half = GROUND_SPAN / 2.0;
    let n = [0.0f32, 1.0, 0.0];
    let v = |x: f32, z: f32, col: [f32; 3]| Vertex { pos: [x, 0.0, z], nrm: n, col };
    let mut base: Vec<Vertex> = Vec::new();

    match scene {
        // 棋盘：训练场景 groundplane 同款双色格（cell 0.5m）
        Scene::Checker => {
            let cell = 0.5f32;
            let ni = (half / cell) as i32;
            let c1 = [0.10f32, 0.15, 0.20];
            let c2 = [0.05f32, 0.10, 0.15];
            for i in -ni..ni {
                for j in -ni..ni {
                    let col = if (i + j) % 2 == 0 { c1 } else { c2 };
                    let (x0, z0) = (i as f32 * cell, j as f32 * cell);
                    let (x1, z1) = (x0 + cell, z0 + cell);
                    base.push(v(x0, z0, col)); base.push(v(x0, z1, col)); base.push(v(x1, z1, col));
                    base.push(v(x0, z0, col)); base.push(v(x1, z1, col)); base.push(v(x1, z0, col));
                }
            }
        }
        // 平面 / 斜坡：实体深色底板
        _ => {
            let base_col = [0.035f32, 0.048, 0.066];
            base.push(v(-half, -half, base_col)); base.push(v(-half, half, base_col));
            base.push(v(half, half, base_col)); base.push(v(-half, -half, base_col));
            base.push(v(half, half, base_col)); base.push(v(half, -half, base_col));
        }
    }
    let base_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ground-base"),
        contents: bytemuck::cast_slice(&base),
        usage: wgpu::BufferUsages::VERTEX,
    });
    // 网格线（棋盘格本身就是格，不再叠线）
    let mut line_count = 0u32;
    let mut line_buf = base_buf.clone();
    if scene != Scene::Checker {
        let ni = (half / GRID_CELL) as i32;
        let mut lines: Vec<Vertex> = Vec::new();
        let c = [0.16f32, 0.23, 0.32];
        for i in -ni..=ni {
            let f = i as f32 * GRID_CELL;
            lines.push(Vertex { pos: [f, 0.001, -half], nrm: n, col: c });
            lines.push(Vertex { pos: [f, 0.001, half], nrm: n, col: c });
            lines.push(Vertex { pos: [-half, 0.001, f], nrm: n, col: c });
            lines.push(Vertex { pos: [half, 0.001, f], nrm: n, col: c });
        }
        line_count = lines.len() as u32;
        line_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX,
        });
    }
    GroundMesh { base_buf, base_count: base.len() as u32, line_buf, line_count }
}

/// 斜坡视觉：实体楔形（三角棱柱），坐在地面上 —— 坡面与 XML 薄盒顶面严格重合。
/// 碰撞体是绕中心旋转 −12° 的薄盒（近端埋地、远端悬空 16cm，下无支撑），
/// 照它画出来就是"浮空斜板"：板下漏空的缝与板面会在低视角把鸭子拦腰切断。
/// 视觉补齐侧面/背面到地面，读起来才是"一个能爬的坡"；物理不动。
pub(crate) fn make_wedge_mesh(center: [f32; 3], euler_y_deg: f32, half: [f32; 3], top_col: [f32; 3], side_col: [f32; 3]) -> Vec<Vertex> {
    let th = euler_y_deg.to_radians();
    let (cy, sy) = (th.cos(), th.sin());
    let ax = [cy, 0.0, -sy];   // 沿坡向上（局部 x）
    let az = [sy, 0.0, cy];    // 坡面法线（局部 z）
    // MuJoCo 是 z-up：算出的角点转到渲染 y-up 世界（同鸭子根部 Rx(-90°)）
    let to_render = |v: [f32; 3]| [v[0], v[2], -v[1]];
    // 坡面中心 = 薄盒顶面中心；近/远坡缘（近端 z≈0，即坡从地面起步）
    let tc = [center[0] + az[0] * half[2], center[1], center[2] + az[2] * half[2]];
    let near = [tc[0] - ax[0] * half[0], tc[2] - ax[2] * half[0]];
    let far = [tc[0] + ax[0] * half[0], tc[2] + ax[2] * half[0]];
    let wy = half[1];
    let mut out: Vec<Vertex> = Vec::new();
    let quad = |a: [f32; 3], b: [f32; 3], d: [f32; 3], e: [f32; 3], n: [f32; 3], col: [f32; 3], out: &mut Vec<Vertex>| {
        let n = to_render(n);
        for p in [a, b, e, a, e, d] {
            out.push(Vertex { pos: to_render(p), nrm: n, col });
        }
    };
    let t = |pm: [f32; 3]| pm;   // 语义别名：参数已按 MuJoCo 坐标给出
    let n0 = [near[0], -wy, near[1]];
    let n1 = [near[0], wy, near[1]];
    let f0 = [far[0], -wy, far[1]];
    let f1 = [far[0], wy, far[1]];
    let g0 = [far[0], -wy, 0.0];   // 远端底缘（落在地面）
    let g1 = [far[0], wy, 0.0];
    // 坡面
    quad(t(n0), t(n1), t(f0), t(f1), az, top_col, &mut out);
    // 两侧三角（水平边沿地面 → 竖直边在远端）
    for (a, bb, c, ny) in [(n0, g0, f0, [0.0, -1.0, 0.0]), (n1, g1, f1, [0.0, 1.0, 0.0])] {
        let n = to_render(ny);
        for p in [a, c, bb] {
            out.push(Vertex { pos: to_render(p), nrm: n, col: side_col });
        }
    }
    // 背面（远端竖直面）
    quad(t(f0), t(f1), t(g0), t(g1), [1.0, 0.0, 0.0], side_col, &mut out);
    out
}

/// 渲染素材（与渲染目标无关）：管线、缓冲、地面/坡体网格、CAD 顶点。
///
/// **实时窗口（surface）与离屏截图（texture）共用它** —— 两条路径只差 render target、
/// 深度/MSAA 纹理与 HUD 叠加；绘制序列只有一份（[`Renderer::draw`]）。
/// 历史上这两条路径各写一份，改了一边漏一边（棋盘图案周期、团影绑定都踩过）。
pub(crate) struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_tri: wgpu::RenderPipeline,
    pipeline_line: wgpu::RenderPipeline,
    bg_global: wgpu::BindGroup,
    bg_model: wgpu::BindGroup,
    buf_global: wgpu::Buffer,
    buf_model: wgpu::Buffer,
    buf_vertex: wgpu::Buffer,
    base_buf: wgpu::Buffer,
    base_count: u32,
    line_buf: wgpu::Buffer,
    line_count: u32,
    scene_buf: wgpu::Buffer,
    scene_count: u32,
    kind: GroundKind,
    cad: Vec<CadBody>,
}

impl Renderer {
    /// 建全部渲染素材。`format` 决定管线颜色目标（窗口与离屏必须用同一格式）。
    /// `samples`：窗口路径用 MSAA（[`SAMPLES`]），离屏渲染目标是单采样纹理时传 1 ——
    /// 管线的多重采样数必须与 render pass 的附件一致，否则 wgpu 直接报错。
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        samples: u32,
        cad: Cad,
    ) -> Result<Self> {
        use wgpu::util::DeviceExt;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("duck"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl_global = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(96),
                },
                count: None,
            }],
        });
        let bgl_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl_global, &bgl_model],
            push_constant_ranges: &[],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
            ],
        };
        let make_pipeline = |topology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                // 不背面剔除：STL 三角汤经抽稀后绕向不可靠，65k 面不在乎双面光栅
                primitive: wgpu::PrimitiveState { topology, cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    bias: Default::default(),
                    stencil: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
                multiview: None,
                cache: None,
            })
        };
        let pipeline_tri = make_pipeline(wgpu::PrimitiveTopology::TriangleList);
        let pipeline_line = make_pipeline(wgpu::PrimitiveTopology::LineList);

        let buf_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cad"),
            contents: bytemuck::cast_slice(&cad.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ground = make_ground(device, Scene::Flat);
        let buf_global = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global"),
            size: 112,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_model = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("models"),
            size: UNIFORM_ALIGN * MAX_MODEL_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_global = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bgl_global,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf_global.as_entire_binding() }],
            label: None,
        });
        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bgl_model,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf_model,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
            label: None,
        });
        queue.write_buffer(&buf_model, SLOT_IDENTITY as u64 * UNIFORM_ALIGN,
            bytemuck::cast_slice(&glam::Mat4::IDENTITY.to_cols_array()));
        Ok(Renderer {
            device: device.clone(),
            queue: queue.clone(),
            pipeline_tri,
            pipeline_line,
            bg_global,
            bg_model,
            buf_global,
            buf_model,
            buf_vertex,
            base_buf: ground.base_buf.clone(),
            base_count: ground.base_count,
            line_buf: ground.line_buf,
            line_count: ground.line_count,
            scene_buf: ground.base_buf.clone(),
            scene_count: 0,
            kind: GroundKind { vis: Scene::Flat, ramp: false },
            cad: cad.bodies,
        })
    }

    /// 每帧相机（view-proj + 双光）
    pub(crate) fn set_camera(&self, view_proj: glam::Mat4) {
        let gu = GlobalU {
            view_proj: view_proj.to_cols_array(),
            light1: [-0.5, -1.0, -0.35, 0.0],
            light2: [0.6, -0.4, 0.5, 0.0],
        };
        self.queue.write_buffer(&self.buf_global, 0, bytemuck::bytes_of(&gu));
    }

    /// 每帧模型矩阵。注意步长是 [`UNIFORM_ALIGN`]（256B）而不是矩阵自身的 64B ——
    /// 着色器按动态偏移 `slot × UNIFORM_ALIGN` 取矩阵，紧凑写会让模型散架。
    pub(crate) fn set_models(&self, models: &[glam::Mat4]) {
        for (i, m) in models.iter().take(MAX_MODEL_SLOTS).enumerate() {
            self.queue.write_buffer(&self.buf_model, i as u64 * UNIFORM_ALIGN,
                bytemuck::cast_slice(&m.to_cols_array()));
        }
    }

    /// 地面跟随矩阵：底板与网格线同用一个变换（吸附结果）
    pub(crate) fn set_ground_mat(&self, m: glam::Mat4) {
        let arr = m.to_cols_array();
        let bytes = bytemuck::cast_slice(&arr);
        self.queue.write_buffer(&self.buf_model, SLOT_GRID as u64 * UNIFORM_ALIGN, bytes);
        self.queue.write_buffer(&self.buf_model, SLOT_GROUNDBASE as u64 * UNIFORM_ALIGN, bytes);
    }

    /// 地面素材按 GroundKind 重建（视觉样式或坡体任一变化时）
    pub(crate) fn ensure_ground(&mut self, kind: GroundKind) {
        use wgpu::util::DeviceExt;
        if self.kind == kind {
            return;
        }
        let g = make_ground(&self.device, kind.vis);
        self.base_buf = g.base_buf;
        self.base_count = g.base_count;
        self.line_buf = g.line_buf;
        self.line_count = g.line_count;
        if kind.ramp {
            // 坡体渲染：实体楔形（坡面与 scene_ramp.xml 的 geom 顶面重合）
            let wedge = make_wedge_mesh(
                [0.855, 0.0, 0.0816], -12.0, [0.5113, 0.5, 0.025],
                [0.20, 0.25, 0.33], [0.10, 0.13, 0.18],
            );
            self.scene_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wedge"),
                contents: bytemuck::cast_slice(&wedge),
                usage: wgpu::BufferUsages::VERTEX,
            });
            self.scene_count = wedge.len() as u32;
        } else {
            self.scene_count = 0;
        }
        self.kind = kind;
    }

    /// 画整个场景：底板 → 坡体楔形 → CAD body → 网格线（顺序即层次）
    pub(crate) fn draw(&self, rpass: &mut wgpu::RenderPass<'_>) {
        rpass.set_pipeline(&self.pipeline_tri);
        rpass.set_bind_group(0, &self.bg_global, &[]);
        rpass.set_bind_group(1, &self.bg_model, &[SLOT_GROUNDBASE as u32 * UNIFORM_ALIGN as u32]);
        rpass.set_vertex_buffer(0, self.base_buf.slice(..));
        rpass.draw(0..self.base_count, 0..1);
        if self.scene_count > 0 {
            rpass.set_vertex_buffer(0, self.scene_buf.slice(..));
            rpass.set_bind_group(1, &self.bg_model, &[SLOT_IDENTITY as u32 * UNIFORM_ALIGN as u32]);
            rpass.draw(0..self.scene_count, 0..1);
        }
        rpass.set_vertex_buffer(0, self.buf_vertex.slice(..));
        for b in &self.cad {
            rpass.set_bind_group(1, &self.bg_model, &[b.slot as u32 * UNIFORM_ALIGN as u32]);
            rpass.draw(b.start..b.start + b.count, 0..1);
        }
        rpass.set_pipeline(&self.pipeline_line);
        rpass.set_vertex_buffer(0, self.line_buf.slice(..));
        rpass.set_bind_group(1, &self.bg_model, &[SLOT_GRID as u32 * UNIFORM_ALIGN as u32]);
        rpass.draw(0..self.line_count, 0..1);
    }
}

struct WindowArc(Arc<Window>);
impl HasWindowHandle for WindowArc {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        self.0.window_handle()
    }
}
impl HasDisplayHandle for WindowArc {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        self.0.display_handle()
    }
}

/// 窗口端：surface + 深度/MSAA 纹理 + 渲染素材（[`Renderer`]）。
pub(crate) struct Gpu {
    pub(crate) name: String,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) depth_view: Option<wgpu::TextureView>,
    pub(crate) msaa_view: Option<wgpu::TextureView>,
    pub(crate) depth_size: (u32, u32),
    pub(crate) renderer: Renderer,
}

/// 建窗口渲染栈：surface/device/queue/config + 渲染素材。
pub(crate) fn init_gpu(window: &Arc<Window>, cad: Cad) -> Result<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(WindowArc(Arc::clone(window)))?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow::anyhow!("没有可用的 GPU 适配器"))?;
    println!("GPU: {:?}", adapter.get_info().name);
    let name = adapter.get_info().name.clone();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))?;
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);
    let renderer = Renderer::new(&device, &queue, format, SAMPLES, cad)?;
    Ok(Gpu {
        name,
        device,
        queue,
        surface,
        config,
        depth_view: None,
        msaa_view: None,
        depth_size: (0, 0),
        renderer,
    })
}

/* ════════════════════════════════════════════════════════════════════════
   8 · 相机 + App
   ════════════════════════════════════════════════════════════════════════ */

pub(crate) struct Camera {
    /// 拖拽 1:1 直接驱动（不平滑 —— 平滑会"飘"）；灵敏度已按逻辑像素归一
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) dist: f32,
    pub(crate) target: Vec3,
    /// 滚轮的目标距离：实际 dist 每帧指数逼近（只有缩放需要平滑）
    pub(crate) g_dist: f32,
}

impl Camera {
    pub(crate) fn new(yaw: f32, pitch: f32, dist: f32, target: Vec3) -> Self {
        Camera { yaw, pitch, dist, target, g_dist: dist }
    }

    /// 滚轮缩放平滑：实际距离向目标距离指数逼近（速率 18/s）
    pub(crate) fn glide(&mut self, dt: f32) {
        let k = 1.0 - (-dt * 18.0).exp();
        self.dist *= (self.g_dist / self.dist).powf(k);
    }

    pub(crate) fn view_proj(&self, aspect: f32) -> glam::Mat4 {
        let eye = self.target + Vec3::new(
            self.dist * self.pitch.cos() * self.yaw.cos(),
            self.dist * self.pitch.sin(),
            self.dist * self.pitch.cos() * self.yaw.sin(),
        );
        let view = glam::Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = glam::Mat4::perspective_rh(42f32.to_radians(), aspect, 0.01, 20.0);
        proj * view
    }
}
