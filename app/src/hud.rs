//! HUD（egui）：面板绘制 + 界面命令队列。
//!
//! 与仿真的边界：**只读** `World` 的状态，改状态一律通过 [`UiCmd`] 队列回写
//! （[`apply_cmd`]）—— 这样 HUD 不持有仿真状态，也不会有面板改一半的半成品。
//!
//! 面板按块拆函数（SCENE / POLICIES / JOINTS / IMU / PD / PERF / KEYS），
//! 每个函数只画自己那一块，改动互不干扰。

use std::path::PathBuf;
use std::time::Instant;

use crate::{
    Mode, Scene, World, DEFAULT_POSITION, JOINT_RANGE, KICK_DURATION, MOUTH_INDEX,
    MAX_ANGULAR, MAX_HEAD, MAX_LINEAR, RISE_SECS, ROULADE_DURATION,
    BODY_KEY_TILT, BODY_TILT_RANGE, BODY_Z_RANGE,
};

/// 站姿倾斜的显示单位（度）：滑条行程与键盘满偏都由常量派生，避免两处各写一份
const TILT_DEG: f32 = (BODY_TILT_RANGE * 180.0 / std::f64::consts::PI) as f32;
const KEY_TILT_DEG: f32 = (BODY_KEY_TILT * 180.0 / std::f64::consts::PI) as f32;
use winit::keyboard::KeyCode;

const CYAN: egui::Color32 = egui::Color32::from_rgb(0, 229, 255);
const ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 138, 42);
const DIM: egui::Color32 = egui::Color32::from_rgb(120, 140, 160);
const VIOLET: egui::Color32 = egui::Color32::from_rgb(187, 134, 252);
const GOOD: egui::Color32 = egui::Color32::from_rgb(80, 220, 120);
const BAD: egui::Color32 = egui::Color32::from_rgb(255, 93, 93);

#[derive(Clone)]
/// 文件对话框选择结果：策略 or 场景（同一条通道，按类型分流）
pub(crate) enum DlgPick {
    /// 策略：点按钮时的目标槽位 + 选中文件（随路径一起带走，避免中途改选送错槽）
    Policy(usize, PathBuf),
    Scene(PathBuf),
}

pub(crate) enum UiCmd {
    ToggleEnable,
    Reset,
    SetMode(Mode),
    Pick,
    Roulade,
    KickL,
    KickR,
    Sit,
    BodyPose,
    Kp(f32),
    Kd(f32),
    /// PD 复位到训练基线（kp 0.55 / kd* 0）
    PdReset,
    SetLoadTarget(usize),
    PickFile(usize),
    /// 浏览选择场景 XML（对话框在 App 侧弹，见 render）
    PickScene,
    ScaleMult(f64),
    ToggleConsole,
    SetScene(Scene),
    /// 选中自动发现的场景文件（scenes 下标）—— 物理 XML 即刻重载
    SelectScene(usize),
    /// body-pose 槽位赋值：(0=z 米, 1=roll rad, 2=pitch rad)
    BodySlot(usize, f32),
    /// 行进基值赋值（拖拽巡航）：(0=vx, 1=vy m/s, 2=ω rad/s)
    TwistBase(usize, f32),
    /// 头颈拖拽基值赋值（持久）：索引同 head_cmd —— 0颈 1头p 2头y 3头r（原始 −1..1）
    HeadBase(usize, f32),
}

pub(crate) fn apply_cmd(w: &mut World, c: UiCmd) {
    match c {
        UiCmd::ToggleEnable => w.sim.enabled = !w.sim.enabled,
        UiCmd::Reset => w.reset_all(),
        UiCmd::SetMode(m) => {
            if w.mode != m { w.toggle_mode(); }
        }
        UiCmd::Pick => w.on_key(KeyCode::KeyG),
        UiCmd::Roulade => w.on_key(KeyCode::KeyX),
        UiCmd::KickL => w.on_key(KeyCode::KeyZ),
        UiCmd::KickR => w.on_key(KeyCode::KeyC),
        UiCmd::Sit => w.on_key(KeyCode::KeyV),
        UiCmd::BodyPose => w.on_key(KeyCode::KeyB),
        UiCmd::Kp(v) => { if let Some(p) = w.phys.as_mut() { p.kp = v as f64; } }
        UiCmd::Kd(v) => { if let Some(p) = w.phys.as_mut() { p.kd = v as f64; } }
        UiCmd::PdReset => {
            // 训练基线值：与 microduck_rl 训练时一致（改大 kp 会偏离训练分布）
            if let Some(p) = w.phys.as_mut() { p.kp = 0.55; p.kd = 0.0; }
        }
        UiCmd::SetLoadTarget(i) => w.load_target = i,
        UiCmd::PickFile(_) => { /* 由 App 处理：需要通道，见 render() */ }
        UiCmd::PickScene => { /* 同上：对话框需要通道 */ }
        UiCmd::ScaleMult(v) => w.scale_mult = v,
        UiCmd::ToggleConsole => w.console_open = !w.console_open,
        UiCmd::SetScene(sc) => w.set_scene(sc),
        UiCmd::SelectScene(i) => w.select_scene(i),
        UiCmd::BodySlot(i, v) => {
            if let Some(slot) = w.body_cmd.get_mut(i) { *slot = v as f64; }
        }
        UiCmd::TwistBase(i, v) => {
            if let Some(slot) = w.sim.twist_base.get_mut(i) { *slot = v as f64; }
        }
        UiCmd::HeadBase(i, v) => {
            if let Some(slot) = w.sim.head_base.get_mut(i) { *slot = v as f64; }
        }
    }
}

fn sep(ui: &mut egui::Ui) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
    ui.painter_at(r).line_segment(
        [r.left_center(), r.right_center()],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(18, 42, 54)),
    );
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(5.0);
    ui.label(egui::RichText::new(format!("▸ {title}")).color(CYAN).size(11.0));
    ui.add_space(2.0);
}

const JOINT_SHORT: [&str; 15] = [
    "L髋yaw", "L髋roll", "L髋pitch", "L膝", "L踝", "颈", "头pitch", "头yaw", "头roll", "嘴",
    "R髋yaw", "R髋roll", "R髋pitch", "R膝", "R踝",
];

fn show_range(i: usize) -> [f32; 2] {
    if i == MOUTH_INDEX { [-0.087, 0.524] } else if i < MOUTH_INDEX { JOINT_RANGE[i] } else { JOINT_RANGE[i - 1] }
}

/// 组标题：彩色小竖标 + 组名 + 延伸细线，给三个分组块一个视觉锚点
fn group_tag(ui: &mut egui::Ui, name: &str, col: egui::Color32) {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(3.0, 10.0), egui::Sense::hover());
        ui.painter().rect_filled(r, 1.0, col);
        ui.label(egui::RichText::new(name).color(col).size(9.0));
        let (rr, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
        ui.painter().line_segment(
            [egui::pos2(rr.left(), rr.center().y), egui::pos2(rr.right(), rr.center().y)],
            egui::Stroke::new(1.0, col.gamma_multiply(0.35)),
        );
    });
}

/// 指令面板一行：名字(右对齐) | 中轴条(细量程线 + 脊线 + 粗色条，右正左负) | 数值紧随其后。
/// 指令面板一行的公共部分：名字(右对齐) + 中轴条(细量程线 + 脊线 + 粗色条，右正左负)，
/// 返回数值单元格矩形 —— 数值由调用方决定怎么画，坐标全部确定，不依赖流式布局。
const ROW_H: f32 = 13.0;

/// 指令面板一行：整行一次分配（行高恒定 = ROW_H），内部全按 rect 坐标绘制 ——
/// 名字(右对齐) | 中轴条(细量程线 + 脊线 + 粗色条，右正左负) | 数值格矩形返回。
/// 不用流式子布局，行高申报与渲染严格一致，多行累积零漂移。
fn bar_track(ui: &mut egui::Ui, label: &str, v: f32, lo: f32, hi: f32, col: egui::Color32, marks: &[f32]) -> egui::Rect {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), ROW_H),
        egui::Sense::hover(),
    );
    let p = ui.painter();
    let cy = rect.center().y;
    // 名字：右对齐到 24px 列
    p.text(
        egui::pos2(rect.left() + 24.0, cy),
        egui::Align2::RIGHT_CENTER,
        label,
        egui::FontId::proportional(9.0),
        egui::Color32::from_rgb(165, 180, 200),
    );
    // 条区：84px。行程可非对称（z 就是 −4/+3cm），零点的屏幕位置按比例算
    let tx0 = rect.left() + 28.0;
    let width = 84.0;
    let x_of = |val: f32| tx0 + ((val - lo) / (hi - lo)).clamp(0.0, 1.0) * (width - 2.0) + 1.0;
    let axis_x = x_of(0.0);
    for (x0, x1) in [(tx0 + 1.0, axis_x - 2.0), (axis_x + 2.0, tx0 + 83.0)] {
        p.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, cy - 1.0), egui::pos2(x1, cy + 1.0)),
            1.0,
            col.gamma_multiply(0.30),
        );
    }
    p.line_segment(
        [egui::pos2(axis_x, rect.top() + 1.0), egui::pos2(axis_x, rect.bottom() - 1.0)],
        egui::Stroke::new(1.0, col.gamma_multiply(0.6)),
    );
    // 训练上限刻度：键盘满偏 = 训练行程，条的量程更宽（滑条可越界实验）——
    // 标出来才不会把"到训练上限"看成"没到头"。
    for m in marks {
        let x = x_of(*m);
        p.line_segment(
            [egui::pos2(x, rect.top() + 2.0), egui::pos2(x, rect.bottom() - 2.0)],
            egui::Stroke::new(1.0, GOOD.gamma_multiply(0.7)),
        );
    }
    // 值条：从零点往当前值画，加粗圆角 —— 每行唯一的重元素
    let vx2 = x_of(v);
    if (vx2 - axis_x).abs() > 0.5 {
        p.rect_filled(
            egui::Rect::from_min_max(egui::pos2(axis_x.min(vx2), cy - 3.0), egui::pos2(axis_x.max(vx2), cy + 3.0)),
            3.0,
            col,
        );
    }
    // 数值格：条区右端 + 6px 起，到行右缘（归零钮在其中右对齐，见 bar_row）
    egui::Rect::from_min_max(
        egui::pos2(tx0 + 88.0, rect.top()),
        egui::pos2(rect.right(), rect.bottom()),
    )
}

/// 可编辑行：数值是 DragValue，锁进固定单元格（左对齐、限宽、字号 9pt）；
/// 返回行末"0"按钮是否被点（调用方据此把该值归零）
fn bar_row(
    ui: &mut egui::Ui,
    label: &str,
    v: f32,
    lo: f32,
    hi: f32,
    col: egui::Color32,
    marks: &[f32],
    value: impl FnOnce(&mut egui::Ui),
) -> bool {
    let vr = bar_track(ui, label, v, lo, hi, col, marks);
    let mut zero = false;
    ui.allocate_new_ui(
        egui::UiBuilder::new()
            .max_rect(egui::Rect::from_min_max(
                egui::pos2(vr.left(), vr.top()),
                egui::pos2(vr.right(), vr.bottom()),
            ))
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
        |ui| {
            ui.visuals_mut().override_text_color = Some(col);
            // DV 的最小尺寸 = interact_size；归零让宽度贴内容，文本才能顶到左缘与只读值对齐
            ui.spacing_mut().interact_size = egui::vec2(0.0, 14.0);
            ui.spacing_mut().button_padding = egui::Vec2::ZERO;
            let st = ui.style_mut();
            st.text_styles.insert(egui::TextStyle::Body, egui::FontId::proportional(9.0));
            st.text_styles.insert(egui::TextStyle::Button, egui::FontId::proportional(9.0));
            value(ui);
            // 归零钮：真 egui Button 顶到格子右缘（与 DragValue 同一套命中机制，不错位）
            let wr = ui.available_rect_before_wrap();
            let zresp = ui.allocate_new_ui(
                egui::UiBuilder::new()
                    .max_rect(egui::Rect::from_min_max(
                        egui::pos2(wr.right() - 14.0, wr.top()),
                        egui::pos2(wr.right(), wr.bottom()),
                    )),
                |ui| {
                    ui.visuals_mut().override_text_color = None;
                    ui.add_sized(
                        [14.0, 12.0],
                        egui::Button::new(egui::RichText::new("0").size(8.5).color(DIM)),
                    )
                },
            ).inner;
            let zresp = zresp.on_hover_text("归零");
            if zresp.clicked() { zero = true; }
        },
    );
    zero
}

/// 定宽 kv 单元：标签右对齐暗色 + 值左对齐等宽着色。
/// 分栏位置由宽度钉死 —— egui Grid 的列宽随内容自适应，做不到这种刀切分栏
fn kv_row(ui: &mut egui::Ui, k: &str, v: &str, vcol: egui::Color32, lw: f32, vw: f32) {
    let (lr, _) = ui.allocate_exact_size(egui::vec2(lw, 14.0), egui::Sense::hover());
    ui.painter().text(lr.right_center(), egui::Align2::RIGHT_CENTER, k,
        egui::FontId::proportional(9.0), DIM);
    let (vr, _) = ui.allocate_exact_size(egui::vec2(vw, 14.0), egui::Sense::hover());
    ui.painter().text(vr.left_center(), egui::Align2::LEFT_CENTER, v,
        egui::FontId::monospace(9.5), vcol);
}

/// 内联可拖数字（指令面板一行式用）：范围限幅 + 前后缀 + 固定小数位。
/// 显示统一带符号（+0.00 / +0.0cm / +0°），与只读值同一格式家族。
fn drag_val<'a>(
    v: &'a mut f32,
    lo: f32,
    hi: f32,
    speed: f32,
    suffix: &'a str,
    decimals: usize,
) -> egui::DragValue<'a> {
    egui::DragValue::new(v)
        .range(lo..=hi)
        .speed(speed)
        .suffix(suffix)
        .max_decimals(decimals)
        .custom_formatter(move |val, _| format!("{:+.*}", decimals, val))
        .custom_parser(|s| {
            let s = s.trim().trim_start_matches('+');
            let s = s.trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.');
            s.parse::<f64>().ok()
        })
}

/// HUD 的 egui 上下文：中文字体（等线→雅黑→黑体）+ 暗色科幻样式。
/// 窗口与离屏截图共用，保证截图里的 HUD 和窗口里一致。
pub(crate) fn build_egui_ctx() -> egui::Context {
    let ctx = egui::Context::default();
    {
        let mut fonts = egui::FontDefinitions::default();
        let cjk = ["C:/Windows/Fonts/Deng.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
            .iter()
            .find_map(|path| std::fs::read(path).ok());
        if let Some(bytes) = cjk {
            fonts.font_data.insert("cjk".into(), egui::FontData::from_owned(bytes).into());
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                if let Some(list) = fonts.families.get_mut(&fam) {
                    list.push("cjk".into());
                }
            }
            println!("HUD CJK font ✓");
        } else {
            println!("HUD CJK font ✗（没找到等线/雅黑/黑体）");
        }
        ctx.set_fonts(fonts);
    }
    let mut st = egui::Style::default();
    st.visuals = egui::Visuals::dark();
    st.visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(6, 10, 14, 216);
    st.visuals.window_fill = egui::Color32::from_rgba_unmultiplied(6, 10, 14, 216);
    st.visuals.extreme_bg_color = egui::Color32::from_rgb(10, 16, 22);
    st.visuals.selection.bg_fill = egui::Color32::from_rgb(0, 90, 110);
    st.visuals.hyperlink_color = CYAN;
    st.override_text_style = Some(egui::TextStyle::Monospace);
    ctx.set_style(st);
    ctx
}

/// HUD 总入口：各面板自行定位（Area / SidePanel），这里只按层调用。
/// 面板改动互不影响 —— 历史上这里是一个 640 行的巨函数。
pub(crate) fn build_hud(
    ctx: &egui::Context,
    w: &World,
    frame_dt: f32,
    tick_hz: f32,
    cmds: &mut Vec<UiCmd>,
) {
    if crate::shot_hud() {
        println!("[hud] build_hud called, console={}", w.console_open);
    }
    corners(ctx);
    boot_overlay(ctx, w);
    status_card(ctx, w, tick_hz, cmds);
    top_msg(ctx, w);
    cmd_panel(ctx, w, cmds);
    hotbar(ctx, w, cmds);
    console(ctx, w, frame_dt, tick_hz, cmds);
}

/// 暗色玻璃卡片（状态卡 / 指令面板通用）
fn card_frame() -> egui::Frame {
    egui::Frame::default()
        .fill(egui::Color32::from_rgba_unmultiplied(6, 10, 14, 185))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(24, 52, 66)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::symmetric(10, 7))
}

/// 屏幕四角取景框（游戏感）
fn corners(ctx: &egui::Context) {
    // ── 屏幕四角取景框（游戏感）──
    egui::Area::new(egui::Id::new("corners"))
        .order(egui::Order::Background)
        .show(ctx, |ui| {
            let sr = ui.ctx().screen_rect();
            let (sw, sh) = (sr.width(), sr.height());
            let p = ui.painter();
            let (len, off) = (26.0f32, 10.0f32);
            let st = egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(0, 229, 255, 110));
            let corner = |x: f32, y: f32, dx: f32, dy: f32| {
                p.line_segment([egui::pos2(x, y), egui::pos2(x + dx * len, y)], st);
                p.line_segment([egui::pos2(x, y), egui::pos2(x, y + dy * len)], st);
            };
            corner(off, off, 1.0, 1.0);
            corner(sw - off, off, -1.0, 1.0);
            corner(off, sh - off, 1.0, -1.0);
            corner(sw - off, sh - off, -1.0, -1.0);
        });

}

/// 中央：开机倒计时（3·2·1 → 物理环境）
fn boot_overlay(ctx: &egui::Context, w: &World) {
    // ── 中央：开机倒计时（3·2·1 → 物理环境）──
    if !w.boot_switched {
        let remain = (3.0f32 - w.boot.elapsed().as_secs_f32()).max(0.0);
        let alpha = (0.35 + 0.65 * (remain.fract())).min(1.0);   // 每秒呼吸
        egui::Area::new(egui::Id::new("bootcount"))
            .anchor(egui::Align2::CENTER_CENTER, [0.0, -40.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new(format!("{:.0}", remain.ceil()))
                        .color(egui::Color32::from_rgba_unmultiplied(0, 229, 255, (alpha * 255.0) as u8))
                        .size(64.0));
                    ui.label(egui::RichText::new("进入物理环境 PHYSICS")
                        .color(egui::Color32::from_rgba_unmultiplied(215, 222, 232, (alpha * 200.0) as u8))
                        .size(14.0));
                });
            });
    }

}

/// 左上：状态徽章 + 环境/电源/复位/控制台开关（同一张卡）
fn status_card(ctx: &egui::Context, w: &World, tick_hz: f32, cmds: &mut Vec<UiCmd>) {
    // ── 左上：状态徽章 + 环境/电源/复位/控制台开关（同一张卡）──
    // 原来状态卡和按钮是两个独立悬浮区，窗口窄于二者宽度之和时互相压盖；
    // 并进同一张卡的同一布局流后结构上不可能重叠
    let busy = w.ctl.roulade.is_some() || w.ctl.kick.is_some() || w.ctl.ground_pick.is_some();
    egui::Area::new(egui::Id::new("status"))
        .anchor(egui::Align2::LEFT_TOP, [18.0, 16.0])
        .show(ctx, |ui| {
            card_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    // 左：状态块
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            let (col, mark) = if w.sim.enabled { (GOOD, "● RUN") } else { (BAD, "○ IDLE") };
                            ui.label(egui::RichText::new(mark).color(col).size(16.0));
                            ui.separator();
                            ui.label(egui::RichText::new(w.ctl.label).color(ORANGE).size(16.0));
                            if busy {
                                ui.label(egui::RichText::new("BUSY").color(ORANGE).size(10.0));
                            }
                        });
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            let mode = if w.mode == Mode::Physics { "PHYSICS" } else { "RIG" };
                            ui.label(egui::RichText::new(mode).color(CYAN).size(10.0));
                            ui.label(egui::RichText::new(format!("gain {}", w.ctl.gain)).color(DIM).size(10.0));
                            ui.label(egui::RichText::new(format!("tick {:.0}Hz", tick_hz)).color(DIM).size(10.0));
                        });
                    });
                    ui.separator();
                    // 右：环境 / 电源 / 复位 / 控制台开关
                    if ui
                        .selectable_label(w.mode == Mode::Rig, egui::RichText::new("支架").size(11.0))
                        .clicked()
                    {
                        cmds.push(UiCmd::SetMode(Mode::Rig));
                    }
                    if ui
                        .selectable_label(w.mode == Mode::Physics, egui::RichText::new("物理").size(11.0))
                        .clicked()
                    {
                        cmds.push(UiCmd::SetMode(Mode::Physics));
                    }
                    ui.separator();
                    if ui.button(egui::RichText::new("PW").size(10.0)).clicked() {
                        cmds.push(UiCmd::ToggleEnable);
                    }
                    if ui.button(egui::RichText::new("RESET ⟳").size(10.0)).clicked() {
                        cmds.push(UiCmd::Reset);
                    }
                    if ui
                        .button(
                            egui::RichText::new(if w.console_open { "隐藏 ◧" } else { "控制台 ◨" })
                                .size(10.0),
                        )
                        .clicked()
                    {
                        cmds.push(UiCmd::ToggleConsole);
                    }
                });
            });
        });

}

/// 顶中：拖拽/加载结果提示（5s 自动消失；Extend 不换行 —— Area 窄宽会折成竖排）
fn top_msg(ctx: &egui::Context, w: &World) {
    // ── 顶中：拖拽/加载结果提示（5s 自动消失；Extend 不换行 —— Area 窄宽会把
    // Label 折成一字一行竖排）──
    egui::Area::new(egui::Id::new("topmsg"))
        .anchor(egui::Align2::CENTER_TOP, [0.0, 14.0])
        .show(ctx, |ui| {
            let fresh = |t: &Instant| t.elapsed().as_secs_f32() < 5.0;
            if let Some(hint) = &w.drop_hint {
                ui.add(egui::Label::new(
                    egui::RichText::new(hint).color(ORANGE).size(12.0),
                ).wrap_mode(egui::TextWrapMode::Extend));
            } else if let Some((ok, msg, at)) = &w.load_msg {
                if fresh(at) {
                    ui.add(egui::Label::new(
                        egui::RichText::new(msg).color(if *ok { GOOD } else { BAD }).size(11.0),
                    ).wrap_mode(egui::TextWrapMode::Extend));
                }
            }
        });

    // （环境/电源/复位/控制台开关已并入左上状态卡 —— 窄窗口不再互相压盖）

}

/// 左下：指令输入面板（obs 48..61 的指令块，twist/head/body 三组）。
/// 颜色即分组；拖拽 = 持久基值、键盘 = 弹簧偏移、行末「0」单独归零。
fn cmd_panel(ctx: &egui::Context, w: &World, cmds: &mut Vec<UiCmd>) {
    // ── 左下：指令输入一行式（obs 48..61 的指令块，twist/head/body 三组一串）──
    // 颜色即分组：橙=twist、青=head、紫=body。twist 键盘持续驱动只读；
    // head 的头p/r 与 body 三轴是内联 DragValue，拖数字即改，仍是可输入面板。
    // 状态头如实标注"谁在写指令块"：行为型策略激活时 twist 段被脚本接管。
    egui::Area::new(egui::Id::new("cmd"))
        // -16 是理论贴底值；Frame/行高的测量偏差 ~24px，实测 -40 正好贴底不裁行
        .anchor(egui::Align2::LEFT_BOTTOM, [18.0, -40.0])
        .show(ctx, |ui| {
            card_frame().show(ui, |ui| {
                ui.set_max_width(196.0);
                ui.spacing_mut().item_spacing.y = 2.0;   // 条行贴紧，脊线近似连续
                // 可拖数值去底框：静态时是纯文字，悬停才见交互框（按钮底色在 weak_bg_fill）
                {
                    let visuals = ui.visuals_mut();
                    for state in [&mut visuals.widgets.inactive, &mut visuals.widgets.hovered, &mut visuals.widgets.active] {
                        state.weak_bg_fill = egui::Color32::TRANSPARENT;
                    }
                    visuals.widgets.inactive.bg_fill = egui::Color32::TRANSPARENT;
                    visuals.widgets.inactive.bg_stroke = egui::Stroke::NONE;
                    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 16);
                    visuals.widgets.active.bg_fill = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 30);
                }
                let (state, col) = match w.ctl.label {
                    "walk" => ("输入开放", GOOD),
                    "stand" => ("twist 归零 · head/body 可输入", GOOD),
                    "sit" | "rise" => ("twist 已接管（姿势旗）", ORANGE),
                    "ground_pick" => ("twist 已接管（相位编码）", ORANGE),
                    "kick_left" | "kick_right" | "roulade" => ("指令块已清零", ORANGE),
                    _ => ("未使能", DIM),
                };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("▸ 指令输入").color(CYAN).size(10.5));
                    ui.label(egui::RichText::new(w.ctl.label).color(CYAN).size(9.5));
                });
                ui.label(egui::RichText::new(state).color(col).size(9.0));

                // ── twist（橙）：条显示 live（基值+键盘），数字拖的是持久基值（巡航）──
                ui.add_space(3.0);
                group_tag(ui, "TWIST", ORANGE);
                let t = w.sim.ema_twist;
                for (i, name, live, max) in [
                    (0usize, "vx", t[0] as f32, MAX_LINEAR as f32),
                    (1, "vy", t[1] as f32, MAX_LINEAR as f32),
                    (2, "ω", t[2] as f32, MAX_ANGULAR as f32),
                ] {
                    let lim = if i == 2 { MAX_ANGULAR as f32 } else { MAX_LINEAR as f32 };
                    let zeroed = bar_row(ui, name, live, -max, max, ORANGE, &[], |ui| {
                        let mut base = w.sim.twist_base[i] as f32;
                        if ui.add(drag_val(&mut base, -lim, lim, lim * 0.02, "", 2)).changed() {
                            cmds.push(UiCmd::TwistBase(i, base));
                        }
                    });
                    if zeroed { cmds.push(UiCmd::TwistBase(i, 0.0)); }
                }

                // ── head（青）：数字拖基值/头p·头r 直接设值；条显示 live（含键盘叠加）──
                ui.add_space(6.0);
                group_tag(ui, "HEAD", CYAN);
                let h = w.sim.ema_head;
                let zeroed = bar_row(ui, "颈", h[0] as f32, -MAX_HEAD as f32, MAX_HEAD as f32, CYAN, &[], |ui| {
                    let mut nb = w.sim.head_base[0] as f32 * MAX_HEAD as f32;
                    if ui.add(drag_val(&mut nb, -2.5, 2.5, 0.05, "", 2)).changed() {
                        cmds.push(UiCmd::HeadBase(0, nb / MAX_HEAD as f32));
                    }
                });
                if zeroed { cmds.push(UiCmd::HeadBase(0, 0.0)); }
                let zeroed = bar_row(ui, "头p", h[1] as f32, -MAX_HEAD as f32, MAX_HEAD as f32, CYAN, &[], |ui| {
                    let mut hp = w.sim.head_base[1] as f32 * MAX_HEAD as f32;
                    if ui.add(drag_val(&mut hp, -2.5, 2.5, 0.05, "", 2)).changed() {
                        cmds.push(UiCmd::HeadBase(1, hp / MAX_HEAD as f32));
                    }
                });
                if zeroed { cmds.push(UiCmd::HeadBase(1, 0.0)); }
                let zeroed = bar_row(ui, "头y", h[2] as f32, -MAX_HEAD as f32, MAX_HEAD as f32, CYAN, &[], |ui| {
                    let mut yb = w.sim.head_base[2] as f32 * MAX_HEAD as f32;
                    if ui.add(drag_val(&mut yb, -2.5, 2.5, 0.05, "", 2)).changed() {
                        cmds.push(UiCmd::HeadBase(2, yb / MAX_HEAD as f32));
                    }
                });
                if zeroed { cmds.push(UiCmd::HeadBase(2, 0.0)); }
                let zeroed = bar_row(ui, "头r", h[3] as f32, -MAX_HEAD as f32, MAX_HEAD as f32, CYAN, &[], |ui| {
                    let mut hr = w.sim.head_base[3] as f32 * MAX_HEAD as f32;
                    if ui.add(drag_val(&mut hr, -2.5, 2.5, 0.05, "", 2)).changed() {
                        cmds.push(UiCmd::HeadBase(3, hr / MAX_HEAD as f32));
                    }
                });
                if zeroed { cmds.push(UiCmd::HeadBase(3, 0.0)); }

                // ── body（紫）：三轴拖数字改，值进观测按 B 生效 ──
                ui.add_space(6.0);
                group_tag(ui, if w.body_active { "BODY · 姿态模式" } else { "BODY" }, VIOLET);
                let zlab = if w.body_active { "● z" } else { "z" };
                let zeroed = bar_row(ui, zlab, w.sim.ema_body[0] as f32, BODY_Z_RANGE[0] as f32, BODY_Z_RANGE[1] as f32, VIOLET, &[], |ui| {
                    let mut zcm = w.body_cmd[0] as f32 * 100.0;
                    if ui.add(drag_val(&mut zcm, (BODY_Z_RANGE[0] * 100.0) as f32, (BODY_Z_RANGE[1] * 100.0) as f32, 0.1, "cm", 1)).changed() {
                        cmds.push(UiCmd::BodySlot(0, zcm / 100.0));
                    }
                });
                if zeroed { cmds.push(UiCmd::BodySlot(0, 0.0)); }
                for (i, name) in [(1usize, "roll"), (2, "pitch")] {
                    // 条 = 生效值（含键盘弹簧与平滑），数字 = 拖拽基值
                    let v = (w.sim.ema_body[i] as f32).to_degrees();
                    let zeroed = bar_row(ui, name, v, -TILT_DEG, TILT_DEG, VIOLET, &[-KEY_TILT_DEG, KEY_TILT_DEG], |ui| {
                        let mut deg = (w.body_cmd[i] as f32).to_degrees();
                        if ui.add(drag_val(&mut deg, -TILT_DEG, TILT_DEG, 1.0, "°", 0)).changed() {
                            cmds.push(UiCmd::BodySlot(i, deg.to_radians()));
                        }
                    });
                    if zeroed { cmds.push(UiCmd::BodySlot(i, 0.0)); }
                }
            });
        });

}

/// 右下：技能键帽竖排（贴右下角锚定；命中区 = 可见区，不能再用 available_width 分配）
fn hotbar(ctx: &egui::Context, w: &World, cmds: &mut Vec<UiCmd>) {
    // ── 右下：技能键帽竖排（贴右下角锚定，任何窗口尺寸都不乱、不挡中央）──
    // 冷却扫描直接压在键帽里（MOBA 语义）：进度 = 键帽上的半透明层高度。
    // 控制台展开时整列左移避让（与右上按钮同款）。
    let lbl = w.ctl.label;
    let right_off = if w.console_open { 346.0 } else { 18.0 };
    egui::Area::new(egui::Id::new("hotbar"))
        .anchor(egui::Align2::RIGHT_BOTTOM, [-right_off, -18.0])
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                // (键, 名, 激活, 冷却剩余比例 0..1, 命令)
                let rows: Vec<(&str, &str, bool, f32, UiCmd)> = vec![
                    ("G", "捡地", lbl == "ground_pick",
                        (w.ctl.ground_pick.unwrap_or(0.0) / 0.7) as f32, UiCmd::Pick),
                    ("X", "滚翻", lbl == "roulade",
                        (w.ctl.roulade.unwrap_or(0.0) / (ROULADE_DURATION + w.ctl.roulade_ext)) as f32,
                        UiCmd::Roulade),
                    ("V", "坐起", lbl == "sit" || lbl == "rise",
                        (w.ctl.rise_remaining / RISE_SECS) as f32, UiCmd::Sit),
                    ("Z", "左踢", lbl == "kick_left",
                        (w.ctl.kick.map(|(_, r)| r).unwrap_or(0.0) / KICK_DURATION) as f32, UiCmd::KickL),
                    ("C", "右踢", lbl == "kick_right",
                        (w.ctl.kick.map(|(_, r)| r).unwrap_or(0.0) / KICK_DURATION) as f32, UiCmd::KickR),
                    ("B", "姿态", w.body_active, 0.0, UiCmd::BodyPose),
                ];
                for (key, name, active, frac, cmd) in rows {
                    let frac = frac.clamp(0.0, 1.0);
                    // 标签条（仅悬停）+ 键帽（可点击）水平并排：
                    // 命中区必须=可见区！原来整行用 available_width 分配 click——
                    // 右锚 Area 的可用宽近全屏，透明点击带横贯屏幕，点左面板会误触技能
                    ui.horizontal(|ui| {
                        ui.allocate_exact_size(egui::vec2(52.0, 30.0), egui::Sense::hover());
                        let (r, resp) = ui.allocate_exact_size(
                            egui::vec2(26.0, 30.0),
                            egui::Sense::click(),
                        );
                        // 键帽 = 命中区本身
                        let cap = egui::Rect::from_min_size(
                            egui::pos2(r.center().x - 13.0, r.center().y - 13.0),
                            egui::vec2(26.0, 26.0),
                        );
                        let p = ui.painter();
                    // 键帽底 + 描边
                    p.rect_filled(cap, 4.0, if active {
                        egui::Color32::from_rgb(34, 24, 10)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 130)
                    });
                    p.rect_stroke(cap, 4.0, egui::Stroke::new(1.5, if active {
                        ORANGE
                    } else {
                        egui::Color32::from_rgba_unmultiplied(215, 222, 232, 105)
                    }), egui::StrokeKind::Inside);
                    // 冷却扫描：从顶部压下的半透明层，高度 = 剩余比例
                    if active && frac > 0.001 {
                        let cr = egui::Rect::from_min_size(
                            cap.min,
                            egui::vec2(cap.width(), cap.height() * frac),
                        );
                        p.rect_filled(cr, 4.0, egui::Color32::from_rgba_unmultiplied(0, 229, 255, 52));
                    }
                    // 键字母（压在扫描层之上）
                    p.text(cap.center(), egui::Align2::CENTER_CENTER, key,
                        egui::FontId::monospace(12.0),
                        if active { ORANGE } else { egui::Color32::from_rgb(215, 222, 232) });
                    // 标签在键帽左侧
                    p.text(egui::pos2(cap.left() - 8.0, cap.center().y),
                        egui::Align2::RIGHT_CENTER, name,
                        egui::FontId::proportional(10.5),
                        egui::Color32::from_rgba_unmultiplied(125, 139, 157, 255));
                    if resp.clicked() {
                        cmds.push(cmd);
                    }
                    });
                }
            });
        });

}


/// 控制台抽屉（TAB / 右上按钮开关）：SCENE / POLICIES / JOINTS / IMU / PD / PERF / KEYS。
/// 面板各自成函数，改一块不影响别块。
fn console(ctx: &egui::Context, w: &World, frame_dt: f32, tick_hz: f32, cmds: &mut Vec<UiCmd>) {
    if !w.console_open {
        return;
    }
    egui::SidePanel::right("console")
        .resizable(false)
        .min_width(332.0)
        .max_width(332.0)
        .frame(
            egui::Frame::default()
                .fill(egui::Color32::from_rgba_unmultiplied(6, 10, 14, 225))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(24, 52, 66)))
                .inner_margin(egui::Margin::symmetric(10, 8)),
        )
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new("◤ M I C R O D U C K ◢").color(CYAN).size(17.0));
                ui.label(egui::RichText::new("NEURAL LOCOMOTION CONSOLE").color(DIM).size(9.0));
            });
            ui.add_space(4.0);
            ui.label(egui::RichText::new("TAB 收起 · 拖 .onnx 入窗口加载").color(DIM).size(9.0));
            sep(ui);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    console_scene(ui, w, cmds);
                    console_policies(ui, w, cmds);
                    console_joints(ui, w);
                    console_imu(ui, w);
                    console_pd(ui, w, cmds);
                    console_perf(ui, w, tick_hz, frame_dt);
                    console_keys(ui);
                });
        });
}

/// SCENE 面板：视觉地面圆点 + 场景文件列表（点行选中）+ 浏览/坡体提示
fn console_scene(ui: &mut egui::Ui, w: &World, cmds: &mut Vec<UiCmd>) {
                // ── SCENE ──
    // 行式布局 = SCENE / POLICIES 共用语言：状态点 ●（着色）| 定宽名称（可点）| 详情
    section(ui, "SCENE · 场景（1/2 视觉 · 3 坡体）");
    ui.horizontal(|ui| {
        for sc in [Scene::Flat, Scene::Checker] {
            let on = w.scene == sc;
            let col = if on { CYAN } else { DIM };
            if ui.add(egui::Button::new(
                egui::RichText::new(format!("{} {}", if on { "●" } else { "○" }, sc.label()))
                    .color(col).size(10.0))
                .frame(false))
                .clicked()
            {
                cmds.push(UiCmd::SetScene(sc));
            }
        }
        ui.label(egui::RichText::new("视觉地面").color(DIM).size(8.5));
    });
    // 自动发现的场景文件（assets/mj/*.xml）：点行选中，物理 XML 即刻重载
    for (i, sf) in w.scenes.iter().enumerate() {
        let active = i == w.scene_sel;
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(if active { "●" } else { "○" })
                .color(if active { GOOD } else { DIM }).size(9.0));
            if ui.add_sized([96.0, 16.0],
                egui::SelectableLabel::new(active, egui::RichText::new(&sf.name).size(10.0)))
                .clicked()
            {
                cmds.push(UiCmd::SelectScene(i));
            }
            let hover = sf.path.display().to_string();
            ui.label(egui::RichText::new(if sf.has_ramp { "坡体 · 物理碰撞" } else { "平地" })
                .color(DIM).size(8.5)).on_hover_text(hover);
        });
    }
    if w.scene_has_ramp() {
        ui.label(egui::RichText::new("坡体有真实碰撞 —— 平面策略爬坡可能摔（R 复位回平地）").color(DIM).size(8.5));
    }
    ui.horizontal(|ui| {
        if ui.button("浏览 .xml…").clicked() {
            cmds.push(UiCmd::PickScene);
        }
        ui.label(egui::RichText::new("↑ 或拖 .xml 入窗口").color(DIM).size(8.5));
    });
}

/// POLICIES 面板：七槽状态点 + 目标槽位选择 + 浏览按钮
fn console_policies(ui: &mut egui::Ui, w: &World, cmds: &mut Vec<UiCmd>) {
    // ── POLICIES ──
                section(ui, "POLICIES · ONNX 拖入窗口或点行选中");
                for (i, slot) in w.pol.slots.iter().enumerate() {
                    let selected = i == w.load_target;
                    let ok = slot.session.is_some();
                    ui.horizontal(|ui| {
                        // ● 已加载 / ○ 失败（不用 ✓✗ —— 中文字体缺该字形会渲染成豆腐块）
                        ui.label(egui::RichText::new(if ok { "●" } else { "○" })
                            .color(if ok { GOOD } else { BAD }).size(9.0));
                        // 角色名：定宽单元格（跨行列对齐）+ 点选为加载目标
                        if ui.add_sized([70.0, 16.0],
                            egui::SelectableLabel::new(selected, egui::RichText::new(slot.role).size(10.0)))
                            .clicked()
                        {
                            cmds.push(UiCmd::SetLoadTarget(i));
                        }
                        let shown = if slot.file.len() > 19 {
                            format!("…{}", &slot.file[slot.file.len() - 18..])
                        } else {
                            slot.file.clone()
                        };
                        let hover = match &slot.err {
                            Some(e) => format!("{}", e),
                            None => slot.path.display().to_string(),
                        };
                        ui.label(egui::RichText::new(shown).color(DIM).size(9.0)).on_hover_text(hover);
                    });
                }
                ui.horizontal(|ui| {
                    if ui.button("浏览 .onnx…").clicked() {
                        cmds.push(UiCmd::PickFile(w.load_target));
                    }
                    ui.label(
                        egui::RichText::new("↑ 点角色名选目标槽位").color(DIM).size(9.0),
                    );
                });
                let mut mult = w.scale_mult as f32;
                let mut mult_changed = false;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("动作缩放 ×").color(DIM).size(10.0));
                    mult_changed = ui
                        .add(egui::Slider::new(&mut mult, 0.2..=2.0).step_by(0.05).text(""))
                        .changed();
                    ui.label(egui::RichText::new(format!("{mult:.2}")).color(ORANGE).size(10.0));
                });
                if mult_changed {
                    cmds.push(UiCmd::ScaleMult(mult as f64));
                }
}

/// JOINTS 面板：15 关节全宽条（青=实际 · 橙刻度=目标 · 红=家位，缝隙即跟踪误差）
fn console_joints(ui: &mut egui::Ui, w: &World) {
    // ── JOINTS ──
    section(ui, "JOINTS · 观测 = 真实反馈");
    ui.label(egui::RichText::new("青条=实际 · 橙刻度=目标 · 红线=家位；走动时刻度与条边的缝隙 = 跟踪误差").color(DIM).size(8.5));
    // 单列全宽条：双列 Grid 的自适应列宽会把条压到十几像素，看不出变动
    for i in 0..15 {
        let [lo, hi] = show_range(i);
        let span = (hi - lo) as f64;
        let t = (((w.sim.q[i] - lo as f64) / span).clamp(0.0, 1.0)) as f32;
        let tg = (((w.sim.targets[i] - lo as f64) / span).clamp(0.0, 1.0)) as f32;
        let hm = (((DEFAULT_POSITION[i] - lo as f64) / span).clamp(0.0, 1.0)) as f32;
        ui.horizontal(|ui| {
            let (lr, _) = ui.allocate_exact_size(egui::vec2(46.0, 9.0), egui::Sense::hover());
            ui.painter().text(
                egui::pos2(lr.right(), lr.center().y),
                egui::Align2::RIGHT_CENTER,
                JOINT_SHORT[i],
                egui::FontId::proportional(9.0),
                DIM,
            );
            let (r, resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width() - 4.0, 8.0),
                egui::Sense::hover(),
            );
            let p = ui.painter_at(r);
            p.rect_filled(r, 2.0, egui::Color32::from_rgb(14, 22, 30));
            let wq = r.width() * t;
            p.rect_filled(
                egui::Rect::from_min_size(r.left_top(), egui::vec2(wq, r.height())),
                2.0, egui::Color32::from_rgb(0, 120, 160),
            );
            // 家位线画在填充之上（半透明）：被青条盖住时也始终可见
            let q_len = r.width() * hm;
            p.line_segment(
                [egui::pos2(r.left() + q_len, r.top()), egui::pos2(r.left() + q_len, r.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(200, 60, 60, 150)),
            );
            let wt = r.width() * tg;
            p.line_segment(
                [egui::pos2(r.left() + wt, r.top() + 1.0), egui::pos2(r.left() + wt, r.bottom() - 1.0)],
                egui::Stroke::new(2.0, ORANGE),
            );
            let _ = resp;
        });
    }
}

/// IMU 面板：气泡水平仪（投影重力解倾角）+ grav/倾角/角速度读数
fn console_imu(ui: &mut egui::Ui, w: &World) {
    // ── IMU ──
    section(ui, "IMU · trunk frame");
    let (g, gw) = w.imu;
    // 由投影重力解倾角（直立 grav = [0,0,-1]）：气泡=倾斜方向，
    // 倾角读数=|倾|，陀螺=角速度模长（掉出平静阈转橙）
    let tilt_x = g[0].atan2(-g[2]).to_degrees();
    let tilt_y = g[1].atan2(-g[2]).to_degrees();
    let tilt_mag = (tilt_x * tilt_x + tilt_y * tilt_y).sqrt();
    let wmag = (gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]).sqrt();
    let tilt_col = if tilt_mag < 8.0 { GOOD } else if tilt_mag < 25.0 { ORANGE } else { BAD };
    let gyro_col = if wmag < 1.0 { GOOD } else { ORANGE };
    ui.horizontal(|ui| {
        // 气泡水平仪：满量程 ±30°，圈内十字 + 半程参考圈
        let (r, _) = ui.allocate_exact_size(egui::vec2(46.0, 46.0), egui::Sense::hover());
        let p = ui.painter_at(r);
        let c = r.center();
        let rad = r.width() * 0.5 - 1.0;
        p.circle_filled(c, rad, egui::Color32::from_rgb(11, 18, 26));
        p.circle_stroke(c, rad, egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 64, 82)));
        let axis = egui::Stroke::new(1.0, egui::Color32::from_rgb(30, 46, 60));
        p.line_segment([egui::pos2(c.x - rad, c.y), egui::pos2(c.x + rad, c.y)], axis);
        p.line_segment([egui::pos2(c.x, c.y - rad), egui::pos2(c.x, c.y + rad)], axis);
        p.circle_stroke(c, rad * 0.5, axis);
        let bx = (tilt_x / 30.0).clamp(-1.0, 1.0) as f32;
        let by = (tilt_y / 30.0).clamp(-1.0, 1.0) as f32;
        let dot = egui::pos2(c.x + bx * (rad - 4.0), c.y - by * (rad - 4.0));
        p.line_segment([c, dot], egui::Stroke::new(1.0, tilt_col.gamma_multiply(0.5)));
        p.circle_filled(dot, 3.5, tilt_col);
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                kv_row(ui, "grav",
                    &format!("{:+.2} {:+.2} {:+.2}", g[0], g[1], g[2]),
                    egui::Color32::from_rgb(200, 214, 232), 28.0, 108.0);
            });
            ui.horizontal(|ui| {
                kv_row(ui, "倾",
                    &format!("x{tilt_x:+4.0}° y{tilt_y:+4.0}°"),
                    tilt_col, 28.0, 108.0);
            });
            ui.horizontal(|ui| {
                kv_row(ui, "ω",
                    &format!("{wmag:4.2} rad/s"),
                    gyro_col, 28.0, 108.0);
            });
        });
    });
}

/// ACTUATOR PD 面板：kp / kd* 滑条 + 规格行（点基线行复位到训练值）
fn console_pd(ui: &mut egui::Ui, w: &World, cmds: &mut Vec<UiCmd>) {
    // ── ACTUATOR PD（物理模式）──
    if let Some(ph) = w.phys.as_ref() {
        section(ui, "ACTUATOR PD · 物理伺服");
        let mut kp = ph.kp as f32;
        let mut kd = ph.kd as f32;
        let mut kp_changed = false;
        let mut kd_changed = false;
        // 标签(定宽右对齐) | 滑条(去内建数值框) | 橙色等宽值
        ui.horizontal(|ui| {
            ui.set_min_height(17.0);
            let (lr, _) = ui.allocate_exact_size(egui::vec2(26.0, 17.0), egui::Sense::hover());
            ui.painter().text(lr.right_center(), egui::Align2::RIGHT_CENTER,
                "kp", egui::FontId::proportional(10.0), DIM);
            kp_changed = ui.add_sized(
                [ui.available_width() - 58.0, 15.0],
                egui::Slider::new(&mut kp, 0.20..=3.0).step_by(0.01).text("").show_value(false),
            ).changed();
            ui.label(egui::RichText::new(format!("{kp:>5.2}")).color(ORANGE).monospace().size(10.5));
        });
        ui.horizontal(|ui| {
            ui.set_min_height(17.0);
            let (lr, _) = ui.allocate_exact_size(egui::vec2(26.0, 17.0), egui::Sense::hover());
            ui.painter().text(lr.right_center(), egui::Align2::RIGHT_CENTER,
                "kd*", egui::FontId::proportional(10.0), DIM);
            kd_changed = ui.add_sized(
                [ui.available_width() - 58.0, 15.0],
                egui::Slider::new(&mut kd, 0.0..=0.30).step_by(0.005).text("").show_value(false),
            ).changed();
            ui.label(egui::RichText::new(format!("{kd:>5.3}")).color(ORANGE).monospace().size(10.5));
        });
        // 参数规格：定宽 kv 行（标签右对齐、值左对齐等宽 —— egui Grid
        // 列宽随内容漂移，这里要的是刀切一样的分栏）
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            kv_row(ui, "τ_max", &format!("±{:.2} N·m", ph.max_force),
                egui::Color32::from_rgb(200, 214, 232), 40.0, 100.0);
        });
        // 训练基线行 = 复位按钮：点击把 kp/kd 调回训练值（0.55 / 0）
        ui.horizontal(|ui| {
            let (rr, resp) = ui.allocate_exact_size(
                egui::vec2(140.0, 14.0), egui::Sense::click());
            let resp = resp.on_hover_text("点击复位到训练基线：kp 0.55 · kd* 0");
            let col = if resp.hovered() {
                CYAN
            } else if (kp - 0.55).abs() > 1e-6 || kd.abs() > 1e-6 {
                ORANGE   // 已偏离基线：橙字提示"可点我复位"
            } else {
                DIM
            };
            let p = ui.painter();
            p.text(egui::pos2(rr.left() + 40.0, rr.center().y),
                egui::Align2::RIGHT_CENTER, "训练基线",
                egui::FontId::proportional(9.0), col);
            p.text(egui::pos2(rr.left() + 42.0, rr.center().y),
                egui::Align2::LEFT_CENTER,
                if resp.hovered() { "⟲ kp 0.55 · kd* 0" } else { "kp 0.55 · kd* 0" },
                egui::FontId::monospace(9.5), col);
            if resp.clicked() { cmds.push(UiCmd::PdReset); }
        });
        if kp_changed { cmds.push(UiCmd::Kp(kp)); }
        if kd_changed { cmds.push(UiCmd::Kd(kd)); }
    }

    sep(ui);
}

/// PERF 面板：infer / tick / render / frame（tick 掉出 45Hz 转橙）+ GPU
fn console_perf(ui: &mut egui::Ui, w: &World, tick_hz: f32, frame_dt: f32) {
    ui.add_space(3.0);
    // ── PERF：性能两行网格（定宽分栏；tick 掉出 45Hz 转橙示警）──
    let fps = if frame_dt > 0.0 { 1.0 / frame_dt } else { 0.0 };
    let tick_col = if tick_hz >= 45.0 { GOOD } else { ORANGE };
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        kv_row(ui, "infer", &format!("{:.2} ms", w.infer_us / 1000.0), GOOD, 40.0, 74.0);
        kv_row(ui, "tick", &format!("{:.0} Hz", tick_hz), tick_col, 34.0, 74.0);
    });
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        kv_row(ui, "render", &format!("{:.0} fps", fps), CYAN, 40.0, 74.0);
        kv_row(ui, "frame", &format!("#{}", w.tick_count), DIM, 34.0, 74.0);
    });
    ui.label(egui::RichText::new(w.gpu_name.clone())
        .color(egui::Color32::from_rgb(96, 112, 130)).size(8.5));

}

/// KEYS 面板：完整键位表（左下浮层只开局显示 12s，这里常驻兜底）
fn console_keys(ui: &mut egui::Ui) {
    // ── KEYS：完整键位表（左下浮层只开局显示 12s，这里常驻兜底）──
    section(ui, "KEYS · 键位");
    ui.label(egui::RichText::new(
        "WASD/QE 行走 · ↑↓←→ 头颈 · Ctrl+方向 头p/r\nG 捡地 · X 滚翻 · V 坐起 · Z/C 踢 · M 嘴\nB 姿态模式（WS 升降 · AD 侧倾 · QE 俯仰）\nSPACE 使能 · P 支架/物理 · R 复位 · TAB 控制台",
    ).color(DIM).size(8.5));
}
