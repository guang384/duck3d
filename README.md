---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '51e06045-e825-4f47-8182-63866a6eb200'
  PropagateID: '51e06045-e825-4f47-8182-63866a6eb200'
  ReservedCode1: '51dcff33-ade4-4321-b5f7-dc61eef2e166'
  ReservedCode2: '51dcff33-ade4-4321-b5f7-dc61eef2e166'
---

# duck3d — microduck 的桌面 3D 控制台

一只在桌面窗口里跑真机策略的机器鸭：wgpu 走 **DX12/Vulkan 硬件渲染**，50Hz 仿真跑在
原生线程定时器上，推理用 **ONNX Runtime**（与真机 robotd 同引擎同版本 `=2.0.0-rc.11`，
构建时自动下载，零配置）。控制调度是 `robotd/src/control.rs` 的移植。

![duck3d](https://img.shields.io/badge/policies-obs%5B1%2C61%5D%20%E2%86%92%20actions%5B1%2C14%5D-yellow)

## 跑起来

```bash
cd app
cargo run --release                    # 打开窗口：3 秒后自动进物理模式，W/A/S/D 直接走
cargo run --release -- --selftest      # 无窗口仿真自检（站/走/坐/踢/滚翻/槽位替换）
cargo run --release -- --shot out.bmp  # 离屏渲一帧（自动化验证）
```

实测（Intel Arc 130T）：推理 **0.02 ms/tick**，全流程 GPU 渲染零卡顿。桌面快捷方式可指向
`app/target/release/duck3d-app.exe`（在项目内任意目录启动都能自动找到 `policies/` 与 `assets/`）。

自动化验证用的环境变量：`DUCK3D_SHOT_HUD=1`（离屏带 HUD）、`DUCK3D_SHOT_PHYSICS=1`
（物理模式）、`DUCK3D_SHOT_SIZE=宽x高`、`DUCK3D_WALKSEC=秒`（按 W 走并打印轨迹）、
`DUCK3D_STRAFE=1`（走的同时按 A，验证侧向碰撞）、`DUCK3D_SCENE=<场景名>`（启动即选场景）。

## 场景

两层，互不冲突：

| 层 | 选项 | 切换方式 |
|---|---|---|
| **视觉地面** | 平面（网格）/ 棋盘（训练场景同款） | `1` / `2` 键，或控制台圆点。纯外观 |
| **场景文件** | `assets/mj/*.xml` 启动时自动扫描列出；可拖 `.xml` 入窗口或「浏览 .xml…」加自定义 | 控制台点行选中；`3` 键一键切坡体场景 |

- 场景文件决定**物理**（碰撞体、摩擦、几何）；文件带坡（名字含 `ramp`）就自动画实体楔形坡。
- 地面跟随鸭子移动，按视觉图案周期吸附（网格 0.25 m、棋盘 1.0 m）—— 视觉上无限延伸，
  且图案不会在行走时跳相。
- 坡体有真实碰撞 —— 平面策略爬坡可能摔（`R` 复位回平地）。

## 操作

| 输入 | 作用 |
|---|---|
| `W/S` `A/D` | 前后 / 左右平移 |
| `Q/E` | 转向 |
| `↑↓` / `←→` | 颈 pitch / 头 yaw |
| `Ctrl+↑↓` / `Ctrl+←→` | 头 pitch / 头 roll |
| `空格` | 使能 / 禁用策略 |
| `R` | 复位（清指令基值 + 关节回家位） |
| `P` | 支架 RIG / 物理 PHYSICS |
| `B` | 姿态模式：`WS` 升降 · `AD` 侧倾 · `QE` 俯仰 |
| `G/X/Z/C/V/M` | 捡地 / 前滚翻（按住连滚）· 左踢 / 右踢 / 坐起 / 呱 |
| 鼠标 | 左键拖旋转（1:1 跟手）· 滚轮缩放（平滑）· 右键/中键平移 |
| `TAB` | 控制台抽屉（默认收起） |

## 左下指令面板

三组就是观测里的指令块（obs 48..61），颜色即分组，每行一个中轴条（右正左负）：

- **TWIST**：vx / vy / ω —— 行走
- **HEAD**：颈 / 头p / 头y / 头r —— 摆头
- **BODY**：z / roll / pitch —— 站姿（按 `B` 生效；x/y/yaw 训练未绑定，恒零）

规则：**拖拽 = 持久值（基值），键盘 = 弹簧**（按住偏离、松开回基值）；行末小「0」按钮
单独归零该项。行为型策略激活时 twist 段被脚本接管（捡地相位编码 `[cos φ, sin φ, 0]`、
坐姿旗 `[1,0,0]`、踢/滚翻清零），面板顶部如实标注当前归属。

## 两种环境（`P` 键或 HUD 切换）

- **支架 RIG** —— 躯干悬空看步态（gait viewer），永不摔。
- **物理 PHYSICS** —— **MuJoCo 真物理**，载入的正是训练仓库的 `scene.xml`
  （`robot_groundcontact` + 地面）—— 与 `microduck_rl` 训练/官方 CPU 部署脚本
  `scripts/infer_policy.py` 同一份模型、同一套 `chosen_actuator` 位置伺服
  （kp 0.55 N·m/rad、τmax ±0.96 N·m）。实测（`--selftest`）：站立 12s 高度恒定、
  直立度 1.000；前进 4s 位移 0.44 m；停走即站定。

两个坑记在这：XML 的 kp 是"软"伺服，无策略时 home 位会塌（官方脚本的 standby 模式
临时把 kp 调到 2.0 才保持姿势）；Rapier 等其它引擎因接触/执行器模型差异会让策略失稳 ——
所以物理后端是 MuJoCo，而不是"差不多"的引擎。

> 坡体场景的坡是**实体楔形**（`scene_ramp.xml` 的 mesh）。旧版是绕中心转 −12° 的薄盒，
> 近端埋地、远端悬空、板下留 5.7~16 cm 净空 —— 鸭子从侧面走过时脚会从板下钻进去造成
> 穿模，已改。

## 界面加载策略

HUD 的 `POLICIES · ONNX` 面板里点角色名选目标槽位，然后把训练好的 `.onnx`
**直接拖进窗口**（或「浏览 .onnx…」）即加载 —— 加载时按 robotd 的规矩校验
`obs[1,61] → actions[1,14]`，不符只在面板上报原因、保留原策略。失败的加载不会打断仿真。
旁边的**动作缩放 ×**滑条对应 robotd 的 `scale_mult`，新策略若需要不同 action_scale，
直接在界面上调。

七个槽位（`ROLES`，与 `robotd-params` 的 `ResolvedPolicy` 一致）：
`walk` / `stand` / `sitstand` / `ground_pick` / `kick_left` / `kick_right` / `roulade`。
槽位是策略调度的语义单元（每个对应一种行为及其触发），固定七个；导入只是替换某个槽的
模型文件，不会新增槽。

## HUD

青/橙暗色科幻风（egui）。默认收起控制台，`TAB` 展开：

- **状态卡**：RUN/IDLE + 当前策略标签 + `PHYSICS/RIG` + gain + tick Hz，右侧
  支架/物理 · PW · RESET · 控制台开关；
- **指令面板**（左下，常驻）：见上；
- **SCENE**：视觉地面圆点 + 场景文件列表（点行选中）+ 浏览按钮；
- **POLICIES**：七槽状态点（● 已加载 / ○ 失败，悬停看原因或路径）+ 目标槽位选择；
- **JOINTS**：15 关节全宽条 —— 青条=实际 · 橙刻度=目标 · 红线=家位，
  刻度与条边的缝隙就是跟踪误差；
- **IMU**：气泡水平仪（由投影重力解倾角，满量程 ±30°）+ grav / 倾角 / 角速度读数，
  按阈值变色；
- **ACTUATOR PD**：kp（位置增益）与 kd\*（速度阻尼，经 `biasprm` 注入）滑条 +
  τ_max / 训练基线规格行 —— **点基线行即复位**到训练值（kp 0.55 / kd\* 0），
  偏离基线时该行变橙提示；
- **PERF**：infer / tick / render / frame（tick 掉出 45 Hz 转橙）+ GPU 型号；
- **KEYS**：键位表（左下浮层只在开局显示 12 秒）。

## 指令 · 观测 · 调度

- **指令模型**：与 `padd` 同款 —— 0.3 m/s、1.5 rad/s、死区 0.1、头部命令 ×2.5 rad
  进观测（不是直接加在输出上）、body z/roll/pitch 行程同款（z −4~+3 cm、倾角 ±0.5 rad）。
- **观测**：`duck-control/src/obs.rs` 的 61 维布局逐位对齐
  （gyro 3 + gravity 3 + pos−home 14 + vel 14 + last action 14 + command 13），
  body x/y/yaw 恒零，body 块顺序 z, roll, pitch。
- **控制调度**：`robotd/src/control.rs` 的忠实移植 ——
  - 优先级链：`roulade > kick > ground pick > sit/rise > stand(按指令幅值) > walk`；
  - 技能窗口：捡地 4 s 相位、0.7 截止；踢腿 0.5 s；滚翻 1 s、0.15 s 链滚窗口；
  - 动作缩放（走路 0.9 / 站立 1.0）、站立增益软化（200→160）；
  - 训练低通：头 0.5 / 腿 0.7，嘴（槽 9）永远跳过。

## 鸭子模型

真机 CAD 外观（73.6 万三角面，14.7 MB），取自 `microduck_rl` 训练仓库的 MuJoCo 模型。
`gen_cad_table.mjs` 离线处理：按 body 合并网格、烘焙变换（含 `ankle_right` 坐标系补偿）、
材质色烘焙为顶点色，打包成 DUCKCAD3（量化 Int16 + 顶点色），另有格式文档在脚本头部。
原生应用另补两颗 CAD 里没有的眼睛。15 个关节与 `duck-ipc-proto::JOINT_NAMES` 一一对应，
home 位 = `DEFAULT_POSITION`（与训练环境的 `HOME_FRAME` 逐值相同）。嘴：训练模型里嘴是
刚性件，真机才有嘴舵机 —— 桌面版把 CAD 下颚（jaw / jaw_soft）挂在铰链上，沿用
`mouth_target` 的 −5°…+30° 行程做开合动画。

## 与真机的关系

桌面版是**离线仿真**，不连任何机器人。真机的同款控制路径是 `robotd` 的 50Hz 循环
（意图来自 `padd` 手柄 / IPC 的 `robot.*` 调用）。

诚实声明：执行器用的是模型 XML 里自带的仿射 PD（gainprm=0.55）近似，训练时的 BAM 电压
模型（负载相关摩擦、延迟）不复现 —— 评估趋势可信，绝对数值以 `microduck_rl` 为准。
想跑完整 BAM 执行器：在训练仓库用 `scripts/infer_policy.py` 或真机验证。

## 文件

```
duck3d/                   独立项目（不依赖 microduck 仓库的其余部分）
├── app/                 原生应用：wgpu 渲染 + ort 推理 + mujoco-rs 物理（独立 crate）
│   └── mujoco-dist/     构建期自动下载的 MuJoCo 3.12（mujoco-rs 的 auto-download，不入库）
├── assets/
│   ├── duck_cad.bin     CAD 打包（DUCKCAD3，全精度，原生应用用）
│   └── mj/              MuJoCo 场景：scene.xml · scene_ramp.xml · robot_groundcontact.xml
├── policies/            默认策略（7 个 alpha_* 槽位 + 轮式对照），随项目自带
├── tools/               模型/烘焙校验脚本（checkbake.py · overlap.py）
├── gen_cad_table.mjs    CAD 生成器：读 microduck_rl/mjmodel.xml → 重打 duck_cad.bin
└── README.md            本文件
```

策略文件是**项目自带**的副本（`policies/`）：microduck 主仓库已把它们改为从 Hub 下载、
不再随仓库携带，因此这里留一份可运行的默认集。要换成你自己的策略，直接在界面上拖入
`.onnx`（按槽位替换）即可。

CAD 更新流程：训练仓库的模型改动后，改 `gen_cad_table.mjs` 顶部的路径跑一遍，
生成的 `duck_cad.bin` 直接落在 `assets/`。

## 来源与许可

**Apache-2.0**（见 `LICENSE`）。本项目的派生成分与来源：

| 内容 | 来源 | 许可 |
|---|---|---|
| 观测布局（`obs[1,61]`）、控制调度、安全链 | `microduck` 的 `duck-control` · `robotd`（同仓库移植） | Apache-2.0 |
| MuJoCo 模型、CAD 网格、默认策略 `policies/*.onnx` | `microduck_rl`（训练仓库）训练与导出 | Apache-2.0 |
| MuJoCo 3.12 运行时 | DeepMind，构建期由 `mujoco-rs` 自动下载（不入库） | Apache-2.0 |
| ONNX Runtime | Microsoft，构建期自动下载 | MIT |

本仓库不包含机器人固件/daemon（`robotd`、`padd`、`duck-ipc-proto` 等）——那些在
`microduck` 主仓库里；这里只保留桌面仿真控制台所需的最小集合。

> AI生成
