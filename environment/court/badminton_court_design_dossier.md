# MuJoCo 羽毛球标准场地设计档案

版本：1.0.0
目标读者：Codex / MuJoCo 集成工程师
单位：SI，长度单位为 metre，角度单位为 radian，MuJoCo XML 中按 metre 建模。

---

## 0. 设计目标

本设计档案用于在 MuJoCo 中创建一个 BWF 标准羽毛球场地资产，包括：

1. 标准场地地板；
2. 双打外边线、单打边线、后场边线、短发球线、双打长发球线、中线；
3. 球网、网柱、顶端白色网带、简化网孔视觉；
4. 可选球网碰撞代理；
5. 落点合法性判定 helper；
6. 可由 Codex 继续改造的参数 JSON、生成脚本和验证脚本。

此包不是体育馆建筑模型，也不包含观众席、灯光、墙壁、天花板和裁判席。核心目标是让羽毛球、球拍、SMPL/机器人身体在同一 MuJoCo 坐标系中拥有正确场地几何和规则判定。

---

## 1. 官方依据

本设计使用 BWF Laws of Badminton 的场地和球网规格作为名义标准。主要规则点：

- 球场由 40 mm 宽的线标出。
- 线应清晰可辨，最好为白色或黄色。
- 所有线都属于其定义的区域。
- 网柱高度为 1.55 m。
- 网柱放在双打边线上，不论比赛是单打还是双打。
- 球网由深色细绳制成，网孔不小于 15 mm 且不大于 20 mm。
- 球网深度为 760 mm，宽度至少 6.1 m。
- 球网上沿有 75 mm 白色网带。
- 网顶中心高度为 1.524 m，双打边线处高度为 1.55 m。
- Diagram A 标准场地可同时用于单打和双打；全场对角线为 14.723 m。

本包中的尺寸还采用 Diagram A 中的常用标准值：

| 项目 | 数值 |
|---|---:|
| 全场长度 | 13.40 m |
| 双打宽度 | 6.10 m |
| 单打宽度 | 5.18 m |
| 线宽 | 0.040 m |
| 短发球线距网 | 1.98 m |
| 双打长发球线距后边界 | 0.76 m |
| 网顶中心高度 | 1.524 m |
| 网顶双打边线处高度 | 1.550 m |
| 球网深度 | 0.760 m |
| 网带宽度 | 0.075 m |

---

## 2. 坐标系

### 2.1 世界坐标

本包采用以下坐标约定：

```text
x 轴：场地长度方向
y 轴：场地宽度方向
z 轴：竖直向上
原点：球网中心点在地面上的投影
球网：x = 0
场地表面：z = 0
正半场：x > 0
负半场：x < 0
中线：y = 0
```

俯视图：

```text
y ↑

          双打宽度 6.10 m
   -3.05 ┌──────────────────────────────┐ +3.05
         │                              │
         │        x < 0 半场             │
         │                              │
x = 0    ├──────────── NET ─────────────┤
         │                              │
         │        x > 0 半场             │
         │                              │
         └──────────────────────────────┘
        -6.70          x →           +6.70

全长 13.40 m
```

### 2.2 关键坐标

| 元素 | 坐标或范围 |
|---|---|
| 双打外边界 legal outer edge | `x = ±6.70`, `y = ±3.05` |
| 单打侧边界 legal outer edge | `y = ±2.59` |
| 短发球线 near edge | `x = ±1.98` |
| 双打长发球线 outer edge | `x = ±5.94` |
| 单打长发球线 | 后边界，即 `x = ±6.70` |
| 中线 | `y = 0`, 从短发球线延伸到后边界 |
| 球网 | `x = 0`, 横跨 `y = -3.05 ... +3.05` |
| 官方网柱位置 | 双打边线处，`y = ±3.05` |

---

## 3. 线宽语义：edge-correct 建模

BWF 规则中线宽为 40 mm，且线属于其定义区域。为了避免“视觉线宽导致场地多出 4 cm”的问题，本包采用 **edge-correct** 语义：

```text
官方尺寸 = 合法区域外缘尺寸
视觉线条 = 宽 40 mm 的 box，中心向场内偏移 20 mm
```

例如：

```text
双打外边界 y = +3.05 是合法区域外缘。
双打右边线视觉 box 中心在 y = +3.03。
该线的外缘在 y = +3.05，内缘在 y = +3.01。
```

同理：

| 线 | 合法外缘 | 视觉线中心 |
|---|---:|---:|
| 后边界正半场 | `x = +6.70` | `x = +6.68` |
| 后边界负半场 | `x = -6.70` | `x = -6.68` |
| 双打正 y 边线 | `y = +3.05` | `y = +3.03` |
| 双打负 y 边线 | `y = -3.05` | `y = -3.03` |
| 单打正 y 边线 | `y = +2.59` | `y = +2.57` |
| 单打负 y 边线 | `y = -2.59` | `y = -2.57` |
| 短发球线正半场 | near edge `x = +1.98` | `x = +2.00` |
| 短发球线负半场 | near edge `x = -1.98` | `x = -2.00` |
| 双打长发球线正半场 | outer edge `x = +5.94` | `x = +5.92` |
| 双打长发球线负半场 | outer edge `x = -5.94` | `x = -5.92` |

这对仿真很重要：落在线上应该被判为 inside，而不是 outside。实际接触仿真中，线条只是视觉几何，不参与碰撞；规则判定由 `src/court_geometry.py` 负责。

---

## 4. 场地线设计

### 4.1 需要绘制的线

| 线名 | MuJoCo geom name | 说明 |
|---|---|---|
| 双打正 y 边线 | `doubles_sideline_pos_y` | 双打右/上边界 |
| 双打负 y 边线 | `doubles_sideline_neg_y` | 双打左/下边界 |
| 单打正 y 边线 | `singles_sideline_pos_y` | 单打右/上边界 |
| 单打负 y 边线 | `singles_sideline_neg_y` | 单打左/下边界 |
| 正 x 后边界 | `back_boundary_pos_x` | 正半场后边界，也是单打长发球线 |
| 负 x 后边界 | `back_boundary_neg_x` | 负半场后边界，也是单打长发球线 |
| 正 x 短发球线 | `short_service_line_pos_x` | 正半场短发球线 |
| 负 x 短发球线 | `short_service_line_neg_x` | 负半场短发球线 |
| 正 x 双打长发球线 | `doubles_long_service_line_pos_x` | 正半场双打发球远端线 |
| 负 x 双打长发球线 | `doubles_long_service_line_neg_x` | 负半场双打发球远端线 |
| 正 x 半场中线 | `centre_service_line_pos_x_half` | 正半场发球区中线 |
| 负 x 半场中线 | `centre_service_line_neg_x_half` | 负半场发球区中线 |

### 4.2 线条是否参与碰撞

默认不参与碰撞：

```xml
contype="0"
conaffinity="0"
```

原因：

1. 真实场地线是油漆或平整贴线，不应该让羽毛球在 2 mm 几何台阶上弹起；
2. 落点合法性不是接触问题，而是规则判定问题；
3. MuJoCo 中把线做成 raised collision box 会让地面接触不稳定，尤其是羽毛球底座接触地面时。

---

## 5. 地板设计

默认地板为一个静态 box：

```text
floor_half_size_x = 7.70 m
floor_half_size_y = 4.05 m
floor_thickness   = 0.010 m
floor_z           = -0.005 m
```

因此可见/碰撞地板比标准场地每边多 1 m 缓冲：

```text
地板总尺寸 = 15.40 m × 8.10 m
标准双打场地 = 13.40 m × 6.10 m
```

地板参与碰撞，线条不参与碰撞。默认地板摩擦参数：

```xml
friction="1.0 0.005 0.0001"
condim="3"
```

对机器人足底、SMPL 足部、羽毛球落地、球拍落地都可以作为初始值。正式任务中可按鞋底、地胶或木地板接触重新标定。

---

## 6. 球网设计

### 6.1 官方尺寸

| 项目 | 数值 |
|---|---:|
| 网宽 | 至少 6.10 m |
| 网深 | 0.760 m |
| 网孔 | 15–20 mm |
| 网带宽度 | 75 mm |
| 网顶中心高度 | 1.524 m |
| 网顶双打边线处高度 | 1.550 m |
| 网柱高度 | 1.550 m |

### 6.2 网顶下垂模型

真实球网顶线会有轻微下垂。本包使用抛物线近似：

\[
z_\text{top}(y)=h_c + (h_p-h_c)\left(\frac{|y|}{3.05}\right)^2
\]

其中：

```text
h_c = 1.524 m
h_p = 1.550 m
y ∈ [-3.05, 3.05]
```

因此：

```text
z_top(0)       = 1.524 m
z_top(±3.05)   = 1.550 m
```

球网底部：

\[
z_\text{bottom}(y)=z_\text{top}(y)-0.760
\]

中心底部约为：

```text
1.524 - 0.760 = 0.764 m
```

### 6.3 网柱位置

规则要求网柱放在双打边线上，且支撑物不得伸入场内。MuJoCo 中圆柱体有半径，如果把圆柱中心直接放在 `y=±3.05`，几何体会伸进场内。因此本包采用：

```text
官方 post site: y = ±3.05
物理/视觉 post cylinder center: y = ±(3.05 + 0.035) = ±3.085
post radius = 0.035 m
```

也就是让圆柱内切边与双打合法外缘 `y=±3.05` 相切，避免支撑物伸入场地。

XML 中有 reference sites：

```xml
<site name="post_official_neg_y_site" pos="0 -3.05 0" .../>
<site name="post_official_pos_y_site" pos="0  3.05 0" .../>
```

---

## 7. 球网碰撞模型

包内提供两个 MJCF：

```text
assets/badminton_court_bwf_visual.xml
assets/badminton_court_bwf_collision_net.xml
```

### 7.1 visual-only 版本

`badminton_court_bwf_visual.xml`：

- 地板参与碰撞；
- 场地线 visual only；
- 球网 visual only；
- 网柱 visual only；
- 适合训练阶段、轨迹预测、无球网接触任务；
- 速度最快，接触最稳定。

### 7.2 collision-net 版本

`badminton_court_bwf_collision_net.xml`：

- 地板参与碰撞；
- 球网顶线用 capsule 参与碰撞；
- 球网主体用薄 box 分段代理参与碰撞；
- 场地线仍然不参与碰撞；
- 适合测试羽毛球是否触网、挂网、不过网等情况。

碰撞代理不是布料仿真，不会产生真实网线缠绕或柔性变形。对机器人控制/RL 来说，这是合理取舍。若需要真实挂网，可以在 Codex 后续任务中增加事件式 net-trap 判定。

---

## 8. 落点规则判定

`src/court_geometry.py` 提供规则 helper。

### 8.1 回合落点

```python
from court_geometry import CourtParams

court = CourtParams.from_json("params/court_bwf_nominal.json")

court.inside_rally(x, y, mode="doubles")
court.inside_rally(x, y, mode="singles")
```

语义：

```text
doubles:
  -6.70 <= x <= 6.70
  -3.05 <= y <= 3.05

singles:
  -6.70 <= x <= 6.70
  -2.59 <= y <= 2.59
```

由于线属于区域，边界值本身返回 `True`。

### 8.2 发球落点

```python
court.inside_service(
    x,
    y,
    mode="doubles",
    court_half="+x",
    lateral_half="+y",
)
```

参数：

```text
mode:
  "singles" 或 "doubles"

court_half:
  "+x" 表示接发球目标半场在正 x 半场
  "-x" 表示接发球目标半场在负 x 半场

lateral_half:
  "+y" 表示目标发球区在 y 正半边
  "-y" 表示目标发球区在 y 负半边
```

发球规则几何：

```text
短发球线 included:
  |x| >= 1.98

双打长发球线 included:
  |x| <= 5.94

单打长发球线 = 后边界 included:
  |x| <= 6.70

中心线 included:
  y=0 线可判入对应发球区
```

示例：

```python
# 正 x 半场、正 y 发球区，双打发球落点
court.inside_service(1.98, 0.01, "doubles", "+x", "+y")  # True
court.inside_service(5.94, 2.00, "doubles", "+x", "+y")  # True
court.inside_service(5.941, 2.00, "doubles", "+x", "+y") # False
```

---

## 9. 与前两个包的集成建议

你已有：

```text
shuttlecock_mujoco_design_package
badminton_racket_mujoco_design_package
```

推荐集成方式：

1. 使用本包的 `badminton_court_bwf_visual.xml` 作为训练默认场地。
2. 羽毛球初始位置使用世界坐标，例如：
   ```text
   serve start: x=-5.8, y=-1.5, z=1.0
   target half: x>0
   ```
3. 球拍/人体/机器人站位用 court helper 生成：
   ```text
   单打后场: x≈±5.5, y≈0
   双打发球: x≈±4.8, y≈±1.0
   ```
4. 回合落点判定使用 `inside_rally`，不要依赖场地边线碰撞。
5. 发球落点判定使用 `inside_service`。
6. 触网判定初版用 `collision_net` 版本；若接触不稳定，退回 visual-only 并用事件式几何判定。

---

## 10. MuJoCo 编译与引用方式

### 10.1 直接打开

```python
import mujoco

model = mujoco.MjModel.from_xml_path(
    "assets/badminton_court_bwf_visual.xml"
)
data = mujoco.MjData(model)
```

### 10.2 作为 include 合入总场景

你的主场景可以写：

```xml
<mujoco model="badminton_scene">
  <include file="assets/badminton_court_bwf_visual.xml"/>
  <!-- include shuttlecock, racket, humanoid/robot here -->
</mujoco>
```

实际使用时注意 MuJoCo include 路径相对主 XML 文件，而不是 Python 工作目录。Codex 可以把场地 XML 拆为 `<worldbody>` snippet，以适配你的项目结构。

---

## 11. 验证清单

运行：

```bash
python src/generate_court_mjcf.py
python src/validate_court_params.py
```

期望输出：

```text
PASS  full court length 13.40 m
PASS  doubles width 6.10 m
PASS  singles width 5.18 m
PASS  line width 40 mm
PASS  short service near edge at |x|=1.98 m
PASS  doubles long service outer edge at |x|=5.94 m
PASS  net centre height 1.524 m
PASS  net sideline height 1.550 m
...
All court design checks passed.
```

Codex 后续改动必须维持这些检查通过，除非明确切换到另一个坐标约定或线宽测量约定。

---

## 12. 文件结构

```text
badminton_court_mujoco_design_package/
├── badminton_court_design_dossier.md
├── assets/
│   ├── badminton_court_bwf_visual.xml
│   └── badminton_court_bwf_collision_net.xml
├── docs/
│   ├── codex_tasks.md
│   └── validation_protocol.md
├── params/
│   └── court_bwf_nominal.json
└── src/
    ├── court_geometry.py
    ├── generate_court_mjcf.py
    └── validate_court_params.py
```

---

## 13. 设计取舍

### 13.1 为什么不用真实 15–20 mm 网孔生成全部网线

真实球网如果按 20 mm 网孔生成：

```text
水平跨度 6.10 m / 0.02 m ≈ 305 列
垂直深度 0.76 m / 0.02 m ≈ 38 行
```

仅网线视觉就会带来数百个 capsule。可视化还可以接受，但碰撞非常没有必要。羽毛球触网的关键不是每个网孔的拓扑，而是：

1. 是否过网；
2. 是否触到网带；
3. 是否被网面阻挡；
4. 是否发生挂网/缠网事件。

因此 XML 默认使用稀疏视觉网格，参数 JSON 保留官方 15–20 mm 网孔规格。需要更真实外观时，让 Codex 调整 `net_visual_mesh_pitch` 或用 mesh 材质贴图。

### 13.2 为什么球网不是柔性布料

MuJoCo 不适合直接把标准球网做成大规模柔性细绳网络并进行高速羽毛球接触。初版采用：

```text
visual cords + thin collision proxy + top cord capsule
```

如果要模拟挂网，可以额外写事件式逻辑：

```text
if shuttle crosses x=0 and z is near net plane and velocity is low:
    classify as net caught / let / fault depending on context
```

### 13.3 为什么场地线不碰撞

场地线是判定区域，不是物理障碍。碰撞应由地板统一处理；落点规则由 `court_geometry.py` 处理。

---

## 14. 建议 Codex 后续任务

见 `docs/codex_tasks.md`。最重要的几个任务：

1. 把本场地 XML 和前两个包的羽毛球、球拍合并成单一 scene。
2. 给 shuttlecock 添加落点事件：当软木球头接触地板时记录 `x,y` 并调用 `inside_rally`。
3. 给发球任务添加 `inside_service` 判定。
4. 给球网添加事件式触网判定，避免依赖复杂软网接触。
5. 生成可视化 overlay：当前合法区域、目标服务区、羽毛球落点。
