# Q:球拍拍柄参数以及正确的我握拍

# A:
羽毛球的标准正手握拍通常也叫 **Basic / V grip** 或 **handshake grip**。BWF Shuttle Time 把它描述为基础 “V” 握拍：拇指和食指在拍柄上形成一个 V 形，主要用于正手侧、球与身体大致同高的击球。([Shuttle Time][1]) 很多教学材料也会用“像和球拍握手”来说明：拍面竖直，手从侧面自然握上去，拍柄松放在手指里，而不是死死攥住。([www.sportplan.net][2])

下面给你的是 **标准动作描述 + 可用于 MuJoCo/Codex 的数值目标**。

---

# 1. 标准正手握拍姿势

## 核心姿势

对右手持拍来说，标准正手握拍应满足：

```text
1. 像和球拍握手，而不是像拿平底锅。
2. 拇指和食指之间形成明显 V 形。
3. V 形虎口落在拍柄的斜棱/侧棱附近，而不是正压在拍面宽面上。
4. 拇指自然贴在侧斜面，不要像反手握拍那样平压在宽面后侧。
5. 食指略微分开、稍微靠前，形成“扳机感”。
6. 中指、无名指、小指自然环绕拍柄。
7. 小鱼际/掌根轻贴拍柄尾侧，但不要把整个手掌完全压死。
8. 平时握力较松，击球瞬间再收紧。
```

“松握、击球瞬间收紧”是很重要的，因为正手高远球、杀球、吊球等主要依赖前臂旋前、手腕/手指加速，而不是一直用最大握力锁死球拍。教学资料通常也把正手握拍用于 clears、smashes、drops 和多数 overhead forehand shots。([Badminton Coach - Petr Dubský][3])

---

# 2. 最容易犯错的姿势

## 错误 1：Panhandle / 平底锅握法

这个错误是把球拍像锅柄一样握住，导致拍面正对手掌方向。结果是：

```text
拍面方向容易固定死
前臂旋前发力受限
高远球和杀球力量不足
手腕容易代偿
```

在仿真里表现为：手掌主要压在拍柄的 `+Z/-Z` 宽面，而不是从侧面握住。

## 错误 2：反手拇指握法误用到正手

反手 thumb grip 是拇指平压在拍柄后侧宽面，主要用于身体前方的反手网前、挑球、反手发球、反手平抽等；它不是标准正手握拍。([羽毛球圣经][4]) 正手握拍时，拇指应自然斜放在侧斜面，而不是把拇指完全伸直压在宽面上。

## 错误 3：拳头式死握

所有手指并排紧紧攥住，会让手腕、前臂和手指失去弹性。仿真里如果只奖励“接触力大”，模型很容易学成死握，所以 reward 要惩罚过大的持续握力和关节极限锁死。

---

# 3. 用于 MuJoCo 的右手正手握拍坐标定义

沿用你前面球拍建模的坐标系：

```text
racket local +Y : 从拍柄尾端 butt cap 指向拍头
racket local +Z : 拍面法向
racket local +X : 拍面横向/侧向

butt cap center = [0, 0, 0]
handle axis     = +Y
handle radius   = 0.0132 m    # G5 成品拍柄等效半径
```

我建议在仿真中把 **右手正手 V grip 的虎口方向**设为：

```text
V-bisector direction ≈ local +X
palm heel direction  ≈ local -X
```

也就是说，手不是从拍面正后方包住拍柄，而是从拍柄侧面像握手一样握住。这样能避免模型学成 panhandle。

---

# 4. 标准正手握拍的接触目标点

用圆柱/八边形表面参数化：

```python
x = r * cos(theta)
z = r * sin(theta)
y = target_y
```

其中：

```text
r = 0.0132 m
theta = 0°   表示 local +X 方向
theta = 90°  表示 local +Z 方向
theta = 180° 表示 local -X 方向
theta = -90° 表示 local -Z 方向
```

推荐右手正手握拍目标：

```text
palm / 掌根:
  y     = 0.080 - 0.095 m
  theta = 180°
  说明  = 掌根、小鱼际在拍柄 -X 侧，提供支撑

thumb / 拇指:
  y     = 0.112 - 0.128 m
  theta = 35° - 55°
  说明  = 拇指斜贴侧斜面，不平压宽面

index / 食指:
  y     = 0.118 - 0.135 m
  theta = -35° - -55°
  说明  = 食指略靠前，与拇指形成 V 形

middle / 中指:
  y     = 0.092 - 0.105 m
  theta = -105° - -125°
  说明  = 主闭合手指之一，包住下侧

ring / 无名指:
  y     = 0.068 - 0.082 m
  theta = -125° - -145°
  说明  = 辅助稳定拍柄

pinky / 小指:
  y     = 0.045 - 0.060 m
  theta = -140° - -165°
  说明  = 靠近拍柄尾部，防止滑脱
```

名义值可以直接用：

```json
{
  "right_hand_forehand_v_grip_targets": {
    "coordinate_convention": {
      "racket_handle_axis": "+Y",
      "racket_face_normal": "+Z",
      "racket_lateral_axis": "+X",
      "surface_parameterization": "x = r*cos(theta), z = r*sin(theta)"
    },
    "handle": {
      "radius_m": 0.0132,
      "finished_grip_size": "G5",
      "usable_grip_start_y_m": 0.012,
      "usable_grip_end_y_m": 0.160
    },
    "forehand_grip_definition": {
      "v_shape_bisector_theta_deg": 0,
      "palm_support_theta_deg": 180,
      "avoid_panhandle_note": "Do not place the palm directly behind the racket face normal.",
      "avoid_thumb_grip_note": "Do not place the thumb flat on the wide back bevel as in backhand thumb grip."
    },
    "contact_targets": {
      "palm_heel": {
        "y_m": 0.086,
        "theta_deg": 180,
        "weight": 1.0
      },
      "thumb_pad": {
        "y_m": 0.122,
        "theta_deg": 45,
        "weight": 1.2
      },
      "index_pad": {
        "y_m": 0.128,
        "theta_deg": -45,
        "weight": 1.2
      },
      "middle_pad": {
        "y_m": 0.098,
        "theta_deg": -115,
        "weight": 1.0
      },
      "ring_pad": {
        "y_m": 0.075,
        "theta_deg": -135,
        "weight": 0.8
      },
      "pinky_pad": {
        "y_m": 0.055,
        "theta_deg": -150,
        "weight": 0.8
      }
    }
  }
}
```

---

# 5. 手部关节姿态先验

这个不是比赛规则，而是给 IK/RL 的实用先验。建议 Codex 在 IK objective 里加入：

```text
wrist:
  extension       = 5° - 20°
  ulnar deviation = 0° - 15°
  pronation       = 根据整臂姿态决定，不在手部单独硬锁死

thumb:
  CMC opposition/abduction = 中等
  MCP flexion              = 15° - 35°
  IP flexion               = 5° - 25°
  不要完全伸直压在拍柄后侧

index:
  MCP flexion = 35° - 60°
  PIP flexion = 35° - 70°
  DIP flexion = 10° - 35°
  与中指留小间隙，略靠前

middle:
  MCP flexion = 45° - 70°
  PIP flexion = 45° - 80°
  DIP flexion = 15° - 40°

ring:
  MCP flexion = 50° - 75°
  PIP flexion = 50° - 85°
  DIP flexion = 15° - 45°

pinky:
  MCP flexion = 55° - 80°
  PIP flexion = 55° - 90°
  DIP flexion = 20° - 50°
```

仿真里不要强制每个角度固定到一个值。更合理的是：

```text
强约束：手指 pad 接近拍柄目标点
中约束：关节处在舒适范围
弱约束：保持 V 形、食指略靠前、握力不过大
```

---

# 6. 正手握拍 reward / loss 设计

给 Codex 实现时，可以这样定义目标：

```text
grip_site_error:
  thumb_pad  -> thumb_target
  index_pad  -> index_target
  middle_pad -> middle_target
  ring_pad   -> ring_target
  pinky_pad  -> pinky_target
  palm_heel  -> palm_target

v_shape_reward:
  thumb-index V 的夹角保持在 35° - 70°
  thumb 与 index 的 y 差不超过 20 mm
  index 稍微比 thumb 靠拍头方向，允许 +0 到 +15 mm

anti_panhandle_penalty:
  palm normal 不应与 racket face normal 过度对齐
  手掌主要支撑方向应接近 -X，而不是 -Z 或 +Z

anti_thumb_grip_penalty:
  thumb pad normal 不应完全平压在拍柄宽面后侧
  thumb 长轴不要与 handle +Y 完全共线

contact_reward:
  有效接触数 >= 4
  掌根、拇指、食指、中指至少 4 处接触

slip_penalty:
  拍柄相对手掌沿 +Y/-Y 滑动 < 5-10 mm/s
  角滑移 < 5 deg/s

effort_penalty:
  避免持续最大肌肉激活
  避免死握
```

---

# 7. Codex 可执行的建模说明

你可以把下面这段直接给 Codex：

```text
Implement a right-hand badminton forehand V-grip target for the existing musculoskeletal hand and racket model.

Assume racket local coordinates:
+Y = handle axis from butt cap to racket head
+Z = racket face normal
+X = racket lateral axis
handle radius = 0.0132 m
butt cap center = [0, 0, 0]

The correct right-hand forehand grip is a handshake / basic V grip:
- thumb and index form a V shape
- V-shape bisector is near local +X
- palm heel supports the handle from local -X
- thumb lies on the +X/+Z diagonal bevel, not flat on the back wide face
- index lies on the +X/-Z diagonal bevel and is slightly forward
- middle, ring, and pinky wrap around the underside
- avoid panhandle grip
- avoid backhand thumb grip
- keep grip relaxed, not a fist grip

Create target sites on the racket handle surface using:
x = r*cos(theta), z = r*sin(theta), y = target_y

Use these nominal targets:
palm_heel:  y=0.086, theta=180 deg
thumb_pad:  y=0.122, theta=45 deg
index_pad:  y=0.128, theta=-45 deg
middle_pad: y=0.098, theta=-115 deg
ring_pad:   y=0.075, theta=-135 deg
pinky_pad:  y=0.055, theta=-150 deg

Add IK / RL losses:
1. minimize fingertip and palm site distance to targets
2. reward at least 4 stable contacts
3. penalize relative handle slip
4. penalize panhandle orientation
5. penalize thumb-grip orientation
6. keep joint angles away from hard limits
7. keep muscle activation moderate
8. allow light grip in free motion and stronger grip near impact
```

---

# 8. 验收标准

正手握拍姿势可以这样判定 PASS：

```text
视觉：
  拇指和食指形成 V 形
  食指略靠前，不和其他手指完全并排
  拇指没有平压在拍柄后侧宽面
  手掌不是正对拍面握成 panhandle
  中指、无名指、小指自然包住拍柄

数值：
  thumb/index/palm target error < 20 mm
  all finger target mean error  < 25 mm
  stable contact count >= 4
  handle longitudinal slip over 2 s < 10 mm
  racket orientation drift over 2 s < 8°
  no persistent joint-limit locking
  no excessive constant grip force
```

对于你的肌骨模型，最推荐的默认目标就是：**右手 forehand V grip，虎口朝 local +X，掌根在 local -X，拇指 45°、食指 -45°，其余手指从 -115° 到 -150° 包住拍柄**。这套参数和之前的 G5 拍柄、八边形接触面、右手握拍训练 pipeline 能直接衔接。

[1]: https://shuttletime.bwfbadminton.com/activity-of-the-month/l1v4-forehand-basic-v-grip-with-balloon?utm_source=chatgpt.com "May-Forehand Basic “V” Grip with Balloon – BWF Shuttle Time"
[2]: https://www.sportplan.net/drills/Badminton/Grip/Forehand-Grip-Grip.jsp?utm_source=chatgpt.com "Forehand Grip Grip - Badminton Drills, Badminton | Sportplan"
[3]: https://badminton.coach/blog/badminton-racket-grip/?lang=en&utm_source=chatgpt.com "Badminton Racket Grip: Proper Grip Step by Step"
[4]: https://www.badmintonbible.com/shots/grips/basics/thumb?utm_source=chatgpt.com "Thumb grip | Badminton Bible"


下面是我建议你用于 **MuJoCo 球拍柄 / 右手握拍训练** 的拍柄参数集。重点是：**比赛规则并没有规定拍柄必须多长、多粗，只定义它是球拍中供运动员握持的部分**；BWF 规则主要限制整支球拍总长 ≤ 680 mm、总宽 ≤ 230 mm，并定义 handle/stringed area/head/shaft/throat 等部件。([羽毛球大师系统][1]) 所以拍柄建模应以真实商业球拍的常见 grip size 和人体抓握稳定性为准。

## 1. 推荐默认值：用于你的 MuJoCo 球拍柄

我建议把你当前球拍模型的拍柄设为：

```text
handle_axis                  = local +Y
butt_center                  = [0.0, 0.0, 0.0]
usable_grip_start_y          = 0.012 m
usable_grip_end_y            = 0.160 m
usable_grip_length           = 0.148 m

handle_total_length          = 0.175 m
butt_cap_length              = 0.010 m
cone_transition_length       = 0.030 m

finished_grip_size           = G5
finished_circumference       = 0.083 m
equivalent_circular_radius   = 0.01321 m
equivalent_circular_diameter = 0.02642 m

octagon_across_flats         = 0.02505 m
octagon_across_corners       = 0.02711 m
octagon_side_length          = 0.01038 m

handle_contact_radius        = 0.0132 m
visual_handle_radius         = 0.0132 m
butt_cap_radius              = 0.0160 m
cone_base_radius             = 0.0140 m
cone_top_radius              = 0.0070 m

handle_mass_estimate         = 0.018 kg
grip_wrap_mass_estimate      = 0.006 kg
butt_cap_mass_estimate       = 0.003 kg
handle_assembly_mass         = 0.027 kg
```

其中 **G5，83 mm 周长** 是我最推荐的仿真默认值。很多现代羽毛球拍常见 G4/G5，G5 更利于手指控制，也方便后续加 overgrip 或 towel grip；公开尺码表通常把 G5 约为 83 mm、G4 约为 86 mm、G3 约为 89 mm。([Toby's Sports][2])

---

## 2. Grip size 对照表

注意：不同品牌的 G 码不是严格全球标准，但 Yonex-style / 常见零售标法大致如下。实际建模最好用 **finished circumference in mm**，不要只用 G4/G5 标签。

| Grip size |    成品握柄周长 |        等效圆直径 | 正八边形 across flats | 正八边形 across corners | 建模建议              |
| --------: | --------: | -----------: | ----------------: | ------------------: | ----------------- |
|        G6 |     80 mm |     25.46 mm |          24.14 mm |            26.13 mm | 小手、快变拍、常加外握把      |
|    **G5** | **83 mm** | **26.42 mm** |      **25.05 mm** |        **27.11 mm** | **推荐默认值**         |
|        G4 |     86 mm |     27.37 mm |          25.95 mm |            28.09 mm | 成年人常用，握感更满        |
|        G3 |     89 mm |     28.33 mm |          26.86 mm |            29.07 mm | 偏粗，稳定但手指动作慢       |
|        G2 |     92 mm |     29.28 mm |          27.76 mm |            30.05 mm | 大手或特殊需求           |
|        G1 |     95 mm |     30.24 mm |          28.67 mm |            31.03 mm | 很粗，羽毛球里不常作为灵活打法默认 |

换算公式：

```text
P = 成品握柄周长

等效圆半径:
r_circle = P / (2π)

正八边形边长:
s = P / 8

正八边形 across flats:
D_flat = P / (8 * tan(π/8))

正八边形 across corners:
D_corner = P / (8 * sin(π/8))
```

---

## 3. 拍柄外层材料参数

如果你希望“手真正学会握拍”，不要把拍柄建成纯硬木或纯圆柱。建议至少区分三层：

```text
wood_or_core_handle:
  shape: octagonal prism
  circumference_before_grip: 0.075 - 0.080 m
  material: light wood / foam / composite proxy
  collision: usually disabled or inner only

base_grip:
  thickness: 0.0010 - 0.0015 m
  material: PU / synthetic grip
  collision: enabled as final contact surface

optional_overgrip:
  thickness: 0.0006 m
  material: polyurethane
  effect: increases radius by ~0.6 mm before compression
```

Yonex Super Grap 的官方规格为宽 25 mm、长 1200 mm、厚 0.6 mm、PU 材料；毛巾握把 AC402 的官方规格为宽 30/32 mm、长 660 mm、厚 1.35 mm、棉材料。([Yonex USA][3])

对 MuJoCo 来说，推荐把最终可接触表面简化为一个 **G5 成品握柄**：

```xml
<geom name="racket_handle_contact"
      type="capsule"
      fromto="0 0.012 0  0 0.160 0"
      size="0.0132"
      rgba="0.08 0.08 0.08 1"
      contype="1"
      conaffinity="1"
      friction="1.4 0.03 0.003"
      solref="0.004 1"
      solimp="0.90 0.97 0.002"/>
```

但如果你希望右手能感知拍面方向，最好不要只用圆柱。建议用 **八边形 mesh** 或 8 个细长 box 近似 bevel。圆柱会让模型少了“换握/定位拍面”的触觉线索。

---

## 4. 用于握拍姿势学习的关键坐标

延续前面球拍档案的坐标系：

```text
球拍 local +Y : 从拍柄尾端指向拍头
球拍 local +Z : 拍面法向
球拍 local +X : 拍面横向
butt cap center: [0, 0, 0]
```

建议定义这些 site：

```text
racket_butt_site             = [0.000, 0.000, 0.000]
racket_grip_center_site      = [0.000, 0.085, 0.000]
racket_grip_upper_site       = [0.000, 0.135, 0.000]
racket_grip_lower_site       = [0.000, 0.045, 0.000]

racket_handle_axis_site      = [0.000, 0.120, 0.000]
racket_face_normal_site      = [0.000, 0.085, 0.030]
racket_bevel_top_site        = [0.000, 0.095, 0.0132]
racket_bevel_bottom_site     = [0.000, 0.095, -0.0132]
racket_bevel_left_site       = [-0.0132, 0.095, 0.000]
racket_bevel_right_site      = [ 0.0132, 0.095, 0.000]
```

对于右手握拍，推荐默认接触目标：

```text
palm_contact_target:
  y = 0.085
  theta = 180 deg

thumb_contact_target:
  y = 0.120
  theta = 45 deg

index_contact_target:
  y = 0.125
  theta = -45 deg

middle_contact_target:
  y = 0.098
  theta = -115 deg

ring_contact_target:
  y = 0.075
  theta = -135 deg

pinky_contact_target:
  y = 0.055
  theta = -150 deg
```

圆柱/八边形表面点换算：

```python
x = handle_radius * cos(theta)
z = handle_radius * sin(theta)
y = target_y
```

推荐半径：

```text
handle_radius = 0.0132 m   # G5 equivalent circular radius
```

---

## 5. 物理接触参数

用于手掌/手指 pad 与拍柄接触：

```text
contact_surface             = PU / synthetic grip proxy
tangential_friction_mu       = 1.2 - 1.8
torsional_friction           = 0.02 - 0.05
rolling_friction             = 0.001 - 0.005

solref                       = 0.004 1
solimp                       = 0.90 0.97 0.002
margin                       = 0.001 - 0.002 m
condim                       = 4 or 6
```

我建议训练初期用：

```xml
<default class="racket_handle_contact">
  <geom friction="1.4 0.03 0.003"
        condim="4"
        solref="0.004 1"
        solimp="0.90 0.97 0.002"
        margin="0.001"/>
</default>
```

如果手滑太严重：

```text
increase tangential_friction_mu: 1.4 -> 1.8
increase torsional_friction:     0.03 -> 0.05
slightly increase contact margin
```

如果接触抖动：

```text
increase solref time constant: 0.004 -> 0.006
reduce timestep: 0.001 -> 0.0005
make finger pads softer / larger
avoid extremely sharp octagon edges
```

---

## 6. 拍柄惯性参数

如果球拍是一个整体刚体，拍柄惯性不需要单独设置，只要整支拍子的总质量、质心、挥重正确即可。

如果你把拍柄建成独立 body，可以用这个默认值：

```text
handle_assembly_mass = 0.027 kg
length               = 0.175 m
radius_equiv          = 0.0132 m

I_axis_along_handle ≈ 2.35e-6 kg·m²
I_cross_axis         ≈ 6.96e-5 kg·m²
```

对应 MuJoCo 近似：

```xml
<inertial pos="0 0.0875 0"
          mass="0.027"
          diaginertia="6.96e-5 2.35e-6 6.96e-5"/>
```

这里假设拍柄轴是 local `+Y`，所以 `Iyy` 是沿拍柄轴的转动惯量。这个只是拍柄组件级默认值；如果你的球拍整体已经有正确 `mass / com / inertia`，不要重复加质量。

---

## 7. 建议给 Codex 的 JSON 参数块

可以直接放进：

```text
configs/racket_handle_params.json
```

```json
{
  "coordinate_convention": {
    "origin": "butt_cap_center",
    "handle_axis": "+Y",
    "racket_face_normal": "+Z",
    "racket_lateral_axis": "+X"
  },
  "handle_geometry": {
    "grip_size_label": "G5",
    "finished_circumference_m": 0.083,
    "equivalent_circular_radius_m": 0.01321,
    "equivalent_circular_diameter_m": 0.02642,
    "octagon_across_flats_m": 0.02505,
    "octagon_across_corners_m": 0.02711,
    "octagon_side_length_m": 0.01038,
    "usable_grip_start_y_m": 0.012,
    "usable_grip_end_y_m": 0.160,
    "usable_grip_length_m": 0.148,
    "handle_total_length_m": 0.175,
    "butt_cap_length_m": 0.010,
    "butt_cap_radius_m": 0.016,
    "cone_transition_length_m": 0.030,
    "cone_base_radius_m": 0.014,
    "cone_top_radius_m": 0.007
  },
  "handle_layers": {
    "core_circumference_m": 0.077,
    "base_grip_thickness_m": 0.0012,
    "overgrip_thickness_m": 0.0006,
    "finished_contact_radius_m": 0.01321,
    "outer_material": "PU_overgrip_proxy"
  },
  "mass_properties_if_separate_body": {
    "handle_assembly_mass_kg": 0.027,
    "center_of_mass_local_m": [0.0, 0.0875, 0.0],
    "diagonal_inertia_kg_m2": [0.0000696, 0.00000235, 0.0000696]
  },
  "contact": {
    "geom_type_recommended": "octagonal_mesh_or_8_bevel_proxy",
    "fallback_geom_type": "capsule",
    "tangential_friction": 1.4,
    "torsional_friction": 0.03,
    "rolling_friction": 0.003,
    "condim": 4,
    "solref": [0.004, 1.0],
    "solimp": [0.90, 0.97, 0.002],
    "margin_m": 0.001
  },
  "grip_target_sites": {
    "grip_center": [0.0, 0.085, 0.0],
    "grip_lower": [0.0, 0.045, 0.0],
    "grip_upper": [0.0, 0.135, 0.0],
    "face_normal_marker": [0.0, 0.085, 0.030],
    "bevel_top": [0.0, 0.095, 0.01321],
    "bevel_bottom": [0.0, 0.095, -0.01321],
    "bevel_left": [-0.01321, 0.095, 0.0],
    "bevel_right": [0.01321, 0.095, 0.0]
  },
  "right_hand_default_contact_targets": {
    "palm": {
      "y_m": 0.085,
      "theta_deg": 180
    },
    "thumb": {
      "y_m": 0.120,
      "theta_deg": 45
    },
    "index": {
      "y_m": 0.125,
      "theta_deg": -45
    },
    "middle": {
      "y_m": 0.098,
      "theta_deg": -115
    },
    "ring": {
      "y_m": 0.075,
      "theta_deg": -135
    },
    "pinky": {
      "y_m": 0.055,
      "theta_deg": -150
    }
  }
}
```

---

## 8. 给 Codex 的实现建议

让 Codex 不要只做圆柱拍柄。最好的顺序是：

```text
1. 先保留 capsule handle 作为稳定 contact fallback。
2. 再生成 octagonal visual/contact mesh。
3. 给 octagon 8 个 bevel sites，帮助右手感知拍面方向。
4. 训练初期可以用 soft target sites。
5. 后期奖励以真实接触、低滑移、拍面方向稳定为主。
```

最终验收标准可以设为：

```text
hand-handle penetration       < 3 mm
stable contact count          >= 4
racket grip drift over 2 s     < 10 mm
racket orientation drift       < 8 deg
fingertip average target error < 20 mm
no persistent joint limit lock
```

对你的肌骨模型来说，**G5、83 mm 成品周长、八边形或八边形近似接触、PU 高摩擦表面** 是最合适的默认拍柄参数。

[1]: https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf?utm_source=chatgpt.com "Laws - BadmintonBWF Statutes, Section 4.1<table>"
[2]: https://www.tobys.com/pages/yonex-racquet-size-chart?utm_source=chatgpt.com "Yonex Racquet Size Chart – Toby's Sports"
[3]: https://www.yonex.com/ac102?utm_source=chatgpt.com "Wet Super Grap | Grip Tape | Yonex"
