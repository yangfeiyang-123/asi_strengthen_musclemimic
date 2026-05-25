# MuJoCo 真实羽毛球拍仿真设计档案

版本：1.0  
目标对象：现代成人竞技羽毛球拍，优先适配 MuJoCo、机器人、SMPL/人体动作、强化学习和你之前的 `shuttlecock_mujoco_design_package`。  
核心思想：**外观上尽量像真实羽毛球拍；动力学上优先匹配尺寸、质量、平衡点、挥重、线床刚度、击球响应，而不是在 MuJoCo 中做完整碳纤维层合板 + 每根线的有限元。**

---

## 1. 设计结论

本包提供两个可用层级：

1. `assets/badminton_racket_rigid.xml`  
   推荐默认模型。单刚体球拍，带 freejoint、真实尺寸、手柄、杆、喉部、椭圆头框、22x21 可视化线床。球拍框/手柄可碰撞，线床默认不碰撞，线床击球由 `src/racket_stringbed.py` 施加代理力。

2. `assets/badminton_racket_flex_proxy.xml`  
   可选柔性代理模型。手柄/下杆和头框/上杆分成两个 body，中间有两个带刚度/阻尼的 hinge joint，用来近似杆身弯曲和回弹。它不是真实复合材料 FEM，但可让机器人挥拍时出现可调的头框滞后和恢复。

默认名义参数：

```text
总长                         0.675 m
总宽                         0.210 m
线床长                       0.253 m
线床宽                       0.188 m
质量                         0.090 kg
平衡点 / 质心距拍柄尾端       0.310 m
挥重轴                       距拍柄尾端 0.090 m，轴向 local +X
目标挥重                     93.0 kg·cm² = 0.00930 kg·m²
Ixx, Iyy, Izz about COM       0.004944, 0.000180, 0.004900 kg·m²
穿线模式                     22 mains × 21 crosses
名义线径                     0.70 mm
名义张力                     26 lbf ≈ 115.65 N
线床中心法向刚度             9600 N/m
5 mm 静态中心压入反力         48 N
推荐 timestep                0.0005 s
推荐 integrator              implicitfast
```

---

## 2. 依据和约束

### 2.1 BWF / 规则尺寸约束

本模型遵守常规羽毛球拍规则约束：

```text
球拍整体长度 <= 680 mm
球拍整体宽度 <= 230 mm
线床整体长度 <= 280 mm
线床整体宽度 <= 220 mm
如线床延伸进入喉部，延伸区宽度 <= 35 mm，线床总长 <= 330 mm
线床必须是平面交叉线 pattern，中心不能比其他区域更稀疏
```

本设计取值：

```text
整体长度 = 675 mm，满足 <= 680 mm
整体宽度 = 210 mm，满足 <= 230 mm
线床长度 = 253 mm，满足 <= 280 mm
线床宽度 = 188 mm，满足 <= 220 mm
```

### 2.2 真实/商用球拍参考

现代竞技球拍常见规格大致为：

```text
长度：约 665–675 mm，部分“加长 10 mm”球拍接近 675 mm
重量：约 4U 83 g、3U 88 g，穿线和手胶后通常进入 85–95 g 区间
张力：高端球拍常见建议约 20–29 lbf
穿线模式：高端 Yonex 示例为 22 × 21
```

本设计不复制某一品牌 CAD，而是构造一个**规则合规、质量和挥重接近真实竞技拍**的通用模型。

### 2.3 研究参考约束

公开研究中，羽毛球拍挥重常以距拍柄尾端约 9 cm 的轴定义。现代羽毛球拍典型质量约 0.085–0.095 kg，典型挥重约 90–97 kg·cm²。球拍击球研究还指出，挥重会影响拍头速度、击球位置和杆身变形贡献，因此在仿真中比“单纯质量”更重要。

公开有限元研究使用了商业羽毛球拍头几何：头部长约 253 mm、宽约 188 mm；线径 0.7 mm；尼龙线材料参数约 `E = 7200 MPa, ν = 0.3, ρ = 1100 kg/m³`。该研究以 26 lbf 线张力、5 mm 压入作为线床碰撞测试，中心击球反力约 47–48 N。因此本包把线床中心静态法向刚度设为：

```text
k_center = 48 N / 0.005 m = 9600 N/m
```

这是代理模型的第一标定点。

---

## 3. 坐标系约定

球拍 body 局部坐标：

```text
origin  = 拍柄尾端 / butt cap 中心
+X      = 线床横向，左右方向
+Y      = 从拍柄尾端指向拍头顶端
+Z      = 垂直线床平面的法向
```

重要 site：

```text
grip_pose_site          pos = [0, 0.090, 0]
                        推荐作为机器人手或 SMPL 手部附着/控制点

butt_site               pos = [0, 0, 0]
                        拍柄尾端

stringbed_center_site   pos = [0, 0.532, 0]
                        线床中心，击球代理平面中心

head_tip_site           pos = [0, 0.675, 0]
                        拍头顶端
```

### 3.1 和 SMPL/机器人手的对接建议

若手部模型有右手掌 body，可以使用两种方式：

1. **freejoint + mocap weld**：把 `grip_pose_site` 对齐到手掌/虎口目标位姿，由 mocap body 或控制器驱动；
2. **equality weld 到手部 body**：把球拍 body 焊接到手部 body，然后让手部关节/根节点驱动球拍。

推荐附着点不是 butt cap，而是 `grip_pose_site`，因为真实握拍的主受力点通常在拍柄尾端往上约 70–110 mm 区间。

---

## 4. 几何设计

### 4.1 分段

```text
拍柄 / handle:
  y = 0.010 到 0.170 m
  radius = 0.014 m
  butt cap radius = 0.018 m

拍杆 / shaft:
  y = 0.170 到 0.385 m
  radius = 0.0045 m

喉部 / throat:
  从 [0, 0.385, 0] 分叉到 [±0.052, 0.407, 0]
  radius ≈ 0.0045 m

头框 / head frame:
  外椭圆中心 y = 0.532 m
  外半宽 = 0.105 m
  外半长 = 0.143 m
  外框 capsule 半径 = 0.006 m
  顶点 y = 0.675 m
  下缘 y = 0.389 m

线床 / string bed:
  椭圆中心 y = 0.532 m
  半宽 = 0.094 m
  半长 = 0.1265 m
  总宽 = 0.188 m
  总长 = 0.253 m
```

### 4.2 外观实现

`assets/badminton_racket_rigid.xml` 中：

```text
手柄      sphere + capsule
拍杆      capsule
喉部      2 根 capsule
头框      40 段 capsule 近似椭圆
主线      22 根 capsule，沿 +Y
横线      21 根 capsule，沿 +X
代理线床  透明 box，仅可视化，默认 contype=0 conaffinity=0
```

线床视觉线全部禁用碰撞：

```xml
contype="0" conaffinity="0"
```

原因：在 MuJoCo 中让几十根细线与羽毛球软木头部高速接触会产生大量接触点、强刚度和不稳定性，而且真实线床的能量吸收来自整体膜/网格变形，不应由每根 capsule 的局部刚性碰撞直接替代。

---

## 5. 质量、平衡点和挥重

### 5.1 目标

```text
总质量 m = 0.090 kg
质心位置 y_com = 0.310 m
挥重轴 y_axis = 0.090 m
目标挥重 I_s = 93 kg·cm² = 0.00930 kg·m²
```

### 5.2 MuJoCo inertial

MJCF 中：

```xml
<inertial pos="0 0.31 0"
          mass="0.09"
          diaginertia="0.004944 0.00018 0.0049"/>
```

### 5.3 挥重验证公式

在本坐标中，典型挥重对应距拍柄尾端 90 mm、轴向 local +X 的转动惯量：

```text
I_s = Ixx_COM + m * (y_com - y_axis)^2
```

代入：

```text
I_s = 0.004944 + 0.090 * (0.310 - 0.090)^2
    = 0.004944 + 0.004356
    = 0.009300 kg·m²
    = 93.0 kg·cm²
```

这比只设置总质量更关键，因为机器人/人体挥拍时的“拍头重不重”主要由这个量决定。

---

## 6. 线床动力学代理

### 6.1 为什么不用每根线的真实接触

真实线床是带预张力的交叉绳网。完整建模需要：

```text
几十根带预张力的柔性索/梁
线-线交叉摩擦
线-框固定约束
线-球头接触
高速 1–2 ms 量级碰撞
拍杆弯曲和回弹
```

这在 FEM 或专门 rope/rod solver 中可做，但在 MuJoCo 里对 RL/人体仿真不划算。推荐采用：

```text
视觉上：22×21 线床 capsule
动力学上：一个椭圆线床平面 + 位置相关 spring-damper + 可选 event rebound
```

### 6.2 椭圆线床内点判定

对羽毛球软木接触点 `p`，先转到球拍局部坐标：

```text
p_local = R_racket^T * (p_world - p_racket_origin)
```

线床归一化半径：

```text
rho² = (x / a)² + ((y - y_center) / b)²
```

其中：

```text
a = 0.094 m
b = 0.1265 m
y_center = 0.532 m
```

若 `rho² <= 1`，接触点投影在线床内部。

### 6.3 法向穿透

```text
signed_z = p_local.z
penetration = cork_radius + stringbed_proxy_thickness - |signed_z|
```

默认：

```text
cork_radius = 0.0135 m
stringbed_proxy_thickness = 0.0015 m
```

若 `penetration > 0`，认为软木头部进入线床代理区域。

### 6.4 位置相关刚度

中心刚度：

```text
k_center = 9600 N/m
```

边缘击球的有效线长更短，刚度更高，所以使用：

```text
k(rho²) = k_center * min(1 + edge_gain * rho², max_multiplier)
edge_gain = 1.20
max_multiplier = 2.50
```

### 6.5 力模型

法向单位向量：

```text
n = sign(signed_z) * racket_local_Z_world
```

相对速度：

```text
v_rel = v_shuttle_contact - v_racket_surface_at_contact
v_n = dot(v_rel, n)
v_t = v_rel - v_n n
```

法向力：

```text
F_n = max(0, k(rho²) * penetration - c_n * v_n)
c_n = 3.0 N·s/m
```

切向阻尼：

```text
F_t = -c_t * v_t
c_t = 0.15 N·s/m
|F_t| <= mu * F_n
mu = 0.08
```

总力：

```text
F_on_shuttle = F_n n + F_t
F_on_racket  = -F_on_shuttle
```

用 `mujoco.mj_applyFT` 分别施加到羽毛球软木接触点和球拍线床接触点。

---

## 7. 高速击球的 event rebound 方案

羽毛球杀球时拍头法向速度可达到几十 m/s，单纯 soft-contact 可能因为 timestep 过大漏检。`src/racket_stringbed.py` 提供一个事件式速度更新函数：

```python
stringbed_rebound_velocity(
    shuttle_velocity_world,
    racket_surface_velocity_world,
    normal_world,
    restitution_normal=0.50,
    tangential_velocity_scale=0.85,
)
```

核心公式：

```text
v_rel = v_shuttle - v_racket_surface
v_rel = v_n n + v_t

若 v_n < 0：
v_rel_after = (-e * v_n) n + gamma * v_t
v_shuttle_after = v_racket_surface + v_rel_after
```

默认：

```text
e = 0.50
gamma = 0.85
```

建议使用策略：

```text
低速/近距离接触：使用 spring-damper force proxy
高速穿透/杀球：检测线床穿越事件，使用 event rebound 修正 freejoint 速度
```

---

## 8. 柔性拍杆代理

`assets/badminton_racket_flex_proxy.xml` 提供一个二体模型：

```text
racket_handle:
  freejoint
  手柄 + 下杆

racket_head:
  子 body
  上杆 + 喉部 + 头框 + 线床
  通过两个 hinge joint 与 handle 连接
```

默认 hinge：

```text
shaft_flex_x:
  axis = local +X
  stiffness = 398.6 N·m/rad
  damping = 0.040 N·m·s/rad
  range = ±8 deg

shaft_flex_z:
  axis = local +Z
  stiffness = 393.4 N·m/rad
  damping = 0.045 N·m·s/rad
  range = ±8 deg
```

解释：

```text
+X hinge 主要近似拍头相对手柄的出平面弯曲
+Z hinge 主要近似线床平面内的侧向弯曲
```

注意：柔性代理的质量/惯量是近似分配。若任务是严格匹配挥重，优先使用 rigid 模型；若任务是观察拍杆滞后、回弹、击球瞬间拍头恢复速度，使用 flex proxy 并重新标定。

---

## 9. 和羽毛球模型的接口

和上一份羽毛球包连接时，建议：

```text
羽毛球 body 名称：shuttle
羽毛球软木接触 site：shuttle_cork_site 或 cork_site
球拍 rigid body 名称：racket
球拍 flex body 名称：racket_head
线床中心 site：stringbed_center_site
```

若你的羽毛球 MJCF 没有 cork site，建议添加：

```xml
<site name="shuttle_cork_site" pos="0 0 0.018" size="0.003"/>
```

这里假设羽毛球局部 +Z 指向软木头部。

在 step loop 中：

```python
from racket_stringbed import apply_stringbed_force
from shuttlecock_aero import apply_shuttle_aero

for _ in range(n_steps):
    data.qfrc_applied[:] = 0.0

    apply_shuttle_aero(model, data, body_name="shuttle")
    contact = apply_stringbed_force(
        model,
        data,
        racket_body_name="racket",          # flex proxy 用 "racket_head"
        shuttle_body_name="shuttle",
        shuttle_contact_site_name="shuttle_cork_site",
    )

    mujoco.mj_step(model, data)
```

---

## 10. 推荐验证协议

### 10.1 几何合规

运行：

```bash
python src/validate_racket_params.py
```

期望：

```text
overall_length_ok        PASS
overall_width_ok         PASS
stringbed_length_ok      PASS
stringbed_width_ok       PASS
mass_target_ok           PASS
swingweight_target_ok    PASS
balance_target_ok        PASS
```

### 10.2 质量和平衡点

检查：

```text
mass = 90 g
balance point = 310 mm from butt
swingweight @ 90 mm = 93 kg·cm²
```

若要匹配某一支真实球拍，应实测：

```text
总质量
平衡点
9 cm 轴挥重 / simple pendulum swingweight
```

然后更新 `params/racket_nominal.json` 中的：

```text
mass_kg
center_of_mass_from_butt_y_m
principal_inertia_about_com_kg_m2.Ixx
mujoco_inertial_pos
mujoco_diaginertia
```

### 10.3 线床静态压入

用软木头部或半球压在线床中心：

```text
压入 5 mm
目标法向反力：45–55 N
默认值：48 N
```

若反力偏低：增大 `static_center_normal_stiffness_n_per_m`。  
若高速碰撞回弹过强：降低 `event_restitution_normal` 或增大 `normal_damping_n_s_per_m`。  
若边缘击球过软：增大 `normal_stiffness_edge_gain`。  
若边缘击球过硬/数值不稳：降低 `normal_stiffness_max_multiplier`。

### 10.4 动态击球

测试条件：

```text
拍面固定，羽毛球以 5, 10, 20, 40 m/s 法向撞击线床
拍面以 10, 20, 40, 60 m/s 法向运动击打静止羽毛球
中心击球和 rho≈0.7 的偏心击球都测试
```

记录：

```text
接触时间 / effective contact duration
最大压入
最大法向力
出球速度
出球方向
羽毛球自旋 / 姿态扰动
球拍 qfrc_applied 峰值
```

期望趋势：

```text
线床张力越高：刚度越高，压入越小，触球时间越短
偏心越大：法向反力方向和切向扰动更明显
挥重越高：同样拍头速度下有效质量更高，但人/机器人可达到的拍头速度可能降低
```

---

## 11. 已知局限

1. 这不是碳纤维层合板 FEM；不预测真实断裂、应力集中、框体局部屈曲。
2. 视觉线床不是动态线；真实线-线摩擦、线移动、回弹声学都被简化。
3. 线床刚度使用中心静态压入标定，不能自动覆盖所有张力/线径/穿线模式。
4. flex proxy 是二自由度代理，不是真实连续杆模态。
5. 真实击球中球拍和羽毛球都在变形；本模型把大部分变形等效到线床代理和可选拍杆 hinge。

这些局限是有意选择的，因为目标是 MuJoCo 中稳定、可调、可和 SMPL/机器人连接，而不是离线工程 FEM。

---

## 12. Codex 建议任务

建议让 Codex 按这个顺序分析/改造：

1. 读取 `params/racket_nominal.json`，确认几何、质量、挥重、线床参数。
2. 运行 `python src/validate_racket_params.py`。
3. 读取 `assets/badminton_racket_rigid.xml`，确认所有 geom/site 名称。
4. 在你的 MuJoCo 场景中 include 或复制 rigid racket body。
5. 将 `grip_pose_site` 对齐到 SMPL 右手或机器人末端执行器。
6. 把上一份羽毛球模型中的 cork site 名称传入 `apply_stringbed_force`。
7. 先用低速碰撞验证 spring-damper，再用高速杀球启用 event rebound。
8. 若需要拍杆弯曲，切换到 `badminton_racket_flex_proxy.xml`，并把 `racket_body_name` 改为 `racket_head`。
9. 用真实视频/运动捕捉数据重新拟合：`event_restitution_normal`、`normal_damping`、`center_normal_stiffness`、`edge_gain`。
10. 若目标是具体品牌型号，实测该拍质量、平衡点和挥重，再更新 inertial。

---

## 13. 文件清单

```text
badminton_racket_mujoco_design_package/
  badminton_racket_design_dossier.md
  README.md
  params/
    racket_nominal.json
  assets/
    badminton_racket_rigid.xml
    badminton_racket_flex_proxy.xml
    scene_racket_stringbed_debug.xml
  src/
    generate_racket_mjcf.py
    racket_stringbed.py
    validate_racket_params.py
  docs/
    validation_protocol.md
    codex_tasks.md
    references.md
```

---

## 14. 参考资料摘要

- BWF/Laws-style racket limits: frame length <= 680 mm, width <= 230 mm; stringed area length <= 280 mm and width <= 220 mm; stringed area must be flat crossed strings.
- Yonex ASTROX 100ZZ product data: 3U/4U variants, 3U average 88 g, stringing advice 21–29 lbs, head-heavy balance, 10 mm longer, 22×21 string pattern.
- Towler, Mitchell, King, Scientific Reports 2023: typical badminton racket mass about 0.085–0.095 kg; typical 9 cm-axis swingweight about 90–97 kg·cm²; racket smash tests used motion capture and studied MoI effects.
- Yin, Chang, Cheng, Proceedings 2020: badminton racket finite-element string-bed model with commercial racket head length 253 mm, width 188 mm; string diameter 0.7 mm; nylon material parameters; 26 lbf string tension; 5 mm impact displacement; reaction force around 47–48 N.
