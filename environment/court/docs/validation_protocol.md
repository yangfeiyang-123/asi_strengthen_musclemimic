# 羽毛球场地 MuJoCo 资产验证协议

版本：1.0.0

---

## 1. 目标

验证 `badminton_court_mujoco_design_package` 是否满足：

1. BWF 标准几何尺寸；
2. MuJoCo XML 可解析；
3. 场地线、球网、网柱关键元素存在；
4. 落点规则 helper 与线宽语义一致；
5. visual-only 与 collision-net 两种资产行为清晰。

---

## 2. 静态尺寸验证

运行：

```bash
python src/validate_court_params.py
```

必须通过：

```text
full court length 13.40 m
doubles width 6.10 m
singles width 5.18 m
line width 40 mm
half length 6.70 m
doubles half-width 3.05 m
singles half-width 2.59 m
short service near edge at |x|=1.98 m
doubles long service outer edge at |x|=5.94 m
net centre height 1.524 m
net sideline height 1.550 m
```

---

## 3. XML 验证

脚本必须确认以下 XML 可解析：

```text
assets/badminton_court_bwf_visual.xml
assets/badminton_court_bwf_collision_net.xml
```

必须存在的 geoms/sites：

```text
floor_collision
back_boundary_pos_x
back_boundary_neg_x
short_service_line_pos_x
short_service_line_neg_x
doubles_long_service_line_pos_x
doubles_long_service_line_neg_x
net_post_pos_y
net_post_neg_y
net_midpoint_site
```

`badminton_court_bwf_visual.xml` 不应包含 `net_collision_proxy_*`。
`badminton_court_bwf_collision_net.xml` 应包含启用 `contype=2` 的 `net_collision_proxy_*`。

---

## 4. 规则判定验证

### 4.1 回合落点

以下必须为 `True`：

```python
court.inside_rally(6.70, 3.05, "doubles")
court.inside_rally(0.00, 2.59, "singles")
```

以下必须为 `False`：

```python
court.inside_rally(6.701, 0.00, "doubles")
court.inside_rally(0.00, 2.591, "singles")
```

### 4.2 发球落点

以下必须为 `True`：

```python
court.inside_service(1.98, 0.01, "doubles", "+x", "+y")
court.inside_service(5.94, 2.00, "doubles", "+x", "+y")
court.inside_service(6.70, -2.00, "singles", "+x", "-y")
```

以下必须为 `False`：

```python
court.inside_service(5.941, 2.00, "doubles", "+x", "+y")
court.inside_service(6.701, -2.00, "singles", "+x", "-y")
```

---

## 5. MuJoCo 运行验证

在项目环境中运行：

```python
import mujoco

for path in [
    "assets/badminton_court_bwf_visual.xml",
    "assets/badminton_court_bwf_collision_net.xml",
]:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(path, model.ngeom, model.nbody)
```

验收标准：

1. 无 XML 编译错误；
2. 地板位于 `z=0` 附近；
3. 球网位于 `x=0`；
4. 正负半场对称；
5. 相机视角下线条清晰可见。

---

## 6. 物理验证

### 6.1 地板接触

将一个 sphere 放在 `z=1.0` 处自由落体：

```text
预期：sphere 在地板上稳定接触，不因边线产生额外台阶反弹。
```

由于线条 visual-only，球落在线附近和普通地板接触行为应一致。

### 6.2 球网接触

使用 `badminton_court_bwf_collision_net.xml`：

1. 让羽毛球以低速撞击网带；
2. 让羽毛球以低速撞击网面；
3. 检查是否发生接触。

初版验收只要求存在阻挡/接触，不要求真实柔性挂网。若需要更真实网面行为，应在后续任务中用事件式 net-trap 逻辑处理。

---

## 7. 集成验证

和羽毛球包/球拍包集成后，验证以下场景：

1. 羽毛球从 `x=-5.0` 飞向 `x=+5.0`，越过球网；
2. 羽毛球低于网顶高度时应被 `collision_net` 版本阻挡；
3. 羽毛球落在单打线外、双打线内时：
   ```python
   inside_rally(x, y, "singles") == False
   inside_rally(x, y, "doubles") == True
   ```
4. 双打发球落在 `x=6.1` 附近时应判出界，因为超过双打长发球线；
5. 单打发球落在 `x=6.1` 附近时仍可判入界。

---

## 8. 常见失败模式

### 8.1 线条参与碰撞导致羽毛球弹跳异常

解决：

```xml
contype="0"
conaffinity="0"
```

场地线只负责视觉。

### 8.2 网柱侵入场地

如果把圆柱中心放在 `y=±3.05` 且 radius > 0，柱体会伸进场内。默认模型把柱体中心放在：

```text
y = ±(3.05 + post_radius)
```

并用 `post_official_*_site` 标记官方线位置。

### 8.3 include 路径错误

MuJoCo `<include file="..."/>` 的路径相对主 XML 文件，不一定相对 Python 当前目录。合并大场景时建议 Codex 把场地 worldbody 拆成 include-friendly snippet。

### 8.4 球网碰撞过硬

解决路线：

1. 训练阶段使用 visual-only 版本；
2. 触网判定用几何事件；
3. 只在评估阶段启用 collision-net；
4. 调小网面 proxy 的 `solref` 或改成 soft contact。
