# Codex 任务清单：羽毛球场地资产

## 0. 最短路径

从包根目录运行：

```bash
python src/generate_court_mjcf.py
python src/validate_court_params.py
```

然后在项目中使用：

```text
assets/badminton_court_bwf_visual.xml
```

作为默认场地。

---

## 1. 合并到完整 badminton scene

### 任务

把以下资产合并为一个 MuJoCo scene：

```text
badminton_court_mujoco_design_package/assets/badminton_court_bwf_visual.xml
shuttlecock_mujoco_design_package/assets/shuttlecock_mujoco.xml
badminton_racket_mujoco_design_package/assets/badminton_racket_rigid.xml
```

### 要求

1. 保持场地坐标系不变；
2. 球网在 `x=0`；
3. 羽毛球 free body 初始位置可配置；
4. 球拍可 attach 到机器人末端或 SMPL 手部；
5. 场地线不参与碰撞。

---

## 2. 落点事件

### 任务

实现 `LandingEventDetector`：

```python
event = detector.update(model, data)
if event is not None:
    x, y = event.xy
    is_in = court.inside_rally(x, y, mode="doubles")
```

### 建议

1. 监听羽毛球 cork 与 `floor_collision` 的首次接触；
2. 记录接触点或球心投影；
3. 去重，避免同一次落地多帧触发；
4. 返回：
   ```python
   {
       "time": data.time,
       "x": x,
       "y": y,
       "mode": "singles" | "doubles",
       "inside": bool,
       "feature": nearest_line_or_zone
   }
   ```

---

## 3. 发球区判定

### 任务

用 `court.inside_service(...)` 实现发球落点 reward。

### 示例

```python
legal = court.inside_service(
    landing_x,
    landing_y,
    mode="doubles",
    court_half="+x",
    lateral_half="+y",
)
```

### 注意

1. `court_half` 表示接发球目标半场；
2. `lateral_half` 表示目标服务区左右半边，当前按 `+y/-y` 表示；
3. 如果项目中使用“球员朝向”的 left/right，需要写 adapter；
4. 线内/线上均为合法落点。

---

## 4. 触网事件

### 任务

实现不依赖复杂软网接触的 `NetCrossingAndContactDetector`。

### 建议逻辑

1. 监听羽毛球从 `x<0` 到 `x>0` 或相反的穿越；
2. 计算穿越 `x=0` 时的插值高度 `z_cross`；
3. 计算该 `y_cross` 处网顶高度：
   ```python
   z_net = court.net_top_height(y_cross)
   ```
4. 如果 `z_cross < z_net + shuttle_radius_margin`，判为未过网或触网；
5. 如果使用 collision-net XML，可融合真实 contact 数据。

---

## 5. 可视化 overlay

### 任务

添加 debug overlay：

1. 当前 rally 合法区域；
2. 当前 service 合法区域；
3. 羽毛球预测落点；
4. 最近一次实际落点；
5. 单打/双打模式切换。

### 实现路线

1. 在 MuJoCo viewer 中增加 semi-transparent geom；
2. 或用 Python matplotlib/top-down camera 输出；
3. 或在 XML 中增加可启用/禁用的 site/box overlay。

---

## 6. 网格密度配置

### 任务

让 `src/generate_court_mjcf.py` 接收命令行参数：

```bash
python src/generate_court_mjcf.py --net-pitch 0.02
```

### 要求

1. 默认仍为稀疏视觉网格；
2. 允许生成 15–20 mm 官方网孔密度；
3. 碰撞代理不随网孔密度暴涨；
4. 高密度网线仍然 visual-only。

---

## 7. 场地材质参数化

### 任务

在 JSON 中增加材质配置：

```json
{
  "materials": {
    "floor_rgba": [0.05, 0.23, 0.12, 1],
    "line_rgba": [1, 1, 0.92, 1],
    "net_rgba": [0.03, 0.03, 0.035, 1]
  }
}
```

并让生成脚本读取。

---

## 8. 与 SMPL / WorldGroundedSMPL 对齐

### 任务

提供场地到人体 root pose 的初始化函数：

```python
def player_start_pose(role: str, mode: str) -> dict:
    ...
```

### 建议站位

```text
singles_base_neg_x: x=-4.8, y=0.0
singles_base_pos_x: x=+4.8, y=0.0
doubles_front_neg_x: x=-2.6, y=±1.0
doubles_back_neg_x: x=-5.0, y=∓1.0
```

这些是训练/初始化默认值，不是官方规则。

---

## 9. 单元测试

新增 pytest：

```text
tests/test_court_geometry.py
tests/test_court_xml.py
tests/test_landing_detector.py
tests/test_service_bounds.py
```

必须覆盖：

1. 线上判入；
2. 线外 1 mm 判出；
3. 单打/双打边界差异；
4. 双打发球长线；
5. 单打发球后边界；
6. 网顶高度插值；
7. XML 中场地左右/正负半场对称。

---

## 10. 交付标准

完成后，项目应能：

1. 编译完整 MuJoCo badminton scene；
2. 运行 10 秒仿真不崩溃；
3. 正确判定羽毛球落点 in/out；
4. 正确区分 singles/doubles；
5. 可选择 visual-only 或 collision-net；
6. 与球拍和羽毛球模型坐标一致。
