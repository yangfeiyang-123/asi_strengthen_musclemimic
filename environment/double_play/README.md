# Double Play — 双人正手高远球对打环境

两个 MyoFullBody 肌骨模型（各 354 条肌肉，去手指、球拍以 exact-child 刚性挂在右手）分别站在 BWF
球场两侧后场（|x| = 4.6 m），共享一个羽毛球，互相击打正手高远球。

## 目录

```
double_play/
├── assets/double_play_scene.xml      # 生成的双人场景（nq=185, nu=708, keyframe "double_ready"）
├── params/double_play_nominal.json   # 场景/物理/发球/奖励/观测的参数文档（单一说明源）
├── src/
│   ├── build_double_play_scene.py    # 场景构建器：单人场景 + 镜像 attach 第二个"人+拍"
│   ├── rally_physics.py              # 双拍子步物理（v2 气动 + 双拍事件反弹/冷却/扫掠检测）
│   ├── double_play_env.py            # DoublePlayRallyEnv：双人回合制拉吊环境
│   └── run_double_play_demo.py       # 无头渲染冒烟（发球弧线视频）
└── tests/                            # 场景结构 / 双拍物理 / 环境行为测试
```

## 重建场景

```bash
.venv/bin/python -m environment.double_play.src.build_double_play_scene
```

P1 保持单人 incoming-hit 场景的全部名字（`root`、`overall_racket`、`overall_shuttle`……），P2 为
`p2_` 前缀 + 绕 z 轴 180° 镜像。碰撞位与单人场景一致：人体 1、拍框 4、MJX 兼容椭球 8、拍面落地代理
16、软木 conaffinity 13。

## 环境

```python
from environment.double_play.src.double_play_env import DoublePlayRallyEnv

env = DoublePlayRallyEnv(seed=0)
obs, info = env.reset()                      # 发球机向接球方后场击出高远球
actions = {"p1": a1, "p2": a2}               # 每人 354 维, [-1, 1]
obs, rewards, terminated, truncated, info = env.step(actions)
```

- **镜像观测**：P2 的观测经绕网线 180° 旋转，双方都"从 -x 半场向 +x 进攻"，可用单一共享策略自博弈
  （reset 时镜像精确到机器精度，此后因求解器全局迭代耦合仅统计对称）。
- **回合规则**：双方必须交替合法击球；落地结分（后场落点最优）、错序击球/倒地判负、
  达到 `max_rally_hits` 视为成功长回合（truncated）。
- **物理 v2（默认开启）**：裙部横流气动 + 压心随动阻尼（翻正 0.3 s 内收敛）、扫掠穿面检测
  （>30 m/s 合速不再穿模）、速度相关恢复系数、软木偏心角冲量（击球后自然翻转）、
  恢复被 `boundinertia` 钳掉的羽毛球真实惯量。
- **域随机化**：`aero_domain_randomization=True` 时每回合从
  `environment/shuttlecock/params/shuttlecock_nominal.json` 的 randomization 区间采样气动参数与风。

## 冒烟 / 可视化

```bash
MUJOCO_GL=egl .venv/bin/python -m environment.double_play.src.run_double_play_demo \
    --video outputs/double_play/double_play_demo.mp4
```

## 测试

```bash
.venv/bin/python -m pytest environment/double_play/tests/ -q
```

## 已知边界

- CPU MuJoCo 环境；MJX 批量版尚未移植（`badminton_physics_mjx.make_params` 对 CPU-only 物理
  开关 fail-closed）。
- 每回合一次发球、落地即止；多球连续发球与计分制可在其上扩展。
- 双方为同一右手持拍模型的 180° 旋转副本（非左右镜像反射），符合"两名右手球员"设定。
