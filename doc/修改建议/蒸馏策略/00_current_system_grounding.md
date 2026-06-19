# 00 当前仓库机制与蒸馏问题定义

## 0.1 当前目标

当前 ForehandClear 训练阶段的目标是：

```text
让 MyoFullBody 肌骨模型在 MuJoCo/MJX 中闭环模仿 GMR retarget 后的 ForehandClear body trajectory。
```

不包含：

```text
球拍模型
羽毛球动力学
拍-球碰撞
击球落点 reward
racket / shuttle residual control
```

因此当前 policy 是一个 **base body policy**，不是完整击球策略。

---

## 0.2 当前数据链路

根据仓库配置，ForehandClear fullbody 训练使用：

```text
Badminton video clips
  -> WHAM SMPL / SMPL-H motion
  -> AMASS-style NPZ
  -> GMR retargeted MyoFullBody reference trajectory
```

在 `fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml` 中，训练 motion path 是：

```yaml
badminton/train/forehand_clear_clip1_merged_poses
badminton/train/forehand_clear_clip2_merged_poses
badminton/train/forehand_clear_clip3_merged_poses
```

并且：

```yaml
retargeting_method: gmr
src_human: smplh
target_fps: 30
```

这些 trajectory 进入 environment 的 trajectory handler `env.th`，作为 goal observation 和 MimicReward 的 reference target。

---

## 0.3 当前 teacher PPO policy

当前 PPO actor-critic 单步闭环为：

```text
o_t -> Actor-Critic -> a_t -> MJX step -> o_{t+1}, r_t, done_t
```

其中：

```text
o_t: policy observation
      = joint state + muscle state + foot contact + goal lookahead

a_t: 354-D normalized muscle controls

r_t: imitation reward comparing current simulated state with reference trajectory
```

当前 `MyoFullBody` / `MjxMyoFullBody` action 是肌肉 actuator 控制；禁用 fingers 后 action 维度为 354。

---

## 0.4 Observation 组成

当前图中使用的 observation decomposition：

```text
joint state
muscle state
foot contact
goal lookahead
```

注意：`touch/contact` 在仓库中具体是 foot contact sensors，不是击球接触。当前 `MyoFullBody` 加入的 touch sensors 是：

```text
r_foot
r_toes
l_foot
l_toes
```

所以在后续文档中统一称为：

```text
foot contact
```

---

## 0.5 Goal lookahead 的角色

`GoalTrajMimic` 会从 trajectory handler 读取当前和未来参考帧，构造 goal observation。当前配置通常包括：

```yaml
n_step_lookahead: 5
n_step_stride: 20
enable_motion_phase: true
use_concise_lookahead: true
```

这意味着 teacher policy 不是单纯根据身体自身状态行动，而是每一步都看到未来 reference target。

这类 policy 更准确地称为：

```text
trajectory-conditioned teacher policy
```

---

## 0.6 Reward 机制

当前 `MimicReward` 是 DeepMimic-style tracking reward，比较：

```text
simulation qpos      vs trajectory qpos
simulation qvel      vs trajectory qvel
simulation root      vs trajectory root
simulation sites     vs trajectory sites
simulation site vel  vs trajectory site vel
```

奖励分项包括：

```text
qpos
qvel
root pos/vel
relative site pos/orient/vel
```

因此即使 student policy 不再看到 future lookahead，environment 仍然可以使用 reference trajectory 计算 reward。

---

## 0.7 PPO 闭环训练机制

当前 PPO runner 做：

```text
1. env.reset -> initial obs, env_state
2. network.apply(obs) -> pi, value
3. sample action from pi
4. env.step_with_transition(env_state, action)
5. store Transition(done, absorbing, action, value, reward, log_prob, obs, info, traj_state, metrics)
6. collect num_steps × num_envs transitions
7. compute GAE -> advantages, targets
8. PPO update actor and critic
9. repeat
```

GAE 输入：

```text
reward
value
done
absorbing
last_value
gamma
gae_lambda
```

输出：

```text
A_t: actor update signal
R_t: critic value target
```

---

## 0.8 为什么要蒸馏

Teacher policy 使用 lookahead，优点是 tracking 稳定、学习难度低。缺点是：

```text
它依赖完整 reference future target。
它更像 trajectory tracker，而不是独立 motor skill。
```

如果目标只是展示 trajectory imitation，teacher 已经成立。若目标是得到可迁移的 ForehandClear body motor skill，则推荐蒸馏成：

```text
student policy: body state + foot contact + motion phase -> muscle action
```

蒸馏后的 student 仍然可以用 reference trajectory 做 reward/fine-tune，但 policy 输入不再包含 future lookahead。

---

## 0.9 推荐第一版目标

第一版不要直接做完全无 phase 的 policy。推荐：

```text
Teacher input:
  joint + muscle + foot contact + full goal lookahead

Student input:
  joint + muscle + foot contact + motion phase

Shared output:
  354-D normalized muscle controls
```

原因：ForehandClear 是长时序动作，完全去掉 phase 会让 student 难以区分引拍、挥拍、随挥等相位。
