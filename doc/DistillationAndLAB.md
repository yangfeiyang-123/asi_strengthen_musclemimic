# LATENT 的蒸馏与 Latent Action Barrier 在肌骨羽毛球模型中的落地方案

> 目标：把 LATENT 中“从不完美人体动作数据中学习可修正 latent action space”的思想，迁移到你的 **视频/WHAM/World-Grounded SMPL → 肌骨模型重定向 → PPO/强化学习** 系统中，用于羽毛球动作，尤其是正手高远球动作复现与后续击球任务。

---

## 0. 一句话总览

LATENT 的核心不是“直接从视频恢复动作”，而是：

1. 先用不完美的人类动作片段训练一个 **motion tracker teacher**；
2. 再把这个 tracker 蒸馏成一个 **latent skill decoder**；
3. 训练高层 policy 时，不让 policy 在 latent space 里随便乱采样，而是通过 **Latent Action Barrier, LAB** 把探索限制在“当前状态下像人类动作”的区域附近；
4. 对最难从数据中可靠学习的末端部分，例如持拍手腕、握拍、拍面角度，单独使用 correction/residual policy。

迁移到你的肌骨模型中，就是：

```text
WHAM/SMPL 轨迹
    ↓
SMPL → 肌骨模型重定向
    ↓
训练 muscle/body tracker teacher
    ↓
蒸馏出 latent muscle/body skill space
    ↓
用 LAB 约束 PPO 的 latent action
    ↓
必要时单独训练右腕/握拍/末端 residual correction
```

---

## 1. 你项目中的问题定义

你的当前项目可以抽象成：

```text
输入：人类羽毛球视频
输出：肌骨模型在仿真中复现羽毛球动作，后续可能持拍击球
```

典型流程为：

```text
video
  → WHAM / Optimized-WHAM
  → World-Grounded SMPL
  → SMPL-to-musculoskeletal retargeting
  → musculoskeletal imitation / PPO policy
```

主要困难是：

1. **视频/SMPL 轨迹本身不完美**：手腕、手部、脚底接触、root 高度、地面约束、身体尺度都可能存在误差。
2. **SMPL 到肌骨模型存在 embodiment gap**：SMPL 关节结构与肌骨模型关节结构不同，动作直接映射后可能不动力学可行。
3. **肌肉控制维度高**：如果 action 是 muscle excitation/activation，直接 PPO 很容易学出非人体式发力。
4. **任务奖励可能导致 hacking**：PPO 为了追踪手部轨迹或击球点，可能用极端肌肉激活、异常关节姿态或不自然的身体协同完成任务。
5. **右手腕/握拍/拍面最不可靠**：这些部位在单目视频和 SMPL 中最难准确恢复，不适合直接强监督模仿。

LATENT 的蒸馏 + LAB 正好对应解决第 3、4、5 点。

---

## 2. LATENT 与你的肌骨系统的变量对应关系

| LATENT 中的概念 | 在你的肌骨模型中的对应 |
|---|---|
| human motion fragments | WHAM/SMPL 得到的羽毛球动作片段，如准备、引拍、挥拍、随挥、步法 |
| humanoid motion retargeting | SMPL-to-musculoskeletal retargeting |
| motion tracker | 肌骨动作跟踪 teacher policy |
| joint action / PD target | muscle excitation、muscle activation、joint torque 或 PD target，取决于你的 action schema |
| latent action space | 肌肉/身体动作技能空间 |
| conditional prior P(z\|s) | 当前肌骨状态下的自然动作 latent 分布 |
| high-level policy | PPO task policy / trajectory policy / future grip policy |
| wrist correction | 右腕、手部、握拍、拍面角度、击球末端 residual correction |
| Latent Action Barrier | 限制 PPO 只能在自然动作 latent 分布附近探索 |

---

## 3. 第一部分：蒸馏怎么做

### 3.1 蒸馏的目的

蒸馏不是直接让模型“学会打羽毛球”。

蒸馏的目的，是把一个已经会模仿动作的 teacher tracker 压缩成一个可被高层策略调用的 latent skill model。

原始 tracker 是：

```text
当前肌骨状态 s_t + 下一帧参考动作 r_{t+1}
      → teacher tracker
      → 低层控制动作 a_teacher
```

蒸馏之后变成：

```text
当前肌骨状态 s_t + latent code z_t
      → decoder
      → 低层控制动作 a_body
```

这样后续 PPO 不需要直接输出上百维肌肉激活，而是输出一个低维 latent residual，再由 decoder 解码成自然的肌骨控制动作。

---

## 3.2 Step 0：准备动作数据

### 3.2.1 数据来源

你的数据可以来自：

```text
羽毛球视频 → WHAM/Optimized-WHAM → World-Grounded SMPL
```

然后通过你的 SMPL-to-musculoskeletal retargeting 得到肌骨参考动作。

建议先不要追求完整比赛序列，而是构建羽毛球动作片段库：

```text
primitive motion fragments:
  - ready pose / 准备姿态
  - forehand clear preparation / 正手高远球引拍
  - trunk rotation / 躯干转体
  - shoulder external rotation / 肩外旋储能
  - forward swing / 向前挥拍
  - follow-through / 随挥
  - recovery / 回位
  - side step / 并步或侧移
  - crossover step / 交叉步
```

这样做的好处是：即使视频轨迹不完美，动作片段仍然可以提供自然动作先验。

### 3.2.2 参考动作格式

每条参考动作建议保存为：

```python
reference_motion = {
    "qpos": ...,                 # 肌骨模型 generalized coordinates
    "qvel": ...,                 # generalized velocities
    "root_pos": ...,             # root/global pelvis position
    "root_rot": ...,             # root orientation
    "joint_pos_world": ...,      # 关键关节世界坐标
    "end_effector_pos": ...,     # 手腕、肘、肩、脚等关键点
    "phase": ...,                # 动作相位，可选
    "contact": ...,              # 足底接触，可选
    "smpl_confidence": ...,      # 来自视频估计的置信度，可选
}
```

如果你的 retargeting 目前不能输出完整 muscle state，没有关系。teacher tracker 可以在仿真中根据模型状态得到 muscle length、fiber velocity、activation 等。

---

## 3.3 Step 1：训练肌骨 Motion Tracker Teacher

### 3.3.1 Teacher 的作用

teacher tracker 是一个动作跟踪 policy。

它不是根据球打高远球，而是学习：

> 给定当前肌骨状态和下一帧/未来短窗口参考动作，输出什么肌肉控制，才能让肌骨模型稳定跟踪参考动作。

### 3.3.2 Teacher 输入

建议输入：

```text
obs_tracker_t = [
    q_t,                         # 关节角
    qdot_t,                      # 关节速度
    root_orientation_t,          # root 朝向
    root_velocity_t,             # root 速度
    projected_gravity_t,         # 重力方向在身体坐标系下的投影
    muscle_activation_t,         # 当前肌肉激活
    muscle_length_t,             # 肌肉长度，可选
    muscle_velocity_t,           # 肌肉速度，可选
    previous_action_t,           # 上一帧 action
    reference_features_t,        # 参考动作目标
]
```

其中 `reference_features_t` 可以包含：

```text
ref_{t+1}: 下一帧目标关节角、关节速度、关键点位置
ref_{t:t+H}: 未来 H 帧短窗口目标
phase: 当前动作相位
```

建议使用短窗口未来目标，而不是只用下一帧：

```text
r_t = [ref_{t+1}, ref_{t+5}, ref_{t+10}]
```

这样 teacher 更容易提前引拍、提前蹬转，而不是逐帧被动追踪。

### 3.3.3 Teacher 输出

根据你当前肌骨环境的 action schema，有三种选择。

#### 方案 A：直接输出 muscle excitation

```text
a_teacher_t ∈ [0, 1]^{N_muscle}
```

优点：最符合肌骨控制。

缺点：维度高，训练难，容易激活抖动。

#### 方案 B：输出低维 muscle synergy

```text
u_t ∈ R^K
activation_t = W u_t
```

优点：更稳定，更符合肌肉协同。

缺点：需要先定义或学习 synergy matrix。

#### 方案 C：输出 joint PD target / residual torque

```text
a_teacher_t = target_joint_position 或 residual torque
```

优点：训练更容易。

缺点：对“肌肉发力真实性”的解释较弱。

### 推荐

如果你当前 musclemimic 框架已经可以用 muscle activation 训练，优先使用 **方案 A**。如果训练不稳定，可以先使用 **方案 C 做验证版本**，再迁移到 muscle excitation。

---

## 3.4 Teacher Tracker 的 reward 设计

Teacher 的 reward 应该是 dense imitation reward。建议：

```text
R_tracker =
    w_pose   * r_pose
  + w_vel    * r_velocity
  + w_root   * r_root
  + w_ee     * r_end_effector
  + w_contact* r_contact
  + w_balance* r_balance
  - w_effort * cost_muscle_effort
  - w_smooth * cost_action_smooth
  - w_limit  * cost_joint_limit
  - w_slip   * cost_foot_slip
```

### 3.4.1 关节角跟踪

```text
r_pose = exp(-k_pose * ||q_t - q_ref_t||^2)
```

### 3.4.2 关节速度跟踪

```text
r_velocity = exp(-k_vel * ||qdot_t - qdot_ref_t||^2)
```

### 3.4.3 root / pelvis 跟踪

```text
r_root = exp(-k_root * (||p_root - p_ref||^2 + d_rot(root, root_ref)^2))
```

### 3.4.4 关键点跟踪

对羽毛球高远球，关键点包括：

```text
right_shoulder
right_elbow
right_wrist
pelvis
left_foot
right_foot
head / thorax
```

但注意：右手腕如果来自视频/SMPL，很可能不准。可以在 teacher 阶段对右手腕使用较低权重，或者只跟踪肩、肘、躯干、骨盆、脚步。

### 3.4.5 肌肉 effort 约束

```text
cost_muscle_effort = ||a_muscle||^2
```

不要把它设得太大，否则模型可能为了省力不挥拍。

### 3.4.6 激活平滑约束

```text
cost_action_smooth = ||a_t - a_{t-1}||^2
```

肌肉控制尤其需要这个项，否则容易出现非生理性高频激活。

### 3.4.7 足底接触与滑动约束

羽毛球动作中步法和重心非常重要。建议加入：

```text
cost_foot_slip = contact_foot * ||v_foot_xy||^2
```

用于防止脚底接触地面时横向滑动。

---

## 3.5 对不可靠部位的特殊处理

LATENT 里面对右手腕采取了特殊处理：不让 body tracker 强行学习不可靠手腕动作，并在训练中加入手腕扰动，使身体控制对手腕修正鲁棒。

迁移到你的项目，可以定义两个关节/肌肉集合：

```text
Body set:
  pelvis, torso, neck, left/right hip, knee, ankle,
  right shoulder, right elbow, scapula 等主链条

Correction set:
  right wrist, right hand, grip, racket pose,
  或者右腕相关肌肉/关节 residual
```

### 处理方式

#### 方式 1：降低右手腕跟踪权重

```text
w_right_wrist << w_shoulder, w_elbow, w_trunk
```

#### 方式 2：teacher action 不输出右腕控制

```text
a_teacher_body = action without right_wrist / grip dimensions
```

右腕后续交给 correction policy。

#### 方式 3：训练时对右腕加入随机扰动

```text
right_wrist_q += noise
right_wrist_qdot += noise
```

让 body policy 在右腕被扰动时仍能保持躯干、肩肘、下肢稳定。

### 推荐

你的项目当前如果主要目标是“无拍/无球动作轨迹复现”，可以先采用方式 1。

如果后续要接入球拍、握拍和击球，建议采用方式 2 + 方式 3：

```text
body latent policy 负责全身自然动作
wrist/grip residual policy 负责末端击球修正
```

---

## 3.6 Step 2：构建蒸馏网络

蒸馏阶段需要三个网络：

```text
Posterior Encoder E_phi(z | s, r)
Conditional Prior P_psi(z | s)
Decoder D_theta(a | s, z)
```

### 3.6.1 Posterior Encoder

```text
E_phi(z | s_t, r_t) = N(mu_e(s_t, r_t), sigma_e(s_t, r_t))
```

它训练时能看到参考动作，所以知道当前应该用什么 skill。

输入：

```text
[s_t, reference_features_t]
```

输出：

```text
mu_e, log_sigma_e
```

然后采样：

```text
z_q = mu_e + sigma_e * epsilon
```

### 3.6.2 Conditional Prior

```text
P_psi(z | s_t) = N(mu_p(s_t), sigma_p(s_t))
```

它只看当前肌骨状态，不看参考动作。

它的意义是：

> 当前这个身体状态下，哪些 latent muscle skill 是自然的。

例如：

```text
准备姿态 → 可以起步、转体、引拍
引拍后期 → 可以继续挥拍、蹬转
击球后 → 可以随挥、恢复平衡
单脚支撑 → 不应该突然产生极端挥拍 latent
```

### 3.6.3 Decoder

```text
D_theta(a_body | s_t, z_t)
```

输入：

```text
当前肌骨状态 s_t
latent code z_t
```

输出：

```text
body action a_body
```

如果 action 是 muscle activation，则输出维度为：

```text
N_body_muscles
```

如果你要把右腕/握拍排除，则 decoder 只输出 body action，右腕 correction 后面另接。

---

## 3.7 蒸馏损失函数

总损失：

```text
L_distill = λ_action L_action + λ_KL L_KL + λ_smooth L_smooth + λ_bound L_bound
```

### 3.7.1 Action reconstruction loss

```text
L_action = ||a_teacher_body - a_student_body||^2
```

其中：

```text
a_student_body = D_theta(s_t, z_q)
```

作用：让 decoder 输出的 action 接近 teacher tracker 的 action。

### 3.7.2 KL loss

```text
L_KL = KL(E_phi(z | s_t, r_t) || P_psi(z | s_t))
```

作用：让 posterior encoder 学到的 latent 分布，靠近 conditional prior。

这一步非常关键，因为后续 LAB 需要使用 prior 的：

```text
mu_p(s_t), sigma_p(s_t)
```

如果 KL 没训好，LAB 就没有可靠的“自然动作中心”和“合理探索范围”。

### 3.7.3 Action smoothness loss

```text
L_smooth = ||a_student_t - a_student_{t-1}||^2
```

肌骨模型建议保留这个项，尤其是 muscle excitation 作为 action 时。

### 3.7.4 Activation bound / regularization

如果 decoder 输出 muscle excitation，可加入：

```text
a_student = sigmoid(raw_output)
```

或者：

```text
a_student = clamp(raw_output, 0, 1)
```

更推荐 `sigmoid` 或 `tanh + scaling`，避免硬截断导致梯度问题。

---

## 3.8 Online Distillation / DAgger 风格训练

不要只在离线 reference state 上训练 decoder。因为后续 policy 一旦偏离参考轨迹，模型会进入训练集中没有的状态。

建议采用 online distillation：

```text
1. 用当前 student decoder 在环境中 rollout
2. 收集 student 实际到达的状态 s_t
3. 在这些状态上调用 teacher tracker，得到 a_teacher_t
4. 把 (s_t, r_t, a_teacher_t) 加入 buffer
5. 训练 encoder/prior/decoder
6. 重复
```

伪代码：

```python
for iteration in range(num_iters):
    trajectories = []

    for env in envs:
        s = env.reset(reference_motion=random_motion())
        for t in range(T):
            r = get_reference_features(t)

            # posterior only used during distillation training
            mu_e, logstd_e = encoder(s, r)
            z = sample_gaussian(mu_e, logstd_e)

            a_student = decoder(s, z)
            s_next = env.step(a_student)

            # query teacher on student's visited state
            a_teacher = teacher_tracker(s, r)

            buffer.add(s, r, a_teacher, a_student)
            s = s_next

    for update in range(num_updates):
        batch = buffer.sample()
        mu_e, logstd_e = encoder(batch.s, batch.r)
        z = reparameterize(mu_e, logstd_e)
        a_pred = decoder(batch.s, z)

        mu_p, logstd_p = prior(batch.s)

        loss_action = mse(a_pred, batch.a_teacher)
        loss_kl = kl_gaussian(mu_e, logstd_e, mu_p, logstd_p)
        loss = lambda_action * loss_action + lambda_kl * loss_kl

        optimize(loss)
```

---

## 3.9 蒸馏完成后需要保存什么

训练完成后，保存：

```text
encoder.pt              # 训练阶段可保留，部署时通常不用
prior.pt                # LAB 必须用
latent_decoder.pt       # 高层 policy 必须用
normalization_stats.pkl # obs/action 归一化参数
action_mask.json        # body/correction action 维度划分
latent_config.yaml      # latent dim, sigma clamp, lambda 等配置
```

部署或高层 PPO 阶段通常使用：

```text
prior + decoder
```

encoder 只在蒸馏训练阶段使用。

---

# 4. 第二部分：LAB 怎么做

## 4.1 LAB 解决什么问题

蒸馏之后，你会得到一个 decoder：

```text
D(s_t, z_t) → a_body_t
```

理论上，高层 PPO 只要输出 `z_t`，decoder 就能生成动作。

但问题是：

```text
latent space 中不是每一个点都对应自然动作
```

如果 PPO 直接自由输出 latent，可能会出现：

```text
- latent 跳到训练分布之外
- 肌肉激活异常变大
- 动作抖动
- 身体姿态不自然
- 为了追踪手部或击球点，牺牲全身协调
- 出现非人体式 muscle hacking
```

LAB 的作用是：

> 不让 PPO 在整个 latent space 里乱探索，而是只能在当前状态下 prior 认为合理的 latent 分布附近做有限调整。

---

## 4.2 LAB 的数学形式

蒸馏阶段已经学到了：

```text
P(z | s_t) = N(mu_p(s_t), sigma_p(s_t))
```

高层 policy 不直接输出 `z_t`，而是输出一个 raw latent residual：

```text
u_t = π_high(obs_t)
```

然后通过 LAB 得到真正送入 decoder 的 latent：

```text
z_t = mu_p(s_t) + λ * sigma_p(s_t) * tanh(u_t)
```

最后：

```text
a_body_t = D(s_t, z_t)
```

这就是 LAB 的核心。

---

## 4.3 每一项的含义

### 4.3.1 mu_p(s_t)

```text
mu_p(s_t)
```

表示当前肌骨状态下最自然、最典型的 latent skill。

例如：

```text
当前处于引拍后期 → mu_p 可能对应继续挥拍
当前处于随挥后期 → mu_p 可能对应减速和回位
当前重心不稳 → mu_p 可能对应恢复平衡
```

### 4.3.2 sigma_p(s_t)

```text
sigma_p(s_t)
```

表示当前状态下每个 latent 维度允许变化的范围。

如果某个维度的 sigma 很小，说明该维度不应该乱变。

如果某个维度的 sigma 很大，说明当前状态下这个维度有多种合理动作选择。

### 4.3.3 tanh(u_t)

```text
tanh(u_t) ∈ [-1, 1]
```

作用是防止 PPO 输出无限大的 latent residual。

### 4.3.4 λ

```text
λ
```

控制 PPO 可以偏离 prior 多远。

```text
λ 小 → 动作更像人，但任务探索更保守
λ 大 → 任务探索更自由，但可能更不自然
```

---

## 4.4 为什么 LAB 比 reward penalty 更强

普通方法可能会加一个 penalty：

```text
reward -= ||z_t - mu_p||^2
```

但这是软约束。PPO 如果发现任务奖励更大，仍然可能远离人类动作分布。

LAB 是硬约束，因为真正执行的 latent 永远满足：

```text
z_i ∈ [mu_i - λ sigma_i, mu_i + λ sigma_i]
```

这意味着：

```text
无论 PPO 输出多大，最终动作都不能离当前自然动作 prior 太远
```

对你的肌骨模型来说，这一点非常重要，因为肌肉控制维度高，PPO 很容易利用仿真漏洞学出不自然激活。

---

## 4.5 在肌骨模型中的 LAB action wrapper

建议把 LAB 做成一个 action wrapper：

```python
class LABActionWrapper:
    def __init__(self, prior, decoder, lambda_lab, sigma_min, sigma_max):
        self.prior = prior
        self.decoder = decoder
        self.lambda_lab = lambda_lab
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, state, high_level_action):
        u_latent = high_level_action["latent"]
        correction = high_level_action.get("correction", None)

        mu, log_sigma = self.prior(state)
        sigma = softplus(log_sigma)
        sigma = clamp(sigma, self.sigma_min, self.sigma_max)

        z = mu + self.lambda_lab * sigma * tanh(u_latent)

        a_body = self.decoder(state, z)

        if correction is None:
            return a_body
        else:
            return combine_body_and_correction(a_body, correction)
```

---

# 5. 高层 PPO 如何接入 LAB

## 5.1 高层 policy 的输入

如果你当前只做动作复现，不加球和球拍：

```text
obs_high_t = [
    s_t,
    phase_t,
    target_keypoints_t,
    target_root_t,
    future_motion_goal_t
]
```

如果你后续做带球拍击球：

```text
obs_high_t = [
    s_t,
    root_global_pose_t,
    racket_pose_t,
    shuttlecock_state_t,
    target_landing_area,
    phase_t
]
```

## 5.2 高层 policy 的输出

推荐：

```text
π_high(obs_t) → [u_latent_t, a_correction_t]
```

其中：

```text
u_latent_t: latent residual，经过 LAB 后送入 decoder

a_correction_t: 右腕/握拍/拍面/末端 residual
```

如果你暂时不做球拍，可以先只输出：

```text
π_high(obs_t) → u_latent_t
```

## 5.3 最终控制链路

```text
obs_high_t
   ↓
PPO high-level policy
   ↓
u_latent_t, correction_t
   ↓
LAB:
   z_t = mu_p(s_t) + λ sigma_p(s_t) tanh(u_latent_t)
   ↓
Decoder:
   a_body_t = D(s_t, z_t)
   ↓
Correction merge:
   a_full_t = merge(a_body_t, correction_t)
   ↓
Musculoskeletal simulator
```

---

# 6. Correction action 在你的项目里怎么设计

## 6.1 为什么需要 correction

你的视频到 SMPL 对以下部分不可靠：

```text
- 右手腕角度
- 手部姿态
- 握拍状态
- 拍面方向
- 击球瞬间的拍头速度
```

如果强行让 body latent 学这些东西，会污染整个 latent skill space。

所以建议：

```text
body latent 负责大部分自然全身动作
correction policy 负责末端高精度修正
```

## 6.2 Correction 的三种形式

### 形式 A：关节 residual

```text
a_wrist = a_decoder_wrist + α * tanh(c_wrist)
```

适合 right wrist / hand DOF。

### 形式 B：肌肉 residual

```text
a_muscle_final = a_decoder + M_corr * α * tanh(c_muscle)
```

其中 `M_corr` 是右前臂、腕部、手部相关肌肉 mask。

### 形式 C：末端目标 residual

```text
racket_target_pose = nominal_racket_pose + residual_pose
```

再由低层控制器或 IK/PD/muscle policy 跟踪。

### 推荐

短期建议：

```text
无拍动作复现阶段：不加 correction，或只加右腕弱 residual
```

中期建议：

```text
带拍但不打球：加入 grip/wrist residual，让手稳定持拍
```

长期建议：

```text
带球击打：加入 racket pose / racket velocity correction
```

---

# 7. 你的项目中建议的训练阶段

## Stage 1：动作数据清洗与重定向

```text
输入：WHAM/SMPL 输出
处理：
  - root 轨迹平滑
  - ground alignment
  - foot contact 修复
  - 关节角滤波
  - SMPL-to-musculoskeletal retargeting
输出：musculoskeletal reference motion dataset
```

## Stage 2：训练 muscle/body tracker teacher

```text
输入：当前肌骨状态 + reference motion target
输出：muscle activation / body action
目标：稳定跟踪参考羽毛球动作
```

## Stage 3：蒸馏 latent muscle/body skill space

```text
teacher tracker → encoder/prior/decoder
```

得到：

```text
prior(s) = [mu_p, sigma_p]
decoder(s, z) = a_body
```

## Stage 4：训练 LAB-constrained high-level PPO

```text
PPO 输出 u_latent
LAB 限制 z_t
Decoder 输出 body action
```

## Stage 5：加入末端 correction

```text
right wrist / grip / racket residual policy
```

## Stage 6：加入羽毛球任务

```text
输入 shuttlecock state / target hitting point
奖励加入击球点、拍面、拍速、球落点
```

---

# 8. 推荐的代码模块划分

建议新增如下模块：

```text
latent_muscle/
  ├── networks/
  │   ├── posterior_encoder.py
  │   ├── conditional_prior.py
  │   ├── latent_decoder.py
  │   └── lab_action_wrapper.py
  │
  ├── train/
  │   ├── train_tracker_teacher.py
  │   ├── collect_distill_buffer.py
  │   ├── train_latent_distillation.py
  │   └── train_highlevel_ppo_lab.py
  │
  ├── data/
  │   ├── motion_dataset.py
  │   ├── reference_feature_builder.py
  │   └── action_mask.py
  │
  └── configs/
      ├── tracker_teacher.yaml
      ├── latent_distill.yaml
      └── ppo_lab.yaml
```

---

# 9. 关键配置建议

## 9.1 latent 维度

建议从小到大尝试：

```text
latent_dim = 16 / 32 / 64
```

如果动作片段较简单，先用 16 或 32。

如果包括多种步法、正手、反手、带拍动作，可以尝试 64。

## 9.2 KL 权重

```text
lambda_KL = 1e-4 ~ 1e-2
```

建议使用 KL warm-up：

```text
前期小 KL，先保证 action reconstruction
后期逐渐增大 KL，让 latent space 更规整
```

## 9.3 LAB lambda

```text
lambda_lab = 0.5 ~ 2.0
```

建议：

```text
动作复现阶段：0.5 ~ 1.0
任务击球阶段：1.0 ~ 2.0
```

如果动作明显僵硬，增大 `lambda_lab`。

如果动作抖动、不自然，减小 `lambda_lab`。

## 9.4 sigma clamp

必须限制 prior 输出的 sigma：

```text
sigma_min = 0.05
sigma_max = 2.0
```

否则会出现：

```text
sigma 太小 → policy 几乎不能探索
sigma 太大 → LAB 失去约束
```

---

# 10. PPO reward 如何设计

如果是动作复现阶段：

```text
R_high =
    w_track * r_tracking
  + w_ee    * r_hand_elbow_shoulder
  + w_root  * r_root
  + w_smooth* r_smooth
  - w_effort* muscle_effort
  - w_corr  * correction_magnitude
```

如果是击球阶段：

```text
R_high =
    w_hit      * r_hit_success
  + w_contact  * r_contact_timing
  + w_racket   * r_racket_velocity_direction
  + w_landing  * r_shuttle_landing
  + w_natural  * r_body_naturalness
  - w_effort   * muscle_effort
  - w_smooth   * action_smoothness
  - w_corr     * correction_magnitude
```

注意：LAB 本身已经是硬约束，不一定需要额外加很强的 latent penalty。但可以加一个轻微项：

```text
cost_latent_residual = ||tanh(u_latent)||^2
```

用于鼓励 policy 不要总是贴着 barrier 边界跑。

---

# 11. 评价指标

为了证明这个方法对你的肌骨模型有效，建议同时评估四类指标。

## 11.1 轨迹复现指标

```text
MPJPE
joint angle error
root position/orientation error
end-effector error
phase alignment error
```

## 11.2 动作自然性指标

```text
joint acceleration
action smoothness
foot slip
self-collision
joint limit violation
```

## 11.3 肌肉合理性指标

```text
mean muscle activation
activation smoothness
peak activation
left-right coordination
proximal-to-distal activation timing
```

对正手高远球，可以重点看：

```text
trunk → shoulder → elbow → wrist 的时序
下肢蹬转 → 躯干旋转 → 上肢挥拍 的发力链
```

## 11.4 任务指标

```text
击球点误差
拍头速度
拍面方向误差
羽毛球落点误差
成功率
```

---

# 12. 消融实验建议

建议至少做以下对比：

```text
1. PPO from scratch
2. 普通 motion tracking PPO
3. tracker + latent distillation, no LAB
4. tracker + latent distillation + LAB
5. tracker + latent distillation + LAB + wrist/grip correction
```

预期结果：

```text
no LAB:
  任务可能能完成，但动作抖动，肌肉激活不自然

with LAB:
  轨迹略保守，但动作更自然，激活更平滑，稳定性更好

with correction:
  末端击球/握拍效果更好，不破坏全身动作
```

---

# 13. 常见问题与解决方案

## 13.1 Teacher tracker 本身学不好

可能原因：

```text
SMPL-to-muscle retargeting 质量差
reward 权重不合理
动作片段 root/foot contact 错误
muscle action 太高维
```

解决：

```text
先降低任务难度，只跟踪准备姿态、引拍、挥拍单片段
先用 joint target action 验证，再切 muscle action
加入 reference motion smoothing
修正 foot contact
```

## 13.2 蒸馏后动作质量下降

可能原因：

```text
latent_dim 太小
KL 权重太大
decoder 容量不足
DAgger buffer 覆盖不够
```

解决：

```text
增大 latent_dim
使用 KL warm-up
增加 decoder hidden size
增加 online rollout 数据
```

## 13.3 LAB 后动作太僵硬

可能原因：

```text
lambda_lab 太小
sigma 被 clamp 得太小
prior 学得过窄
```

解决：

```text
增大 lambda_lab
增大 sigma_min
减小 KL 权重
```

## 13.4 LAB 后仍然不自然

可能原因：

```text
lambda_lab 太大
sigma 太大
teacher 本身不自然
reward 中任务项过强
```

解决：

```text
减小 lambda_lab
减小 sigma_max
加强 action smoothness / muscle effort
提高动作数据质量
```

## 13.5 右手腕/握拍影响全身稳定

解决：

```text
body decoder 不控制右腕
训练 body tracker 时对右腕加扰动
correction residual 加幅度限制和平滑约束
```

---

# 14. 最小可行版本 MVP

如果现在要最快落地，不建议一次性做完整系统。建议分三步。

## MVP-1：只做 body tracker

目标：

```text
肌骨模型能够稳定复现正手高远球无拍动作
```

不做 latent，不做 LAB。

## MVP-2：做 tracker distillation

目标：

```text
latent decoder 可以生成接近 teacher 的 body action
```

验证：

```text
decoder rollout 的 MPJPE / joint angle error 接近 teacher
```

## MVP-3：加入 LAB high-level PPO

目标：

```text
PPO 在 latent space 中调整动作，但不会产生不自然肌肉激活
```

验证：

```text
with LAB 比 no LAB 更平滑、肌肉激活更合理、跌倒率更低
```

---

# 15. 可以直接写进论文的方法描述

本文借鉴 LATENT 中的 correctable latent action space 思想，将视频重建得到的不完美人体运动轨迹作为动作先验，而不是作为必须严格跟踪的完整真值。首先，我们基于 World-Grounded SMPL 轨迹重定向得到肌骨模型参考运动，并训练一个动作跟踪型 teacher policy，使肌骨模型能够在物理仿真中稳定复现羽毛球基础动作片段。随后，我们通过带条件变分瓶颈的在线蒸馏方法，将 teacher policy 压缩为一个 latent muscle skill space。该空间由 posterior encoder、conditional prior 和 decoder 组成，其中 decoder 根据当前肌骨状态和 latent code 输出肌肉控制信号，conditional prior 则建模当前状态下自然动作 latent 的分布。为避免高层强化学习策略在 latent space 中采样分布外动作，我们进一步引入 Latent Action Barrier，将高层策略输出限制在 conditional prior 均值附近、由状态相关标准差自适应缩放的区域内。最终，高层策略只需在自然动作先验附近进行有限探索，从而在完成羽毛球动作复现或击球任务的同时，保持肌骨动作的稳定性、自然性与肌肉发力合理性。

---

# 16. 最核心公式汇总

## Teacher tracker

```text
a_teacher_t = π_tracker(s_t, r_t)
```

## Posterior encoder

```text
E(z | s_t, r_t) = N(mu_e(s_t, r_t), sigma_e(s_t, r_t))
```

## Conditional prior

```text
P(z | s_t) = N(mu_p(s_t), sigma_p(s_t))
```

## Decoder

```text
a_body_t = D(s_t, z_t)
```

## Distillation loss

```text
L = λ_action ||a_teacher - a_body||²
  + λ_KL KL(E(z | s, r) || P(z | s))
  + λ_smooth ||a_t - a_{t-1}||²
```

## LAB

```text
raw_sigma_p = prior_scale_head(s_t)
sigma_p(s_t) = clamp(softplus(raw_sigma_p), sigma_min, sigma_max)
z_t = mu_p(s_t) + λ_lab * sigma_p(s_t) * tanh(u_t)
```

## Final action

```text
a_full_t = merge(D(s_t, z_t), a_correction_t)
```

---

# 17. 最终结论

对于你的肌骨羽毛球项目，LATENT 中最值得借鉴的不是“网球任务本身”，而是其处理不完美动作数据的控制结构：

```text
不完美视频/SMPL 动作
    → 先训练可执行的肌骨 tracker
    → 再蒸馏成 latent muscle skill space
    → 再用 LAB 约束高层 PPO
    → 最后对右腕/握拍/击球末端做 residual correction
```

它可以帮助你的模型避免两个常见问题：

```text
1. 直接追踪视频轨迹导致动力学不可行
2. PPO 为了任务奖励学出非人体肌肉发力
```

因此，在你的项目中，推荐把蒸馏 + LAB 作为 **SMPL-to-musculoskeletal imitation policy 的高级版本**，而不是简单附加 reward。它应该进入 action space 设计和 policy architecture 设计的核心部分。

---

# 18. 当前仓库融合审查与必须满足的接口契约

截至当前实现，`musclemimic.latent_muscle` 已经提供了 LATENT/LAB 的基础模块：

```text
PosteriorEncoder
ConditionalPrior
LatentDecoder
LABActionWrapper
ActionMask
```

现有仓库也已经具备 teacher rollout、BC/DAgger distillation、PPO checkpoint、body/grip layered policy 等基础设施。因此整体路线是可行的，但不能把“模块已存在”误解为“训练闭环已经完整”。真正训练 latent skill 前，必须满足下面几个契约。

## 18.1 high-level obs 与 LAB state 必须分离

高层 policy 的输入可以包含：

```text
s_t, phase, target keypoints, racket/shuttle state, landing target
```

但 `ConditionalPrior` 和 `LatentDecoder` 应该只接收训练时使用的肌骨状态表示 `s_t`。如果把完整 high-level obs 直接送进 prior/decoder，会产生隐蔽维度错配，甚至让 prior 学到任务目标泄漏而不是自然动作先验。

当前 `LatentBodyPolicy` 支持 `state_adapter`：

```python
LatentBodyPolicy(
    high_level_policy=policy,
    lab_wrapper=lab,
    state_adapter=extract_body_state,
)
```

`extract_body_state(obs_high)` 必须输出与 prior/decoder 训练时完全一致的 state 向量。

## 18.2 latent 蒸馏数据必须包含 reference_features

`PosteriorEncoder` 的输入是：

```text
E(z | s_t, reference_features_t)
```

现有 teacher/DAgger shard 默认保存 `student_obs` 和 `teacher_action`，但旧 shard 不一定有 `reference_features`。如果缺少该字段，只能做普通 BC，不能训练完整的 posterior/prior/decoder latent distillation。

新的 dataset writer/loader 会在存在 `reference_features` 时保留该字段，并在 metadata 中记录：

```text
reference_features_dim
```

后续 collector 需要显式写入 `reference_features`，或者从保存的 full obs/goal group 中离线重建。

## 18.3 body action 与 correction action 必须按 actuator 名称对齐

右腕、手指、握拍、拍面 correction 不能和 body decoder 同时控制同一个 actuator。否则 PPO 会通过两个通道互相抵消或放大，导致训练不稳定。

当前应使用：

```python
ActionMask.from_layered_router(router)
```

或在加载 mask 后调用：

```python
mask.assert_matches_partitions(...)
```

确保 latent body decoder 与 grip/correction policy 的 actuator partition 与运行时 `LayeredActuatorRouter` 完全一致。

## 18.4 LAB sigma 的实现约定

文档中的 `sigma_p(s_t)` 是正的标准差，不建议网络直接输出无约束 sigma。实现中应输出 raw scale：

```text
raw_sigma_p = scale_head(s_t)
sigma_p = clamp(softplus(raw_sigma_p), sigma_min, sigma_max)
```

这样既避免负标准差，也避免 sigma 过小导致无法探索，或 sigma 过大导致 LAB 约束失效。

## 18.5 当前仍未完成的训练闭环

当前融合已经具备数学模块和运行时 action 接口，但还没有完成以下训练脚本级闭环：

```text
1. collector 显式保存 reference_features
2. latent_distillation trainer 用 posterior/prior/decoder 优化 latent_distillation_loss
3. high-level PPO 的 action space 改为 raw latent residual
4. PPO rollout 中通过 LABActionWrapper 解码 body action
5. grip/wrist/racket correction 与 body action 合并后进入 MuJoCo
```

因此短期结论是：

```text
基础方案可行；
当前仓库已具备安全接入点；
但完整 LAB-constrained PPO 训练还需要一个专门的 latent_distillation trainer 和 PPO action wrapper。
```
cd /data3/yangfeiyang/WorkSpace/optimized_wham



IN="/data3/yangfeiyang/WorkSpace/optimized_wham/BadmintonVideos/forehand_clear/wrong"

OUT="/data3/yangfeiyang/WorkSpace/optimized_wham/output/forehand_clear/InsufficientArmExtension"



mkdir -p "$OUT"



find "$IN" -maxdepth 1 -type f -name "*.mp4" -print0 | sort -z | while IFS= read -r -d '' video; do

  base="$(basename "$video")"

  stem="${base%.*}"

  seq_out="$OUT/$stem"

  mkdir -p "$seq_out"



  python scripts/video_to_fixed_smpl_to_opensim.py \

    --video "$video" \

    --output-pth "$OUT" \

    --fps 30 \

    --device cuda \

    --pose-backend rtmpose \

    --world-grounded \

    --world-grounded-out-dir "$seq_out/world_grounded" \

    --optimize-lower-body \

    --lower-body-out-dir "$seq_out/lower_body_corrected" \

    --contact-preserving \

    --export-reference-bundle \

    --reference-bundle-out-dir "$seq_out/reference_bundle" \

    --quality-tier all \

    --skip-ik \

    --retarget-out-dir "$seq_out/opensim_retarget" \

    > "$seq_out/pipeline.log" 2>&1

done