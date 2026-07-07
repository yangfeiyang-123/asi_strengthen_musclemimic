# Q:
我想在后训练损失函数中加一项：肌肉总激活最小，这个可以吗？你再分析一下，还有什么符合人体动作的约束，也可以做为损失函数的一部分呢？详细调研并分析
# A:
可以加，而且你现在的代码里已经有这项的雏形：MimicReward 里有 activation_energy_coeff，并在 reward 中计算 activation_energy = mean(square(data.act))，再作为负 penalty 加进总 reward；只是你的 fullbody 默认配置里 activation_energy_coeff: 0.0，所以目前没有启用。 ￼  ￼  ￼

但要注意：“肌肉总激活最小”不能作为无条件强约束。对正手上网挑球、杀球、急停、弓步这类羽毛球动作，短时间高激活是合理的；如果权重过大，策略会为了省力继续学成“原地挑球”或“动作幅度变小”。所以正确做法不是“让激活越小越好”，而是：在 root、击球手、脚步接触已经满足的前提下，惩罚多余激活、抖动激活和不必要共同收缩。

⸻

1. 肌肉总激活最小化是否合理？

合理。肌骨系统里肌肉数量多于关节自由度，力分配本身是冗余问题。生物力学里常用“最小肌肉努力、最小代谢能量、最小激活平方”等准则来解释肌肉力共享。相关研究也指出，显式肌肉/肌腱模型比纯关节力矩模型更容易得到人类可实现的动作，且代谢能量或激活相关代价可以被用于轨迹优化和策略优化。 ￼

你的框架尤其适合做这件事，因为 MyoFullBody 是 free-root 肌骨模型，README 里写了它有 416 条肌肉，关闭手指肌肉时约 354 条，并使用 Hill-type muscle dynamics 和 activation dynamics。也就是说，data.act 不是普通机器人的 torque command，而更接近肌肉激活状态。 ￼

推荐基础形式是：

J_{\text{act}}=\frac{1}{T}\sum_t \frac{1}{N_m}\sum_i a_i(t)^2

对应 reward penalty：

r_t = r_{\text{mimic}} - \lambda_{\text{act}} J_{\text{act},t}

在你代码里，这基本就是现有的：

activation_energy = mean(square(data.act))
activation_energy_penalty = -activation_energy
total_penalties += activation_energy_coeff * activation_energy_penalty

⸻

2. 但不要直接用“sum”，要做归一化和分阶段

你的模型肌肉数量很多，如果直接 sum(a_i^2)，reward scale 会随肌肉数量变化；如果不同动作/配置关闭手指肌肉，loss scale 也会变。更稳的是：

J_{\text{act}} = \text{mean}(a_i^2)

你当前代码已经用了 mean(square(data.act))，这一点是对的。 ￼

我建议 post-train 先从很小的系数开始：

experiment:
  env_params:
    reward_params:
      activation_energy_coeff: 0.001

然后网格试：

0.0003, 0.001, 0.003, 0.01

如果 err/root_xyz、err/right_hand_abs 或击球动作幅度变差，就说明权重过大。你代码里还把 total_penalities 裁剪到不低于 -1.0，因此很大的 penalty 会饱和，反而失去区分度。 ￼

⸻

3. 我更推荐“激活预算”而不是“无限最小化”

对羽毛球，最好把肌肉能耗写成超过合理预算才惩罚：

J_{\text{act-budget}} =
\max(0,\bar a^2(t)-b(t))^2

其中 b(t) 是该动作阶段的合理激活预算。预算可以来自三种方式：

1. 用旧策略成功 rollout 的 mean(a²) / p75(a²) 作为 budget。
2. 用动作 phase 分段：准备期、移动期、击球期、恢复期各一个 budget。
3. 用参考速度自适应：root/hand 速度越大，允许更高激活。

一个简单的 phase-aware 版本：

\lambda_{\text{act}}(t)
=
\frac{\lambda_0}{1 + \alpha\|v_{\text{root,ref}}\| + \beta\|v_{\text{hand,ref}}\|}

含义是：移动和击球越剧烈，省力约束越弱；静止、准备、恢复阶段省力约束更强。这样不会把杀球、上网、急停这种本来需要爆发力的动作压扁。

⸻

4. PPO 里应该把它加到 reward，而不是随便加到 actor loss

标准 PPO 不通过物理仿真器反传梯度。rollout 后的 data.act 已经是采样得到的环境状态；如果你只在 _update_network 里把 mean(data.act**2) 加到 loss，它通常对策略参数没有正确梯度，除非你把它变成 reward/advantage 的一部分，或者直接对当前策略输出的 action/excitation distribution 做可微正则。

所以建议分两类：

肌肉 activation、foot slip、joint acceleration、contact 这类依赖仿真状态的项：
  加到环境 reward 里。
action mean、policy std、excitation magnitude 这类直接依赖网络输出的项：
  可以作为 actor auxiliary loss。

对你现在最稳的做法是：直接启用/扩展 MimicReward 里的 activation penalty。

⸻

5. 除了总激活，还可以加入哪些人体动作约束？

我按“最适合你当前 badminton_mimic post-train”的优先级排列。

⸻

A. 肌肉激活变化率最小化：非常推荐

仅最小化 a_i^2 只能减少平均激活，但不能直接减少“肌肉忽高忽低”的抖动。人体肌肉有 activation dynamics，不会每一帧剧烈跳变。MyoSuite 这类肌骨控制环境也强调肌肉动力学、接触和真实动作任务中的神经肌肉控制复杂性。 ￼

推荐：

J_{\Delta a}
=
\frac{1}{N_m}\sum_i
\left(\frac{a_i(t)-a_i(t-1)}{\Delta t}\right)^2

或者不除以 \Delta t，只做 frame-to-frame 平滑：

J_{\Delta a}
=
\frac{1}{N_m}\sum_i
(a_i(t)-a_i(t-1))^2

这项比纯 activation energy 更不容易让动作幅度变小，因为它主要惩罚抖动，而不是惩罚正常发力。

代码上需要在 MimicRewardState 里加：

last_activation: Union[np.ndarray, jnp.ndarray]

初始化：

last_activation = data.act

reward 里：

activation_rate = backend.mean(backend.square(data.act - reward_state.last_activation))
activation_rate_penalty = -activation_rate

然后更新：

reward_state = reward_state.replace(last_activation=data.act)

建议系数：

activation_rate_coeff: 0.001 ~ 0.01

⸻

B. 肌肉力 / 关节力矩 / 机械功率约束：推荐，但要谨慎

你当前代码已经有 joint_torque_coeff，它用的是 data.qfrc_actuator，这是 actuator 作用到 generalized coordinates 后的力/力矩，不是每条肌肉自己的 force。 ￼

更生物力学的形式有三种：

1. 肌肉 force penalty

如果 MJX data 能拿到 data.actuator_force：

J_F = \frac{1}{N_m}\sum_i
\left(\frac{F_i}{F_{i,\max}}\right)^2

这比单纯激活更接近“肌肉实际用力”。但如果没有可靠的 F_{i,\max}，先不要强行做归一化。

2. 关节力矩 penalty

J_{\tau}=\frac{1}{N_q}\sum_j \tau_j^2

这会减少不自然的大力矩，但太强会让动作软、root 跟不上。你已经有 joint_torque_coeff，post-train 可以试很小值：

joint_torque_coeff: 1e-6 ~ 1e-4

3. 功率 / 代谢近似

J_{\text{power}} =
\sum_j |\tau_j \dot q_j|

或者只惩罚正功：

J_{\text{positive-power}} =
\sum_j \max(0,\tau_j \dot q_j)

近年的代谢代价研究也显示，力矩大小和力矩变化率都会影响能耗，因此 torque 与 torque_rate/activation_rate 的组合比单独 activation² 更合理。 ￼

⸻

C. 不必要共同收缩 co-contraction penalty：可选，但不能太强

人体确实会共同收缩拮抗肌来提高关节刚度，尤其是急停、击球、落地、保护膝踝时。所以不能简单地把 co-contraction 全部消掉。

如果你有 flexor/extensor 肌肉配对表，可以加：

J_{\text{coact}}
=
\sum_{(f,e)}
a_f a_e

或者：

J_{\text{coact}}
=
\sum_{(f,e)}
\min(a_f,a_e)^2

这项适合惩罚“全身同时绷紧”的假动作，但不要在击球瞬间、落地瞬间强惩罚。建议只在低速 phase 或站稳 phase 开启。

⸻

D. 关节极限和关节中心化：强烈推荐

这类约束对“动作像人”很重要，尤其是肩、肘、腕、髋、膝、踝。形式：

J_{\text{joint-limit}}
=
\sum_j
\text{ReLU}(q_j-q_j^{\max}+m)^2
+
\text{ReLU}(q_j^{\min}-q_j+m)^2

其中 m 是 margin，比如 5%～10% 的关节范围。不是等超过极限才惩罚，而是接近极限就开始惩罚。

这对你的视频 SMPL imitation 很重要，因为 SMPL/WHAM 的某些肩膀、手腕、髋关节姿态可能视觉上能解释 2D，但肌骨模型里不自然。关节极限约束可以避免 PPO 用“反关节”“过伸”“手腕乱翻”去骗 site reward。

建议权重：

joint_limit_coeff: 0.01 ~ 0.1

这项目前你的 MimicReward 里没有，需要新增。

⸻

E. 关节速度、加速度、jerk 平滑：推荐

你的代码已经有 joint_acc_coeff，计算的是当前 joint velocity 与上一帧 joint velocity 的差。 ￼

可以分三级：

J_{\dot q}=\sum_j \dot q_j^2

J_{\ddot q}=\sum_j (\dot q_j(t)-\dot q_j(t-1))^2

J_{\text{jerk}}=\sum_j
(q_j(t)-3q_j(t-1)+3q_j(t-2)-q_j(t-3))^2

对 post-train，建议先用你已有的 joint_acc_coeff，不要马上加 jerk，因为 jerk 需要更多历史状态，代码复杂度更高。

推荐：

joint_acc_coeff: 1e-5 ~ 1e-3
action_rate_coeff: 0.001 ~ 0.01

但注意你当前 acceleration_norm = sum(square(joint_vel - last_joint_vel) / env.dt)，不是除以 dt²，量纲不是严格加速度平方。它更像“速度变化惩罚”，可用，但系数要通过日志调。

⸻

F. 足底接触、穿地、滑动：对上网挑球非常重要

羽毛球上网挑球的问题很可能不是“手臂不会摆”，而是脚步推进、支撑脚落地、弓步制动没有学好。接触约束应该成为 post-train 的核心项之一。最新的 BadmintonGRF 数据集专门针对羽毛球这种非周期性 court sport，记录多视角 RGB、力台、Vicon，并构建从 2D pose 估计 GRF 的 benchmark，这也说明羽毛球动作里的 ground reaction force / impact segment 是非常关键的物理量。 ￼

推荐三项：

1. 足底穿地

J_{\text{penetration}}
=
\sum_{\text{foot sites}}
\text{ReLU}(z_{\text{ground}}-z_{\text{foot}})^2

2. 接触时脚滑

J_{\text{slip}}
=
c_{\text{foot}}(t)
\cdot
\|v_{\text{foot},xy}(t)\|^2

其中 c_{\text{foot}}(t) 可以来自参考轨迹，或者用规则估计：

foot height < threshold 且 vertical velocity 小 => contact = 1

3. swing phase 脚尖离地

J_{\text{clearance}}
=
(1-c_{\text{foot}}(t))
\cdot
\text{ReLU}(h_{\text{clear}}-z_{\text{toe}})^2

这能减少拖脚。对上网挑球，优先加 penetration + slip，clearance 可以后加。

⸻

G. 支撑面 / COM / 动态平衡约束：有用，但要做成软约束

人体站立或低速移动时，COM 投影应大致落在支撑区域内；动态移动时可以短暂超出，但需要通过迈步、弓步或躯干角动量恢复平衡。机器人和人体运动控制文献中常用 ZMP、support polygon、capture point 等方式刻画这种约束；例如 biped NMPC 工作会通过约束 ZMP 在 support polygon 内来生成鲁棒步态，同时允许 CoM 高度变化和角动量调节。 ￼

简单可落地版本：

J_{\text{COM}}
=
\text{dist}(
\text{proj}_{xy}(\text{COM}),
\text{support polygon}
)^2

只在以下阶段启用：

双脚站立
单脚支撑稳定期
击球后恢复期

不要在上网冲刺和弓步落地前强制 COM 在支撑面内，否则会阻止自然前倾和跨步。

更适合你当前框架的简化版本：

如果当前有接触脚：
  惩罚 pelvis/root 与支撑脚距离过大
  惩罚 torso pitch/roll 过大
  惩罚 root height 非自然突变

⸻

H. Root、躯干、头部姿态约束：推荐

羽毛球动作里，躯干不是随便摆的。上网挑球需要：

root 朝向接近球场/击球方向
pelvis 和 torso 协调旋转
躯干前倾但不能塌腰
头部相对稳定，视线大致朝击球方向

可加：

J_{\text{torso-roll}}
=
\text{roll}_{\text{torso}}^2

J_{\text{torso-pitch}}
=
\text{ReLU}(|\text{pitch}_{\text{torso}}|-p_{\max})^2

J_{\text{head-stability}}
=
\|\omega_{\text{head}}\|^2

但这类项不要替代 reference tracking。它们只负责“防止离谱姿态”。

⸻

I. 手 / 球拍末端约束：对羽毛球必须加

如果目标是“真实击球动作”，只优化肌肉激活和身体自然性是不够的。必须保证右手或球拍末端在击球时刻到达正确空间位置，并有合理速度方向。

建议新增：

J_{\text{hand-pos}}
=
\|p_{\text{right hand}} - p_{\text{ref right hand}}\|^2

J_{\text{hand-vel}}
=
\|v_{\text{right hand}} - v_{\text{ref right hand}}\|^2

如果能估计球或球拍头：

J_{\text{racket-head}}
=
\|p_{\text{racket head}} - p_{\text{target impact}}\|^2
+
\|v_{\text{racket head}} - v_{\text{target}}\|^2

DeepMimic 的思想本来就是 imitation objective 和 task objective 可以组合；AMP 也强调用 motion prior 保持风格，而用任务 reward 指定目标。对羽毛球，right hand / racket endpoint 就是非常自然的 task objective。 ￼

⸻

J. 参考动作可执行性 / retargetability 约束：很适合你的 pipeline

你现在的问题之一是：SMPL、retarget、PPO 三步目标不完全一致。可以在 post-train 中加入“retarget 后可执行性”相关指标：

root_xy_error
right_hand_abs_error
foot_slip
activation_energy
joint_limit_violation
early_termination_rate

而不是只看 SMPL pose error。AddBiomechanics 这类大规模数据集的动机也正是：仅有 pose 不足以恢复人体物理量，还需要力、力矩、外力等动力学信息。 ￼

⸻

6. 我建议你的 post-train 总 reward 写成这样

对你当前阶段，我建议：

R_{\text{post}}
=
R_{\text{mimic-root-heavy}}
+
R_{\text{right-hand}}
-
\lambda_a J_{\text{act}}
-
\lambda_{\Delta a}J_{\Delta a}
-
\lambda_{\text{slip}}J_{\text{foot-slip}}
-
\lambda_{\text{limit}}J_{\text{joint-limit}}
-
\lambda_{\text{acc}}J_{\text{joint-acc}}

也就是：

必须保留/加强：
  root tracking
  right hand / racket endpoint tracking
  relative pose tracking
再加入人体约束：
  muscle activation effort
  activation smoothness
  foot slip / penetration
  joint limit
  joint acceleration / action rate

不要一开始加入十几项，否则很难判断哪项有效。

⸻

7. 最推荐的第一版配置

先用你现有最好 checkpoint 做 post-train：

experiment:
  resume_from: <your_best_checkpoint>
  reset_lr_schedule_on_resume: true
  lr: 5e-5
  env_params:
    reward_params:
      # 保持或加强 root / 手部，不然 effort 会让动作变小
      root_pos_w_sum: 0.30
      root_vel_w_sum: 0.20
      rpos_w_sum: 0.40
      qpos_w_sum: 0.05
      qvel_w_sum: 0.05
      rvel_w_sum: 0.10
      # 先打开已有项
      activation_energy_coeff: 0.001
      action_rate_coeff: 0.001
      joint_acc_coeff: 0.0001
      joint_torque_coeff: 0.0
  ppo_config:
    num_steps: 64
    update_epochs: 2
    init_std: 0.5
    ent_coef: 0.0

如果 root 开始变差：

activation_energy_coeff 降到 0.0003
或者只在 err_root_xyz < 0.3 时启用 effort penalty

如果动作还抖：

action_rate_coeff 提到 0.003
新增 activation_rate_coeff

如果脚滑明显：

优先加 foot_slip，而不是继续加 muscle effort

⸻

8. 我建议优先实现的新增 penalty

第一优先级：activation rate

因为你已有 activation_energy_coeff，下一步最自然是加：

activation_rate_coeff

它比单纯 energy 更能改善自然性，而且不容易损害动作幅度。

第二优先级：foot slip / penetration

这对上网挑球、弓步、急停最关键。没有这个，策略可能靠奇怪的 root 或脚部滑动来骗 reference。

第三优先级：joint limit margin

这项能防止肩、肘、膝、踝出现视觉上不明显但肌骨上不合理的姿态。

第四优先级：right-hand / racket absolute tracking

这不是“人体约束”，但对羽毛球任务是核心。否则策略可以省力地做一个像样的相对姿态，却没有真实击球空间位置。

⸻

9. 不建议第一版加入的项

暂时不建议先加：

强 co-contraction penalty
严格 COM 必须在 support polygon 内
强 torque penalty
强 metabolic power penalty
严格左右对称约束

原因是羽毛球动作本来就是非对称、爆发、强接触、强制动的运动。过强的“人体自然性”正则会把动作变成慢速康复步态，而不是运动员击球动作。

⸻

10. 评价指标必须同步加上

每次 post-train 不要只看 reward。建议至少记录：

tracking:
  err_root_xyz
  err_root_yaw
  err_right_hand_abs
  err_site_abs
  err_rpos
muscle:
  mean_activation
  mean_activation_sq
  p95_activation
  max_activation
  activation_rate
contact:
  foot_slip_left/right
  foot_penetration
  stance_contact_duration
motion quality:
  joint_limit_violation
  joint_acc
  action_rate
  torso_pitch/roll
  root_xy_displacement_ratio

其中最关键的 Pareto 图是：

x-axis: err_root_xyz / err_right_hand_abs
y-axis: mean_activation_sq

你要找的是“tracking 不变或更好，同时 activation 降低”的点，而不是 activation 最低的点。

⸻

11. 最终建议

你的想法是对的，但我建议把“肌肉总激活最小”改成：

肌肉多余激活最小
+ 肌肉激活变化率最小
+ 脚底接触真实
+ 关节不过限
+ root/右手/球拍任务目标不退化

也就是：effort 是后训练的正则项，不是主目标。主目标仍然应该是“上网到位、击球到位、身体姿态接近参考”；effort 只负责让策略不要用过度肌肉激活、抖动激活和不必要共同收缩去完成这个动作。

# Q:
我有个问题：就是我的本身的SMPL，轨迹是存在一点问题的，我是否可以针对PPO的一些训练约束，或者后训练的训练约束，去学出更合理的动作呢
# A:
可以。如果 SMPL 本身有一点问题，PPO/post-train 确实可以学出比原始 SMPL 更合理的动作，但前提是：你不能让 policy 只做“逐帧追踪 SMPL”。你需要把训练目标改成：

\text{合理动作}
=
\text{可信的 SMPL imitation}
+
\text{物理可执行性}
+
\text{人体生物力学约束}
+
\text{羽毛球任务约束}

也就是说，PPO 可以把有噪声的 SMPL 投影到一个物理可执行、肌骨合理、任务成立的动作流形上。但是它不能从完全缺失的信息里凭空恢复真实动作。例如，如果 SMPL 的 root 前移本来就丢了，而你又没有加入 court/root/击球点/脚步约束，那么 PPO 很可能还是会学出“原地挑球”。这不是 PPO 的问题，而是目标函数没有给它“必须上网”的信号。

DeepMimic 这类工作已经证明，可以把 motion imitation objective 和 task objective 组合，让物理角色既模仿参考动作，又满足额外任务目标；AMP 进一步用 motion prior 约束动作风格，而不用死追逐帧参考；PHC 则专门强调可以从 noisy video pose 这类有噪声输入中训练鲁棒 humanoid controller。 ￼

⸻

1. 先说结论：可以修，但只能修“目标函数能表达的问题”

你可以把 flawed SMPL 分成几类。

第一类：物理不合理，但真实意图还在。
比如脚轻微穿地、脚滑、膝盖有些过伸、root 高度抖动、肌肉激活过大、动作不平滑。这类问题非常适合用 PPO/post-train 修。因为物理仿真器和肌骨模型本身会提供约束，policy 会自动找到更可执行的动作。

第二类：SMPL 局部姿态有噪声，但任务信息还在。
比如手臂角度略偏、膝盖角度不准、躯干姿态有点抖，但 root 路径、击球时刻、手部大致轨迹还对。这类也可以修。方法是降低有噪声关节的 tracking 权重，加入手部/球拍末端、脚接触、root、平衡约束。

第三类：SMPL 关键任务信息缺失。
比如上网挑球的 root 前移在 SMPL 里就没有，或者 camera scale 错导致人根本没走到网前。这种情况下，PPO 不能只靠“人体自然性约束”自动恢复真实 root。因为“站在原地挑球”可能更省力、更稳定、更容易追踪局部姿态。你必须额外加入：court 坐标、击球点、右手/球拍末端位置、root target、脚步接触序列等任务信号。

所以你的目标不是“PPO 修 SMPL”，而是：

\pi^*
=
\arg\max_\pi
\Big[
R_{\text{trusted imitation}}
+
R_{\text{badminton task}}
+
R_{\text{physics}}
+
R_{\text{human biomechanics}}
-
C_{\text{unnecessary effort}}
\Big]

⸻

2. 你现在的代码为什么容易学到 flawed SMPL？

你当前 MimicReward 本质上是 DeepMimic-style tracking reward：qpos、qvel、root_pos、relative site position、relative site orientation、relative site velocity、root velocity 等一起加权。代码里也已经有 action out-of-bounds、joint acceleration、joint torque、action rate、activation energy 这些 penalty 接口。 ￼

但默认 fullbody 配置里：

qpos_w_sum: 0.1
qvel_w_sum: 0.1
root_pos_w_sum: 0.1
root_vel_w_sum: 0.1
rpos_w_sum: 0.6
rquat_w_sum: 0.01
rvel_w_sum: 0.1
action_rate_coeff: 0.0
activation_energy_coeff: 0.0

也就是说，当前 reward 最重的是 root-relative site pose，肌肉激活、动作平滑、能耗约束默认没有打开。 ￼

这就会导致一个问题：如果 SMPL 的 root 轨迹不可靠，而局部身体姿态还可以，policy 很容易学成“相对姿态对、全局位移错”。正手上网挑球里你看到的“像站在原地挑球”，就是这种 reward 结构的典型结果。

⸻

3. 怎么让 PPO 学出比 flawed SMPL 更合理的动作？

我建议你把后训练目标分成四层。

⸻

第一层：可信 SMPL imitation，而不是全量 SMPL imitation

不要把 SMPL 每一帧、每个关节、每个 root 都同等相信。应该做 confidence-weighted imitation。

R_{\text{imit}}
=
\sum_j c_j(t)
\exp
\left(
-\beta_j
\left\|
x_j^{\text{sim}}(t)-x_j^{\text{ref}}(t)
\right\|^2
\right)

其中 c_j(t) 是可信度。

可以这样设：

高可信：
  手腕/手部在视频中清楚时
  躯干和头部
  支撑脚接触帧
  root 高度
低可信：
  被遮挡的手臂
  快速挥拍时的手腕
  单目深度不确定的 root 前后方向
  明显脚滑或穿地的帧
  WHAM confidence 低的帧

如果你暂时没有 WHAM confidence，可以用规则估计：

foot penetration 大 -> lower-body reference confidence 降低
foot sliding 大 -> foot position confidence 降低
root velocity 突然异常 -> root reference confidence 降低
joint angular velocity 爆炸 -> 对应关节 confidence 降低
2D reprojection error 大 -> 对应 joint confidence 降低

这样 PPO 就不会被错误 SMPL 死死拉住。

⸻

第二层：物理可执行性约束

这部分是最适合修 flawed SMPL 的。

3.1 足底不穿地

C_{\text{penetration}}
=
\sum_{f}
\text{ReLU}
\left(
z_{\text{ground}}-z_f
\right)^2

作用：修复 SMPL/retarget 中脚穿地、落地不稳的问题。

对你的羽毛球动作，这项非常重要。上网挑球通常有跨步、弓步、前脚支撑。如果脚穿地或接触错误，policy 会用奇怪的 root/膝盖姿态去追 reference。

⸻

3.2 接触脚不滑

C_{\text{foot-slip}}
=
\sum_f
c_f(t)
\left\|
v_{f,xy}(t)
\right\|^2

其中 c_f(t) 是 foot contact mask。

contact mask 可以来自：

1. SMPL/WHAM 原始 contact，如果有；
2. 脚高度 < threshold 且竖直速度小；
3. retarget 后 foot site 靠近地面；
4. 人工标注关键动作帧。

这项比单纯肌肉激活更重要。因为如果没有 foot-slip penalty，policy 可能通过“脚滑”实现 root 跟踪，看起来能上网，但物理上不像人。

⸻

3.3 root 高度和平滑

C_{\text{root-height}}
=
(z_{\text{root}}-z_{\text{root,ref}})^2

C_{\text{root-acc}}
=
\left\|
v_{\text{root}}(t)-v_{\text{root}}(t-1)
\right\|^2

作用：减少 root 抖动、突然漂移、过高或过低。

但是注意：对上网挑球，root 前后/左右方向不要过度平滑。否则会把真实的快速启动和急停磨掉。

⸻

第三层：人体生物力学约束

这部分就是你前面问的“肌肉总激活最小”所属的层。

4.1 肌肉总激活最小：可以，但只能作为软正则

你现在代码已经支持：

C_{\text{act}}
=
\frac{1}{N_m}
\sum_i a_i(t)^2

代码中对应：

activation_energy = backend.mean(backend.square(data.act))
activation_energy_penalty = -activation_energy

然后乘 activation_energy_coeff 加到 reward penalty 里。 ￼

建议先试：

activation_energy_coeff: 0.0003
activation_energy_coeff: 0.001
activation_energy_coeff: 0.003

不要一上来设太大。因为羽毛球动作本来需要爆发力，特别是上网、急停、弓步、杀球、反手发力。如果这项过强，policy 会选择更省力的错误动作，例如：

不往前跨
不充分转髋
手臂挥拍幅度变小
击球动作变慢
root 不跟踪

所以我更推荐“多余激活最小”，而不是“所有激活都最小”。

可以做成 gated penalty：

C_{\text{act-gated}}
=
\mathbf{1}
[
e_{\text{root}} < \epsilon_r
\land
e_{\text{hand}} < \epsilon_h
]
\cdot
\frac{1}{N_m}
\sum_i a_i^2

意思是：只有 root 和手部已经跟得上时，才强烈惩罚肌肉用力。否则前期不要让“省力”压过“动作到位”。

⸻

4.2 肌肉激活变化率：比总激活更推荐

C_{\Delta a}
=
\frac{1}{N_m}
\sum_i
(a_i(t)-a_i(t-1))^2

这项非常适合 post-train。它主要减少抖动和神经控制不自然，而不像总激活那样容易让动作变小。

建议你新增：

activation_rate_coeff: 0.001 ~ 0.01

代码上需要在 MimicRewardState 里保存 last_act，类似你现在保存 last_qvel 和 last_action。

⸻

4.3 肌肉共同收缩 penalty：谨慎使用

人体会有共同收缩，尤其是落地、急停、击球瞬间，所以不能简单地消灭共同收缩。

如果你能建立 antagonist muscle pair，例如屈肌/伸肌配对，可以加：

C_{\text{coact}}
=
\sum_{(i,j)\in \mathcal{A}}
a_i a_j

但只建议在低速恢复阶段启用，不建议在击球瞬间启用。

⸻

4.4 关节极限 margin

C_{\text{joint-limit}}
=
\sum_j
\text{ReLU}(q_j-q_j^{\max}+m)^2
+
\text{ReLU}(q_j^{\min}-q_j+m)^2

这项很重要。因为 flawed SMPL 或 retarget 可能让肩、腕、髋、膝、踝接近极限，policy 为了追 site reward 也可能学出不自然关节姿态。

推荐先加：

joint_limit_coeff: 0.01 ~ 0.1

对羽毛球尤其要关注：

肩外旋/内旋
肘过伸
腕极限翻转
髋外展/内收
膝过伸
踝过度内翻/外翻

⸻

4.5 关节速度、加速度、jerk

你现在代码已经有 joint_acc_coeff，它用 last_qvel 和当前 qvel 计算 joint velocity change penalty。 ￼

建议：

joint_acc_coeff: 0.0001 ~ 0.001
action_rate_coeff: 0.001 ~ 0.01

如果动作仍然抖，再考虑 jerk：

C_{\text{jerk}}
=
\left\|
q(t)-3q(t-1)+3q(t-2)-q(t-3)
\right\|^2

但 jerk 需要更多历史状态，第一版不必加。

⸻

第四层：羽毛球任务约束

这是最关键的一层。如果 SMPL 有问题，任务约束比人体自然性约束更能修正动作。

5.1 右手 / 球拍末端世界坐标约束

如果你只追 root-relative pose，policy 可以站在原地把手摆得像参考。但真实击球要求右手或球拍末端到达世界空间中的击球点。

建议加：

R_{\text{hand-abs}}
=
\exp
\left(
-\beta_h
\left\|
p_{\text{right hand}}^{\text{sim}}
-
p_{\text{right hand}}^{\text{ref}}
\right\|^2
\right)

如果你能估计球拍头：

R_{\text{racket}}
=
\exp
\left(
-\beta_r
\left\|
p_{\text{racket head}}^{\text{sim}}
-
p_{\text{impact}}
\right\|^2
\right)

这对上网挑球尤其有用。因为如果 root 不上网，右手世界位置很难到达正确击球点。

⸻

5.2 root/court 位置约束

如果你能从视频标定羽毛球场地，建议把 root 放到 court coordinate 里。

R_{\text{court-root}}
=
\exp
\left(
-\beta_c
\left\|
p_{\text{root},xy}^{\text{sim}}
-
p_{\text{root},xy}^{\text{court/ref}}
\right\|^2
\right)

对于正手上网挑球，可以定义几个关键帧：

准备帧：root 在中场/后场某位置
启动帧：root velocity 朝网前
击球帧：root 到达网前附近
恢复帧：root 稳定或回撤

这比盲目相信 SMPL root 更可靠。

⸻

5.3 击球相位约束

加入 motion phase 或 impact phase：

R_{\text{impact-phase}}
=
\exp
\left(
-\beta
(t_{\text{hit}}^{\text{sim}}-t_{\text{hit}}^{\text{ref}})^2
\right)

更简单的是，在击球前后窗口提高右手、躯干、root、前脚接触的权重：

t_hit - 10 到 t_hit + 10:
  right_hand_abs 权重高
  root_pos 权重高
  front_foot_contact 权重高
  effort penalty 降低

这符合运动规律：击球瞬间本来就需要较大肌肉激活，不应该强行省力。

⸻

6. 后训练时推荐的总目标

我建议你的 post-train reward 设计成：

R_{\text{post}}
=
R_{\text{trusted-SMPL}}
+
R_{\text{root/court}}
+
R_{\text{right-hand/racket}}
+
R_{\text{contact}}
+
R_{\text{balance}}
-
\lambda_{\text{act}}C_{\text{act}}
-
\lambda_{\Delta a}C_{\Delta a}
-
\lambda_{\text{limit}}C_{\text{joint-limit}}
-
\lambda_{\text{slip}}C_{\text{foot-slip}}
-
\lambda_{\text{acc}}C_{\text{joint-acc}}

更具体一点：

必须有：
  root tracking / court target
  right hand or racket endpoint tracking
  relative body pose tracking
  foot contact / no slip
作为正则：
  muscle activation energy
  activation rate
  action rate
  joint acceleration
  joint limit margin

不要只加 activation_energy。如果只加肌肉总激活，policy 最容易变得更省力但更不像真实动作。

⸻

7. 训练策略：不要一次性全加

我建议分三阶段 post-train。

⸻

阶段 1：纠正 root 和击球点

目标是先让动作“到位”。

root_pos_w_sum: 0.30
root_vel_w_sum: 0.20
rpos_w_sum: 0.35
qpos_w_sum: 0.05
qvel_w_sum: 0.05
rvel_w_sum: 0.10
activation_energy_coeff: 0.0
action_rate_coeff: 0.001
joint_acc_coeff: 0.0001

如果你已经有 right-hand absolute reward，则这一阶段也打开：

right_hand_abs_w_sum: 0.2 ~ 0.4

这一阶段不要强行省力。

⸻

阶段 2：加入人体自然性约束

当 root 和右手基本到位后，再打开：

activation_energy_coeff: 0.0003 ~ 0.001
activation_rate_coeff: 0.001 ~ 0.003
joint_limit_coeff: 0.01
foot_slip_coeff: 0.01

这一阶段的目标是让动作更平滑、更像人、更少肌肉乱用。

⸻

阶段 3：精修肌肉效率

最后才逐步提高：

activation_energy_coeff: 0.001 -> 0.003
activation_rate_coeff: 0.003 -> 0.01

但必须监控：

err/root_xyz
err/right_hand_abs
root_xy_displacement
foot_slip
mean_activation_sq

如果 root 或右手误差变大，就说明 effort 太强。

⸻

8. 也可以用 constrained PPO，而不是固定 reward 权重

如果你不想手调一堆 penalty 权重，可以把问题写成 constrained MDP：

\max_\pi
\mathbb{E}
[
R_{\text{task}} + R_{\text{imit}}
]

subject to:

\mathbb{E}[C_{\text{activation}}] < b_a

\mathbb{E}[C_{\text{foot-slip}}] < b_f

\mathbb{E}[C_{\text{joint-limit}}] < b_j

然后用 Lagrangian：

R'
=
R
-
\lambda_a(C_a-b_a)
-
\lambda_f(C_f-b_f)
-
\lambda_j(C_j-b_j)

如果某个约束超预算，自动增大对应 \lambda；如果约束满足，则减小 \lambda。P3O 这类工作就是把 PPO 和约束代价结合，避免完全靠固定 penalty 手调。 ￼

对你来说最适合的是：

主 reward：
  tracking + root/court + hand/racket
cost：
  activation_energy
  activation_rate
  foot_slip
  joint_limit
  joint_acc

这样不会出现“为了省力牺牲击球”的问题，因为省力是约束预算，不是主目标。

⸻

9. 如果 SMPL 错得比较明显，该怎么处理 reference？

这时我建议你不要让 policy 直接追 flawed SMPL，而是做一个“参考轨迹修正层”。

方法 A：robust tracking loss

把 MSE 换成 Huber / truncated loss：

\rho(e)
=
\begin{cases}
e^2, & |e| < \delta \\
2\delta |e|-\delta^2, & |e| \ge \delta
\end{cases}

大错误不会把 policy 强行拉向错误 reference。

方法 B：reference confidence

对每个时间、每个 body part 加权：

c_root(t)
c_left_foot(t)
c_right_foot(t)
c_right_hand(t)
c_torso(t)

比如 foot sliding 明显的帧，降低 foot reference 权重，但提高 no-slip physics penalty。

方法 C：policy-generated corrected reference

可以迭代：

1. 用 flawed SMPL 训练初始 policy。
2. 用 post-train constraints 得到更物理的 rollout。
3. 把 rollout 的 qpos/site_xpos 作为 corrected trajectory。
4. 再用 corrected trajectory 做一轮 imitation fine-tune。

这相当于用物理策略“过滤”有噪声 SMPL。SimPoE 这类方法也是把视觉/kinematic estimate 和 physics-based control 结合起来，用 dynamics 约束得到更物理合理的人体运动。 ￼

⸻

10. 最适合你当前问题的具体建议

针对你说的“SMPL 轨迹存在一点问题”，我建议你按这个顺序做。

先加，不容易出问题的项

1. action_rate penalty
2. joint_acc penalty
3. activation_rate penalty
4. joint_limit margin
5. foot penetration
6. foot slip during contact

这些一般会让动作更自然，不太会改变动作语义。

谨慎加的项

1. activation_energy
2. joint_torque
3. co-contraction
4. COM support polygon

这些过强会让动作保守、慢、幅度小。

必须加的羽毛球任务项

1. root/court position
2. right hand absolute position
3. right hand velocity or racket head velocity
4. impact phase
5. front foot contact / lunge stance

如果你不加这些，PPO 只能修“像不像人”，但不能保证“是不是上网挑球”。

⸻

11. 推荐第一版 post-train reward

你可以先做一个非常实用的版本：

R
=
0.30R_{\text{root-pos}}
+
0.20R_{\text{root-vel}}
+
0.35R_{\text{rpos}}
+
0.10R_{\text{rvel}}
+
0.20R_{\text{right-hand-abs}}
-
0.001C_{\Delta a}
-
0.001C_{\text{action-rate}}
-
0.0001C_{\text{joint-acc}}
-
0.01C_{\text{foot-slip}}
-
0.01C_{\text{joint-limit}}

然后先不要开 activation_energy。等动作到位后，再加：

activation_energy_coeff: 0.0003

如果没退化，再试：

activation_energy_coeff: 0.001

⸻

12. 你要避免的错误

最危险的是这样：

flawed SMPL tracking
+ muscle activation minimum
+ no root/court/hand task constraint

这会鼓励 policy 学出：

更省力
更慢
更小幅度
更稳定
但不一定是真实羽毛球动作

也就是你已经遇到的“原地挑球”会更严重。

正确方向是：

先让动作任务成立：
  root 到位
  手到位
  脚步接触合理
再让动作更像人：
  不穿地
  不脚滑
  不关节过伸
  不肌肉乱激活
  不抖动

⸻

13. 最终回答

可以通过 PPO/post-train 约束学出比原始 SMPL 更合理的动作。
但它不是自动修复，而是一个“带物理和任务先验的优化问题”。

如果 SMPL 只是有小噪声，下面这些约束可以明显改善动作：

muscle activation energy
activation rate
joint limit
joint acceleration
action rate
foot penetration
foot slip
root height
torso/head stability
COM/support balance

如果 SMPL 的 root 或击球点有问题，则必须额外加入：

root/court target
right-hand/racket endpoint
impact phase
front-foot contact
root velocity direction

对你的羽毛球任务，我最建议的后训练策略是：

第一阶段：root + right hand + foot contact 修正动作语义
第二阶段：joint limit + foot slip + activation rate 修自然性
第三阶段：小权重 muscle activation energy 修肌肉效率

这样 policy 才有机会从“不完美的 SMPL”中学出“更合理的人体动作”，而不是更省力地复现 SMPL 的错误。

---

# 动作分层：哪些动作适合初训，哪些适合后训练

第一版不要只按动作名字决定训练阶段，而要按“root 位移、root 峰值速度、root yaw、动作标签和人工 hint”做保守分层。SMPL 可观测性、脚步接触质量、手部末端是否可信这几类信息，当前不是自动完整识别，而是通过 `action_stage_hints.yaml` 里的 `contact_unreliable`、`endpoint_unreliable`、`fine_hand_dominant` 等 hint 显式标出来。

适合放进初训/base training 的动作：

- 正手高远球、ForehandClear。
- Root 位移中等、脚步稳定的 Backhand。
- 原地或小步吊球。
- 不含明显起跳和重落地的站立杀球。
- 基础侧移、启动、恢复、split-step、小后撤，前提是 retarget 后脚步和 root 稳定。

适合用已有 checkpoint 后训练/post-train 的动作：

- ForehandNetLift、上网挑球、网前上步。
- 弓步、跨步、并步、急停、恢复步法。
- 后场后撤接吊、后撤高远、后撤后恢复。
- 起跳杀球、重落地杀球。
- 大幅转体的 backhand 或 smash。

暂时不作为主目标的动作：

- 很细小的搓球。
- 主要由手腕、手指、拍面角度决定的网前小技术。
- 身体几乎不动、SMPL 无法观测关键差异的假动作。

推荐先运行：

```bash
.venv/bin/python musclemimic/badminton/scripts/recommend_action_stages.py \
  --cache-root caches/AMASS/MyoFullBody/gmr \
  --manifest manifests/ForehandClear/raw_list.txt \
  --manifest manifests/Backhand/best_list.txt \
  --manifest manifests/ForehandNetLift/best_list.txt \
  --manifest manifests/Smash/best_list.txt \
  --hints manifests/action_stage_hints.yaml \
  --output outputs/action_stage/recommendations.json \
  --summary-output outputs/action_stage/summary.json
```

再生成训练阶段 manifest：

```bash
.venv/bin/python musclemimic/badminton/scripts/build_stage_manifests.py \
  --recommendations outputs/action_stage/recommendations.json \
  --output-dir manifests/generated
```

再生成论文 claim 证据模板：

```bash
.venv/bin/python musclemimic/badminton/scripts/build_claim_evidence_template.py \
  --output outputs/action_stage/claim_evidence_template.json
```

解释标准：

- `base`：适合初训，用来学习通用肌骨控制和基础羽毛球身体模式。
- `posttrain`：适合从已有 checkpoint 微调，通常需要更强 root、右手末端、足底接触和自然性约束；如果后续有可靠球拍标定，再加入球拍末端约束。
- `repair`：动作有价值，但 reference root、脚步、接触或手部末端不可信，应先修数据。
- `exclude`：当前 SMPL 表达不了关键技术细节，不适合作为主要训练目标。

当前自动生成结果在：

- `manifests/generated/base_general_list.txt`
- `manifests/generated/posttrain_general_list.txt`
- `manifests/generated/posttrain_net_frontcourt_list.txt`
- `manifests/generated/posttrain_rotation_list.txt`
- `manifests/generated/repair_list.txt`

`outputs/action_stage/recommendations.json` 是被 `.gitignore` 忽略的中间诊断报告，用来追溯每条 motion 的 `metrics`、`hints`、`stage`、`family`、`reasons` 和 `cache_file`；真正提交给训练流程交接的是 `manifests/generated/*.txt`。每次重新生成后，都应该检查 `manifests/generated/*.txt` 的 git diff，确认动作分桶变化是预期的。

增强后的 recommendation report 还包含 `confidence`、`failure_modes`、`review_required` 和 `required_action`。`confidence=high` 表示当前自动分桶规则下可直接使用；`confidence=medium` 通常表示靠近阈值的边界样本，不应该单独支撑强论文结论；`confidence=low` 表示应优先 repair、人工复核或排除。`outputs/action_stage/summary.json` 是快速审计入口：如果 `review_required_count` 很高，先检查这些动作，再把 manifest 用于训练。

论文 claim 应该用 `outputs/action_stage/claim_evidence_template.json` 作为训练结果和结论之间的契约。在 `all_mix`、`action_name_grouping`、`metric_gated_staging` 以及模板中的 ablation 都有指标之前，不要声称分阶段训练优于简单方案。如果 ablation 与假设矛盾，应该削弱 claim，而不是只调阈值。

如果某个 bucket 没有动作，例如当前没有生成 `exclude_list.txt` 或 `posttrain_smash_list.txt`，这是正常的：生成脚本只写非空 manifest，避免空文件被误用。
