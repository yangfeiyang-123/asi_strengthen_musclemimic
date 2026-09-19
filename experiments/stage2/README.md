# 阶段二：技能蒸馏

目标：把依赖未来参考的 T3 teacher（经球拍课程后的 100% 档 checkpoint）压成只看当前状态的低维技能，部署时不需要未来参考与在线肌电。

## 五个 arm（以代码命名为准）

| arm | 形式 | EMG context | dropout |
|---|---|---|---:|
| S2-A | direct：BC → 3 轮 DAgger → fresh-optimizer PPO | 无 | — |
| S2-B | latent，无肌电；在此选定并锁定 architecture | 无 | — |
| S2-C | latent，真实 EMG context（主方法） | 真实 | 0.30 |
| S2-D | latent，跨样本 shuffled context（负对照） | shuffled | 0.30 |
| S2-E | latent，真实 context，无 dropout | 真实 | 0 |

## 入口

- 一次且仅一次的 shared teacher collection：`musclemimic-distill-collect-teacher`（需 `--save_emg_reference`）。
- S2-A：`musclemimic/distill/stage2_direct_lifecycle.py`，配置 `fullbody/config_specific_task/distill/conf_fullbody_forehandclear_*`。
- S2-B..E：`musclemimic-latent-synergy-sweep --stage2-arm <S2-B|S2-C|S2-D|S2-E>`，family gate 在 `musclemimic/badminton/stage2_context_family.py`。
- 完整顺序与证据门：`docs/runbooks/peasd_implementation_guide.md` §4，方法细节 `docs/narrative/02_三阶段方法与肌电参与机制.md` §10–15。

## 状态

尚无任何 run（2026-09-18）。前置：阶段一 T3 晋级 + 球拍课程 100% 档 promotion。run 记录产生后放本目录，并同步 `../EXPERIMENT_LOG.md`。
