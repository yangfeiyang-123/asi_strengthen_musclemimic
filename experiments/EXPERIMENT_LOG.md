# PEASD 实验进度总表（Stage 1 主表 + Stage 2 调度）

> 本表保留服务器 9 的原始实验主表与最新运行记录；路径整理不构成新的实验验收。另一工作站随 `74bf895` 提交的记录单独保存在 [EXPERIMENT_LOG_msclab.md](archive/msclab/EXPERIMENT_LOG_20260918.md)，其中本机路径和排期不适用于服务器 9。

> Stage 1 最后核对：2026-10-08；旧版15/15 DONE，v2 9/9 DONE（T3/T4 seed2恢复后终点核验通过）；Stage 2 最后核对：2026-09-02 12:58。
> 范围：训练叶进度分母仍**只统计 Stage 1**。同时保留 15 个 seed-0 工程叶口径，并新增正式 seeds `0/1/2` 的 45 叶口径；Stage 2 在第 11 节单独记录，不进入 Stage 1 分母。
> 统计单位：一个 `action × arm × seed` 是一个独立训练叶；validation frame、episode、trajectory、旧 checkpoint 和分析终点都不是额外 seed。
> 重要限制：旧版 Forehand Clear 的 T0–T4 × seeds0/1/2 共 15 叶已完成；anchor v2 的 T1/T3/T4 × seeds0/1/2 共 9 叶单独统计，不能用旧版结果替代。


## 最后两组正式完成（2026-10-08 14:58，北京时间）

**anchor-v2所需9/9组现已全部完成；旧版15/15完成统计不变。** 本次T3/T4 seed2恢复修复后均精确到达800,010,240步，finalized checkpoint为update39063；原始运行、10月4日恢复及本次终点恢复三段累计各65次数值验证，完整终点证据与manifest/source/config/checkpoint内容绑定核验通过。

| 实验 | GPU / 本次PID | 最终步数 | 累计验证 | 终点finalized | 正常退出 |
|---|---|---:|---:|---|---|
| T3 v2 seed2 | 2 / 3297507 | 800,010,240 | 65 | 14:57:07 | 14:57:47，exit=0 |
| T4 v2 seed2 | 3 / 3297591 | 800,010,240 | 65 | 14:57:12 | 14:57:52，exit=0 |

14:58宿主机复核：两张GPU均无计算进程，两个tmux pane均exit=0；W&B均finished且最终step=800010240。恢复于14:51:54派发，包含保存与验证约6分钟；修复准备与回归检查另计。没有使用超预算39650 checkpoint，保留全部旧结果；本次未上传新权重，也不代表teacher promotion。

修复在隔离SHA `3ebd56f6d92fa27e07b576c277dd4502c182fb20`中完成：强制绝对总预算、恢复前断言实际剩余更新和终点、为39040父checkpoint绑定完整祖先历史。奖励、终止、数据和PPO更新规则不变，优化器及计数精确恢复；视频继续关闭、数值验证开启，Orbax save/restore各4GB。测试为119通过、3项旧视频资产测试跳过；源码检查717通过；完整GPU预检、dry-run、实际预算检查和五项启动验收通过。

- [最终完成核验，含两组manifest/checkpoint完整路径与内容身份](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/completion_status.json)
- [宿主机退出与W&B完成状态](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/final_host_status.json)
- [T3日志](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/logs/t3_s2_v1.log) · [T4日志](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/logs/t4_s2_v1.log)
- W&B：[T3 seed2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s2r3) · [T4 seed2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s2r3)。

以下14:54启动记录及更早失败记录仅为历史，当前以本节完成核验为准。

## 最后两组终点恢复（2026-10-08 14:54，北京时间）

**T3/T4 v2 seed2已恢复并通过五项启动验收，v2仍为7/9完成、2/9训练中。** 用户本次明确要求修复恢复。两组在本机空闲GPU2/3从各自update39040 / 799,539,200步继续，保留模型与优化器；日志确认train_step与schedule_count均4,997,120、schedule_offset=0，实际剩余23次更新、绝对终点800,010,240步。关闭视频、保留数值验证；三段历史累计应为65条，不新增seed。Orbax save/restore各4GB。

新隔离工作树`.local/worktrees/anchor_v2_endpoint_20261008`，SHA `3ebd56f6d92fa27e07b576c277dd4502c182fb20`，fingerprint `82f52c82ae7aa238d21459550c6c7e680dd6f49c5f1ce4b95232300afdadebe0`。修复仅涉及绝对预算防护、内容绑定恢复边界和累计验证历史；旧冻结工作树和所有checkpoint不变，reward/data/termination/PPO更新规则不变。119项针对性测试通过、3项旧视频测试跳过；717项源码检查通过；两卡完整预检、dry-run与实际预算解析通过。

截至14:54:34，两组W&B均running且到799,600,640步；各80+20条轨迹本地命中、manifest/source身份正确，优化器精确恢复且在指定GPU，无fatal/NaN/OOM。T3 PID3297507、T4 PID3297591。尚未生成正式终点，不能提前记为完成。

- W&B：[T3 seed2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s2r3) · [T4 seed2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s2r3)。
- tmux socket：`/tmp/musclemimic_anchor_v2_recovery_20261008.sock`；sessions：`stage1_fc_anchor_v2_recovery_t3_s2_20261008`、`stage1_fc_anchor_v2_recovery_t4_s2_20261008`。
- 运行包：`artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008`；日志：`logs/t3_s2_v1.log`、`logs/t4_s2_v1.log`。
- [启动验收及完整manifest路径](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/startup_acceptance.json) · [运行合同](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/matched_contract.json) · [修复与验证记录](../artifacts/stage1_fc_anchor_v2_local9_endpoint_20261008/release_status.json)。

## 两台服务器训练复核（2026-10-08 14:28，北京时间）

**v2仍为7/9完成；本机剩余T3/T4 seed2均已异常退出，没有达到正式终点。** 旧版15/15完成沿用既有核验。本机宿主机实时检查确认两组Python PID已消失，两个tmux pane均dead、exit=1，GPU2/3均无计算进程（利用率0%、各15MiB）。退出记录分别为10月6日02:24:21、02:43:42。

两组最后预算内且已记录验证的checkpoint均为update39040 / 799,539,200步（99.94%），父子验证累计64/65条；manifest及验证历史的身份绑定通过本次核对。两组随后均保存了超预算update39650 / 812,032,000步，并报`Stage1 validation history entry is off the fixed schedule`。两组均不存在checkpoint_39063，不能将超预算checkpoint计为正式完成。恢复预算根因沿用10月6日复现结论；后续需修正绝对预算及恢复边界合同，再从39040补23次更新和末次验证。

远端A100 T1 seed1/seed2已有10月4日终点核验：均800,010,240步、65次验证，分别于10月2日05:38和10月3日04:24完成；10月5日权重上传及远端文件哈希核验也已完成。本次SSH到172.18.22.7认证失败，W&B宿主机查询ReadTimeout，故远端完成结论沿用已有终点与上传证据，未重新确认远端当前进程或新增任务。

本次仅检查并同步记录，未启动、停止或修改训练，未上传权重。

证据：[本机checkpoint与历史](../artifacts/experiment_status_20261008/local.json) · [宿主机与访问限制](../artifacts/experiment_status_20261008/host.json) · [W&B查询](../artifacts/experiment_status_20261008/wandb_host.json) · [远端既有完成证据](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/remote_progress_20261004.json) · [已上传权重校验](../artifacts/hf_completed_upload_20261005/remote_verification.json)。

## 恢复预算根因与宿主机复核（2026-10-06 02:29）

v2仍7/9完成；剩余两组均已越过合同预算，不能按正常训练中或正式完成报告。02:28 W&B：T3 failed、812,032,000步；T4 running、809,840,640步。02:29宿主机实查：T3 PID1221938已消失、tmux退出码1、GPU2无计算进程；T4 PID1221866仍在GPU3，利用率100%，约5988MiB显存。

前次无法读取GPU是执行沙箱隔离：沙箱中无`/dev/nvidia*`且看不到训练PID；沙箱外只读`nvidia-smi`和`ps`均成功。这不是此次训练异常的驱动故障证据，前节实时状态未知的限制已由本节解除。

根因已通过实际manifest与冻结源码函数的CPU复现确认：显式resume令`apply_resume_resets=True`，进而`preserve_checkpoint_target=False`；两组配置缺少`total_timesteps_is_absolute`，预算解析走`configured_total_timesteps + base_global_ts0`，得到T3 1,212,272,640步、T4 1,199,779,840步。`extend_completed_run=false`不约束该显式恢复分支。启动日志两组均显示remaining_updates=39063，正确应分别18933/19543。恢复预检及启动验收仅核对声明预算，未断言解析后的实际终点，导致漏检。

因此从39040继续跑到了下一常规验证点39650，而不是在39063执行末次23更新与终点验证。T3在39650的验证历史追加被固定调度校验拒绝；截至检查时T4尚未出现同一报错，但已处于相同超预算路径。数值训练并非因NaN/OOM在此失败。

两组39040 checkpoint及累计64/65条验证记录保留，39063不存在。恢复修复应固定绝对预算，从预算内checkpoint补23次更新和末次验证，并为新恢复边界建立新的内容绑定、运行身份及门禁；当前恢复合同仅接受旧20130/19520两个父边界，不能只改一个命令参数就宣称可安全恢复。不得使用39650作为正式终点或覆盖冻结工作树。本次仅诊断与记录，未停训、重启或修改训练源码。

证据：[宿主机](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/host_diagnosis_20261006.json) · [W&B](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/wandb_diagnosis_20261006.json) · [预算复现](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/budget_root_cause_20261006.json)。

## 训练末尾异常复核（2026-10-06）

v2仍为7/9已完成；T3/T4 seed2均未发现正式终点checkpoint_39063，不能记为完成。此前7组与旧版15组沿用既有完成核验。

两组均已保存并记录验证至update39040 / 799,539,200步（99.94%），父子历史合计均64/65条。T3随后越过800,010,240步预算，保存update39650 / 812,032,000步，并在追加验证历史时抛出`Stage1 validation history entry is off the fixed schedule`；该超预算checkpoint不属于合规终点。T4可见日志截至10月6日01:17，尚无终点或相同报错。两组history中的target_global_timestep分别为1,212,272,640和1,199,779,840，与固定总预算不一致，需诊断恢复预算/末次验证调度。

本次执行环境无法访问NVIDIA驱动，未能确认宿主机实时GPU/进程或W&B状态，不能据此断言T4正常运行或已停止。未启动、停止、修改训练或上传权重；保留全部checkpoint。历史ETA已失效。

[本次文件证据](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/progress_20261006.json)。

## 中午训练复核（2026-10-05 12:16）

v2仍为7组完成、2组正常运行，尚未全训完。

| 实验 | 实时步数 | 实时进度 | 已保存进度 |
|---|---:|---:|---:|
| t3_s2 | 704,593,920 | 88.07% | 87.45% |
| t4_s2 | 686,694,400 | 85.84% | 84.33% |

累计数值验证T3为56/65、T4为54/65；两组W&B步数较前次增长、指定GPU进程存在，未检出训练异常。近期速度外推完成时间：t3_s2 2026-10-05T23:02:44.358339+08:00, t4_s2 2026-10-06T01:19:30.713446+08:00，仅供参考。[实查证据](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/progress_20261005_noon.json)。

## 新增两组v2最终权重已上传（2026-10-05）

T1 v2 seed1、seed2已上传并通过远端逐文件大小与LFS SHA256/Git blob hash校验。完成时间：2026-10-05T03:24:13.412274+08:00；[提交](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/commit/ec9e8dda4930d91674110f65978cbd4fb1ffcae9)。

| 实验 | 最终步数 | 验证次数 | Hugging Face |
|---|---:|---:|---|
| T1 v2 seed1 | 800,010,240 | 65 | [checkpoint](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed1/T1/checkpoint_39063) |
| T1 v2 seed2 | 800,010,240 | 65 | [checkpoint](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed2/T1/checkpoint_39063) |

本次新增34个文件、803,305,189字节，包含31个原生Orbax checkpoint文件及索引/校验和；模型、优化器、config、metadata和隐藏元数据完整保留。原有338个内容文件保持不变，另核验`.gitattributes`只追加7条新文件LFS规则。未上传私有原始数据或资产。

目前已上传22组独立最终权重（旧版15+v2 7），另有1份旧版重复运行备份，共23份完成权重。T3/T4 v2 seed2仍在训练，中间checkpoint未上传。本次没有停止或修改训练。

[上传回执](../artifacts/hf_completed_upload_20261005/remote_verification.json)。

## 训练进度复核（2026-10-05）

v2仍为7/9完成、2/9运行；两组关闭视频后的恢复训练正常推进，无新增完成组。

| 实验 | GPU | 已保存步数 | 已保存进度 | 累计数值验证 | 最近保存 |
|---|---|---:|---:|---:|---|
| t3_s2 | 4090 / 2 | 612,147,200 | 76.52% | 49/65 | 01:49:32 |
| t4_s2 | 4090 / 3 | 599,654,400 | 74.96% | 48/65 | 02:14:53 |

实时核验时间：2026-10-05T02:55:11.230983+08:00；W&B分别为621,752,320、605,409,280步，两个指定进程在线，GPU利用率均100%，日志未检出NaN/OOM/Traceback。恢复后各新增16次数值验证，未再次卡在视频。按近期保存速度估计T3在10月5日23:04、T4在10月6日01:20完成，仅为速度外推。A100两组完成沿用10月4日终点核验，本次未重新检查远端进程。

[本次证据](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/progress_20261005.json)。

## 空卡恢复训练（2026-10-04，北京时间）

**v2 共9组：7组已完成，剩余T3 seed2、T4 seed2已在本机空闲4090 GPU2/3启动恢复。** A100 T1 seed1/2终点均为800,010,240步、65次验证，身份核验通过；此前完成的5组不重复运行。

| 实验 | 主机 / GPU | 状态 | 恢复起点或最终步数 | 已保存进度 |
|---|---|---|---:|---:|
| T3 seed2 | 本机4090 / 2 | 恢复训练中（启动验收通过） | 412,262,400 | 51.53% |
| T4 seed2 | 本机4090 / 3 | 恢复训练中（启动验收通过） | 399,769,600 | 49.97% |
| T1 seed1 | A100 / 2 | 已完成，10月2日05:38 finalized | 800,010,240 | 100% |
| T1 seed2 | A100 / 3 | 已完成，10月3日04:24 finalized | 800,010,240 | 100% |

截至2026-10-04T03:17:01.222075+08:00，W&B已上报T3 seed2 412,549,120步、T4 seed2 400,056,320步，均超过父checkpoint；下表的51.53%/49.97%为已保存恢复起点，新运行尚未到下一次保存边界。两组80训练+20验证本地轨迹、manifest/source身份、正确GPU进程、优化器恢复和在线推进五项验收通过，未检出NaN/OOM/Traceback。

用户授权恢复参数、优化器与步数，并仅关闭视频输出，数值验证保持开启。旧PID1883497/1883506和CUDA context已消失；本次启动前GPU2/3无进程，各空闲24,067MiB。未停止其他用户任务或重置GPU。

恢复工作树为`.local/worktrees/anchor_v2_resume_20261004`，SHA `ed8be942190033a0c5d330e3bac974dd844581f1`；新W&B run为`l9av2t3s2r2`和`l9av2t4s2r2`。已核验日志中的exact optimizer restore，T3 train_step/schedule_count=2,576,640，T4=2,498,560，schedule_offset=0。总预算不扩展，父子验证历史合计仍为65条。

首次恢复被旧视频写回配置造成的动作合同标识差异拦截，未还原或更新权重。重建训练/视频环境确认原因后，在新隔离checkout加入仅对这两个内容绑定checkpoint有效的恢复检查；原checkpoint、manifest和失败日志均保留。针对性测试119通过/3资产相关跳过，源码检查717通过，两卡完整预检与dry-run通过。

[启动验收](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/startup_status.json) · [恢复交接与诊断](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/HANDOFF.md) · [A100完成证据](../artifacts/stage1_fc_anchor_v2_local9_resume_novideo_20261004/remote_progress_20261004.json)。A100新增两个终点尚未上传HF。

以下为历史记录，早期“5组完成/两组停滞”已被本节取代。

## 训练复核（2026-09-30 10:36，北京时间）

v2仍为5组完成、2组运行、2组停滞。没有新增完成组。

| 实验 | 主机 | 已保存进度 | 验证次数 | 状态 |
|---|---|---:|---:|---|
| T3 seed2 | 本机 | 51.53% | 33/65 | 停滞约48小时 |
| T4 seed2 | 本机 | 49.97% | 32/65 | 停滞约48小时 |
| T1 seed1 | A100 | 65.59% | 42/65 | 运行中 |
| T1 seed2 | A100 | 56.22% | 36/65 | 运行中 |

本机两组日志/检查点均仍停在9月28日上午，主线程仍等待down_interruptible；NVIDIA modeset错误持续。GPU2/3整卡利用率100%不能证明本用户任务恢复。A100两组检查点较昨日分别增加14.05/12.49个百分点；近期速度外推T1 seed1约10月2日01:57、seed2约10月3日22:31完成，后者较昨日减速，预计时间不稳定。本次未停训、重启或重置GPU。

[本次证据](../artifacts/experiment_status_20260930/训练进度.md)。

## 训练进度复核（2026-09-29 16:42，北京时间）

诊断补充（09-29 16:47）：两组均阻塞于验证视频rollout阶段；本机存在持续NVIDIA modeset错误，GPU0/1/4状态异常，另有明确GPU Reset Required记录。驱动/图形路径故障为最强嫌疑，尚未取得调用栈确认具体阻塞调用。当前同卡其他任务晚于停滞启动，不能视为初始原因。[完整诊断](../artifacts/experiment_status_20260929/停滞诊断.md)。本次未中断训练或重置GPU。


**v2 共9组：5组已完成、A100两组运行、本机两组停滞、0组待启动。** 此前完成的T1 seed0、T3 seed0/1、T4 seed0/1已于9月27日上传并通过远端校验；本次没有新增完成组。旧版15/15完成的历史核验不变。

| 实验 | 主机 / GPU | 状态 | 已保存步数 | 进度 | 验证次数 | 最新checkpoint时间 |
|---|---|---|---:|---:|---:|---|
| T3 seed2 | 本机 / 2 | 停滞 | 412,262,400 | 51.53% | 33/65 | 2026-09-28T11:00:15 |
| T4 seed2 | 本机 / 3 | 停滞 | 399,769,600 | 49.97% | 32/65 | 2026-09-28T10:35:11 |
| T1 seed1 | A100 / 2 | 运行中 | 412,262,400 | 51.53% | 33/65 | 2026-09-29T16:06:34 |
| T1 seed2 | A100 / 3 | 运行中 | 349,798,400 | 43.72% | 28/65 | 2026-09-29T15:52:15 |

本机T3/T4 seed2的checkpoint和日志已约29.7/30.1小时未更新，均停在验证视频400步rollout入口；Python主线程为sleeping、等待点down_interruptible，实查GPU2/3利用率均0%。原PID1883497/1883506仍在指定GPU，不能仅凭PID存活认定训练正常推进。两张卡另有其他用户进程4021581/4144559，但尚无充分证据认定其为停滞原因。本次未停止、恢复或重启任何训练，也未操作其他用户进程。

A100 PID530302/530230仍在指定GPU2/3，两卡利用率100%，checkpoint保存于今日16:06/15:52。按最近三个保存间隔估算，T1 seed1约10月2日01:49、seed2约10月2日16:26完成；仅作当前速度外推。本机两组停滞，不给出可靠完成时间。四组manifest/source/config/metadata与验证历史绑定检查通过，近期日志未检出训练NaN/OOM/Traceback。未查询W&B API，本次进度统一使用已finalized checkpoint。

[本机证据](../artifacts/experiment_status_20260929/local.json) · [A100证据](../artifacts/experiment_status_20260929/remote.json) · [汇总](../artifacts/experiment_status_20260929/summary.json)。

## 五组v2完成权重已上传（2026-09-26）

<!-- HF_UPLOAD_20260926_BEGIN -->
**新增5组anchor-v2最终checkpoint已上传并通过远端逐文件验证**（2026-09-27T15:38:31.229467+08:00）。[提交](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/commit/77ca53e06d0f6407c4e730ff66b7d7314ac283a6)。

共73个新增payload文件，其中70个checkpoint文件、2,007,892,604字节。所有payload均按远端大小及LFS SHA256/Git blob hash核验；上传前已有266个文件已全部核对，265个内容保持不变，`.gitattributes`如有变化仅为已校验的新checkpoint LFS规则追加。没有覆盖旧版权重。

| v2实验 | seed | 来源 | Hugging Face权重 |
|---|---:|---|---|
| T1 | 0 | local9 | [checkpoint_39063](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed0/T1/checkpoint_39063) |
| T3 | 0 | remote7 | [checkpoint_39063](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed0/T3/checkpoint_39063) |
| T3 | 1 | local9 | [checkpoint_39063](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed1/T3/checkpoint_39063) |
| T4 | 0 | remote7 | [checkpoint_39063](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed0/T4/checkpoint_39063) |
| T4 | 1 | local9 | [checkpoint_39063](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/tree/main/anchor_v2/seed1/T4/checkpoint_39063) |

每组均为finalized update39063 / 800,010,240 steps、65条验证历史，source SHA38bb5c7及manifest/config/metadata绑定核验通过。保留完整Orbax train_state、config、metadata、optimizer与隐藏元数据。源checkpoint、日志和正在训练的四组v2保持原状；没有上传原始轨迹/EMG、SMPL或私有资产清单。完成训练不等于teacher promotion。

本次上传由用户重新明确授权，取代此前“暂不上传权重”的安排。仓库现已覆盖旧版15组独立终点及1组A100重复备份，加上本次5组v2独立终点。

[v2完成索引](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/ANCHOR_V2_COMPLETED_20260926.md) · [v2身份与文件哈希](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/ANCHOR_V2_COMPLETED_20260926.json) · [校验和](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/ANCHOR_V2_SHA256SUMS_20260926)。

本地回执：[上传状态](../artifacts/hf_completed_upload_20260926/status.json) · [远端逐文件校验](../artifacts/hf_completed_upload_20260926/remote_verification.json) · [准备清单](../artifacts/hf_completed_upload_20260926/preparation.json)。
<!-- HF_UPLOAD_20260926_END -->

## A100接跑完成（2026-09-26 14:47）

<!-- A100_STARTUP_20260926_1447_BEGIN -->
**v2所需9组全部已启动：5组完成、4组运行、0组待启动。** 本机T1 seed0已核验800,010,240 steps、65条验证历史；A100剩余T1 seed1/seed2于14:37派发，14:47:07通过五项启动验收。旧版15/15完成的统计不变。

| A100新实验 | 物理GPU | Python PID | 最新W&B步数 | W&B |
|---|---|---:|---:|---|
| T1 v2 seed1 | 2（用户批准共享） | 530302 | 430,080 | [r7av2t1s1a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t1s1a) |
| T1 v2 seed2 | 3 | 530230 | 696,320 | [r7av2t1s2a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t1s2a) |

用户明确批准SIGTERM后，于14:25仅向已完成的PID45627/45555各发送一次SIGTERM；两秒后确认PID和CUDA context均消失，最终checkpoint与完整日志保留。没有使用SIGKILL，也没有再次向旧pane发送Ctrl-C。此前退出卡住的原因说明已更正：W&B在12:22完成退出，之后Python主进程仍未结束；其更深层原因未定。

14:27完整GPU门禁通过后，启动前发现GPU2已有其他用户PID495366，启动器拒绝派发，未形成新run。用户随后明确授权“占着一起用就好”；共享仅适用于本批次T1 seed1、物理GPU2及既有PID495366，启动时空闲显存78,350MiB。新增精确UUID/PID白名单与至少16GiB空闲显存检查，GPU3仍要求无其他计算进程；没有停止或修改既有PID495366。共享条件下GPU/JAX完整预检、身份/数据/QC/预算/dry-run/未使用W&B ID检查重新通过，已有145项CPU测试沿用（冻结源码未变），共享门禁6项模拟用例通过。

两组均80 train + 20 validation全部本地命中；manifest/config/source/reward/terminal/fresh optimizer正确；实际Python位于指定物理GPU；已进入训练循环，无fatal/NaN/OOM；W&B running且两次步数从266,240/471,040增至430,080/696,320。冻结SHA与fingerprint、每组800,010,240 steps / 39,063 updates / 65次验证、fresh/no-resume及Orbax save/restore各4GB均保持合同值。源码没有修改；仅在本批次运行包增加单组调度接口与共享显卡门禁。

用户要求暂不上传权重，本次未启动HF权重上传。四组仍运行的v2为本机T3/T4 seed2及A100 T1 seed1/seed2；不需要自动排队器。A100训练tmux socket为`/data3/yangfeiyang/tmp/tmux_fc_anchor_v2_20260926_t1.sock`，session分别`stage1_fc_anchor_v2_remote7_t1_s1_v1`、`stage1_fc_anchor_v2_remote7_t1_s2_v1`。保留本机W&B代理控制socket`/tmp/mm_wandb_proxy_20260926.sock`与反向转发；本机T1 seed0在14:40快照中虽已完成但Python仍在，本次未操作该本机进程。

运行身份与远端持续证据：

- `forehand_clear_aug100_80train20val_peasd_38bb5c7_remote7_anchor_v2_direct800_v1_t1av2_s1`；config hash `b13b57d232e7`；manifest `/data3/yangfeiyang/WorkSpace/musclemimic/datasets/forehandClear_standard/training_aug100_80train20val_anchor_v2_remote7_38bb5c7_20260926_t1_s1/checkpoints/260926T063859-pid530302-65e66a/manifest.json`；日志 `/data3/yangfeiyang/WorkSpace/musclemimic/artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/logs/t1_s1_v1.log`。
- `forehand_clear_aug100_80train20val_peasd_38bb5c7_remote7_anchor_v2_direct800_v1_t1av2_s2`；config hash `8b3a2bab0ff0`；manifest `/data3/yangfeiyang/WorkSpace/musclemimic/datasets/forehandClear_standard/training_aug100_80train20val_anchor_v2_remote7_38bb5c7_20260926_t1_s2/checkpoints/260926T063859-pid530230-8d09e7/manifest.json`；日志 `/data3/yangfeiyang/WorkSpace/musclemimic/artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/logs/t1_s2_v1.log`。

证据：[A100启动验收](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/startup_acceptance_20260926.json) · [A100运行合同](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/matched_contract.json) · [共享授权](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/gpu_sharing_authorization.json) · [全部门禁](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/preflight/verification.json) · [SIGTERM与释放回执](../artifacts/experiment_status_20260926/release_followup/sigterm_completed_approved.json) · [九组状态](../artifacts/experiment_status_20260926/recheck_after_remote_start/summary.json) · [CSV](../artifacts/experiment_status_20260926/recheck_after_remote_start/实验记录_1447.csv)。
<!-- A100_STARTUP_20260926_1447_END -->

## A100退出状态更正（2026-09-26 14:16）

<!-- A100_EXIT_FOLLOWUP_20260926_BEGIN -->
W&B子进程已退出，T3/T4上传流分别在12:22:49/12:22:34关闭成功。此前根据旧日志沿用“仍等待W&B收尾”的说明已过时。14:16实查：已完成训练的Python PID45627/45555仍在，均无子进程，主线程为R且占用约89%–96% CPU，分别占GPU2/3约6770/6766MiB；当前卡在Python退出阶段，更深层原因尚未定位。

两组finalized update39063 / 800010240 steps、65条验证历史和manifest绑定再次通过。原pane在11:46已各收到一次Ctrl-C，未重复发送。用户要求先训练、暂不上传权重；T1 v2 seed1/seed2运行包、145项测试和dry-run均已准备且冻结源码未变，但GPU空闲门禁仍不通过。已询问是否允许对上述两个已完成进程使用SIGTERM作为退出方式例外；截至本记录尚未发送SIGTERM，未启动新训练，也未安装自动队列。收到答复后优先完成GPU释放、GPU/JAX预检与两组启动验收。
<!-- A100_EXIT_FOLLOWUP_20260926_END -->


## 最新实验表（2026-09-26 13:29）

<!-- LIVE_AUDIT_20260926_1329_BEGIN -->
**旧版15/15完成；v2为4/9完成、3/9运行、2/9待启动。** 本机原批次和A100实查于13:28，新seed2在线状态核验于13:29。

| v2实验 | seed | 主机 | 物理GPU | 状态 | 进度 | 验证历史 |
|---|---:|---|---|---|---:|---:|
| T1 | 0 | 本机 | 1 | 训练中 | 98.38% | 63/65 |
| T1 | 1 | A100 | 计划2 | 待启动：GPU仍被占用 | 0.00% | 0/65 |
| T1 | 2 | A100 | 计划3 | 待启动：GPU仍被占用 | 0.00% | 0/65 |
| T3 | 0 | A100 | 2 | 已完成，收尾仍占卡 | 100.00% | 65/65 |
| T3 | 1 | 本机 | 已释放 | 已完成 | 100.00% | 65/65 |
| T3 | 2 | 本机 | 2 | 训练中（今日新启动） | 1.55%* | 0/65 |
| T4 | 0 | A100 | 3 | 已完成，收尾仍占卡 | 100.00% | 65/65 |
| T4 | 1 | 本机 | 已释放 | 已完成 | 100.00% | 65/65 |
| T4 | 2 | 本机 | 3 | 训练中（今日新启动） | 1.52%* | 0/65 |

进度分母为800,010,240 steps；星号为W&B实时步数，其余运行/完成组使用已finalized checkpoint。新seed2尚未记录首轮验证历史，不能因此视为训练失败；旧审计脚本因假设history已存在而无法直接读取新组，本次使用manifest/GPU/日志/W&B联合核验。T1 seed0已保存787,046,400步，63/65次验证，预计今天14:31完成（估计值）。

A100 T3/T4完成进程仍占GPU2/3；先前已各发送一次Ctrl-C，本次未重复发送。T1 seed1/seed2仍受GPU空闲门禁阻塞，未启动。新本机T3/T4 seed2 W&B分别更新至12,410,880和12,144,640步，进程/manifest/本地80+20数据/日志无fatal均核验通过。

证据：[汇总](../artifacts/experiment_status_20260926/recheck_1327/summary.json) · [本机原批次](../artifacts/experiment_status_20260926/recheck_1327/local_original.json) · [A100](../artifacts/experiment_status_20260926/recheck_1327/remote_original.json) · [本机seed2](../artifacts/experiment_status_20260926/recheck_1327/local_seed2_live.json) · [CSV](../artifacts/experiment_status_20260926/recheck_1327/实验记录_1329.csv)。
<!-- LIVE_AUDIT_20260926_1329_END -->

## 最新训练进度与接跑（2026-09-26 12:07）

<!-- LIVE_AUDIT_20260926_BEGIN -->
**v2 共9组：4组完成、3组运行、2组因GPU未释放而待启动。** 旧版15/15完成的统计不变。原批次本机/A100分别实查于12:06:32/12:06:44，新本机两组于12:07:28完成五项启动验收。

| 主机 | 实验 | 状态 | 进度 / 验证 | GPU / PID |
|---|---|---|---|---|
| 本机 | [T1 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) | 运行中 | 774,553,600 / 96.82% / 62/65 | 1 / 2306382 |
| 本机 | [T3 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) | 已完成；原GPU已释放 | 800,010,240 / 100.00% / 65/65 | 已释放 |
| 本机 | [T4 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) | 已完成；原GPU已释放 | 800,010,240 / 100.00% / 65/65 | 已释放 |
| A100 | [T3 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 已完成；W&B收尾仍占卡 | 800,010,240 / 100.00% / 65/65 | 2 / 45627 |
| A100 | [T4 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 已完成；W&B收尾仍占卡 | 800,010,240 / 100.00% / 65/65 | 3 / 45555 |
| 本机 | [T3 v2 seed2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s2a) | 新启动；验收通过 | W&B 245,760 steps；尚无验证终点 | 2 / 1883497 |
| 本机 | [T4 v2 seed2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s2a) | 新启动；验收通过 | W&B 225,280 steps；尚无验证终点 | 3 / 1883506 |
| A100 | T1 v2 seed1 | 待启动；GPU占用门禁未通过 | 尚未启动 | 计划GPU2 |
| A100 | T1 v2 seed2 | 待启动；GPU占用门禁未通过 | 尚未启动 | 计划GPU3 |

四组完成结果均核验为 finalized update39063 / 800,010,240 steps、65条验证历史，manifest/source/config/指标与checkpoint身份绑定通过。最终checkpoint保存时间：本机T3 seed1为09-26 08:46:48、T4 seed1为09-26 09:07:14；A100 T3 seed0为09-26 07:55:23、T4 seed0为09-25 22:47:56。完成训练不等于已通过teacher promotion。

原批次只剩本机T1 seed0，已保存774,553,600步（96.82%）、62/65次验证；按最近checkpoint间隔外推预计09-26 14:31完成，时间会随实际速度变化。该进程保留继续训练。

本机两组完成进程各收到一次Ctrl-C后，已确认PID/CUDA context消失，日志和checkpoint完整保留。GPU2/3已分别接跑T3/T4 seed2，沿用冻结SHA38bb5c7及原v2合同：fresh/no-resume、800,010,240 steps、39,063 updates、65次验证、Orbax save/restore各4GB。145项focused tests、完整GPU/JAX/QC预检及dry-run均通过。两组均80 train + 20 validation本地命中，manifest有效，实际训练循环无fatal/NaN/OOM，W&B running且两次观测从40,960分别增长至245,760/225,280步。新组进度来自W&B，与原批次finalized checkpoint口径分开。

A100两组虽然达到endpoint，Python仍等待wandb-core收尾。11:46已各向唯一pane发送一次Ctrl-C，禁止重复发送；当前PID45555/45627及CUDA context仍在。W&B代理已恢复，客户端日志有200成功上传，也反复出现503（mysql_metadata_filestream bulkhead full）与超时。新T1 seed1/seed2的145项CPU测试和两组dry-run通过；GPU空闲检查失败，未执行生产启动，没有与旧进程挤占同一张卡。本次未创建自动排队器，须在确认旧PID/CUDA消失后继续GPU预检、verify_prelaunch、start.sh及五项启动验收。

代理交接：本机后台SSH控制socket `/tmp/mm_wandb_proxy_20260926.sock`，反向转发A100 `127.0.0.1:11089`到本机`127.0.0.1:11083`，仍供远端收尾使用，不要关闭。新本机训练tmux socket为`/tmp/musclemimic_anchor_v2_local9_20260926.sock`；A100待启动包计划socket为`/data3/yangfeiyang/tmp/tmux_fc_anchor_v2_20260926_t1.sock`。

| 已完成v2 | 最终帧覆盖率 ↑ | 提前终止率 ↓ | EMG anchor相关性 ↑ |
|---|---:|---:|---:|
| 本机 T3 seed1 | 90.06% | 25% | 0.404 |
| 本机 T4 seed1 | 95.28% | 15% | 0.163 |
| A100 T3 seed0 | 88.25% | 30% | 0.444 |
| A100 T4 seed0 | 91.62% | 30% | 0.222 |

上述为单seed最终验证值；帧覆盖率不是任务成功率，不能据此声称v2整体优于旧版。九组尚未全部完成。此前旧版15组及1组重复备份的HF上传已验收；本次新完成的四组v2尚未上传，不能算入旧上传回执。

证据：[实验记录CSV](../artifacts/experiment_status_20260926/实验记录_20260926.csv) · [本机快照](../artifacts/experiment_status_20260926/local_latest.json) · [A100快照](../artifacts/experiment_status_20260926/remote_latest.json) · [完成与绑定核验](../artifacts/experiment_status_20260926/verification.json) · [本机启动验收](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260926_seed2/startup_acceptance_20260926.json) · [A100阻塞回执](../artifacts/experiment_status_20260926/remote_launch_blocked.json) · [本机运行合同](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260926_seed2/matched_contract.json) · [A100待启动合同](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260926_t1/matched_contract.json)。
<!-- LIVE_AUDIT_20260926_END -->

## 最新训练进度（2026-09-25 10:32）

<!-- LIVE_AUDIT_20260925_1032_BEGIN -->
本机实查：**2026-09-25 10:32:15**；A100实查：**10:31:42**（北京时间）。**五组v2均持续运行；0/9完成、5/9运行、4/9待启动。** 原PID仍对应指定GPU，tmux存活，checkpoint较凌晨增长，family/config/manifest/预算和验证指标绑定核验通过，日志无训练fatal、NaN或OOM。

| 主机 | 实验 | GPU / PID | 已保存步数 | 进度 | 较01:07增加 | 验证次数 | 预计完成（北京时间） |
|---|---|---|---:|---:|---:|---:|---|
| 本机 | [T1 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) | 1 / 2306382 | 587,161,600 | 73.39% | 9.37个百分点 | 47/65 | 09-26 10:11 |
| 本机 | [T3 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) | 2 / 2306301 | 612,147,200 | 76.52% | 10.93个百分点 | 49/65 | 09-26 07:14 |
| 本机 | [T4 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) | 3 / 2306373 | 599,654,400 | 74.96% | 9.37个百分点 | 48/65 | 09-26 09:05 |
| A100 | [T3 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 2 / 45627 | 662,118,400 | 82.76% | 7.81个百分点 | 53/65 | 09-26 07:31 |
| A100 | [T4 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 3 / 45555 | 737,075,200 | 92.13% | 6.25个百分点 | 59/65 | 09-25 18:31 |

A100 T4 seed0已finalized 737,075,200步、59/65条验证历史，尚未达到800,010,240步/65条历史的完成条件，保留继续训练。按最近4个checkpoint（3个间隔）的速度，预计今天18:32左右达到预算；T3速度约6.40M steps/hour，预计明天07:31。预计完成时间会随实际速度变化，不能当作完成验收。

最新验证帧覆盖率/提前终止率：本机T1/T3/T4分别为96.19%/10%、83.46%/25%、85.26%/35%；A100 T3/T4为82.72%/20%、91.54%/30%。这是不同阶段的单次验证快照，不作为最终方法优劣结论。

本次最初SSH查询超时；独立连接确认主机可响应，随后排除本地timeout包装引起的SSH终端暂停，已成功完成远端全量只读审计。该检查问题不代表训练中断。A100 W&B代理127.0.0.1:11089仍connection refused，因此进度以实际进程、日志及checkpoint为准。

没有新增训练达到endpoint，本次未启动、停止或恢复训练。待启动仍为T1 seed1、T1 seed2、T3 seed2、T4 seed2；旧版15/15已完成的统计不变。

[本次检查记录](../artifacts/experiment_status_20260925/recheck_0958/实验进度_20260925_1032.md) · [CSV](../artifacts/experiment_status_20260925/recheck_0958/实验进度_20260925_1032.csv) · [本机证据](../artifacts/experiment_status_20260925/recheck_0958/local_latest.json) · [A100证据](../artifacts/experiment_status_20260925/recheck_0958/remote_snapshot.json) · [身份与指标绑定核验](../artifacts/experiment_status_20260925/recheck_0958/verification.json)。
<!-- LIVE_AUDIT_20260925_1032_END -->

## 最新训练进度（2026-09-25 01:07）

<!-- LIVE_AUDIT_20260925_0107_BEGIN -->
本机实查：**2026-09-25 01:04:41**；A100实查：**01:07:20**（北京时间）。**五组v2均持续运行；0/9完成、5/9运行、4/9待启动。** 五个原PID仍在指定GPU，tmux存活，checkpoint较昨日增加，family/config/manifest/预算和验证指标绑定核验通过，日志无训练fatal、NaN或OOM。旧版15/15完成的统计不变。

| 主机 | 实验 | GPU / PID | 已保存步数 | 进度 | 较昨日11:52增加 | 验证次数 | 预计完成（北京时间） |
|---|---|---|---:|---:|---:|---:|---|
| 本机 | [T1 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) | 1 / 2306382 | 512,204,800 | 64.02% | 14.05个百分点 | 41/65 | 09-26 10:11 |
| 本机 | [T3 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) | 2 / 2306301 | 524,697,600 | 65.59% | 14.05个百分点 | 42/65 | 09-26 07:17 |
| 本机 | [T4 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) | 3 / 2306373 | 524,697,600 | 65.59% | 15.62个百分点 | 42/65 | 09-26 09:05 |
| A100 | [T3 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 2 / 45627 | 599,654,400 | 74.96% | 9.37个百分点 | 48/65 | 09-26 16:10 |
| A100 | [T4 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 3 / 45555 | 687,104,000 | 85.89% | 10.93个百分点 | 55/65 | 09-25 18:33 |

进度统一使用已finalized checkpoint / 800,010,240 steps，尚在计算的步数不计入。预计完成时间按最近4个checkpoint（3个间隔）速度外推，会随GPU竞争及验证耗时变化。A100 T4进度最高，预计今天9月25日18:33完成；T3近期速度约5.02M steps/hour，低于昨日6.41M，按该速度外推为9月26日16:10，不能沿用昨日05:00的估计。当前GPU2/3只看到各自训练PID，尚未确定近期减速原因；没有修改训练进程。

最新验证帧覆盖率/提前终止率：本机T1 seed0为83.04%/40%，T3 seed1为81.87%/40%，T4 seed1为84.40%/30%；A100 T3 seed0为86.65%/30%，T4 seed0为88.58%/30%。这些是中途结果，不作为跨组或跨seed最终优劣结论。

A100 W&B代理127.0.0.1:11089仍连接超时，本次以实际进程、GPU、日志与checkpoint为准。没有新训练达到endpoint，本次未启动、停止或恢复训练。待启动仍为T1 seed1、T1 seed2、T3 seed2、T4 seed2。

[本次检查记录](../artifacts/experiment_status_20260925/实验进度_20260925.md) · [CSV](../artifacts/experiment_status_20260925/实验进度_20260925.csv) · [本机证据](../artifacts/experiment_status_20260925/local_snapshot.json) · [A100证据](../artifacts/experiment_status_20260925/remote_snapshot.json) · [身份与指标绑定核验](../artifacts/experiment_status_20260925/verification.json)。
<!-- LIVE_AUDIT_20260925_0107_END -->

## 最新训练进度（2026-09-24 11:52）

<!-- LIVE_AUDIT_20260924_1152_BEGIN -->
本机实查：**2026-09-24 11:51:46**；A100实查：**11:52:33**（北京时间）。**五组v2均持续运行；0/9完成、5/9运行、4/9待启动。** 五个原PID仍在指定GPU，tmux存活，checkpoint较昨日增加，family/config/manifest/预算和验证指标绑定检查通过，日志无训练fatal、NaN或OOM。旧版15/15已完成的统计不变。

| 主机 | 实验 | GPU / PID | 已保存步数 | 进度 | 较昨日13:59增加 | 验证次数 | 预计完成（北京时间） |
|---|---|---|---:|---:|---:|---:|---|
| 本机 | [T1 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) | 1 / 2306382 | 399,769,600 | 49.97% | 23.42个百分点 | 32/65 | 09-26 10:12 |
| 本机 | [T3 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) | 2 / 2306301 | 412,262,400 | 51.53% | 23.42个百分点 | 33/65 | 09-26 07:12 |
| 本机 | [T4 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) | 3 / 2306373 | 399,769,600 | 49.97% | 23.42个百分点 | 32/65 | 09-26 09:00 |
| A100 | [T3 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 2 / 45627 | 524,697,600 | 65.59% | 12.49个百分点 | 42/65 | 09-26 04:59 |
| A100 | [T4 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 3 / 45555 | 599,654,400 | 74.96% | 17.18个百分点 | 48/65 | 09-25 18:34 |

进度统一使用已finalized checkpoint / 800,010,240 steps，尚在计算的步数不计入。预计完成时间依据最近4个checkpoint（3个间隔）的速度外推，随资源竞争和验证耗时变化。A100 GPU2目前只看到T3 PID45627，近期速度约6.41M steps/hour（昨日约3.86M）；预计完成由昨日的9月27日14:02提前至9月26日04:59。T4速度约6.38M steps/hour，预计9月25日18:34完成。

最新验证帧覆盖率/提前终止率：本机T1 seed0为75.02%/40%，T3 seed1为79.97%/65%，T4 seed1为81.01%/60%；A100 T3 seed0为74.83%/40%，T4 seed0为88.65%/40%。本机T3 seed1较昨日64.94%/100%有改善，但仍是中途结果，不能据不同训练阶段的单次指标认定最终优劣。

A100 W&B代理127.0.0.1:11089仍connection refused，本次以实际进程、GPU、日志及checkpoint为准。没有新训练达到endpoint，本次未启动、停止或恢复训练。待启动仍为T1 seed1、T1 seed2、T3 seed2、T4 seed2。

此前完成的15组独立旧版终点及1组A100重复备份已上传Hugging Face，9月23日已验证218个payload文件，提交c98fb500bc674a4b8fe8abd1f8df91557523a980；本次读取已有校验回执，没有把仍在训练的v2记为已完成上传。

[本次检查记录](../artifacts/experiment_status_20260924/实验进度_20260924.md) · [CSV](../artifacts/experiment_status_20260924/实验进度_20260924.csv) · [本机证据](../artifacts/experiment_status_20260924/local_snapshot.json) · [A100证据](../artifacts/experiment_status_20260924/remote_snapshot.json) · [身份与指标绑定核验](../artifacts/experiment_status_20260924/verification.json)。
<!-- LIVE_AUDIT_20260924_1152_END -->

## 最新训练进度（2026-09-23 13:59）

<!-- LIVE_AUDIT_20260923_1359_BEGIN -->
本机实查：**2026-09-23 13:57:40**；A100实查：**13:59:28**（北京时间）。**五组v2均在运行且checkpoint比凌晨继续增长；0/9完成、5/9运行、4/9待启动。** 原PID与指定GPU对应，tmux存活，family/config/manifest/预算核验通过，日志无训练fatal、NaN或OOM。旧版15/15已完成的统计不变。

| 主机 | 实验 | GPU / PID | 已保存步数 | 进度 | 较02:12增加 | 验证次数 | 预计完成（北京时间） |
|---|---|---|---:|---:|---:|---:|---|
| 本机 | [T1 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) | 1 / 2306382 | 212,377,600 | 26.55% | 12.49个百分点 | 17/65 | 09-26 09:57 |
| 本机 | [T3 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) | 2 / 2306301 | 224,870,400 | 28.11% | 12.49个百分点 | 18/65 | 09-26 06:59 |
| 本机 | [T4 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) | 3 / 2306373 | 212,377,600 | 26.55% | 12.49个百分点 | 17/65 | 09-26 09:07 |
| A100 | [T3 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 2 / 45627 | 424,755,200 | 53.09% | 6.25个百分点 | 34/65 | 09-27 14:02 |
| A100 | [T4 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 3 / 45555 | 462,233,600 | 57.78% | 9.37个百分点 | 37/65 | 09-25 18:40 |

进度统一采用finalized checkpoint / 800,010,240 steps；实际正在计算的步数未计入。预计时间以最近4个checkpoint（3个间隔）的速度外推，会受GPU竞争、验证耗时和网络情况影响，不是完成承诺。A100 T3当前约3.86M steps/hour、T4约6.36M；GPU2另有PID2561089占用28,542 MiB，存在资源竞争，未据此认定唯一减速原因，也未修改任何进程。

最新验证：本机T1/T3/T4的帧覆盖率为74.83%/64.94%/73.76%，提前终止率为70%/100%/45%；A100 T3/T4帧覆盖率73.17%/83.83%，提前终止率均40%。这些是不同训练阶段的最新快照，不作为跨组最终优劣结论。

A100 W&B代理127.0.0.1:11089仍connection refused；本次状态来自实际进程、GPU、日志和checkpoint，不依据可能过期的页面进度。没有新的训练达到endpoint，未启动、停止或恢复训练；待启动仍为T1 seed1、T1 seed2、T3 seed2、T4 seed2。

[本次检查记录](../artifacts/experiment_status_20260923/recheck_1355/实验进度_20260923_1359.md) · [CSV](../artifacts/experiment_status_20260923/recheck_1355/实验进度_20260923_1359.csv) · [本机证据](../artifacts/experiment_status_20260923/recheck_1355/local_snapshot.json) · [A100证据](../artifacts/experiment_status_20260923/recheck_1355/remote_snapshot.json) · [身份与指标绑定检查](../artifacts/experiment_status_20260923/recheck_1355/verification.json)。
<!-- LIVE_AUDIT_20260923_1359_END -->

## 上传完成与训练效果检查（2026-09-23）

<!-- LIVE_AUDIT_20260923_BEGIN -->
**实查时间：2026-09-23 02:12（北京时间）。旧版15/15完成；v2共9叶，0完成、5运行、4待启动。** 五个原PID均在指定GPU，checkpoint继续增长，身份/预算/manifest绑定核验通过，日志无训练fatal、NaN或OOM。各项进度统一使用已finalized checkpoint，不把正在计算的步数当作已保存。

| 主机 | 实验 | GPU / PID | 已保存步数 / 进度 | 验证次数 | 验证帧覆盖率 ↑ | 提前终止率 ↓ |
|---|---|---|---:|---:|---:|---:|
| 本机 | [T1 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) | 1 / 2306382 | 112,435,200 / 14.05% | 9/65 | 67.28% | 80% |
| 本机 | [T3 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) | 2 / 2306301 | 124,928,000 / 15.62% | 10/65 | 38.55% | 100% |
| 本机 | [T4 v2 seed1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) | 3 / 2306373 | 112,435,200 / 14.05% | 9/65 | 56.65% | 100% |
| A100 | [T3 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 2 / 45627 | 374,784,000 / 46.85% | 30/65 | 76.78% | 40% |
| A100 | [T4 v2 seed0](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 3 / 45555 | 387,276,800 / 48.41% | 31/65 | 78.81% | 40% |

效果判断：**训练仍在推进，但尚不能认定v2优于旧版或已收敛。** 本机T3 seed1最近3次验证帧覆盖率约36.9%–40.8%，提前终止率均100%，是当前需要继续观察的薄弱点；T4 seed1最新提前终止率同样100%。T1 seed0最近3次提前终止率从100%降至90%、80%，仍有较大改进空间。帧覆盖率为已覆盖帧数/参考总帧数，不是任务成功率；提前终止率为提前结束的episode比例。

A100同seed0、同update18300（374,784,000步）比较：T3/T4的anchor相关性为0.283/0.137，真实参考synergy loss为0.329/0.487，显示T3在肌电对齐上有积极信号；提前终止率40%/45%，帧覆盖率76.78%/83.26%，运动表现并未全面占优。这只是单seed中途比较，不能作为最终显著性结论。T4是相位打乱对照，比较使用real_reference loss，而不是它自己的打乱目标loss；v1/v2的anchor loss定义不同，不直接比较数值大小。

按最近3个checkpoint间隔粗估：本机三组约9月26日07:09–09:51完成；A100 T4约9月25日18:31，T3约9月27日17:34。T3近期速度约3.74M steps/hour，且GPU2现另有其他用户PID2211183占用约28GB，存在资源竞争，减速原因未完全定位，预计时间可能继续变化。没有停止或修改任何训练或其他用户进程。

W&B同步仍有异常：本机日志有HTTP500/mysql guard deadline，A100有上传超时；本次以SSH、本机进程及checkpoint为依据，不把页面状态当作训练成功/失败证据。v2待启动仍为T1 seed1、T1 seed2、T3 seed2、T4 seed2；当前五组未完成，本次没有派发新训练。

Hugging Face已完成：新增3组本机终点和1组A100重复运行备份，57个文件共1,606,255,654字节。218个payload文件已按远端大小及哈希全部核验；原208个文件保持不变，.gitattributes仅追加7条已核对的新权重LFS规则。此前收尾断言把此正常规则更新误报为失败，现已修正并生成正式校验回执；没有重传已有权重。

旧版800,010,240步终点的描述性汇总（独立seed均值，排除A100重复run）：

| 旧版arm（各3 seeds） | 平均帧覆盖率 ↑ | 平均提前终止率 ↓ | 平均rpos误差 ↓ |
|---|---:|---:|---:|
| T0 | 83.36% | 40.00% | 0.0979 |
| T1 | 83.20% | 45.00% | 0.1046 |
| T2 | 90.99% | 30.00% | 0.1002 |
| T3 | 87.69% | 30.00% | 0.0978 |
| T4 | 88.36% | 30.00% | 0.0922 |

旧版T2/T3的平均帧覆盖率高于T0、平均提前终止率更低，但不据此宣称已通过pairwise gate或teacher promotion。T0关闭肌电统计，其零值不是肌电效果优秀。

[本次完整记录](../artifacts/experiment_status_20260923/实验运行检查_20260923.md) · [CSV](../artifacts/experiment_status_20260923/实验运行检查_20260923.csv) · [本机证据](../artifacts/experiment_status_20260923/local_snapshot.json) · [A100证据](../artifacts/experiment_status_20260923/remote_snapshot.json) · [指标绑定核验](../artifacts/experiment_status_20260923/metric_binding_checks.json) · [上传回执](../artifacts/hf_completed_upload_20260922/remote_verification.json) · [HF完成索引](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/COMPLETED_20260922.md)。
<!-- LIVE_AUDIT_20260923_END -->

## 完成权重增量上传 Hugging Face（2026-09-22）

<!-- HF_UPLOAD_20260922_BEGIN -->
**上传完成并通过远端逐文件校验**（2026-09-23T02:13:31.185278+08:00）。提交：[c98fb500bc674a4b8fe8abd1f8df91557523a980](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/commit/c98fb500bc674a4b8fe8abd1f8df91557523a980)。

仓库现在包含旧版 Forehand Clear **15组独立完成终点 + 1组A100完成的duplicate**。本次新增本机T0 seed1、T0 seed2、T1 seed2，以及A100旧T1 seed1 v4；A100重复run位于独立duplicates/remote7/seed1/T1/r7t1s1v4目录，不覆盖本机同seed。仍运行的anchor v2没有作为完成权重上传。

本次新增**57个文件、1,606,255,654字节**，其中54个checkpoint文件共1,606,123,336字节，另有3个完整索引/校验清单。全量16个run共215个checkpoint文件、6,424,244,229字节；已验证218个payload文件的远端大小及LFS SHA256/Git blob哈希，原仓库208个文件保持不变；另一个.gitattributes文件仅由HF追加7条新权重的LFS规则，已核对原规则完整保留。

每组均为finalized update39063 / 800010240 steps、65条验证历史；完整保留Orbax train_state/config/metadata和optimizer状态。未上传原始轨迹、EMG、SMPL或私有资产清单。上传不影响正在运行的训练。

[Hugging Face完成索引](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/COMPLETED_20260922.md) · [完整身份与文件SHA256](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/COMPLETED_20260922.json) · [校验和](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/blob/main/SHA256SUMS_20260922) · [本地上传状态](../artifacts/hf_completed_upload_20260922/status.json) · [远端逐文件验证](../artifacts/hf_completed_upload_20260922/remote_verification.json)。
<!-- HF_UPLOAD_20260922_END -->

## 本机完成与 v2 调度（2026-09-22）

<!-- LIVE_AUDIT_20260922_BEGIN -->
本机启动验收：**2026-09-22T11:49:55.310209+08:00**；A100 实查：**2026-09-22T11:46:21.782309+08:00**（北京时间）。

**旧版 Forehand Clear 15/15 DONE；anchor v2 共9叶，0 DONE、5 RUNNING、4未启动。** 本次完成旧版 T0 seed1、T0 seed2、T1 seed2 的 endpoint 核验，并已在释放的物理 GPU1/2/3 启动三组新 v2。此前12叶沿用09-18审计，本次3叶均 finalized update39063 / 800,010,240 steps、65条验证历史、W&B finished。原进程各一次 Ctrl-C 后消失，完整日志与 checkpoint 保留。

| 主机 | 实验 | 物理 GPU / Python PID | 状态 / 进度 | W&B |
|---|---|---|---|---|
| 本机 | T0 seed1 旧版 | 已释放 | DONE；800,010,240 steps（100.00%） | [l9t0s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s1v1) |
| 本机 | T0 seed2 旧版 | 已释放 | DONE；800,010,240 steps（100.00%） | [l9t0s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s2v1) |
| 本机 | T1 seed2 旧版 | 已释放 | DONE；800,010,240 steps（100.00%） | [l9t1s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s2v1) |
| 本机 | T1 v2 seed0 | 1 / 2306382 | RUNNING；163,840 steps（0.020%） | [l9av2t1s0a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t1s0a) |
| 本机 | T3 v2 seed1 | 2 / 2306301 | RUNNING；163,840 steps（0.020%） | [l9av2t3s1a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t3s1a) |
| 本机 | T4 v2 seed1 | 3 / 2306373 | RUNNING；163,840 steps（0.020%） | [l9av2t4s1a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9av2t4s1a) |
| A100 | T3 v2 seed0 | 2 / 45627 | RUNNING_WANDB_STALE；312,320,000 steps（39.04%） | [r7av2t3s0b](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) |
| A100 | T4 v2 seed0 | 3 / 45555 | RUNNING_WANDB_STALE；287,334,400 steps（35.92%） | [r7av2t4s0b](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) |
| A100 | T1 seed1 旧版重复运行 | 无占用 | DONE_DUPLICATE；800,010,240 steps（100.00%） | [r7t1s1v4](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7t1s1v4) |

本机新三组均通过五项启动验收：80 train + 20 validation 轨迹全部命中本地文件，manifest/config hash/source/reward/terminal/预算/fresh optimizer 正确，W&B running 且连续两次观测步数增长，Python PID 位于指定物理 GPU，日志进入实际训练循环且无 fatal/NaN/OOM/HF下载。表中新启动步数来自 W&B；A100进度来自已 finalized checkpoint，两者口径不同。

A100 T3/T4 v2 seed0 保留继续训练，已验证25/65、23/65次；近期速度外推分别约09-25 13:53、18:28完成，速度变化会影响估计。其 W&B 代理127.0.0.1:11089仍 connection refused，页面进度失真，以SSH进程/日志/checkpoint为准；本次未重启或停止远端训练。远端旧T1 seed1已DONE且不占GPU，重复run不增加唯一叶。

**v2待启动：T1 seed1、T1 seed2、T3 seed2、T4 seed2。** 本次没有创建自动排队器，后续空闲卡仍按实际状态和同一family门禁派发。

新批次冻结SHA：38bb5c7ee492df20f8695de42e96a81c50ada0b5；fingerprint：43f28acd7771a72b08b7ad71e59ef17554970e5dbaa3255061b14834a2fcd5d7。专用venv为.local/workspace/venvs/anchor_v2_38bb5c7；每组800,010,240 steps / 39,063 updates / 65次验证，fresh/no-resume，promotion.auto_stop=false，Orbax save/restore各4GB。145项focused tests无失败/跳过，三卡完整preflight/JAX/dry-run通过。

| 新 v2 run ID | Config hash | 持续证据 |
|---|---|---|
| `forehand_clear_aug100_80train20val_peasd_38bb5c7_local9_anchor_v2_direct800_v1_t1av2_s0` | `610f04ba3d1f` | [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_anchor_v2_local9_38bb5c7_20260922_t1_s0/checkpoints/260922T034705-pid2306382-249e8d/manifest.json) · [日志](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260922/logs/t1_s0_v1.log) |
| `forehand_clear_aug100_80train20val_peasd_38bb5c7_local9_anchor_v2_direct800_v1_t3av2_s1` | `af0762585e44` | [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_anchor_v2_local9_38bb5c7_20260922_t3_s1/checkpoints/260922T034705-pid2306301-0d6382/manifest.json) · [日志](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260922/logs/t3_s1_v1.log) |
| `forehand_clear_aug100_80train20val_peasd_38bb5c7_local9_anchor_v2_direct800_v1_t4av2_s1` | `56cd5d377e21` | [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_anchor_v2_local9_38bb5c7_20260922_t4_s1/checkpoints/260922T034705-pid2306373-036c65/manifest.json) · [日志](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260922/logs/t4_s1_v1.log) |

tmux socket：/tmp/musclemimic_yangfeiyang_anchor_v2_local9_20260922.sock；sessions：stage1_fc_anchor_v2_local9_t1_s0_v1、stage1_fc_anchor_v2_local9_t3_s1_v1、stage1_fc_anchor_v2_local9_t4_s1_v1。start.sh已执行，不得重复派发。

[五项启动验收](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260922/startup_acceptance_20260922.json) · [运行合同](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260922/matched_contract.json) · [预检](../artifacts/stage1_fc_anchor_v2_local9_38bb5c7_20260922/preflight/verification.json) · [本机完成核验](../artifacts/experiment_status_20260922/local_completion.json) · [A100实查](../artifacts/experiment_status_20260922/remote_snapshot_1146.json) · [CSV实验表](../artifacts/experiment_status_20260922/实验运行检查_20260922.csv)。
<!-- LIVE_AUDIT_20260922_END -->

## 历史：GPU3 调度请求复核（2026-09-21 05:14，北京时间）

用户要求释放A100 GPU3的已完成任务并运行所需v2实验。实时核验发现GPU3唯一训练进程是**尚未完成的T4 anchor v2 seed0，PID45555，124,928,000/800,010,240 steps（15.62%），10/65条验证历史**；已完成的旧T1 seed1 v4 PID2237783不存在。T3v2在GPU2已推进到137,420,800 steps、11/65条验证历史。

**用户已明确选择：保留T4 v2继续训练，不停止、不换跑。** 本次没有发送停止信号，也没有启动替代任务；此前GPU3释放/换跑请求已由本次选择澄清，不得据此中止T4v2或派发T1v2。沿用原run ID、PID45555及800,010,240步合同。未来新任务仍须独立启动授权、fresh optimizer、全部启动门禁及W&B在线验收。确认记录时间：2026-09-21T05:18:08.362693+08:00。本次仅登记用户选择，未重新采样GPU或进度。

[本次GPU/PID/manifest/checkpoint证据](../artifacts/experiment_status_20260921/gpu3_reassignment_identity_check.json)。

## 历史：本机与 A100 复核（2026-09-21 05:03，北京时间）

<!-- LIVE_AUDIT_20260921_BEGIN -->
**本机3组、A100 2组均在运行。** 本机T0 seed1 / T0 seed2 / T1 seed2已finalized进度为**79.64% / 81.20% / 79.64%**，验证历史**51 / 52 / 51条（目标65）**；GPU1/2/3与原PID存活，W&B心跳正常。预计9月21日21:02–23:45达到预算，仍需endpoint验收。

A100 T3/T4 anchor v2已实查为**RUNNING_WANDB_STALE**：GPU2/3利用率100%，PID45627/45555与tmux存活；均已finalized **update6100 / 124,928,000 steps（15.62%）**，各**10/65条验证历史**，身份/预算/anchor v2参数匹配，日志无fatal/NaN/OOM。预计完成为**9月25日12:17 / 14:45**，仅为近期速度外推。

W&B仍显示crashed，最后心跳为09-20 05:44:45 / 06:20:01；实际训练没有停止。debug日志持续报告`127.0.0.1:11089 connection refused`，05:03实查该代理端口无监听。这是当前同步失败的直接原因，代理退出原因未确定；本次未修复代理或重启训练。

远端旧T1 seed1 v4已核验**DONE**：finalized update39063 / 800,010,240 steps、65条验证历史、manifest/checkpoint绑定与W&B finished一致，原PID已退出。它是重复运行，不增加唯一训练叶。旧版累计**12 DONE + 3 RUNNING**（12个DONE沿用09-18审计）；新版9叶为**0 DONE、2 RUNNING、7按既有排期未启动**。

[完整实验记录表](../artifacts/experiment_status_20260921/实验运行检查_20260921.md) · [CSV](../artifacts/experiment_status_20260921/实验运行检查_20260921.csv) · [本机证据](../artifacts/experiment_status_20260921/local_snapshot.json) · [A100证据](../artifacts/experiment_status_20260921/remote_snapshot.json) · [W&B证据](../artifacts/experiment_status_20260921/wandb_snapshot.json) · [代理端口证据](../artifacts/experiment_status_20260921/remote_proxy_probe.json)。本次未启动、重启或停止训练；密码未写入仓库或记录。
<!-- LIVE_AUDIT_20260921_END -->

## 历史：本机三组训练复核（2026-09-20 03:15，北京时间）

<!-- LOCAL_REMAINING_20260920_BEGIN -->
**本机三组均在训练，旧版 Forehand Clear 仍为 12/15 DONE、3/15 RUNNING。** 原 Python PID 和 tmux pane 存活，物理 GPU1/2/3 利用率均为 100%，每组显存约 6 GB；未发现训练 fatal traceback、NaN、OOM。

下表为已经 finalized 的 checkpoint 进度，实际训练可能已继续推进：

| 实验 | GPU / PID | 已保存步数 / 800,010,240 | 进度 | 验证次数 / 65 | 预计剩余 | 预计完成时间 |
|---|---|---:|---:|---:|---:|---|
| T0 seed1 | GPU1 / 4109420 | 412,262,400 | 51.53% | 33 | 44.6 小时 | 09-21 23:52 |
| T0 seed2 | GPU2 / 4109499 | 424,755,200 | 53.09% | 34 | 41.8 小时 | 09-21 21:00 |
| T1 seed2 | GPU3 / 4109416 | 424,755,200 | 53.09% | 34 | 43.5 小时 | 09-21 22:45 |

预计时间由最近 4 个 finalized checkpoint（3 个间隔）的实际速度外推：约 8.50–8.78M steps/hour，约 9 月 21 日 21:00–24:00 完成。速度变化会影响估计，正式完成仍须验收 endpoint 与 65 条验证历史。

**W&B 监控异常与训练状态分别记录：** T0 seed1、T1 seed2 的页面状态为 crashed，但本机训练并未停止；其 file_stream 持续超时，T1 seed2 还收到服务端 HTTP500 / mysql query exceeded guard deadline。最后心跳分别是北京时间 9 月 19 日 19:45、17:09，页面步数已过期。T0 seed2 的 W&B 为 running，查询时为 433,336,320 steps（54.17%）。没有为修复监控而停止或重启训练。

本次保持原 c1ccd93 / Aug100 grouped 80+20 / 800,010,240 步合同；T1 seed2 属于旧肌电 anchor v1，与远端 anchor v2 分开统计。
[本次进度与监控诊断证据](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/progress_20260920.json)（含 manifest、日志、checkpoint、W&B 心跳、速度计算依据）。
<!-- LOCAL_REMAINING_20260920_END -->

## 历史：Anchor v2 中断与 GPU2/3 重跑（2026-09-20）

<!-- ANCHOR_V2_REMOTE7_20260920_BEGIN -->
状态：**RUNNING，五项启动验收通过，尚未训练完成**。验收时间：2026-09-20T02:46:45.952841+08:00（北京时间）。

远端 172.18.22.7 本次启动时间为 2026-09-20 01:46:04；旧 PID 和 tmux 已消失。旧 W&B 同步日志约 01:01 停止，训练日志未发现 Python traceback、NaN 或 OOM。已确认服务器发生重启，但重启触发因素以及重启前是否另有中断因素尚不明确。
旧 T3/T4 最后 finalized checkpoint 均为 update 610 / 12,492,800 steps（1.56%）；旧权重、日志、manifest 和原启动验收全部保留，旧运行状态已记为 INTERRUPTED。

38bb5c7 的 engine.py 正式 matched-arm 门禁要求 fresh optimizer，并禁止 auto_resume/resume_from。因此本次以新 run ID 从头重跑原 800,010,240 步合同；没有把旧 checkpoint 当作恢复起点。按用户指定使用启动时空闲的物理 GPU2/3。

| 实验 | 物理 GPU / Python PID | W&B | 验收时步数 | Config hash | 证据快照 |
|---|---|---|---:|---|---|
| T3 anchor v2 seed0 | GPU2 / 45627 | [r7av2t3s0b](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0b) | 163,840 | c5821f40b4df | [manifest](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/manifest_t3.json)、[日志](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/logs/t3_s0_v2.log) |
| T4 anchor v2 seed0 | GPU3 / 45555 | [r7av2t4s0b](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0b) | 163,840 | 1e3dbc756c32 | [manifest](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/manifest_t4.json)、[日志](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/logs/t4_s0_v2.log) |

两组均命中本地 80 train + 20 validation 轨迹，manifest/config hash/source/reward/预算匹配；W&B 在线且步数增长，Python PID 位于指定物理 GPU，日志进入实际训练循环，未发现 fatal traceback、NaN、OOM 或 HF 轨迹下载。
沿用同一冻结源码及 venv 下 9 月 19 日通过的 145 项测试证据；本次重新执行 GPU2/3 完整 preflight、JAX、canonical launcher dry-run 与 W&B 身份检查，全部通过。旧代理 11088 不再监听后，仅将本次运行包代理改为 http://127.0.0.1:11089，复核通过后启动。

Git SHA：38bb5c7ee492df20f8695de42e96a81c50ada0b5；source fingerprint：43f28acd7771a72b08b7ad71e59ef17554970e5dbaa3255061b14834a2fcd5d7。
冻结 worktree：/data3/yangfeiyang/WorkSpace/ENV/tmp/musclemimic_anchor_v2_38bb5c7_remote7。源代码、seed0、Aug100 grouped 80/20、anchor v2 参数与原尝试相同。
每组 800,010,240 steps / 39,063 updates / 65 条预期验证历史；promotion.auto_stop=false，Orbax save/restore 各 4 GB。生产入口仍为 scripts/run_fullbody_training.sh。

完整 run ID：

- T3：forehand_clear_aug100_80train20val_peasd_38bb5c7_remote7_anchor_v2_direct800_v2_t3av2_s0
- T4：forehand_clear_aug100_80train20val_peasd_38bb5c7_remote7_anchor_v2_direct800_v2_t4av2_s0

tmux socket：/data3/yangfeiyang/tmp/tmux_fc_anchor_v2_20260920_rerun2.sock；sessions：stage1_fc_anchor_v2_remote7_t3_s0_v2、stage1_fc_anchor_v2_remote7_t4_s0_v2。
启动已经完成，不要重复执行 start.sh。持续日志及完整远端 manifest 路径见验收 JSON；本地文件为验收时的快照。

[五项启动验收](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/startup_acceptance_20260920.json)、[中断审计](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/interruption_audit.json)、[运行合同](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/matched_contract.json)、[运行状态](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/status.json)、[预检核验](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260920_rerun2/preflight/verification.json)。

新版 T1/T3/T4 × seeds0/1/2 共 9 个独立训练叶：**0 DONE、2 RUNNING、7 未启动**；待跑 T1v2 s0/s1/s2、T3v2 s1/s2、T4v2 s1/s2。旧尝试不增加 seed 或训练叶；anchor v1/v2 记录分别保留。本次未修改本机原训练；旧版 12/15 DONE 的统计仍以 9 月 18 日核对为准，未将其写成今日实时结果。
<!-- ANCHOR_V2_REMOTE7_20260920_END -->

## 历史：Anchor v2 两组启动（2026-09-19 15:59，远端 GPU1/2；9月20日中断）

<!-- ANCHOR_V2_REMOTE7_20260919_BEGIN -->
历史验收状态（9月19日）：**RUNNING，五项启动验收全部通过**；当前状态（9月20日）：**INTERRUPTED**，替代运行见上方 GPU2/3 记录。
用户明确授权GPU1/2共享运行；关闭XLA显存预分配，保留其他用户进程。
145项相关测试全部通过，GPU1/2完整预检、JAX检查、canonical launcher dry-run和W&B身份检查均通过。
两组均从本地加载80 train +20 validation，manifest/config hash/reward/source identity匹配，指定GPU存在新Python PID，
日志已进入实际训练循环且W&B步数推进；未检出fatal traceback、NaN、OOM或远程轨迹下载。
完整证据：[五项启动验收](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260919/startup_acceptance_20260919.json)，检查时间2026-09-19T15:59:13.674801+08:00。

| 实验 | 物理GPU | Python PID | W&B | 验收时步数 | Config hash |
|---|---:|---:|---|---:|---|
| T3 anchor v2 seed0 | 1 | 1662614 | [r7av2t3s0a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t3s0a) | 143,360 | c5821f40b4df |
| T4 anchor v2 seed0 | 2 | 1662613 | [r7av2t4s0a](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7av2t4s0a) | 122,880 | 1e3dbc756c32 |

Git SHA：38bb5c7ee492df20f8695de42e96a81c50ada0b5（origin/double_play_relay，anchor v2引入提交830c206）。
Source fingerprint：43f28acd7771a72b08b7ad71e59ef17554970e5dbaa3255061b14834a2fcd5d7。
Run id统一前缀：forehand_clear_aug100_80train20val_peasd_38bb5c7_remote7_anchor_v2_direct800_v1，
分别追加_t3av2_s0、_t4av2_s0。
独立worktree：/data3/yangfeiyang/WorkSpace/ENV/tmp/musclemimic_anchor_v2_38bb5c7_remote7。
每组800010240 timesteps /39063 updates，65条预期验证历史；fresh optimizer、no resume、promotion.auto_stop=false；Orbax save/restore各4GB。
共享GPU会影响吞吐，当前不以启动阶段推算完成时间。

**旧版/新版肌电结果同时保留**：原c1ccd93的T1/T3/T4记录为anchor v1，已有结果不覆盖；
本机T0s1/T0s2/T1s2保持原合同，当前任务未修改或中断它们。T0/T2的奖励定义未变，无需因anchor v2重跑。
新版T1/T3/T4各seeds0/1/2共9叶独立统计：**0 DONE、2 RUNNING、7未启动**。
后续待跑：T1v2 s0/s1/s2、T3v2 s1/s2、T4v2 s1/s2；未安装自动排队。先评估T3v2是否恢复TRIlat/PT及保持协同指标，再决定后续排期。

| 肌电版本 | anchor带内权重 | burst权重 | 形状权重 | scale floor | channel cap | anchor_max_penalty_each |
|---|---:|---:|---:|---|---|---:|
| v1，原实验保留 | 0 | 0 | 0 | 未覆盖参考tube | 无 | 1 |
| v2，新实验 | 0.5 | 2.0 | 0.5 | 0.05 | 4.0 | 4 |

anchor_weight=0.02、synergy_weight=0.05；T3相位偏移0，T4相位偏移10 bins。
远端新分支中“T0s1/s2未完成”的历史描述已与本机在跑状态区分，没有重复派发T0。

- [新旧肌电差异审查](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260919/change_review.md)
- [运行合同](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260919/matched_contract.json)、[运行状态](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260919/status.json)
- 本地日志快照：[T3](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260919/logs/t3_s0_v1.log)、[T4](../artifacts/stage1_fc_anchor_v2_remote7_38bb5c7_20260919/logs/t4_s0_v1.log)；持续日志位于远端同名运行包。
- tmux socket：/data3/yangfeiyang/tmp/tmux_fc_anchor_v2_20260919.sock；sessions：stage1_fc_anchor_v2_remote7_t3_s0_v1、stage1_fc_anchor_v2_remote7_t4_s0_v1。
- 完整manifest路径见验收JSON；start.sh已执行，禁止重复启动。
<!-- ANCHOR_V2_REMOTE7_20260919_END -->



## 1. 当前结论

- **旧版 Forehand Clear seeds0/1/2 均 5/5 DONE，合计 15/15 完成、0 运行中、0 未启动。** 09-18 审计完成的 12 叶加上 09-22 补齐的 T0 seed1、T0 seed2、T1 seed2；各叶 finalized update39063 / 800010240 steps、65 条验证历史，W&B finished。anchor v2 的 9 叶单独统计。
- seed0 T0/T4 的 local9 replacement 已完成，T1/T2/T3 使用 full5 v3 已完成 endpoint；旧 full5 T0 中断与 T4 失败记录继续保留，不重复计叶。
- **本机 T1 seed1 已完成**；9 月 11 日 T2/T3/T4 seed1 也已完成。T0 seed1 本次已完成；远端 T1 不计作 T0 或额外 seed。
- seed2 的 T0–T4 均已完成 exact endpoint；09-22 已核对最后三组完成并释放 GPU，获授权在本机 GPU1/2/3 启动 T1v2 seed0、T3v2 seed1、T4v2 seed1。
- 远端GPU3的T1 seed1 v4是额外重跑，9月21日已核验800010240步、finalized update39063、65条验证历史与W&B finished；记DONE duplicate，不增加唯一叶数量，也不覆盖本机已完成T1 seed1的证据。远端 T0/T4 seed0 已于 9 月 1 日按用户要求停止，属于 `INTERRUPTED` duplicate。
- Stage2 smoke-scale v4 的既有结论仍为 `COMPLETED_SMOKE_SCALE_ACCEPTANCE_FALSE`；本次没有重跑 Stage2 或将其计入 Stage1 分母。训练叶完成不等于 pairwise gate、blind review 或 teacher promotion 已通过。
- 上述旧版 family 为用户批准的 `c1ccd9328e40 / f377a61e6166 / Aug100 grouped 80+20 / 800010240 steps`，fresh optimizer、no-resume、promotion.auto_stop=false；不与原始 22/5、320M 或历史 40/10 结果混算。

09-18 的 12 叶审计：[audit.json](../artifacts/experiment_log_sync_20260918/audit.json)，包含完整 run ID、config hash、source snapshot、checkpoint identity、manifest/validation 哈希、release/QC 绑定、日志、终态指标和 W&B 状态。
seed1 完成与 seed2 启动证据另见 [completion](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/completion_20260912.json)、
[startup acceptance](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/startup_acceptance_20260912.json)。

### 1.1 进度记分板

| 统计范围 | 计划训练叶 | 完成 | 运行中 | 未启动/阻塞 | 完成率 | 已触达率 |
|---|---:|---:|---:|---:|---:|---:|
| Forehand Clear seed0 | 5 | 5 | 0 | 0 | 100% | 100% |
| Forehand Clear seed1 | 5 | 5 | 0 | 0 | 100% | 100% |
| Forehand Clear seed2 | 5 | 5 | 0 | 0 | 100% | 100% |
| Forehand Clear：5 arms × seeds 0/1/2 | 15 | 15 | 0 | 0 | 100% | 100% |
| Stage1 单 seed 总队列：3 actions × 5 arms | 15 | 5 | 0 | 10 | 33.33% | 33.33% |
| Stage1 旧版总队列：3 actions × 5 arms × 3 seeds | 45 | 15 | 0 | 30 | 33.33% | 33.33% |

已封存的唯一完成叶合计 **12000153600 steps**（15 × 800010240）。
远端重跑、replacement 的旧失败尝试、历史 checkpoint 和 Stage2 均不增加以上分母或完成数；
Lift/ChinaJump 状态沿用原合同记录，本次未复核其私有资产。


### 1.2 “完成”的唯一判定

一个训练叶只有同时满足以下条件才可标为 `DONE`：

1. 达到 resolved contract 的 exact fixed endpoint；
2. 最新 checkpoint 已 finalized，且 immutable checkpoint identity 可重建；
3. endpoint held-out validation evidence 已封存；
4. run manifest、config hash、source fingerprint、split/tube binding 与同 family 一致；
5. W&B final state、日志和异常审计已登记；
6. 不是 dry-run、单元测试、历史 checkpoint、无效启动或不同 family 的结果。

### 历史：实时进度复核（2026-09-18 06:50，北京时间）

Forehand Clear 仍为 **12/15 DONE、3/15 RUNNING、0未启动**。本机 GPU1/2/3 的原训练 PID 均存活，利用率99%/100%/100%；W&B 三组均 running，日志未检出 traceback、NaN 或 CUDA OOM。

| 实验 | GPU | W&B 实时步数 | 预算进度 | 已 finalized checkpoint |
|---|---:|---:|---:|---|
| T0 seed1 | 1 | 40,919,040 | 5.11% | update1830 / 37,478,400 steps |
| T0 seed2 | 2 | 41,902,080 | 5.24% | update1830 / 37,478,400 steps |
| T1 seed2 | 3 | 41,267,200 | 5.16% | update1830 / 37,478,400 steps |

各有3/65条验证历史。按 checkpoint610→1830 的实际保存间隔估算，剩余约89–91小时，约9月22日凌晨完成；当前仍处早期，速度变化会影响估计。远端重复run本次未复核。
证据：[实时进度](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/progress_20260918.json)、[W&B快照](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/live_wandb_20260918.json)。

### 1.3 历史：最后三组启动验收（2026-09-18 01:58）

T0 seed1、T0 seed2、T1 seed2 已通过96项测试、完整逐卡preflight/JAX、dry-run及W&B身份检查。
每组命中本地80+20条轨迹，manifest/seed/reward/800010240预算/fresh optimizer正确；
已进入实际训练循环，W&B在线且步数推进，未发现fatal/NaN/OOM或远程下载。
当前 Forehand Clear **12/15 DONE、3/15 RUNNING、0未启动**；启动验收不等于训练完成。

| 实验 | 物理GPU | Python PID | W&B | Config hash |
|---|---:|---:|---|---|
| T0 seed1 | 1 | 4109420 | [l9t0s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s1v1) | `43dcd580d816` |
| T0 seed2 | 2 | 4109499 | [l9t0s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s2v1) | `675fd5b5e847` |
| T1 seed2 | 3 | 4109416 | [l9t1s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s2v1) | `124ba2c7050f` |

Run id 前缀：`forehand_clear_aug100_80train20val_peasd_c1ccd93_local9_multiseed_direct800_v1`，
分别追加 `_t0_s1`、`_t0_s2`、`_t1_s2`。
冻结源码 c1ccd9328e40、source fingerprint f377a61e…，80/20来源分组不重叠，
每组800010240步、39063 updates、65条验证；Orbax save/restore各4GB。

- [五项验收快照，含完整manifest/日志路径和W&B心跳](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/startup_acceptance_20260918.json)
- [运行合同](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/matched_contract.json)；[持续启动监控](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/production_status.json)
- 日志：运行包的 `logs/t0_s1_v1.log`、`logs/t0_s2_v1.log`、`logs/t1_s2_v1.log`。
- tmux socket：`/tmp/musclemimic_yangfeiyang_stage1_fc_aug100_local9_remaining_20260918_v1.sock`；
  sessions：`stage1_fc_aug100_local9_t0_s1_v1`、`stage1_fc_aug100_local9_t0_s2_v1`、`stage1_fc_aug100_local9_t1_s2_v1`。
- 运行包 `start.sh` 已执行，不得重复派发。后台监控继续观察稳定窗口。
- 先同步实验主表再启动；旧seed2任务在finalized endpoint/W&B finished后各一次Ctrl-C清理，全部结果保留。

## 已完成权重上传 Hugging Face（2026-09-18，上传已验证）

<!-- HF_SERIAL_STATUS_BEGIN -->
本机串行上传已完成并通过全部远端文件大小及哈希验证（2026-09-18T05:12:28.272515+08:00）。提交：[1386b0990e1f7d5a29923f54f0f2e18d8a5c7f41](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903/commit/1386b0990e1f7d5a29923f54f0f2e18d8a5c7f41)。
<!-- HF_SERIAL_STATUS_END -->

用户指定目标：[yangfy0627/musclemimic-checkpoints-20260903](https://huggingface.co/yangfy0627/musclemimic-checkpoints-20260903)。
已整理12组DONE的finalized checkpoint_39063，共161个checkpoint文件、4818120893字节，
每个文件已计算SHA-256；完整保留Orbax train_state/config/metadata结构，未复制原始轨迹、SMPL资产或私有资产清单。
沿用seed0/seed1/seed2与T0–T4目录，旧中途checkpoint保留；新索引为COMPLETED_20260918.json/md和SHA256SUMS_20260918。
历史上传排障记录（已由上方成功提交取代）：用户已完成HF登录且凭据被服务器接受。当时仓库已有40个同哈希文件，需新增121个checkpoint文件（约3.61GB）和3个索引。先后尝试本机与远端转发代理的Xet和HTTP/LFS；Xet出现Request TimedOut，HTTP/LFS大文件持续停顿，直连也超时。**上传尚未完成、没有成功的新提交**；已停止本次上传进程并关闭临时转发，原仓库文件、完整本地payload和日志均保留。需稳定上传网络后重试，状态以JSON为准。

按用户要求再次关闭全部代理直连（2026-09-18T03:33:29.876182+08:00）：HF 主站与 LFS 存储均在 TCP 建连后 TLS 握手超时（15秒，curl 28，HTTP 000），因此未进入文件上传。见[直连检查记录](../artifacts/hf_completed_upload_20260918/direct_connect_check.json)。

- [本地上传状态](../artifacts/hf_completed_upload_20260918/status.json)
- [上传清单和SHA-256](../artifacts/hf_completed_upload_20260918/payload/COMPLETED_20260918.json)
- [上传与远端逐文件验证脚本](../artifacts/hf_completed_upload_20260918/upload.py)

## 2. 状态字典与统计规则

| 状态 | 含义 | 是否计入“完成” |
|---|---|:--:|
| `DONE` | exact endpoint、checkpoint、validation evidence 和 provenance 全部封存 | 是 |
| `RUNNING` | Python PID 和 CUDA context 存在，日志仍在推进 | 否 |
| `INTERRUPTED` | 未到 exact endpoint，当前无训练 PID；保留了 finalized checkpoint | 否 |
| `TODO` | 合同已锁定但尚未启动 | 否 |
| `BLOCKED` | 上游训练、数据、tube、review 或 lineage 未满足 | 否 |
| `ASSET/CONTRACT-BLOCKED` | 数据可能存在，但本 family 的 split/config/tube/evidence 尚未在本机锁定 | 否 |
| `HISTORICAL-NONFORMAL` | 有用的历史/诊断结果，但不属于当前 matched family | 否 |
| `INVALID` | 共享输出目录、错误 lineage、dry-run、旧 optimizer 或其他合同违规 | 否 |
| `FAILED` | 正式运行发生不可接受错误且没有可用 endpoint | 否 |

进度公式：

- `训练叶完成率 = DONE 训练叶数 / 计划训练叶数`；
- `已触达率 = (DONE + RUNNING + INTERRUPTED + FAILED) / 计划训练叶数`；
- `训练步数进度 = Σ min(最新 global timestep, target) / Σ target`；仅在各叶预算已锁定时计算；
- preflight、tests、dry-run、start validation、post-hoc、evidence index、blind review、pairwise gate 和 promotion 单独登记，不计作训练叶。

## 3. Lineage 与 source-of-truth

### 3.1 当前不可混合的 lineage

| Lineage | 用途 | Git SHA / source fingerprint | 数据与 split | Seed / budget | 当前状态 | 结论 |
|---|---|---|---|---|---|---|
| `release-250aa` | 根目录规范定义的默认正式发布 family | `250aa53777af...` / `0a61388a6ac0...` | 原始 release：22 train + 5 held-out validation | seed 0；预算由正式 resolved config 决定 | 新 matched family 尚无完成叶 | 原 formal release 控制口径；本次运行不属于它 |
| `aug100-full5-direct800-v3` | 首轮 T0–T4 单 seed matched family | `c1ccd9328e40...` / `f377a61e6166...` | `raw_smooth_v1_aug100`；按 source group 切分 80 train + 20 held-out validation | seed 0；`800,010,240` steps / arm | T1/T2/T3 `DONE`；T0 `INTERRUPTED`；T4 `FAILED` | T1/T2/T3 endpoint 可用；T0/T4 旧 optimizer/run 不恢复 |
| `aug100-local9-t0t4-direct800-v1` | **当前本机 T0/T4 replacement family** | `c1ccd9328e40...` / `f377a61e6166...` | 与 v3/remote7 语义等价的 Aug100 80/20 grouped split | seed 0；`800,010,240` steps / arm | T0/T4 `DONE`；各 65 条 validation、exact endpoint、W&B finished | 使用全新 run/W&B/checkpoint/optimizer；用于补齐 T0/T4 唯一叶，已补齐 seed0；旧失败/中断尝试不计数 |
| `aug100-local9-t1-s1-direct800-v1` | **当前本机 T1 第二个独立 seed** | `c1ccd9328e40...` / `f377a61e6166...` | 与 T1 seed 0 相同的 Aug100 80/20 grouped split | seed 1；`800,010,240` steps | `DONE`；65 条 validation、exact endpoint、W&B finished | 全新 optimizer/run/W&B/checkpoint 身份；是正式多 seed 扩展的一叶 |
| `aug100-local9-seed1-remaining-direct800-v1` | 本机 T2/T3/T4 第二个 seed | `c1ccd9328e40 / f377a61e6166` | Aug100 grouped 80/20 | seed1；每叶 800010240 steps | T2/T3/T4 DONE | 三叶各自 fresh optimizer、65 条 validation、W&B finished |
| `aug100-local9-seed2-t234-direct800-v1` | 本机 T2/T3/T4 第三个 seed（已完成） | `c1ccd9328e40 / f377a61e6166` | Aug100 grouped 80/20 | seed2；每叶 800010240 steps | 三叶 DONE | 各65条validation和exact endpoint，W&B finished |
| `aug100-remote7-t1-s1-direct800-v4` | 远端同 seed 额外重跑 | `c1ccd9328e40 / f377a61e6166` | Aug100 grouped 80/20 | seed1；800010240 steps | DONE；800010240 steps / 65 条 validation；原 PID 已退出 | duplicate；不增加唯一叶，不替换本机已完成 T1 seed1 |
| `aug100-remote7-t0t4-direct800-v2` | 服务器7辅助 T0/T4 duplicate family | `c1ccd9328e40...` / `f377a61e6166...` | 与服务器9 v3 语义等价的 Aug100 80/20 grouped split | seed 0；`800,010,240` steps / arm | `INTERRUPTED`；T0 update 25,010，T4 update 24,400 | 2026-09-01 按用户要求优雅停止；独立 duplicate lineage，不重复计数唯一训练叶 |
| `aug100-direct800-v2` | 上一 T0/T1/T2 family | `c1ccd9328e40...` / `f377a61e6166...` | 同为 Aug100 80/20，但 run/checkpoint optimizer lineage 不同 | seed 0；`800,010,240` steps / arm | T0/T1/T2 中断 | 历史中途证据；禁止 resume/merge 到 v3 |

控制证据：

- 正式服务器合同：[`../AGENTS.md`](../AGENTS.md)
- 当前 v3 family 合同：[`matched_contract.json`](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/matched_contract.json)
- 当前 v3 硬门禁验收：[`verification.json`](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/verification.json)
- 当前 v3 生产启动状态：[`production_status.json`](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/production_status.json)
- 服务器7 T0/T4 合同：[`matched_contract.json`](../artifacts/stage1_forehand_clear_aug100_remote7_t0t4_direct800_c1ccd93_20260824/matched_contract.json)
- 服务器7 T0/T4 启动状态：[`production_status.json`](../artifacts/stage1_forehand_clear_aug100_remote7_t0t4_direct800_c1ccd93_20260824/production_status.json)
- 本机 replacement T0/T4 启动包：[`README.md`](../artifacts/stage1_forehand_clear_aug100_local9_t0t4_direct800_c1ccd93_20260831/README.md)
- 本机 replacement T0/T4 启动状态：[`production_status.json`](../artifacts/stage1_forehand_clear_aug100_local9_t0t4_direct800_c1ccd93_20260831/production_status.json)
- 本机 T1 seed-1 启动包：[`README.md`](../artifacts/stage1_forehand_clear_aug100_local9_t1_s1_direct800_c1ccd93_20260831/README.md)
- 本机 T1 seed-1 启动状态：[`production_status.json`](../artifacts/stage1_forehand_clear_aug100_local9_t1_s1_direct800_c1ccd93_20260831/production_status.json)
- 上一 v2 family 合同：[`matched_contract.json`](../artifacts/stage1_t0_t1_t2_aug100_direct800_c1ccd93_20260817/matched_contract.json)
- Aug100 数据说明：[`新服务器Aug100增广数据说明.md`](../docs/runbooks/server9/新服务器Aug100增广数据说明.md)

### 3.2 当前 lineage 的质量结论

| 检查 | 结果 | 严重度 | 对进度统计的影响 |
|---|---|---|---|
| v3 endpoint 审计 | T1/T2/T3 均有 checkpoint 39063、sealed endpoint validation、W&B `finished`，日志无 fatal/NaN/OOM/download | 低 | 三臂可标 `DONE` |
| v3 T0/T4 终态 | T0 最后 finalized checkpoint 为 update 610，随后中断；T4 出现 `CUDA_ERROR_ILLEGAL_ADDRESS` 且无 finalized checkpoint | Critical | 两臂不能标 `DONE`，不得恢复旧 optimizer |
| local9 replacement 配置等价性 | 与 remote7/v3 保持同 source、split、seed、budget、reward、terminal、validation 与 fresh/no-resume 合同；仅主机身份不同 | 低 | 可作为 T0/T4 replacement 继续训练 |
| local9 replacement 完成审计 | T0/T4 finalized checkpoint 39063、65 条 validation、W&B finished，无训练 fatal | 低 | 两叶均 DONE，见本次 audit.json |
| T1 seed-1 完成审计 | 本机 exact endpoint、65 条 validation、W&B finished；远端另有同 seed 重跑 | 低 | 唯一叶计 1 个 DONE，远端不额外计数 |
| 服务器7 v2 当前状态 | W&B API 与真实进程状态不一致；现场确认两进程仍训练，随后按用户要求各一次 Ctrl-C，Python/CUDA context 均消失 | High | 改记 `INTERRUPTED`；保留 update 25,010/24,400 checkpoint，不作为额外 seed 或 replacement endpoint |
| 上一 v2 停止原因是否可审计 | 日志末尾没有 traceback、NaN、OOM 或正常 completion marker；停止原因未登记 | High | 只保留历史，不能假设正常完成，也不能 resume 到 v3 |
| 是否满足 `AGENTS.md` 的默认发布 identity | 不满足；用户已显式批准本次 `c1ccd932/f377a61` 覆盖 | High | v3 按独立授权合同汇报，不计入 `release-250aa` 完成率 |
| Lift/ChinaJump 当前本机是否具备可重建 verified tube/gate | 计划文档称已完成，但当前资产清单和 `artifacts/` 中仅发现 Clear verified tube | High | 两个泛化动作先标 `ASSET/CONTRACT-BLOCKED`，不依据旧文档直接标可运行 |

## 4. Stage 1 实验设计矩阵

所有 arm 都训练 full-354 muscle tracking teacher；sEMG 只约束 measured subspace，不是 354 维肌肉真值。

| Arm | Activation anchor | Synergy anchor | Treatment | 主要用途 | 必要比较 |
|---|---:|---:|---|---|---|
| T0 | 0 | 0 | Tube-free tracking baseline；endpoint 后做 post-hoc physiology | 无 EMG 训练基线 | T3 vs T0 |
| T1 | 0.02 | 0 | Activation-only | 分解 measured activation anchor 的作用 | 诊断性 T1 vs T0/T3 |
| T2 | 0 | 0.05 | Real synergy-only | 分解 synergy 项的作用 | 诊断性 T2 vs T0/T3 |
| T3 | 0.02 | 0.05 | Real PEASD-Lite | 主方法与 deployment teacher 候选 | T3 vs T0、T3 vs T4 |
| T4 | 0.02 | 0.05 | Synergy phase 循环平移 10/20 bins；activation anchor 不移动 | 负对照，检验真实 phase structure | T3 vs T4（预注册主对比） |

统一合同要求：同 action 内必须保持相同 source snapshot、grouped split、预算、seed、validation schedule、optimizer freshness 与 promotion thresholds。T1/T2 是 decomposition diagnostics，不能替代 T3 vs T4 主 gate。

## 5. Stage 1 单 seed 项目主表（15 个训练叶）

### 5.1 P0 主结果：Forehand Clear

| ID | Arm | Seed | 合同 | 状态 | 完成进度 | 证据 |
|---|---|---:|---|---|---|---|
| FC-S1-T0-S0 | T0 | 0 | local9_t0t4_direct800_v1 | `DONE` | 39063 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_local9_t0t4_direct800_c1ccd93_20260831/logs/t0_s0_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205114-pid3929763-e2006c/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205114-pid3929763-e2006c/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205114-pid3929763-e2006c/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s0v1) |
| FC-S1-T1-S0 | T1 | 0 | full5_direct800_v3 | `DONE` | 39063 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/logs/t1_s0.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083626-pid458486-f86af6/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083626-pid458486-f86af6/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083626-pid458486-f86af6/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t1s0v3) |
| FC-S1-T2-S0 | T2 | 0 | full5_direct800_v3 | `DONE` | 39063 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/logs/t2_s0.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458396-ac6856/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458396-ac6856/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458396-ac6856/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t2s0v3) |
| FC-S1-T3-S0 | T3 | 0 | full5_direct800_v3 | `DONE` | 39063 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/logs/t3_s0.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458414-35b177/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458414-35b177/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458414-35b177/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t3s0v3) |
| FC-S1-T4-S0 | T4 | 0 | local9_t0t4_direct800_v1 | `DONE` | 39063 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_local9_t0t4_direct800_c1ccd93_20260831/logs/t4_s0_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205116-pid3930267-510219/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205116-pid3930267-510219/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205116-pid3930267-510219/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s0v1) |


### 5.2 P1 泛化：China Jump

| ID | Arm | Seed | 状态 | 启动前缺口 | 完成后用途 |
|---|---:|---:|---|---|---|
| CJ-S1-T0-S0 | T0 | 0 | `ASSET/CONTRACT-BLOCKED` | 锁定新 family source identity、Aug100 grouped split、budget、run id；本机复核 verified tube/gate 资产 | body-only/phase-free 泛化基线 |
| CJ-S1-T1-S0 | T1 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；activation-only treatment | decomposition |
| CJ-S1-T2-S0 | T2 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；synergy-only treatment | decomposition |
| CJ-S1-T3-S0 | T3 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；real PEASD-Lite | 泛化主方法 |
| CJ-S1-T4-S0 | T4 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；phase-shift 负对照 | 泛化因果对照 |

历史 ChinaJump bootstrap/continuity diagnostics 不属于上述 T0–T4，不能替代这些叶节点。

### 5.3 P2 泛化：Forehand Lift

| ID | Arm | Seed | 状态 | 启动前缺口 | 完成后用途 |
|---|---:|---:|---|---|---|
| FL-S1-T0-S0 | T0 | 0 | `ASSET/CONTRACT-BLOCKED` | 锁定新 family source identity、Aug100 grouped split、budget、run id；本机复核 verified tube/gate 资产 | 泛化基线 |
| FL-S1-T1-S0 | T1 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；activation-only treatment | decomposition |
| FL-S1-T2-S0 | T2 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；synergy-only treatment | decomposition |
| FL-S1-T3-S0 | T3 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；real PEASD-Lite | 泛化主方法 |
| FL-S1-T4-S0 | T4 | 0 | `ASSET/CONTRACT-BLOCKED` | 同上；phase-shift 负对照 | 泛化因果对照 |

### 5.4 Forehand Clear 正式多 seed 扩展

seed 编号从 0 开始；每个 arm/seed 的 optimizer、run、日志和 checkpoint 身份独立。
T0 seed1、T0/T1 seed2 已于9月22日完成 endpoint 核验，原进程已退出，全部15叶 DONE。

| ID | Arm | Seed | 状态 | 进度 / 验证历史 | 证据 |
|---|---|---:|---|---|---|
| FC-S1-T0-S1 | T0 | 1 | `DONE` | 800010240 / 800010240；65 条 validation；W&B finished；原 PID 已退出 | [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_remaining_20260918_direct800_v1/checkpoints/260917T175607-pid4109420-a1cd88/manifest.json) · [log](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/logs/t0_s1_v1.log) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s1v1) |
| FC-S1-T1-S1 | T1 | 1 | `DONE` | 800010240 / 800010240；65 条 validation | [log](../artifacts/stage1_forehand_clear_aug100_local9_t1_s1_direct800_c1ccd93_20260831/logs/t1_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t1_s1_direct800_v1/checkpoints/260830T211611-pid3949440-b91190/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t1_s1_direct800_v1/checkpoints/260830T211611-pid3949440-b91190/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t1_s1_direct800_v1/checkpoints/260830T211611-pid3949440-b91190/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s1v1) |
| FC-S1-T2-S1 | T2 | 1 | `DONE` | 800010240 / 800010240；65 条 validation | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/logs/t2_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162022-pid810175-b4fd51/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162022-pid810175-b4fd51/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162022-pid810175-b4fd51/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s1v1) |
| FC-S1-T3-S1 | T3 | 1 | `DONE` | 800010240 / 800010240；65 条 validation | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/logs/t3_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162017-pid810026-63b951/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162017-pid810026-63b951/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162017-pid810026-63b951/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s1v1) |
| FC-S1-T4-S1 | T4 | 1 | `DONE` | 800010240 / 800010240；65 条 validation | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/logs/t4_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162020-pid810017-9c515c/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162020-pid810017-9c515c/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162020-pid810017-9c515c/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s1v1) |
| FC-S1-T0-S2 | T0 | 2 | `DONE` | 800010240 / 800010240；65 条 validation；W&B finished；原 PID 已退出 | [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_remaining_20260918_direct800_v1/checkpoints/260917T175607-pid4109499-595c95/manifest.json) · [log](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/logs/t0_s2_v1.log) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s2v1) |
| FC-S1-T1-S2 | T1 | 2 | `DONE` | 800010240 / 800010240；65 条 validation；W&B finished；原 PID 已退出 | [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_remaining_20260918_direct800_v1/checkpoints/260917T175607-pid4109416-5c1928/manifest.json) · [log](../artifacts/stage1_forehand_clear_aug100_local9_remaining_direct800_c1ccd93_20260918/logs/t1_s2_v1.log) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s2v1) |
| FC-S1-T2-S2 | T2 | 2 | `DONE` | 800010240 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/logs/t2_s2_v1.log) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed2_t234_direct800_v1/checkpoints/260911T170342-pid3893205-25a4da/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s2v1) |
| FC-S1-T3-S2 | T3 | 2 | `DONE` | 800010240 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/logs/t3_s2_v1.log) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed2_t234_direct800_v1/checkpoints/260911T170339-pid3893214-ae58d4/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s2v1) |
| FC-S1-T4-S2 | T4 | 2 | `DONE` | 800010240 / 800010240；65 条 validation；W&B finished | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/logs/t4_s2_v1.log) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed2_t234_direct800_v1/checkpoints/260911T170339-pid3893184-f689cc/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s2v1) |


## 6. 当前 Forehand Clear T0–T4 运行明细

### 6.1 v3 终态审计

| Arm | Run ID / W&B | 物理 GPU / Python PID | Target update / steps | Checkpoint root / config hash | tmux session | 启动验收状态 |
|---|---|---|---:|---|---|---|
| T0 | `...full5_direct800_v3_t0_s0` · [W&B c8t0s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t0s0v3) | 原 GPU 0 / PID 已退出 | target 39,063 / 800,010,240；最后 finalized 610 / 12,492,800 | `260823T083625-pid458413-92e23a` / `0191dcd12b60` | `forehand_clear_aug100_full5_v3_t0_s0`（已关闭） | `INTERRUPTED`；W&B `failed`，旧 optimizer 禁止恢复 |
| T1 | `...full5_direct800_v3_t1_s0` · [W&B c8t1s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t1s0v3) | 原 GPU 1 / PID 已退出 | 39,063 / 800,010,240 | `260823T083626-pid458486-f86af6` / `9737112577fe` | `forehand_clear_aug100_full5_v3_t1_s0`（已关闭） | `DONE`；endpoint/validation sealed，W&B `finished` |
| T2 | `...full5_direct800_v3_t2_s0` · [W&B c8t2s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t2s0v3) | 原 GPU 2 / PID 已退出 | 39,063 / 800,010,240 | `260823T083624-pid458396-ac6856` / `e367c098794a` | `forehand_clear_aug100_full5_v3_t2_s0`（已关闭） | `DONE`；endpoint/validation sealed，W&B `finished` |
| T3 | `...full5_direct800_v3_t3_s0` · [W&B c8t3s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t3s0v3) | 原 GPU 3 / PID 已退出 | 39,063 / 800,010,240 | `260823T083624-pid458414-35b177` / `e8becc5b0fc6` | `forehand_clear_aug100_full5_v3_t3_s0`（已关闭） | `DONE`；endpoint/validation sealed，W&B `finished` |
| T4 | `...full5_direct800_v3_t4_s0` · [W&B c8t4s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t4s0v3) | 原 GPU 4 / PID 已退出 | 未产生 finalized checkpoint | `260823T083625-pid458634-dd6bba` / `4202cd70c56e` | `forehand_clear_aug100_full5_v3_t4_s0`（已关闭） | `FAILED`；`CUDA_ERROR_ILLEGAL_ADDRESS`，W&B `failed` |

旧统一 tmux socket 已关闭。五臂均以 fresh optimizer、`auto_resume=false`、`resume_from=null`、`promotion.auto_stop=false` 启动；终态以 endpoint checkpoint、sealed validation、日志和当前 W&B 状态联合判定，旧启动状态文件本身不是当前运行证据。

### 6.1B 本机完成叶与 seed2 当前运行身份（2026-09-12）

完成时间取 finalized endpoint commit；运行中步数取本次 W&B API，不混作已保存 checkpoint。

| Arm / Seed | W&B | 状态 | Endpoint 保存时间 / 当前 history step | Config hash | 证据 |
|---|---|---|---|---|---|
| T0-S0 | [l9t0s0v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s0v1) | DONE | 2026-09-04T02:23:56.683699+08:00 | `0191dcd12b60` | [log](../artifacts/stage1_forehand_clear_aug100_local9_t0t4_direct800_c1ccd93_20260831/logs/t0_s0_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205114-pid3929763-e2006c/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205114-pid3929763-e2006c/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205114-pid3929763-e2006c/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t0s0v1) |
| T1-S0 | [c8t1s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t1s0v3) | DONE | 2026-08-27T15:48:11.012291+08:00 | `9737112577fe` | [log](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/logs/t1_s0.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083626-pid458486-f86af6/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083626-pid458486-f86af6/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083626-pid458486-f86af6/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t1s0v3) |
| T2-S0 | [c8t2s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t2s0v3) | DONE | 2026-08-27T13:31:52.437866+08:00 | `e367c098794a` | [log](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/logs/t2_s0.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458396-ac6856/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458396-ac6856/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458396-ac6856/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t2s0v3) |
| T3-S0 | [c8t3s0v3](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t3s0v3) | DONE | 2026-08-27T15:00:16.078229+08:00 | `e8becc5b0fc6` | [log](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/logs/t3_s0.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458414-35b177/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458414-35b177/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_full5_direct800_v3/checkpoints/260823T083624-pid458414-35b177/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/c8t3s0v3) |
| T4-S0 | [l9t4s0v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s0v1) | DONE | 2026-09-04T00:09:11.376162+08:00 | `4202cd70c56e` | [log](../artifacts/stage1_forehand_clear_aug100_local9_t0t4_direct800_c1ccd93_20260831/logs/t4_s0_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205116-pid3930267-510219/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205116-pid3930267-510219/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t0t4_direct800_v1/checkpoints/260830T205116-pid3930267-510219/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s0v1) |
| T1-S1 | [l9t1s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s1v1) | DONE | 2026-09-04T02:14:25.605591+08:00 | `39337a20791e` | [log](../artifacts/stage1_forehand_clear_aug100_local9_t1_s1_direct800_c1ccd93_20260831/logs/t1_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t1_s1_direct800_v1/checkpoints/260830T211611-pid3949440-b91190/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t1_s1_direct800_v1/checkpoints/260830T211611-pid3949440-b91190/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_t1_s1_direct800_v1/checkpoints/260830T211611-pid3949440-b91190/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t1s1v1) |
| T2-S1 | [l9t2s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s1v1) | DONE | 2026-09-11T22:09:09.656631+08:00 | `0aab46adcc5d` | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/logs/t2_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162022-pid810175-b4fd51/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162022-pid810175-b4fd51/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162022-pid810175-b4fd51/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s1v1) |
| T3-S1 | [l9t3s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s1v1) | DONE | 2026-09-11T19:43:43.750160+08:00 | `ec904c2abe77` | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/logs/t3_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162017-pid810026-63b951/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162017-pid810026-63b951/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162017-pid810026-63b951/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s1v1) |
| T4-S1 | [l9t4s1v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s1v1) | DONE | 2026-09-11T21:48:35.260814+08:00 | `760f788d65e2` | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed1_remaining_direct800_c1ccd93_20260906/logs/t4_s1_v1.log) · [endpoint](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162020-pid810017-9c515c/checkpoint_39063) · [validation](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162020-pid810017-9c515c/stage1_peasd_validation_history.json) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed1_remaining_direct800_v1/checkpoints/260907T162020-pid810017-9c515c/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s1v1) |
| T2-S2 | [l9t2s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s2v1) | DONE | 2026-09-15T23:07:50.968364+08:00 | `453a747b9464` | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/logs/t2_s2_v1.log) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed2_t234_direct800_v1/checkpoints/260911T170342-pid3893205-25a4da/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t2s2v1) |
| T3-S2 | [l9t3s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s2v1) | DONE | 2026-09-15T20:25:10.319021+08:00 | `5649ecd4d447` | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/logs/t3_s2_v1.log) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed2_t234_direct800_v1/checkpoints/260911T170339-pid3893214-ae58d4/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t3s2v1) |
| T4-S2 | [l9t4s2v1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s2v1) | DONE | 2026-09-15T22:01:10.368810+08:00 | `0671796d7370` | [log](../artifacts/stage1_forehand_clear_aug100_local9_seed2_t234_direct800_c1ccd93_20260912/logs/t4_s2_v1.log) · [manifest](../datasets/forehandClear_standard/training_aug100_80train20val_peasd_local9_seed2_t234_direct800_v1/checkpoints/260911T170339-pid3893184-f689cc/manifest.json) · [W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/l9t4s2v1) |

原 T2/T3/T4 seed2 均已完成；旧GPU/PID仅是历史位置。当前GPU123任务见第1.3节。

### 6.1C 已完成 seed0/seed1/seed2 的 endpoint held-out 指标

以下均来自各 run 最后一次 fixed-endpoint validation；不做跨 seed 聚合，
不将单次 endpoint 指标直接判作 last-3/pairwise promotion gate。
T0 未训练 EMG，EMG 可比结论仍待 post-hoc physiology。

| Arm / Seed | Coverage ↑ | Early term ↓ | RPos (m) ↓ | Action saturation ↓ | Activation energy ↓ | Anchor loss ↓ | Real-reference synergy loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0-S0 | 86.94% | 40.00% | 0.0977 | 59.88% | 0.4495 | N/A | N/A |
| T1-S0 | 77.27% | 75.00% | 0.1120 | 56.61% | 0.4011 | 4.4414 | 0.9868 |
| T2-S0 | 91.67% | 30.00% | 0.0944 | 61.81% | 0.4080 | 1.9568 | 0.1728 |
| T3-S0 | 89.89% | 40.00% | 0.1079 | 62.53% | 0.3987 | 0.9573 | 0.1957 |
| T4-S0 | 85.56% | 30.00% | 0.0901 | 64.41% | 0.4053 | 1.1105 | 0.4849 |
| T1-S1 | 80.32% | 40.00% | 0.1043 | 62.28% | 0.4231 | 5.4598 | 1.0990 |
| T2-S1 | 92.26% | 30.00% | 0.0926 | 59.36% | 0.4115 | 1.2970 | 0.1371 |
| T3-S1 | 87.69% | 20.00% | 0.0866 | 64.99% | 0.4035 | 1.4029 | 0.1950 |
| T4-S1 | 87.34% | 40.00% | 0.0938 | 60.56% | 0.4010 | 1.1169 | 0.5562 |
| T2-S2 | 89.05% | 30.00% | 0.1138 | 58.56% | 0.4162 | 1.5160 | 0.1260 |
| T3-S2 | 85.48% | 30.00% | 0.0990 | 57.89% | 0.4089 | 0.4875 | 0.1855 |
| T4-S2 | 92.19% | 20.00% | 0.0928 | 58.35% | 0.3897 | 1.4868 | 0.5391 |

### 6.1D 远端 T1 seed1 重跑（不增加唯一叶计数）

`r7t1s1v4` / GPU3 / PID2237783 于 2026-09-12 01:28 确认运行，
finalized update8540 / 174899200 steps（21.86%），14 条验证历史；
本机 `l9t1s1v1` 已完成，同 arm/seed 仅计一叶，不自动替换已登记 endpoint。
[远端进度证据](../artifacts/stage1_forehand_clear_aug100_remote7_t1_s1_direct800_c1ccd93_20260910_rerun4/progress_20260912.json)；
[W&B](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7t1s1v4)。


### 6.1A 服务器7辅助 T0/T4 v2 终态复核（不重复计数唯一叶）

| Arm | Run ID / W&B | 物理 GPU / Python PID | Target update / steps | Checkpoint root / config hash | tmux session | 启动验收状态 |
|---|---|---|---:|---|---|---|
| T0 | `...remote7_t0t4_direct800_v2_t0_s0` · [W&B r7t0s0v2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7t0s0v2) | 原 GPU 1 / PID 4175945（已退出） | 25,010 / 512,204,800（64.02%） | `260824T035845-pid4175945-614c98/checkpoint_25010` / `57a5a399c426` | socket 已自然退出 | `INTERRUPTED`；用户要求停止，finalized checkpoint 保留 |
| T4 | `...remote7_t0t4_direct800_v2_t4_s0` · [W&B r7t4s0v2](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/r7t4s0v2) | 原 GPU 1 / PID 4175837（已退出） | 24,400 / 499,712,000（62.46%） | `260824T035845-pid4175837-dca986/checkpoint_24400` / `0429efa1e55f` | socket 已自然退出 | `INTERRUPTED`；用户要求停止，finalized checkpoint 保留 |

两臂曾共享物理 GPU1，且启动时均为 fresh optimizer、`auto_resume=false`、`resume_from=null`、`promotion.auto_stop=false`。2026-09-01 01:27 现场确认 W&B `crashed` 状态不能代表进程已退出：两个 Python PID 和 CUDA context 当时仍存在。按用户指令分别只发送一次 Ctrl-C 后，终端回到 shell，随后 Python PID、CUDA context 与 tmux socket 均消失；未强杀、未删除 checkpoint。证据：[历史启动状态](../artifacts/stage1_forehand_clear_aug100_remote7_t0t4_direct800_c1ccd93_20260824/production_status.json) · [T0 log](../artifacts/stage1_forehand_clear_aug100_remote7_t0t4_direct800_c1ccd93_20260824/logs/t0_s0_v2.log) · [T4 log](../artifacts/stage1_forehand_clear_aug100_remote7_t0t4_direct800_c1ccd93_20260824/logs/t4_s0_v2.log)。

11:33 的首次 v1 启动因隔离 venv 缺少锁定 CUDA JAX extra，在 W&B、轨迹加载、optimizer 和 checkpoint 之前自行失败，状态为 `INVALID`，不计作实验。其根目录和完整日志已归档；有效 v2 重启使用全新的 run/W&B/training-root/Hydra/cache/log/tmux 身份，并增加实际 GPU matmul 硬门禁。

### 6.2 上一 v2 family 的中断进度（不计入 v3）

| Arm | 旧 Run / W&B | 最后 update / steps | 进度 | 最后 finalized checkpoint | 状态 |
|---|---|---:|---:|---|---|
| T0 | `...direct800_v2_t0_s0` · [b8t0s0c1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/b8t0s0c1) | 26,230 / 537,190,400 | 67.15% | `260817T110221-pid549397-f2172b/checkpoint_26230` | `INTERRUPTED` |
| T1 | `...direct800_v2_t1_s0` · [b8t1s0c1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/b8t1s0c1) | 27,450 / 562,176,000 | 70.27% | `260817T110424-pid552736-aab5d6/checkpoint_27450` | `INTERRUPTED` |
| T2 | `...direct800_v2_t2_s0` · [b8t2s0c1](https://wandb.ai/f70331658-university-of-chinese-academy-of-sciences/musclemimic/runs/b8t2s0c1) | 26,840 / 549,683,200 | 68.71% | `260817T110424-pid552817-63d14e/checkpoint_26840` | `INTERRUPTED` |

旧 v2 三臂合计为 `1,649,049,600 / 2,400,030,720` steps，仅用于历史审计；v3 没有从这些 checkpoint 恢复。

### 6.3 上一 v2 family 的最新 held-out validation 中途快照

| Arm | 快照 steps | coverage ↑ | early term ↓ | RPos (m) ↓ | action saturation ↓ | activation energy ↓ | anchor loss ↓ | real synergy loss ↓ | shape cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 537,190,400 | 75.10% | 40.0% | 0.0963 | 53.54% | 0.4143 | N/A（tube-free） | N/A（tube-free） | N/A |
| T1 | 562,176,000 | 82.76% | 40.0% | 0.0948 | 53.66% | 0.4083 | 4.9195 | 0.9816（诊断值，weight=0） | 0.5934 |
| T2 | 549,683,200 | 89.52% | 40.0% | 0.0930 | 54.01% | 0.3684 | 1.1594（诊断值，weight=0） | 0.2209 | 0.8031 |

这些数值不能用于正式 matched claim，原因是：

- 三臂并非同一 global timestep；
- 均不是 fixed endpoint；
- T0 需要 endpoint post-hoc physiology 才有可比 measured-EMG 指标；
- T3/T4 缺失，无法执行预注册主对比；
- 单 seed 不能替代原计划的三独立 seed 统计单位。

### 6.4 上一 v2 快照对绝对 promotion gate 的距离

| Gate | 阈值 | T0 | T1 | T2 | 当前判断 |
|---|---:|---:|---:|---:|---|
| Early termination | ≤ 5% | 40.0% ✗ | 40.0% ✗ | 40.0% ✗ | 三臂均未通过 |
| Frame coverage | ≥ 95% | 75.10% ✗ | 82.76% ✗ | 89.52% ✗ | 三臂均未通过 |
| Relative site position error | ≤ 0.09 m | 0.0963 ✗ | 0.0948 ✗ | 0.0930 ✗ | 三臂均未通过 |
| Action saturation | ≤ 5% | 53.54% ✗ | 53.66% ✗ | 54.01% ✗ | 三臂均未通过 |
| Activation energy | ≤ 0.35 | 0.4143 ✗ | 0.4083 ✗ | 0.3684 ✗ | 三臂均未通过 |

`promotion.auto_stop=false`，因此中途 gate 失败本身不等于正式实验失败；但它表明当前快照远未达到 promotion 条件。正式判定还要求 fixed endpoint、最近 3 次验证、pairwise evidence 和 blind review。

## 7. Stage 1 评估、gate 与 promotion 清单

下列项目不进入 15 个训练叶分母，但没有它们就不能宣称 Stage 1 完成。

| ID | Action | 任务 | 状态 | 必需输入 | 完成证据 |
|---|---|---|---|---|---|
| FC-S1-PHYS-S0 | Clear | T0 seed-0 endpoint post-hoc physiology | `BLOCKED` | T0 exact endpoint + verified tube | sealed validation evidence |
| FC-S1-INDEX | Clear | T0–T4 seed-0 evidence index | `BLOCKED` | 5 个 exact endpoint evidence leaves | self-fingerprinted evidence index |
| FC-S1-BLIND | Clear | T3 seed-0 opaque human review | `BLOCKED` | T3 endpoint reviewer package | completed `review.json` |
| FC-S1-GATE | Clear | pairwise gate | `BLOCKED` | evidence index + blind review + mapping | passed/failed pairwise gate artifact |
| FC-S1-PROMOTE | Clear | T3 seed-0 teacher promotion | `BLOCKED` | passed pairwise gate | immutable teacher promotion artifact |
| CJ-S1-EVAL | ChinaJump | post-hoc/index/blind/gate/promotion | `ASSET/CONTRACT-BLOCKED` | 5 个 CJ endpoint + action-specific evidence | action-specific Stage 1 promotion |
| FL-S1-EVAL | Lift | post-hoc/index/blind/gate/promotion | `ASSET/CONTRACT-BLOCKED` | 5 个 FL endpoint + action-specific evidence | action-specific Stage 1 promotion |

主要成功条件：

- T3 对 T4 的 real-reference synergy loss 至少改善 5%；
- T3 anchor loss 严格优于 T0 post-hoc；
- measured-activation 五项指标不退化；
- tracking、safety、effort 通过绝对 gate 和相对 guardrail；
- T3 seed-0 opaque blind review 通过。

## 8. 不计入当前 family 的历史或排除证据

### 8.1 历史 40-train/10-val T2/T3/T4 family

下列分析是可复用的历史诊断结果，但数据 split、训练预算和当前 v3 T0–T4 family 不同，所以状态为 `HISTORICAL-NONFORMAL`。

| 历史编号 | 日期 | Family / 终点 | Arms | Seed | 关键结论 | Gate | 报告 / 数据 |
|---|---|---|---|---:|---|---|---|
| E01 | 2026-08-15 | `c1ccd93` Aug100 40train/10val @320M | T2/T3/T4 | 0 | T3 vs T4 real-reference synergy loss 相对改善 +57.6%；无 arm 过 gate | 全失败 | [报告](stage1/stage1_t2t3t4_320m_report.md) · [快照](stage1/stage1_t2t3t4_320m_snapshot.json) |
| E02 | 2026-08-17 | 同 family @640M | T2/T3/T4 | 0 | T3 vs T4 改善 +61.6%；T2 在后段反超 T3 synergy；跟踪指标退化；无 arm 过 gate | 全失败 | [报告](stage1/stage1_t2t3t4_640m_report.md) · [快照](stage1/stage1_t2t3t4_640m_snapshot.json) |

| 历史实验 | Arm | Steps | synergy loss ↓ | shape cosine ↑ | RPos (m) ↓ | coverage ↑ | early term ↓ | action sat ↓ | energy ↓ | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| E01 | T2 | 320M | 0.2843 | 0.8027 | 0.1225 | 72.5% | 60% | 50.6% | 0.4100 | ✗ |
| E01 | T3 | 320M | 0.2734 | 0.8027 | 0.1157 | 78.5% | 60% | 50.8% | 0.3937 | ✗ |
| E01 | T4 | 320M | 0.6442 | 0.5230 | 0.1194 | 66.5% | 60% | 51.3% | 0.4080 | ✗ |
| E02 | T2 | 640M | 0.1997 | 0.8374 | 0.1253 | 68.8% | 65.0% | 55.1% | 0.4118 | ✗ |
| E02 | T3 | 640M | 0.2437 | 0.8308 | 0.1280 | 74.2% | 60.0% | 56.9% | 0.3980 | ✗ |
| E02 | T4 | 640M | 0.6341 | 0.5194 | 0.1308 | 75.6% | 66.7% | 56.2% | 0.4177 | ✗ |

### 8.2 明确排除的运行

| 运行/证据 | 状态 | 排除原因 |
|---|---|---|
| Aug100 direct800 v2 的 `b8*` T0/T1/T2 | `HISTORICAL-INTERRUPTED` | 未到 endpoint 且进程已消失；v3 使用新 run/W&B/checkpoint root 与 fresh optimizer，禁止 resume/merge |
| Aug100 direct800 非 v2 的 `a8*` T0/T1/T2 attempts | `INVALID` | 共享 Hydra output directory；已停止并保留 provenance，禁止 resume/merge/report |
| 服务器7 `remote7_t0t4_direct800_v1` CPU-JAX attempt | `INVALID` | 隔离 venv 缺少 CUDA JAX extra；在 W&B/data/optimizer/checkpoint 前失败；完整证据已归档，禁止计数或 resume |
| `forehand_clear_aug100_40train10val_peasd_c1ccd93` T2/T3/T4 | `HISTORICAL-NONFORMAL` | 40/10 split 与 320M/640M 终点不同，不能与当前 80/20 800M family 混合 |
| 旧 `250aa`、`e8de859`、`e721e4c` T0/T1 日志 | `HISTORICAL-NONFORMAL` 或失败尝试 | SHA、数据、run identity 或完成状态不匹配当前 family |
| ChinaJump bootstrap/continuity diagnostics | `HISTORICAL-NONFORMAL` | 不是 PEASD T0–T4 matched arm，不能用于 promotion |
| dry-run、单元测试、preflight、启动验收 | 辅助证据 | 证明运行合同或入口可用，但不是训练结果 |

## 9. 下一步顺序与停止条件

### 9.1 当前必须先解决

1. T2/T3/T4 seed2 已完成并完成终态核验。9 月 18 日准备启动最后三组 T0 seed1、T0 seed2、T1 seed2（GPU1/2/3），以本次新运行包验收状态为准。
2. 最后三组 T0 seed1、T0 seed2、T1 seed2 已启动；持续核验其endpoint，不重复派发，也不把远端duplicate计入唯一叶。
3. seed0 五个训练叶已完成；仍须整理 T0 post-hoc physiology、evidence index、T3 opaque review、pairwise gate 和 promotion。没有相应 gate artifact 就不标通过。
4. 远端 T0/T4 为已停止 duplicate，远端 T1 seed1 v4 为运行中的 duplicate；各自保留证据，不恢复不兼容 optimizer，不重复计叶。
5. 训练异常按规范仅向对应 pane 发一次 Ctrl-C；保留日志和 finalized checkpoint。Lift/ChinaJump 继续等待各自 action-specific 合同与资产复核。


### 9.2 硬停止条件

- 发现 source fingerprint、split、tube、mapping、reward、termination、budget 或 seed 漂移：停止并建立新 family；
- 任何 Hugging Face/remote trajectory 下载尝试：停止；
- NaN、OOM、fatal traceback 或未 finalized checkpoint：停止并登记 incident；
- T3 不优于 T4：停止 Stage 2，先审计 phase、mapping、tube 与 reward delivery；
- pairwise gate 或 blind review 未通过：不得 promotion。

## 10. 每次更新本表时填写

### 10.1 训练叶登记模板

| 字段 | 必填内容 |
|---|---|
| ID | `ACTION-S1-T<arm>-S<seed>` |
| 日期与状态 | start/end time；`RUNNING/DONE/INTERRUPTED/FAILED` |
| 身份 | Git SHA、source fingerprint、config hash、run id |
| 数据 | action、namespace、train/validation 数、split hash、QC/tube binding |
| 训练合同 | seed、target steps/updates、fresh optimizer、resume fields、promotion settings |
| 运行位置 | 物理 GPU、PID、tmux socket/session、append-only log |
| 在线记录 | W&B id、URL、final state |
| Endpoint | exact update/global timestep、checkpoint path、content hash |
| Validation | evidence path/hash、last-3 gate、主要指标 |
| 异常 | NaN/OOM/traceback/download/video failure/中断原因 |
| 结论 | 只写 evidence 支持的结果；中途快照必须标 provisional |

### 10.2 运行结束核对清单

- [ ] exact immutable checkpoint leaf 已 finalized；
- [ ] global timestep 与 resolved fixed endpoint 完全一致；
- [ ] run manifest、config hash、source fingerprint 已登记；
- [ ] W&B id/URL/final state 已登记；
- [ ] append-only log 路径和结束原因已登记；
- [ ] validation evidence path/content hash 已登记；
- [ ] release/QC/tube/split binding 可重建；
- [ ] NaN、OOM、fatal traceback、HF download、video failure 已审计；
- [ ] gate 结果来自真实 endpoint，而非 best checkpoint 或中途快照；
- [ ] 若状态为 `DONE`，本表第 1 节的计数和百分比已同步更新。

## 11. Stage 2 Forehand Clear 调度记录

### 11.1 2026-09-01 服务器7 GPU1 dispatch

| ID | 任务 | 目标服务器 / GPU | 状态 | 已完成 | 当前硬门禁 | 证据 |
|---|---|---|---|---|---|---|
| FC-S2-DISPATCH-R7-G1-V1 | 双服务器配置与源代码对齐 | `172.18.22.7` / 物理 GPU1 | `CONFIG-READY` | 两端共同 Git SHA `c1ccd932...`；4 个关键 source/config SHA-256 一致；focused tests `52 passed`；9 个配置包文件逐字节一致 | 无 | [配置包](stage1/racket_curriculum/forehand_clear_remote7_gpu1_20260901/README.md) · [门禁状态](stage1/racket_curriculum/forehand_clear_remote7_gpu1_20260901/preflight_status.json) |
| FC-S2-STAGE1R-003-R7-G1-V1 | Stage1R 0.03 首个 GPU step | `172.18.22.7` / 物理 GPU1 | `BLOCKED` | 正式入口、cache key、append-only log、tmux socket 和 GPU identity 已配置 | T3 production promotion 缺失；远端 T3 checkpoint 缺失；blind review 缺失 | [共同合同](stage1/racket_curriculum/forehand_clear_remote7_gpu1_20260901/experiment.env) · [远端覆盖](stage1/racket_curriculum/forehand_clear_remote7_gpu1_20260901/server_remote7.env) |
| FC-S2-SHARED | Stage1R → event → mass 025/050/075/100 → physical collection/QC → basis/decoder seal | `172.18.22.7` / 物理 GPU1 | `BLOCKED` | 预算锁定：train/val 各 `1,000,000` transitions | Stage1 promotion、Clear event manifests/banks、Clear frozen decoder/fingerprints；后续每个 mass rung 还需真人 visual review | 同上 |
| FC-S2-A | Direct BC → 3×DAgger → fresh PPO → held-out compare/seal | `172.18.22.7` / 物理 GPU1 | `BLOCKED` | seeds `0/1/2`；BC `200,000` steps；DAgger 每轮 `500,000` transitions；PPO `102,400,000` timesteps；batch `4096` | 必须先生成 immutable `stage2_shared_inputs.json` | 同上 |
| FC-S2-B/C/D/E | latent baseline / real / shuffled / no-dropout context family | 待 S2-A promotion 后调度 | `BLOCKED` | treatment 与 matched-family 顺序已按正式计划登记 | 必须先通过 S2-A 三 seed family promotion 与 S2-B architecture lock | [正式计划](../docs/plans/PEASD正式实验计划.md) |

双服务器配置包位置：

- server9：`/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/Experiments/stage2/forehand_clear_remote7_gpu1_20260901`；
- server7：`/data3/yangfeiyang/WorkSpace/musclemimic/Experiments/stage2/forehand_clear_remote7_gpu1_20260901`；
- 两端 9 文件逐文件 SHA-256 完全一致；排序后的 hash 清单 fingerprint 为
  `7a9ee7eddb1bb2078de7b774f5b963da54f4bec481a1af57b8ead5412289e514`；
- target GPU1 为 A100 80 GB，preflight 时 `76,664 MiB` free；另有非本项目 PID `1093233`
  占用约 `4,346 MiB`，按边界不终止它。

### 11.2 Stage2 未启动的决定性证据

1. 本地 T3 seed-0 candidate 的 exact checkpoint `checkpoint_39063` 存在，但
   `stage1_blind_review_package/review.json` 的 reviewer、总判定与所有 clip 判定仍为 `null`；
2. server9/server7 均没有 `stage1_peasd_teacher_promotion.json`，代码的 production distill
   validator 会拒绝未 promotion teacher；
3. Forehand Clear 正式 Stage1 需要 T0--T4 × seeds `0/1/2` 和 paired gate，目前未形成 15 个
   completed evidence leaves；
4. server7 尚未同步 exact T3 checkpoint；两端尚无 Clear action-specific event manifests/banks
   与 frozen body decoder/fingerprints；
5. `--test-only-allow-unpromoted-teacher`、旧 `/raid` waiver、legacy checkpoint 和旧 optimizer
   在共同合同中显式禁用。因此本次结果是 **preflight fail closed，Stage2 Python/CUDA PID、W&B
   run、optimizer 与 checkpoint 均未创建**。

Stage2 解除阻塞的最短合法顺序是：完成 Stage1 15 叶 → 真人 blind review → evidence index 与
pairwise gate → T3 promotion → 同步 T3/promotion 与 action-specific 资产 → 两端 full preflight
和同入口 dry-run → 服务器7 GPU1 启动 `stage1r_train`。任何一步失败都不得降低 gate 标准。

### 11.3 正式提前启动的 Stage2 流程实验

2026-09-01 02:55 后，用户明确调整调度：其余 Stage1 实验继续独立运行，同时使用当前已完成的
T3 seed-0 endpoint 先完成一次真实 Stage2 全链路。该调整不改写第 11.1/11.2 节所记录的历史
preflight 结论，而是建立独立 run UID 和输出根的新实验。

| ID | Server / GPU | 实验身份 | Teacher | 真实预算 | 当前状态 | 证据 |
|---|---|---|---|---|---|---|
| FC-S2-FORMAL-EARLY-S0-R7-G1-V1 | `172.18.22.7` / 物理 GPU1 | 修复前流程尝试；seed 0 | 同一 T3 `checkpoint_39063` | train 128 + val 64 已成功采集；optimizer 未启动 | `INVALID-BEFORE-OPTIMIZER`；先发现 train-only 无法形成 motion holdout，补 val 后又暴露 BC loader 未合并 nested PPO keys；完整数据/日志保留但不作为 v2 训练结果 | [配置与说明](stage1/racket_curriculum/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/README.md) |
| FC-S2-FORMAL-EARLY-S0-R7-G1-V2 | `172.18.22.7` / 物理 GPU1 | 同上；fresh v2 | 同一 T3 endpoint | BC 10 steps 已完成 | `INVALID_DAGGER_SCHEMA_ABI`；BC train/val MSE `0.100438/0.307862`；DAgger shard 因 observation-filter 默认值未纳入 canonical hash 而 fail-closed，committed DAgger samples=0 | [状态](stage1/racket_curriculum/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/status.json) |
| FC-S2-FORMAL-EARLY-S0-R7-G1-V3 | `172.18.22.7` / 物理 GPU1 | 同上；fresh v3 | 同一 T3 endpoint | BC 10 steps 已完成 | `INVALID_TEACHER_TARGET_SEMANTICS_ABI`；BC train/val MSE `0.100580/0.284868`；22 个 ABI 键仅 teacher target semantics 不同，DAgger committed samples=0 | [状态](stage1/racket_curriculum/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/status.json) |
| FC-S2-FORMAL-EARLY-S0-R7-G1-V4 | `172.18.22.7` / 物理 GPU1 | `FORMAL_EARLY_START_PENDING_UPSTREAM_ACCEPTANCE`；seed 0；fresh v4 | T3 `checkpoint_39063` / `800,010,240`；13 文件 fingerprint `c093b894...` | teacher train 128 + held-out val 64；BC 10；DAgger 1×(64+10)；4 held-out motions × 100 | `COMPLETED_SMOKE_SCALE_ACCEPTANCE_FALSE`；256 条样本提交、BC/DAgger checkpoint 与统一球拍闭环评估均完成；短预算 student 未达到 acceptance | [结果说明](stage1/racket_curriculum/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/README.md) · [机器状态](stage1/racket_curriculum/forehand_clear_formal_early_start_remote7_gpu1_20260901_v1/status.json) |

v4 固定运行身份：

- run UID：`fc-s2-formal-early-start-r7g1-60bff94-s0-v4`；
- 训练 code SHA：`60bff946a826c0fdbc9b1adcea4f44ad7900ba84`，source fingerprint
  `0cca47eb379849abb4a7d8a66f6844d50db9b10f313219728b91956bd1ba8adf`；
- 只读评估 code SHA：`9f6976d93ff976b7f4bab9f62c04fed106235979`，source fingerprint
  `45d45b49cb3dbd851b3fe29391bb78e5f62ca84a6c7c0489b7303cbb60761f56`；该修复只把 teacher/BC/DAgger
  放入相同刚性球拍评估环境，不改训练 checkpoint；Stage2 focused regression `132 passed`；
- teacher parent manifest SHA-256：`17864350fb45e606aa20a01d34c48df11797b1a967cb8726b268e0caa0b7325e`；
- 远端 80/80 train、20/20 validation motion 均为本地文件，无需下载；
- 正式 v4 preflight 时 GPU1 为 A100 80GB，约 `76,678 MiB` free；评估重试前曾因另一用户进程只余
  `19,786 MiB` 而按 20GB 门禁等待，未终止该进程；释放后以 `81,033 MiB` free 通过同一门禁；
- 训练入口依次为 teacher collection → canonical BC → canonical DAgger retrain → canonical
  held-out compare；日志与 `experiment_events.jsonl` append-only。

首次实际启动在 `03:05:18` 进入 teacher collection，`03:10:17` 完成 128 条 train transitions
与两个 shard。`03:10:57` BC 在 optimizer 创建前发现 train 数据仅包含一个唯一 `motion_uid`，
按 motion-level 防泄漏合同退出；CUDA context 随进程正常消失，无 NaN/OOM。修正为在同一
immutable manifest 中追加一条显式、稳定 ID 且与 train 不相交的 validation motion（64
transitions），不关闭 strict motion identity，不做 sample-level 随机切分。

显式 val collection 于 `03:20:26` 完成。随后 BC 再次在 optimizer 前 fail-fast：Hydra config
已经 compose，但 `gamma` 仍只位于 `experiment.ppo_config`，环境 wrapper 读取顶层兼容键。
修复 commit `fa8cfa1a...` 复用 PPO factory 的 canonical merge，新增回归测试并建立全新的 v2
run/output/cache/tmux 身份；v1 采样资产和失败日志只用于审计，不混入 v2 结果。

v2 已创建 fresh optimizer 并完成 BC 10 steps，得到 train/validation MSE `0.100438/0.307862`；
随后 DAgger 在提交首个 shard 前发现 observation-filter 默认值未进入 canonical ABI hash，按严格
schema 合同退出，committed DAgger samples 为 0。v3 修复该缺口并再次使用 fresh run/output，BC
10 steps 得到 train/validation MSE `0.100580/0.284868`；DAgger 又在 commit 前准确发现 22 个 ABI
键中 teacher target semantics 唯一不一致，因此 v3 同样在无 DAgger 样本污染的情况下终止。两次
失败均保留为实现审计，不作为 v4 checkpoint 或指标来源。

v4 在训练 code `60bff946...` 上统一 teacher target semantics 后，从 fresh optimizer 开始完成全部
短预算。数据集中 teacher train `128`、teacher held-out validation `64`、DAgger train `64`，合计
`256` 条；BC step 10 的 train/validation MSE 为 `0.100429/0.298325`，确定性复算 validation MSE
为 `0.293729`。DAgger 完成 1 次 `64` 条采集与 10 个更新步，step 10 的 train/validation MSE 为
`0.121572/0.189141`，确定性复算为 `0.188480`，相对 BC 的确定性 held-out MSE 降低约 `35.8%`。

首次 compare 暴露的是评估环境不一致：teacher 在 bare-hand `MimicReward` 下没有球拍误差字段；
训练和 checkpoint 本身没有失败。只读评估修复 `9f6976d...` 增加显式
`--racket_tracking_eval`，使三个策略都在相同 `MjxMyoFullBodyRacket + RacketMimicReward` 环境中
评估，并通过 `132` 项 focused regression。统一闭环评估使用 4 条 held-out motion、每条最多 100
steps，结果如下：

| 策略 | Mean return | Completion ratio | Early terminations | Root-pos error | Racket-pos error | Racket-rot error | Acceptance |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Teacher T3 | 148.4865 | 0.9945 | 0 | 0.09782 | 0.24705 | 0.51941 | 参考项 |
| BC step 10 | 57.0667 | 0.5675 | 1 | 0.14483 | 0.49079 | 1.13366 | false |
| DAgger step 10 | 52.9195 | 0.5234 | 1 | 0.14560 | 0.51565 | 1.13093 | false |

因此 v4 状态为 `COMPLETED_SMOKE_SCALE_ACCEPTANCE_FALSE`：正式流程、资产绑定、训练、checkpoint
和比较评估都已完成，短预算 student 的闭环性能结论则是尚未达到 acceptance。成功日志中没有
fatal traceback、NaN、OOM、Hugging Face 或远程下载尝试；tmux、Python PID 与 GPU1 CUDA context
均已自然退出。首次 compare 的 traceback 继续保留在旧日志中作为事故审计，成功结果以
`compare_racket_eval_v2.log` 和对应 metrics/acceptance JSON 为准。

本实验是 smoke-scale 的真实训练与正式流程记录，用于先验证数据、checkpoint、学生训练和评估
是否全链路可用。它不与后续 full-budget（1M collection / 200k BC / 3×500k DAgger）结果混写；
等其余 Stage1 实验结束后，按 exact checkpoint hash 做上游身份核对并在本节补记结果。

## 12. 参考合同

- [`AGENTS.md`](../AGENTS.md)
- [`PEASD正式实验计划.md`](../docs/plans/PEASD正式实验计划.md)
- [`实验运行跟踪表.md`](../docs/archive/PEASD实验运行跟踪表_20260811.md)
- [`新服务器部署与训练执行手册.md`](../docs/runbooks/server9/新服务器部署与训练执行手册.md)
- [`peasd_implementation_guide.md`](../docs/runbooks/peasd_implementation_guide.md)
- [`新服务器Aug100增广数据说明.md`](../docs/runbooks/server9/新服务器Aug100增广数据说明.md)
- [当前 Aug100 full5 v3 启动包](../artifacts/stage1_forehand_clear_aug100_full5_direct800_c1ccd93_20260823/README.md)
