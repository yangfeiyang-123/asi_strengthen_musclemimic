# Python 源码

此目录使用标准 Python src 布局；`src` 本身不是 Python 包。

| 包 | 内容 |
| --- | --- |
| `musclemimic` | 算法、环境、奖励、蒸馏、评估和 `grip` 手指握拍工具 |
| `loco_mujoco` | 轨迹、重定向与环境基础组件 |
| `environment` | 球场、球拍、羽毛球及单/双人场景；公共资产随对应模块保存 |
| `fullbody` | 训练/评估入口与 Hydra 配置 |
| `analysis` | 离线分析工具 |

主要包的 import 名称不变；原 `src.grip` 统一为 `musclemimic.grip`。
使用模块入口（例如 `python -m fullbody.eval`）或根目录的正式 launcher。
训练配置导航：[fullbody/config_specific_task/README.md](fullbody/config_specific_task/README.md)。
文档、实验定义、私有数据与输出位于仓库根目录的对应目录，不写入源码包。
