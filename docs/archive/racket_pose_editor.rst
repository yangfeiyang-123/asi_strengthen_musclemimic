全局握拍状态编辑器
==================

这个工具直接加载训练使用的 ``MyoFullBodyRacket``。你只需要调整一次：

* 球拍相对右手 ``thirdmc_r`` 的固定朝向；
* 球拍相对右手 ``thirdmc_r`` 的固定 XYZ 坐标；
* 右手 20 个握拍关节的目标角度。

保存后会得到一个带 SHA-256 指纹的全局握拍 preset。训练环境在每条训练、验证轨迹
reset 时都使用同一组手指目标，``RacketMimicReward`` 也使用同一组目标。轨迹只用于
预览，工具不会改写轨迹 NPZ，也不再保存逐帧球拍 sidecar。

当前默认加载已经验收并晋升的
``configs/racket_grip/forehand_clear_grip_v2_custom.json``，其球拍 attachment 为
``configs/racket_attachment/forehand_clear_rigid_v4_custom.json``。

启动
----

::

   scripts/run_racket_pose_editor.sh

浏览器打开 ``http://127.0.0.1:8088``。若通过 SSH 使用服务器，在本机做端口转发：

::

   ssh -L 8088:127.0.0.1:8088 <server>

可以启动时加载一条服务器轨迹，也可以在网页中填写 NPZ 路径或从浏览器上传：

::

   scripts/run_racket_pose_editor.sh \
     --trajectory datasets/<action>/muscle_trajectory/<motion>.npz \
     --frame 0

fingerless 训练轨迹会按关节名映射到带完整右手手指的预览模型。切换轨迹或帧不会改变
正在编辑的握拍状态；当前手指目标始终应用到预览轨迹的所有帧。

调整与保存
----------

* 拖动右手处的旋转环调整球拍。球拍自身 ``+Y`` 是拍柄方向，``+Z`` 是拍面法向。
* “拍柄 Y±”用于调整拍面绕拍柄的扭转。
* 在“球拍坐标 XYZ（手局部，m）”中直接输入坐标，或用 X/Y/Z 按钮按毫米精调。
  坐标控制只修改 attachment 的 ``position_m``，不会修改当前欧拉角或四元数。
* 展开拇指、食指、中指、无名指和小指面板，逐关节调整握拍角度。
* 点击“保存并应用于所有轨迹”。默认输出两个文件：

  * ``configs/racket_attachment/forehand_clear_rigid_v5_custom.json``；
  * ``configs/racket_grip/forehand_clear_grip_v3_custom.json``。

第二个文件使用 ``musclemimic.racket_grip_preset.v1`` schema，并绑定第一个 attachment
contract 的 fingerprint。原 attachment contract、原 grip preset 和所有轨迹都不会被覆盖。

默认已从最新 v2 preset 开始编辑；也可显式写出完整命令：

::

   scripts/run_racket_pose_editor.sh \
     --preset configs/racket_grip/forehand_clear_grip_v2_custom.json \
     --output configs/racket_attachment/forehand_clear_rigid_v5_custom.json \
     --preset-output configs/racket_grip/forehand_clear_grip_v3_custom.json

使用上面的命令会保留当前默认朝向和坐标。保存到 v5/v3 新文件可保留 v4/v2 作为可回退
基线。

接入所有训练轨迹
----------------

默认环境已经使用最新 v2 preset，不需要额外覆盖。若希望显式固定训练配置，可在目标 Hydra
配置的 ``experiment.env_params`` 中加入：

.. code-block:: yaml

   racket_grip_preset: configs/racket_grip/forehand_clear_grip_v2_custom.json

环境会从 preset 自动加载并校验它绑定的 attachment contract。因此不需要为每条轨迹配置，
也不需要同时填写 ``racket_attachment_contract``。如果两项都显式填写，fingerprint 必须一致，
否则环境会拒绝启动。

若训练配置使用 ``disable_fingers: true``，球拍固定朝向仍生效，但模型没有手指自由度，
所以手指角度会被忽略；要训练真实握拍手指，应使用 ``disable_fingers: false``、
``RacketGripInitialStateHandler`` 和带 finger-grip 项的 ``RacketMimicReward``。

新的球拍朝向和手指目标会改变训练 contract。请使用新的 ``run_id`` 和新的优化器，不能恢复
旧握拍状态下的不兼容 checkpoint。生产训练仍须遵守仓库根目录 ``AGENTS.md``，从
``scripts/run_fullbody_training.sh`` 启动。
