# STG 自动化测试与 AI 训练方案

完整接口、算法和验收指标见同目录的 `stg_ai_testing.md`。本文记录面向 SR 的实施顺序和设计约束。

## 核心原则

工具分成两个后端，不能混淆测试结论：

1. LuaSTG 桥接后端运行原始 Lua、对象池、碰撞、激光和自机逻辑，是游戏内容测试的最终依据。
2. Python 独立后端只近似模拟关键机制，用于高速开发规划器、视觉模型和外部记忆。

独立仿真中通关不代表原符卡已经通关。最终结论必须明确标注为“独立仿真”或“引擎验证”。

## 桥接协议与真实引擎回归

桥接器使用一问一答的 JSONL protocol v2。`catalog` 从与 Spell Practice 菜单相同的 Boss 卡片表实时生成目录；顶层 `attacks` 中每项都包含 `scenario`、从 1 开始的 `attack`、原始 `card_index` 和 `label`。当前 SR 正式目录应为 22 个场景、53 个攻击，测试端必须遍历实时目录，不能另存一份容易失配的卡片序号表。桥接协议和真实引擎报告 schema 为 v2，内嵌 catalog 仍保持独立的 schema v1。

`ping` 除协议版本、帧号和能力列表外，还返回：

- `session_id`：由每轮启动时的 `SR_TEST_SESSION_ID` 指定；
- `process_nonce`：每个桥接实例自行生成，用于识别是否误用了同一引擎进程。
- `runtime_identity.process_id`：通过 Win32 FFI 读取的真实操作系统 PID；
- `runtime_identity.executable_path/executable_crc32`：当前引擎可执行文件路径与 CRC32；
- `runtime_identity.source_crc32`：真实 LuaSTG 进程对入口、兼容层、自机、背景和测试桥接器共 18 个实际加载的 SR Lua 文件计算的 CRC32。

Python 端会对相同的 18 个实际加载 SR Lua 文件独立计算 CRC32 与 SHA-256，并为全部 `stg_lab` Python 模块生成稳定的 `implementation_sha256` 源码指纹。正式验收的两轮报告要求 session、nonce 和正数 PID 都不同，二进制 CRC、运行时 Lua CRC、Python 本地源码 SHA-256 和当前实现指纹一致。启用 `SR_TEST_MODE` 时，`SR_TEST_STARTUP_ACCEPT_TIMEOUT` 默认是 30 秒（非测试模式为 0）；桥接器可在安装阶段等待首个客户端，使 lockstep/headless 在第一次 Render/Present 前生效。`SR_TEST_SOURCE_ROOT` 指定运行时 Lua CRC 的文件根目录。
安装 wheel 后，Python 端可用 `STG_LAB_MOD_ROOT` 指向同一模组目录；未设置时会从当前目录逐级向上发现。

`reset.options.player_protect_frames` 会在关卡初始化、自机对象创建后，把 `protect` 至少提高到指定帧数，不替换原引擎碰撞逻辑。53 项内容回归同时使用 99 残机和 `player_protect_frames=frames_per_attack+600`，避免测试仅因被弹提前中止。

观测中的 `enemy_bullets`、`enemies`、`nontjt_enemies`、`indestructibles`、`lasers` 即使为空也必须编码成 JSON 数组 `[]`，对应 `counts` 必须等于数组长度。目录数组、场景内攻击数组和曲线激光采样点同样遵守此契约，不能把空数组编码成 `{}`。激光会同时保留在所属碰撞组数组中，因此 `lasers` 是有意重复的索引视图。

可见性过滤不参与符卡结束判断。`attack_complete` 直接检查未经可见性过滤的 `GROUP_ENEMY` 与 `GROUP_NONTJT` 原始对象池，避免 Boss 暂时隐藏或移出画面时误判结束。

### 原生 replay 录制

符卡 `reset` 或最终关 `reset_stage` 可以带便携文件名 `replay_name`。桥接器会创建私有 `plus.ReplayFrameWriter`，在每个逻辑帧的 THlib `GetInput()` 完成后记录最终 `KeyState`，但不会占用全局 `replayWriter`，因此后续转场不会误访问 THlib 私有的关卡录像表。客户端使用下列命令结束录制：

```json
{"id":5,"command":"save_replay","finish":false,"reason":"attack_complete"}
```

桥接器将标准 STGR v1 文件写到 `userdata/replay/<setting.mod>/analysis/<replay_name>.rep`，随后通过 `plus.ReplayManager.ReadReplayInfo` 读回并核对文件头、序列化初态、关卡、seed、玩家和帧数；它还会实际读取全部声明的输入字节，并要求最后一帧恰好结束于 EOF。响应包含文件大小、已验证帧字节数、结束原因和 CRC32；瞬时写入、读回或校验失败会保留内存 writer 以便重试。符卡练习固定写 `group_finish=0`；只有严格成功的最终关可以写 `group_finish=1`。普通非最终关以及 ghost、关闭碰撞、额外保护帧等原生录像无法复现的测试选项会被拒绝。

这里的 `verified=true` 表示 STGR 结构、元数据和帧区完整，不表示已经在原生 replay 菜单中完成了一次语义回放。需要保留旧录像时应使用不同名称；THlib 会覆盖同名文件。

`engine-mpc-play --replay-name NAME` 会自动完成上述流程，并把结果放进同轮 JSON 的 `native_replay`。死亡或达到帧上限时也会保存，但 `finish=false`。`.rep` 只包含序列化初态和输入字节，不包含 MPC 的观测、预测或决策理由；分析时必须保留同轮 JSON。需要完整可见对象快照时再加 `--record-observations-from-frame 0`。

2026-08-03 的最终原生 arm64 macOS headless 验证使用 Okuu Boss #3、seed `20260730`、五帧观察延迟、60 帧 horizon 和 `bullet-group-expert`。控制器在 3363 个受控帧后由引擎返回 `attack_complete`，Boss HP 为 0、自机 `death=0`、射击指令率为 1.0。3891 字节录像含中性 reset 帧共 3364 帧，`group_finish=0`、CRC32 为 `5f964c53`、SHA-256 为 `4d6be63144e7706de04112c2204e2960dffb78fda57cc19a4a0423a2b8aa85d2`。独立 STGR 解析器从 JSON 的动作保持记录重建了全部 3364 个预期输入字节，错位为 0，且 3363 个受控帧全部开枪。配套 JSON 保存 1121 次决策及完整观测，SHA-256 为 `20822e937020f0137e3d5e064e9e663f5d9c4134426f2ac698c56db3aa268170`。两者都是被忽略的本地分析产物，不作为源码 fixture 提交。

## 人类化观测

部署模型只能接收可由画面获得的信息：

- 全屏视野，用于整体弹幕密度、旋转方向和安全区域变化；
- 玩家附近的高分辨率视野，用于近身避弹；
- 最近若干帧，用于估计速度、加速度和转向；
- 观察延迟、按键保持和少量感知误差。

模型不能读取 Lua 类名、脚本计时器、随机数状态或未来弹幕位置。教师规划器可以读取精确状态来生成训练标签，但这些字段不得进入学生模型输入。

visible-v2 实现中，神经网络推理只能接收 `DelayedVision` 产生的延迟全局/局部数组。冷启动历史为全零空白帧，不能把 reset 帧反向复制到尚未看见的过去；水平/垂直运动通道只能由相邻可见位置的位移估计，不能读取仿真器或引擎 `vx/vy`。占用通道使用 `log1p` 密度，重叠运动使用与对象顺序无关的加权平均。

`experiments/rebuild_visible_datasets.py` 会把按 episode 分组的教师动作重新通过这条观测路径回放，生成 `canonical_train_visible_v2.npz`、`canonical_heldout_visible_v2.npz` 和 `visible_dataset_v2_manifest.json`。标准配置为四帧历史、五帧观察延迟、六个通道、`48x56` 全局视野和半径为 72 世界单位的 `40x40` 局部视野。

## 安全区与路径

每个未来时刻都计算一张风险图。风险不仅取决于弹幕数量，还要包含：

- 子弹和激光的未来扫掠区域；
- 速度乘以观察与反应延迟得到的安全膨胀量；
- 轨迹估计误差；
- 当前安全区是否即将缩小或断开；
- 到达下一个安全区所需的时间。

规划对象是玩家可达区域，不是单颗子弹轨迹。实现会先把每个未来采样层离散成分级风险栅格，再从一层的可达 cell 向下一层做动态规划，并对移动扫过的栅格段做检查。每个 cell 只保留按“最高危险等级、累计风险、移动距离”排序的最佳代价。连通域仅用于区域标注和诊断，并没有显式构造或搜索 `(component,time)` 图。因此在安全区即将断开时，规划器仍可选择峰值代价更低的短暂次危险通道。

Python Stage 5 Boss #3 场景只近似原符卡后段的移动、周期扩张危险体和旋转扇形弹，用于测试安全区迁移和跨区时机；它不是原符卡完整脚本或逐帧等价模型。Stage 5 Boss #4 使用旋转发射源测试提前绕行，不能依靠贴脸后的局部反应完成。

近似参数直接对照仓库 Lua：自机中心边界为左右 8、下 16、上 32，Reimu 判定半径为 0.5；#3 Lunatic 使用全部 10 个核弹源、高危半径 28，且不添加原作不存在的 warning 通道；#4 等待 120 帧生成星体，每个方向扇发射 9 发。forecast 会保留采样间出生又消失威胁的首帧、最大尺寸帧和末帧 swept 几何。

## Engine MPC 的相位与拓扑记忆

当前 Engine MPC 的跨局策略记忆只描述从在线可见信息学到的安全区动力学，不保存一局游戏的走法。Boss #3 使用可见核弹半径的中位数来抵抗对象重排、新生大核弹和平台微小摆动，并按固定顺序识别四相：

`expanding -> maximum_hold -> contracting -> minimum_hold`

区域动力学记忆 v2 的可加载内容严格限于场景/攻击身份、半径上下限、扩张/收缩率、四相顺序与相对持续时间、半径周期，以及以下横向流规则：

```json
"lateral_flow": {
  "cycle_frames": 360.0,
  "safe_side_rule": "opposite_incoming_lateral_flow"
}
```

`opposite_incoming_lateral_flow` 表示根据下一次扩张时可见核弹排的来向选择反侧开放区，并不保存“左、右、左、右”之类的侧别序列。半径分支用观测帧号计算可见转折之间的时间差；横向流分支从相邻记录画面中最高的可碰撞核弹排位置差估速，匹配标识只用于相邻画面对应且不会写入记忆。两个分支都不读取原始 `dx/vx`、Lua 类名、脚本 timer、隐藏 phase 或 RNG 状态。

控制器用当前可见相位推算下一次扩张，并把相邻可见画面估计出的核弹排运动投影到扩张开始、中点和结束三个事件切片；它比较左右外侧在三个切片中仍然开放的数量与净空，提前选取扩张后仍连通的一侧。区域目标只表达要到达的安全连通域，局部 beam 搜索可为眼前子弹另走非直线绕行路径，二者不会互相冒充。v1 记忆仍可加载以复现旧实验，但因为没有横向流契约，不启用这项相位拓扑预测。

单局工作状态跨帧对应当前安全连通域、目标连通域和动态选出的 `portal`。同一外侧连通空间即使跨过不同核弹排也保持同一语义身份；portal 的实际位置每次都由当前可见几何重算。component、portal 和短期动作承诺在每次 `reset()` 时全部清空，不写入跨局 artifact。相位、导航模式、当前域、目标域或 portal 改变时，旧的短期动作承诺立即失效。以下内容绝不属于策略记忆：

- 绝对 episode 帧触发点；
- 固定世界坐标或 waypoint；
- 预录制侧别序列、固定左右交替规则；
- 固定动作片段或整局动作序列；
- 只对某一次录制有效的过关路线。

当前 `engine-mpc-play` 没有录制动作 loader 或播放分支。可选的原生 replay 只作为输出，绝不会向控制器提供动作或其他输入。原生诊断报告仍保存逐帧动作、位置和绝对帧以便定位故障，但当前代码不会把报告或 `.rep` 重新解释为策略输入；正式运行只能从 reset 开始由 live MPC 连续闭环控制。

完全省略 `--region-dynamics-memory` 时，报告中的 memory 路径、SHA-256 和控制器 memory 均为 `null`，也不会读取路线、动作前缀、checkpoint 或 SQLite。首批墙出现后，最近四个连续稳定的可见半径样本只建立“当前形状稳定”这一事实，不会擅自命名最大/最小平台；控制器用当前墙列位置和相邻画面估速，在 `0 / horizon/2 / horizon` 三个时刻投影左右开放数量，并选择预测窗内实际开放且最宽的同侧目标。第一次扩张/收缩后，半径包络、变化率、相位持续时间和 180 帧周期都在本局内在线学出。

短期拓扑工作状态会在几帧几何无法区分左右时继续保持已经由可见信息选出的外侧连通域，直到新观测明确给出反侧或自机到达目标；每次 reset 都会清空，绝不形成跨局左右序列或绝对帧触发。当直达路线受阻且最迟换区时刻进入 60 帧视野时，真实碰撞与普通子弹危险仍排在最前，区域迁移进度可以暂时高于通常的 8 单位强制区域余量，从而允许必要的低等级危险区穿越，但不会接受预测碰撞。

最终实现指纹为 `0c0b25f53ab677500a830d40e0f38377151e75e2b1ed7fcd63ffa621d9c0f268`。三次独立原生 arm64 macOS headless 进程都使用五帧观察延迟、60 帧 horizon、持续开枪、弹组空隙预测，并且命令中完全没有 memory 参数：

| seed | 控制帧 / 决策 | 严格结果 | replay CRC32 / 帧数 | JSON SHA-256 |
| ---: | ---: | --- | --- | --- |
| 20260730 | 3327 / 1109 | `attack_complete`、Boss `6000 -> 0`、death 0 | `05ab4af6` / 3328 | `a6344535eb1eee7364ac8b4a87dc85b0a02b9d09785a00ee817dc0122ac603ba` |
| 20260731 | 3334 / 1112 | `attack_complete`、Boss `6000 -> 0`、death 0 | `ce651750` / 3335 | `2334dd12cd4c9dc51b99be86bdccc7e122f2442473a7763b5372cffd566b0539` |
| 20260732 | 3331 / 1111 | `attack_complete`、Boss `6000 -> 0`、death 0 | `a12fea5c` / 3332 | `d6d2422c03ebe4f8807e8bce8129e5214e6da800d3a469f8426affe100c3851d` |

三局全部决策都是 `live_mpc`，全部动作保持开枪并禁用 spell，均在本局在线学出 180 帧周期；三份原生 replay 都是 `saved=true`、`verified=true`、`group_finish=0`。这是已执行 seed 的 `3/3` 严格 live-MPC-teacher 证据，不外推到其他 seed、Boss #4 或神经 checkpoint。

连续全关测试使用独立的 `engine-mpc-campaign` 命令。它只发送一次
`reset_campaign`，让原生流程自行从同一难度的 Stage 1 连续推进到 Stage 5，
并在 `observation.campaign.stage_transition_count` 增加时清空场景局部轨迹、
延迟画面、区域/空隙状态和 committed plan；控制器实例和游戏资源保持连续。
campaign 元数据只负责生命周期边界，传入 MPC 前会被删除。命令不提供区域
记忆、路线、checkpoint、动作前缀、reset option 或 replay 参数：

```bash
cd tools/stg_lab
uv run stg-lab engine-mpc-campaign \
  --host 127.0.0.1 --port 24816 \
  --difficulty Lunatic --seed 20260804 \
  --profile bullet-group-expert \
  --max-frames 120000 --horizon-frames 60 --observation-delay 5 \
  --no-render \
  --output artifacts/no-memory-lunatic-campaign-seed20260804.json
```

只有顺序完成五关、每关均观察到真实活动敌人、最终原生转场返回
`campaign_complete`、有限数值 `death=0`、恰好四次控制器边界清理、全部决策
来源为 `live_mpc`、持续开枪且 `spell=false`、所有外置来源字段为 `null` 时才
通过。达到帧上限、中弹、跳关、活动内容缺失或证据结构异常都会失败。THlib
需要为每关分别记录初态，当前协议明确不支持把连续 campaign 保存成单个 replay。

连续测试会在原生重置前和运行结束后各计算一次 Python 源码指纹；运行过程中若
源码发生变化，即使游戏流程完成也会令严格结果失败。报告会额外保留终止关卡
最后 24 帧原始权威观测和最后 8 次延迟 MPC 输入。这个有界窗口在每次原生转场
时清空，只用于事后诊断，不会回送控制器，也不会成为跨关记忆。

首轮 seed `20260804` 原生 Lunatic 诊断 campaign 在连续总帧 12550 完成
Stage 1，随后于总帧 20173（Stage 2 timer 7624）中弹。该轮只有一次重置、
同一个 MPC 实例、持续开枪、禁用 spell，且全部外置记忆字段为 `null`。最后
两次决策预测碰撞 ETA 为 29 和 22，但实际弹丸在 6 个实时帧内命中，因而将
“延迟画面中仍静止、当前帧已经启动”的弹丸预测定位为下一类失败。
`no-memory-lunatic-campaign-gap-commit-v1-seed20260804.json` 只作为诊断证据：
该轮运行中 Python 源码发生过变化，且当时尚未实现起止双指纹拒绝规则，不能
作为严格验收产物。

### 2026-08-04 无外置记忆全关工作暂停点

本节记录暂停继续优化时的实际状态，不替代上述验收定义。本轮
“无外置记忆”继续严格指：仅一次 `reset_campaign`、同一个实时 MPC、
五帧观察延迟、三帧动作保持，且不读取 region memory、route、checkpoint、
action prefix、replay 或 SQLite。每个动作必须 `shoot=true`、`spell=false`；
只有五关按顺序完成、最终 `campaign_complete=true` 且有限数值 `death=0`
才记为通过。

当前已落地的通用实现包括：

- `engine-mpc-campaign` 保留原生 Stage 1 -> 5 资源和流程，仅在真实关卡
  边界清空轨迹、延迟队列、region/gap 状态和短期 plan；关卡元数据不进入
  避弹输入。
- 普通场景和 region 场景的 beam 都保留首动作与空间格多样性；当前
  profile 为 `beam_width=512`、`beam_cell_size=8`。恒加速度外推默认关闭，
  仅保留为显式消融开关，因为局部转向不能证明 60 帧内保持恒加速度。
- 延迟启动弹模板只由本局可见的“静止 -> 运动”转换学习；当新弹在
  学到的启动期限后仍静止时，预测会立即过期，不再把它永久当作即将启动的弹。
- 平行弹组空隙、region 安全连通域、边界净空和 committed gap plan 均在
  每次新可见几何上重验；关卡名、脚本 timer、固定帧号和预录坐标不参与决策。
- live runner 已改为让轨迹估计器消费每一个成熟的延迟 source frame，
  但 beam 仍每三帧只决策一次。重复的延迟填充帧会去重，同 source frame
  的 `observe()`/`select()` 复用威胁缓存，campaign 转关时重置 feed。定向
  MPC/play/campaign 集合共 150 项 Python 测试通过。

截至暂停时的原生引擎证据如下。“存活延长”仍然记为失败：

| 目标 / 配置 | 结束帧 | 严格终态 | 结论 |
| --- | ---: | --- | --- |
| Okuu Lunatic #4，关闭恒加速度 | 4295 | `attack_complete`、death 0 | 通过 |
| Koishi Normal #1，512/8 空间 beam | 977 | `player_hit`、death 100 | 失败 |
| Koishi Normal #1，边角余量 96 / 权重 0.5 | 1061 | `player_hit`、death 100 | 失败 |
| Koishi Normal #1，逃生 plan 重验消融 | 1136 | `player_hit`、death 100 | 失败 |
| Stage 1 Normal，早期无记忆基线 | 12838 | `stage_complete`、death 0 | 通过，历史实现 |
| Stage 1 Lunatic，edge-gap v2 | 12974 | `stage_complete`、death 0 | 通过，历史实现 |
| 当前 Lunatic campaign v4 | 11073 | Stage 1 `player_hit`、death 100 | 失败，0 次转关 |

Okuu #4 报告为
`no-memory-okuu4-no-accel-current-seed20260804.json`（SHA-256
`e8f932add4bae7d8fad26f4477dc97433511e82bfc18c4f76ef8dff4e94fdba6`）。
Koishi 三份报告的 SHA-256 依次为
`fced5708fbe9ba575eb556f79eafc6401f2671de0e22006e16f607b4e0a1c74e`、
`f872d103e12c9e741a8b3986328f759eb885cb2ac0769454ee042ee7c4ac4bef`、
`0c550790fff653547097173d4424c9e5ce343ebf821256d124e865643319cfff`。
campaign v4 报告 SHA-256 为
`4c8f59135aa03b1e793d8c1da0dbfc9a72a39d7e61230d46a3c112d4c2cfe34b`。

还没有新实现下的全关通过证据。较早的 Normal/Lunatic campaign 曾在完成
Stage 1 后进入 Stage 2，分别于总帧 21970/20173 中弹；这只证明当时
实现的 Stage 1 路径，不能与当前源码混合为“已通过前两关”。

Koishi #1 的当前失败不是报告与引擎结果不一致。原生观测显示，多个静止
可碰撞 `nontjt_enemies` 排成从中心伸向边界的六条链，把画面分成扇形连通域；
内外两组心形弹同时沿径向非匀速运动。在 source frame 995 已找到零预测碰撞
的长计划，但只执行首个三帧动作后，frame 998 的滚动重规划改向；旧计划剩余
部分在新观测上仍为零碰撞，而新计划已预测必撞。通用逃生提交消融将存活
从 1061 延长到 1136 帧，但最终仍把自机逼入右下角，因此没有合入为
“已解决”策略。90 帧 horizon 诊断于 1061 帧失败，加强中心/纵向 anchor
也只延长到 1068 帧。

Stage 1 的另一个明确缺口是观察延迟窗口内才出生的弹。例如一次致命弹在
source frame 10511 尚不存在，到权威 frame 10518 才出生；只用当局截止
10511 的可见历史，可在线恢复约 2 帧发射周期、60 单位相对半径和
`-21.5 deg/frame` 相位速度，对 10518 出生点的误差约 0.37 像素。
工作树中已有一个匿名出生族预测原型：它尝试只从新可见轨迹、邻近可见 anchor、
相对偏移、周期和相位生成 `spawn_forecast_inferred` 威胁，不应读取 class、timer、
image、rot 或原始速度。但该原型在本次暂停前尚未完成 ID 打乱/复用、漏发撤销、
邻近多发射体和原生 Stage 1 回归，不得将其写作已验证功能。

暂停时没有保留运行中的训练、参数消融或 LuaSTG 服务器进程。下次继续时应先为
匿名出生原型补完纯可见数据契约和单元测试，再独立回归 Stage 1、延迟启动弹、
Koishi #1 与 Okuu #4；只有这些都不回归时才重启 Normal/Lunatic 连续全关。

同一实现随后在原生 macOS OpenGL 窗口中用 seed `20260730` 可见重跑。控制命令仍完全省略 memory，覆盖层报告 `enabled=true`、`data_source=controller`、revision 3320，并使用控制器的 60 帧 horizon 和 16/20/8 阈值。结果为 3327 控制帧、1109 次 `live_mpc` 决策、`attack_complete`、Boss `6000 -> 0`、death 0；与同 seed 的 headless 决策轨迹逐条一致。报告 `tools/stg_lab/artifacts/engine-mpc-boss3-no-memory-visible-demo-20260804-seed20260730.json` 的 SHA-256 为 `963df911ffa73f9509f97e48463ec517166ff8a8c3ffc1d2704db96c6e845bff`；已验证 replay 的 SHA-256 为 `9aa6f4afa27306660fe84f0dfafef77eace07ef1b09bb0b2c843c833bc6b76bb`，CRC32 为 `c3faa620`。可见逐帧覆盖层运行的全程 FPS 中位数为 27.73，`OBJ >= 300` 密集段为 27.35，最低 14.89；这不影响 lockstep 正确性，但不能声称本轮可见演示达到 60 FPS。

`train-region-dynamics` 从原生报告中读取 `source_frame`、可见区域半径，以及已记录控制器输入中的可碰撞不可摧毁物 `id/x/y`；`id` 只用于相邻画面匹配，产物不保存对象身份。训练器拒绝带录制动作、非 `live_mpc` 决策、authority shield 或未强制 `spell=false` 的来源，并将训练 provenance 与可加载记忆分开保存：

```bash
cd tools/stg_lab
uv run stg-lab train-region-dynamics \
  --input artifacts/engine-mpc-boss3-heldout-v37-d5-memory-no-actions.json \
  --memory-output artifacts/region-dynamics-boss3-v2.json \
  --report-output artifacts/region-dynamics-boss3-v2-training.json
```

正式 v2 记忆 SHA-256 为 `dcdcddeeed840d733e144477934f217b21f6795ab9a83790d7f3813d273546f7`，训练报告 SHA-256 为 `95f4b3e45952f476f430158416d6f13b7617a4a5d25dbe07e522fc0107ac8e99`。输入 v37 报告 SHA-256 为 `60c1dd6bb0cfdece73d7170b3c968736479470ba87fed6e96fa68e42290134f5`，提供 357 个半径样本和 303 个横向流样本。拟合结果为半径 `7/28`、变化率 `0.7/0.7`、四相 `30/30/30/90`、半径周期 `180` 和横向流周期 `360`；360 帧重复的 201 对样本相关系数为 1、归一化 RMSE 为 0，180 帧反号的 252 对样本同样为 1 和 0。整体平移输入时间轴不会改变 memory 或相对样本统计。

### 平行弹组空隙预测

live Engine MPC 现在可以把安全区预测与通用弹幕空隙预测结合使用。只有
`enemy_bullets` 能参与空隙分组；激光、敌人、不可摧毁物和强制位移体仍会
进入常规碰撞与入口路径检查，但不能定义空隙。实现先按速度方向和速度大小
聚类近似平行的运动子弹，再沿运动方向按纵深拆成独立波前。对每个波前在
垂直运动方向的轴上排序，相邻两弹之间才构成候选走廊。整个过程只使用当前
可见几何与运动估计，不读取攻击专用帧号、脚本相位、固定坐标或录制路线。

走廊宽度表示自机中心真正可用的净宽，而不是两弹中心距。计算会扣除两侧
子弹半径，并在两边分别扣除自机半径、10 单位安全余量和观察延迟造成的位移
误差。候选走廊必须在到达与保持阶段的多个未来采样时刻持续开放；边界弹
次序交换、未来闭合或存续时间不足都会使其失效。波前覆盖率阈值会排除无法
代表整体弹幕屏障的孤立弹对。实现先为全部孔洞计算几何，但只对活动孔洞及
最多 8 个与 region 相容且距离较近的候选执行入口认证。认证先检查可执行的
三帧动作块直达路线，若被其他威胁阻挡，再用小型多样化 action beam 搜索
绕行。完整路线对全部威胁的最低余量必须达到 4 单位；region anchor 活动时
提高到 8 单位强制区域余量。这允许必要时短暂穿越较低等级危险区；选中空隙
内部仍保留每侧 10 单位加观察延迟误差的安全带。因此某一平行弹组的孔洞不
能掩盖横穿入口的另一组子弹、激光、敌人或强制位移体。

空隙 anchor 与 region anchor 同时存在：region anchor 指向全局安全连通域或
portal，空隙 anchor 为迎面波前提供局部走廊，候选选择会优先考虑与区域目标
相容的走廊。碰撞预测仍严格早于 gap 入口；在 `enter` 状态，控制器会输出
已认证路线的第一个动作，因此可以有意接受较低的普通弹净空，但不会接受碰
撞；两个 anchor 同时活动时仍执行更高的 region 入口余量。进入 `hold` 后，
整个可用区间都是软目标，不会追逐精确中心。当原地停留仍满足普通安全余量
时，检测到的空隙只进入 `observe`。活动 gap 跨重规划保持身份并延续
`enter -> hold -> exit`，而不是每个决策切换到最新孔洞。

`engine-mpc-play` 默认启用这项逻辑；`--gap-prediction` 可显式开启，
`--no-gap-prediction` 可用于确定性消融。每个 JSON 决策记录
`gap_bullet_group_count`、`gap_corridor_count`、`gap_selected_center`、
`gap_selected_width`、`gap_selected_lifetime_frames` 和
`gap_navigation_mode`；最后一项取 `inactive`、`observe`、`enter`、`hold`
或 `exit`。报告顶层的 `gap_prediction` 汇总还记录 `enabled`、检测/选中
决策数、各模式决策数、`maximum_bullet_group_count` 与
`maximum_corridor_count`。

现在提供三档确定性的 MPC 弹组处理能力；三档保持普通移动评分和持续射击完全
一致，只改变波前识别、可接受空隙、未来采样与入口搜索能力：

| 档位 | 弹组感知 | 可接受空隙 | 入口搜索能力 |
| --- | --- | --- | --- |
| `bullet-group-novice` | 至少 5 发；方向容差 5 度；速度容差 6%/0.15 | 每侧预留 18、净宽至少 10、持续至少 24 帧、覆盖屏幕 65% | 每 12 帧采样、认证 2 个入口、绕行 beam 12 |
| `bullet-group-intermediate` | 至少 4 发；方向容差 8 度；速度容差 12%/0.25 | 每侧预留 14、净宽至少 6、持续至少 18 帧、覆盖屏幕 55% | 每 9 帧采样、认证 4 个入口、绕行 beam 24 |
| `bullet-group-expert` | 至少 3 发；方向容差 12 度；速度容差 20%/0.35 | 当前的每侧预留 10、净宽至少 4、持续至少 12 帧、覆盖屏幕 45% | 每 6 帧采样、认证 8 个入口、绕行 beam 48 |

这些档位限制处理能力，而不是随机制造错误。合成回归中，3/4/5 发波前分别仅
由专家、至少中级、全部档位识别；方向总跨度 20/12/0 度时也呈现相同分级；
标准五弹探针的间距为 32/40/50 时，分别只有专家、至少中级、全部档位接受
其空隙。`engine-mpc-play --profile` 可选择单档，`engine-mpc-matrix` 重复提供
`--profile` 可在相同符卡和 seed 上对比三档。它们描述 MPC 教师能力，不表示
当前发布的神经 checkpoint 已经学会对应的分级处理。

随后在当前源码上，以同一个 Okuu #3 seed `20260730` 对三档进行了原生对照。
三局的 runner 配置、区域动态记忆、运行时身份和已校验运行时源码映射完全相同，
控制器只在上述 15 个弹组能力字段上不同。三份报告均绑定实现 SHA-256
`394d8298f5b42c6a42d586a75ab908bac0fdb7281d6cd5fd54478a83e048d99b`。

| 档位 | 严格结果 | 弹组处理 | 移动 | 报告 SHA-256 |
| --- | --- | --- | --- | --- |
| novice | `attack_complete`；3,360 帧；HP `6000 -> 0`；death 0；射击 `3360/3360` | 最多形成 6 个弹组，无可接受走廊，所有 gap 模式均为 0 | 路径 9500.6747；变向 308；反向 26；急转 122；ABA 6；保持中位数 9 | `8fb9d002e7b0c8a6e993edd602995cf1d5e5fbcfdd4a457fb73a8bfc37c77850` |
| intermediate | `attack_complete`；3,360 帧；HP `6000 -> 0`；death 0；射击 `3360/3360` | 42 次检测并 observe，0 次选中；最大弹组/走廊 `30/3` | 本局与 novice 轨迹完全相同 | `dd73c9fdb36e1e0a34c4dbf4c39ffd1e30749f667953d46eeae6594d6a217c57` |
| expert | `attack_complete`；3,363 帧；HP `6000 -> 0`；death 0；射击 `3363/3363` | 检测 172、选中 44；observe/enter/hold/exit `119/44/0/9`；最大 `54/13` | 路径 9621.0025；变向 316；反向 26；急转 112；ABA 7；保持中位数 9 | `3370da8b2debfdb5b77e05fb7f023d3c3181e0e3894a88e0e5dde6ef8eaf0047` |

中级档已识别走廊，但这些时刻普通 MPC 本来就安全，因此没有为了展示差异而
强行移动。专家入口规划在与低档共同的 1,120 次决策中改变了 802 个输出动作
和 749 个方向选择。三档在这一个 seed 上恰好都严格击破，证明的是分级处理已
生效且没有回归，并不能据此标定真人成功率。

确定性的四弹波前回归还验证了 gap anchor 不只是遥测：自机位于自然形成的
10 单位走廊外侧时，开启预测会把无碰撞首动作从直下改为朝空隙的向左，关闭
预测则保持直下；另一组探针在直线路径上加入静止障碍，并验证控制器能找到
由多个三帧动作块组成的安全绕行。这些都是合成集成探针，不是原生符卡成功
率证据。

本地合成性能探针包含 299 发子弹、23 个波前和 276 条走廊：几何阶段中位
2.92 ms，开启/关闭空隙预测的完整决策中位分别为 135.24/131.66 ms，差值
3.58 ms。几何完成后只会认证活动走廊及最多 8 个排序后的候选，不再对全部
276 条走廊逐一执行 Python 路径搜索。这些数字不是原生闭环 A/B、实时吞吐
或符卡成功率结果。

此前同源码、连续射击的原生 A/B 使用 Okuu #3 seed `20260730`。两份报告均
绑定实现 SHA-256
`002d770a1e4d10ad98a2ce00f21796dd7deeddc931b08ab638a3e07e0bbefb86`，
顶层运行配置和运行时源码映射完全相同，控制器配置的唯一差异是
`gap_prediction_enabled`。

| 空隙预测 | 严格引擎终态 | 空隙遥测 | 移动诊断 | 报告 SHA-256 |
| --- | --- | --- | --- | --- |
| 开启 | `attack_complete`、`passed=true`；3,363 帧 / 1,121 次决策；Boss HP `6000 -> 0`；death 0；射击命令 `3363/3363`（1.0）；预测碰撞移动规划 522 帧 | 检测 172 次、选中 44 次；observe/enter/hold/exit 为 `119/44/0/9`；最大弹组/走廊数 `54/13` | 路径 9621.0025；变向 316 次（每千帧 93.9637）；精确反向 26 次；大于 90 度急转 112 次；ABA 7 次；保持 min/median/mean/max 为 `3/9/10.6088/291` 帧 | `7dc328637957f0682974d97e0227475bea10f4eb79994334bea9599b76b18ea1` |
| 关闭 | `attack_complete`、`passed=true`；3,360 帧 / 1,120 次决策；Boss HP `6000 -> 0`；death 0；射击命令 `3360/3360`（1.0）；预测碰撞移动规划 528 帧 | 已禁用，所有 gap 计数均为 0 | 路径 9500.6747；变向 308 次（每千帧 91.6667）；精确反向 26 次；大于 90 度急转 122 次；ABA 6 次；保持 min/median/mean/max 为 `3/9/10.8738/291` 帧 | `93f27f603bb852021ccab8f62e87285072140a1788cab510a02d69ed51c15e3a` |

在共同的 1,120 次决策中，逐项比较有 802 个输出动作不同，其中 749 个移动
选择不同、947 个规划动作数组不同、952 个决策边界自机位置不同。开启组还多
一个终止决策，动作与位置第一次分歧均出现在第 168 次决策。因此空隙预测确实
被启用并改变了原生移动，同时两边都严格通过。开启组的平滑度数字只描述本局，
不是普遍结论。由于没有记录到 `hold`，本次原生证据不证明持续驻留；确定性
回归测试覆盖了 `enter -> hold`。schema 3 已弃用 unsafe-shot 指标，所以
`unsafe_shot_frames=null` 不表示计数为零。由于这里只运行一个 seed，且两边
本来都能严格通过，这组结果证明的是功能启用与无回归，不是成功率提升。这两份
报告是 `acceptance_claim=false` 的 live MPC 教师证据，不是学习模型结果。

该预测器是通用视觉教师规则，不是写入已发布神经网络 checkpoint 的符卡专用
记忆。本次 A/B 报告尚未转换为 gap-aware DAgger 或 demonstration archive，
当前发布的 stream-v1 checkpoint 也尚未从此类档案重新训练。

### MPC 预筛选的精确等价与性能

`experiments/benchmark_engine_mpc_beam.py` 在原生报告 `engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json`（SHA-256 `e7577aa475ed9a9de6542fedfba8a193dca1b3d8a927e139371e22f41b2d94ef`）的记录观测上，对比保守候选 AABB 威胁预筛选和不筛选参考 beam。已记录的三次运行中位数如下：

| 来源帧 | 模式 | 威胁数 | 不筛选中位耗时 | 预筛选中位耗时 | 加速 | beam 输出 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 995 | 局部规划 | 364 | 0.3653 秒 | 0.1292 秒 | 2.83x | 精确相等 |
| 1292 | 区域规划 | 193 | 0.3809 秒 | 0.1209 秒 | 3.15x | 精确相等 |

这里的等价不是只比较最终动作或终态。所有候选的动作、`collided`、`collision_frames`、`earliest_collision_frame`、精确浮点 `minimum_margin`、边界惩罚、Boss 对齐惩罚，以及完整 20 段动作计划都逐字段相等。来源帧 1292 另以 60 帧预热重放了 21 次有状态决策，优化与参考控制器、提交动作、计划动作和报告碰撞字段全部一致。同机五次重复运行的实测范围约为 2.7x-3.4x，因此这些耗时是会波动的基准样本，不是保证下限。

### MPC 净空、变向迟滞与网格消融

当前 live MPC 仍每 3 帧重规划。普通弹使用 20 单位硬净空，强制位移区域使用独立的 8 单位硬净空；二者分别记分，狭窄强制通道不会掩盖普通子弹净空不足。16 单位危险优先层排在 20 单位目标之前，普通弹净空在 48 单位以内继续获得有上限的软奖励，角落逃生也有 48 单位软余量；这两个 48 都是偏好，不是不可穿越边界。束搜索对换向、180 度反向、`A -> B -> A` 和同方向速度档切换分别累计 `3/9/6/0.75` 代价。

方向通常至少保持 12 帧。近距离碰撞、保持期后段获得至少 8 单位净空增益、立即离开危险角落，或限时 `evacuate` 且新方向在边界/Boss 对齐代价上取得至少 3 单位真实路线进展时才提前释放。`evacuate` 不再仅因模式名绕过迟滞，`preposition` 也不会绕过；旧 committed plan 必须通过相同安全门，不能在安全释放后恢复过时方向。碰撞、首次碰撞、碰撞帧数、16/20 单位净空层级始终先于平滑偏好。

区域目标 `4/6/8/12` 在 32 个 `hold/preposition/evacuate/settle` 记录样本上的碰撞帧均为 15，8 个 crossing 样本均可达且无碰撞；中位净空约为 `4.35/6.33/8.29/12.18`。目标 12 会降低 crossing 推进并把最终锚点 L1 误差从目标 8 的约 0.87 增至 1.25，因此采用 8，而不是在狭窄入口强求 12。

较早的 v46 同 seed 原生闭环用于量化平滑改动：相邻决策变向为 `835 -> 487`，完全反向 `65 -> 20`，`A -> B -> A` 为 `154 -> 20`，实际连续非零位移夹角大于 90 度为 `236 -> 154`；预测最小净空中位数为 `1.43 -> 8.27`，净空不高于 4 的决策为 `815 -> 215`，预测碰撞帧为 `2346 -> 1009`。这些预测字段含 60 帧滚动未来，并不表示实际碰撞；两局当时均为 `death=0`。该证据早于当前 12 帧迟滞与撤离释放修复，属于历史对照，不能替代当前源冻结后的原生复验。

`experiments/benchmark_engine_mpc_grid.py` 还在来源帧 `488/1292/2102/2801/3695` 的同一批真实五帧延迟观测上，对 8/12/16 单元的时间分层网格和连续 beam 做开环对照。网格层仍每 3 帧一个，但每层接收该时间段内每个逻辑帧的 swept threat 占用；同一观测内所有方案截断为共同最短动作时长，再用 Engine MPC 的逐逻辑帧连续圆形几何复算，不能用网格自身等级给自己作证。这比较的是完整规划器，不是只隔离栅格化。公平对照如下：

| 方案 | 预测碰撞计划 | 碰撞帧率 | 最小净空中位数 | 每 60 帧计划内变向 | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 连续 beam 20/8 | 3/5 | 9.67% | -4.47 | 6.20 | 0.120 秒 |
| 8 单元中心网格 | 2/5 | 9.00% | 1.90 | 7.80 | 1.326 秒 |
| 12 单元中心网格 | 3/5 | 10.00% | -3.04 | 9.20 | 0.698 秒 |
| 16 单元中心网格 | 4/5 | 45.33% | -26.29 | 10.60 | 0.460 秒 |

半格对角线膨胀的整格保守版本也未反超：8/12/16 单元的碰撞帧率分别为 11.67%/15.67%/52.33%。8 单元中心网格只比连续 beam 少 2 个碰撞帧（`27/300` 对 `29/300`），不足以证明 episode 级闭环优势；它的平均规划耗时约为连续 beam 的 11 倍，且变向更多。结论是保留网格用于可视化、连通域和全局区域提示，但最终局部动作继续使用逐帧连续几何、迟滞和 committed-plan 安全门；纯网格不替换当前控制器。工作报告 SHA-256 为 `ffc69082eefd2501d473981689eddd4fcfb67a0152a5723b9f099a46c7bbd901`。这是固定五观测上的开环完整规划器消融，不是 `attack_complete` 成功率。

旧 standalone v2 的 SQLite 单路线/多路线库和对应哈希仍可用于复现历史基准，但它们包含录制路线，不再代表当前 Engine MPC 的策略记忆，也不能证明原生引擎泛化。

## 跨平台完整引擎、高速测试目标与渲染

当前 LuaSTG-Sub 源码树已经不再保留“完整引擎依赖 DirectX”的旧前提。完整可执行文件可在 Windows、macOS 和 Linux 使用 SDL2 窗口/输入/音频与 OpenGL 4.1 GPU 精灵/FBO 渲染器，也可组合 null 图形/窗口/音频执行不限帧 headless 测试；两条路径都运行真实 Lua 脚本、资源系统、碰撞逻辑和测试桥。OpenGL 路径会批量提交旧顶点/索引，在 GPU 缓存源纹理，以 FBO 承载 RenderTarget，并直接显示主目标，不再逐帧上传 CPU 画布；受支持构建均不编译或链接 DirectX。

当前边界也必须保留：任意旧 HLSL 效果尚未翻译为 GLSL，只能退化为可见的 RenderTarget 直通合成；model 绘制和现代 MeshRenderer 尚未完全等价。上文 CrossOver/DXVK 结果验证的是原 Windows 可执行文件，属于独立证据，不是当前跨平台源码的运行依赖。

完整 macOS 引擎的 preset 会依次配置、构建、执行无 DirectX 审计并运行对应测试：

```bash
LUA_STG_SUB_ROOT=/path/to/LuaSTG-Sub
cd "$LUA_STG_SUB_ROOT"
cmake --workflow --preset macos-headless-release
cmake --workflow --preset macos-opengl-release
```

仓库仍保留较小且隔离的 `LuaSTGPortableTest` target。它复用原引擎 `XCollision` 的圆形/旋转椭圆判定，支持完全不初始化视频设备的不限帧 headless 模式，以及 SDL2 软件简化碰撞/风险区视图。这个 target 适合算法测试，但不能当作完整 macOS 游戏运行时。

```bash
LUA_STG_SUB_ROOT=/path/to/LuaSTG-Sub
cd "$LUA_STG_SUB_ROOT"
cmake --preset portable-native
cmake --build --preset portable-native-release
ctest --preset portable-native
build/portable-native/portable/LuaSTGPortableTest \
  --scenario pulse --frames 100000
build/portable-native/portable/LuaSTGPortableTest \
  --scenario orbit --frames 600 --analyze-risk
build/portable-native/portable/LuaSTGPortableTest \
  --scenario pulse --frames 180 --render --render-every 6 \
  --screenshot build/portable-native/portable/pulse-macos.bmp
SDL_VIDEODRIVER=dummy \
build/portable-native/portable/LuaSTGPortableTest \
  --scenario orbit --frames 700 --render-every 6 \
  --screenshot build/portable-native/portable/orbit-dummy.bmp
```

`pulse` 覆盖安全区扩张、保持、收缩和最小保持，`orbit` 覆盖旋转强制位移体与旋转扇形弹；简化画面绘制与碰撞体一致的椭圆轮廓、自机判定和四级风险栅格。本机 arm64 macOS 已完成全新配置、Release 构建并由 CTest `1/1` 通过。已记录的 10 万帧 pulse 无渲染实测约为 1,511,690 逻辑帧/秒；包含 710,400 个风险采样的 600 帧 orbit 实测约为 496,620 逻辑帧/秒。macOS SDL2 软件截图 `build/portable-native/portable/pulse-macos.bmp` 和 SDL dummy 软件截图 `build/portable-native/portable/orbit-dummy.bmp` 都是非空 `480x560` BMP。该 target 是算法开发/碰撞可视化部分模块，不执行任意 SR Lua，不能替代完整引擎的 `attack_complete` 验收。

引擎根目录的 `game/plugins/SafetyZoneVisualizer` 提供原生游戏内 F7 分级覆盖层；也可设置 `SR_SAFETY_ZONE_OVERLAY=1` 在首次游戏渲染前开启。覆盖层保留 16 单位显示网格，但每个色块只表示格心采样，不再把半格对角线加入自机半径。运行 `engine-mpc-play` 时，每次决策会直接发布 EngineMPC 实际使用的 `PredictedThreat`、当前 profile 阈值、修正后的自机边界与判定半径、区域导航状态，以及逐未来帧的区域半径包络；桥接器在三帧动作保持的首帧发布一次。覆盖层直接检查未来第 1 帧到配置 horizon 的静止格心风险，不再自行猜测速度或 Boss #3 周期相位。运行状态会报告 `data_source=controller` 及已消费的 revision。

四色使用 MPC shortfall 的严格边界：任一预测净空 `<= 0` 为红色；普通威胁满足 `0 < margin < 16` 时为橙色，`16 <= margin < 20` 时为黄色，`margin >= 20` 时为绿色。活动强制区域的独立目标为 8，`0 < region_margin < 8` 映射到黄色，`>= 8` 满足该层目标。等于 `16/20/8` 时 shortfall 已为 0。分级只取未来所有威胁的最差净空，子弹数量和局部密度不再升级颜色；4 单位 emergency margin 只解除移动方向保持，不是额外颜色等级。绿色内部仍有到 48 单位为止的连续净空奖励，四色无法完整表示这项软偏好。

发布的数据已经采用 MPC 的 `max(a,b)` 保守圆及直线/曲线激光 16--32 单位分段圆覆盖。敌弹和不可摧毁物保留完整运动 horizon，敌机、`GROUP_NONTJT` 和激光只外推 9 帧，尺寸趋势只外推 6 帧；可见但尚未开启碰撞的敌弹和激光预警仍会进入风险场。是否启用 8 单位区域层由控制器当前真正选出的 region anchor 决定；没有 anchor 时，不可摧毁物仍使用普通 16/20 阈值。Boss #3 的 7/28、0.7 单位/帧和收缩期保持包络也直接来自控制器。

没有控制器发布状态时，插件保留只读本地回退：它按桥接器的可见性规则，以 5 帧延迟样本和再早 3 帧的样本估速，并复用相同圆覆盖、9/60 运动上限及 6 帧尺寸上限；但不声称拥有控制器已学习的区域相位。只有这种回退模式在中途按 F7 时需要短暂积累历史。

覆盖层的空间索引只登记可能触及 MPC 阈值的投影圆。独立测试包含严格边界真值表、密度反例、未来第 1..60 帧契约、9/60 帧运动上限、6 帧尺寸上限、延迟估速、激光圆覆盖、控制器状态接入、区域拓扑和 350 发弹幕逐格 full-scan 对照。当前实跑仅有 `51257/14112000` 次查询进入精确净空计算，即 0.36%；672 个栅格单元合并为 179 个同级渲染矩形，即 `179/672`。主插件 SHA-256 为 `8f3d3941c5fa644b5391902935dacad5df3b964222d918ccf6300538ede09178`。覆盖层仍是只读静态风险诊断，不会修改碰撞、对象、输入、RNG、AI 动作或记忆，也不包含 MPC 的移动轨迹、边界、Boss 对齐、portal、空隙入口和变向迟滞评分。

### 原生渲染性能与指标边界

当前控制器直连覆盖层已在原生 arm64 macOS、可见 `640x480` OpenGL 窗口、VSync、逐帧渲染条件下重跑 Okuu Stage 5 Boss #3，seed 为 `20260730`，profile 为 `bullet-group-expert`。引擎在 3363 个控制帧后返回 `attack_complete`，Boss HP 从 6000 降到 0，自机 0 死亡，射击指令率为 1.0。最终引擎遥测证明覆盖层开启且使用 `data_source=controller`，消费 revision 3356、60 帧 horizon 以及 16/20/8 阈值。3364 个有效显示样本中，`OBJ >= 300` 密集段有 2540 个样本，中位 59.989 FPS、P10 59.765 FPS、最低 51.364 FPS，对象峰值 504。报告为 `tools/stg_lab/artifacts/engine-mpc-boss3-controller-overlay-rendered-validation.json`，SHA-256 为 `399a8f6358703e03bd7bd0ba4d3dbf81cdc65a916685e6d3b88dd4520294b4ac`；artifacts 仍是被忽略的本地验证数据，不属于源码提交。

以下保留的同配置基准早于当前 60 帧 MPC 对齐覆盖层，只能作为旧 24 帧整格/密度版本和渲染器的历史性能数据。它使用 LuaSTG Sub v0.21.129、原生 arm64 macOS 15.7.3、Apple M4 Max 40 核 GPU、`640x480` 可见窗口和开启的 VSync。四轮都运行 Stage 5 Boss #3、seed `20260730`，记录 1200 个有效样本、0 个无效样本，逐帧渲染，决策间隔 3、观察延迟 5；密集段定义为 `OBJ >= 300`，每轮都有 273 个密集样本，峰值为 421 OBJ。

| 渲染器 / 覆盖层 | 密集段中位 FPS | P10 FPS | 最低 FPS | 报告 SHA-256 |
| --- | ---: | ---: | ---: | --- |
| CPU 软件 / 关闭 | 30.252 | 29.390 | 28.241 | `7cfcd17ca4ac9e86d3815ca9f302a33b4a6bb704f88b78c6e7949db1d1c61a4f` |
| CPU 软件 / 优化覆盖层 | 29.159 | 28.290 | 25.463 | `b8bf286e2bc68ae6319dee67a7cc80140356b33e24101e844577db16def3d9a0` |
| OpenGL GPU/FBO / 关闭 | 59.990 | 59.867 | 58.911 | `74482a01232db58d21fa75dc51f7a9e10b8ff2f3844b3484ade9f9bf9747d126` |
| OpenGL GPU/FBO / 优化覆盖层 | 59.988 | 59.984 | 52.265 | `67f5c5335b317d11584c01e392e91ff1e9b039723498d437a05614fdd43e94b7` |

旧版严格 GPU 加覆盖层击破报告还覆盖 3816 个有效显示样本，峰值 532 OBJ；其中 2844 个密集样本中位为 59.988 FPS、P10 为 59.492 FPS，并且最终达到 `attack_complete`。该报告同样使用的是旧覆盖层版本。

`render_performance` 只用于报告和诊断。其来源是 `lstg.GetFPS` 的原生 60 显示帧平均值以及 `lstg.GetnObj`，报告明确写入 `reporting_only_not_controller_input=true`；这些字段不进入 AI 观测、MPC 选择或区域动力学记忆。显示 FPS 表示原生渲染/Present 节奏，会受 VSync 影响；lockstep throughput 则是 Python/桥接控制下每秒推进的逻辑帧数或决策数，必须单独按 wall-clock 统计，不能称作显示 FPS 或 AI 推理帧率。

### Windows 可见演示与防闪烁

原生 LuaSTG 每个显示循环都会先把交换链目标清为黑色，再调用 Lua `RenderFunc`，最后执行 Present。可见 lockstep 等待 Python 时不能通过提前返回来减少重复绘制，否则这些已清空的缓冲仍会提交，形成完整画面与黑帧交替的高频闪烁。桥接器现已在所有可见显示循环重绘当前逻辑状态；`--render-every` 只作为旧协议兼容提示保留，不再在 Lua 层跳过绘制。无渲染模式仍会抑制 Lua 场景绘制。

Windows 必须先关闭正在运行的 LuaSTG，再把修改后的桥接器同步到本地游戏副本；从映射盘运行控制器不影响连接。先把 `SR_SOURCE_ROOT` 指向包含 `game` 的源码根目录：

```powershell
$SourceRoot = [IO.Path]::GetFullPath($env:SR_SOURCE_ROOT)
$Mod = 'SR_Subterrain_Reanimation_v100'
Copy-Item -LiteralPath "$SourceRoot\game\mod\$Mod\compat\testing\bridge.lua" `
  -Destination "C:\stg-win-demo\mod\$Mod\compat\testing\bridge.lua" -Force
```

可见演示使用 `SR_TEST_HEADLESS=0`、`SR_TEST_LOCKSTEP=1` 和 `--render --render-every 1`。启动参数应设置 `setting.vsync=true`；这可禁止允许 tearing 的提交，但仅开 vsync 不能修复旧桥接器产生的黑帧。修改 Lua 文件后必须重启引擎，因为文件只在启动时加载。若只有风险色块闪动而游戏底图稳定，可按 F7 关闭覆盖层并单独排查；整屏明暗交替则应先检查本地 `bridge.lua` 是否已经同步。

完成本地游戏复制和 `.venv-win` 安装后，可直接双击 `tools\stg_lab\run-win-boss3.cmd`。它只启动现有文件，不复制、覆盖或安装游戏与 Python 环境。脚本会校验本地桥接器与当前源码一致，默认读取已纳入版本库的 `models\region_dynamics_boss3_v2.json`，设置测试环境变量，以 `setting.vsync=true` 启动 `C:\stg-win-demo\LuaSTGSub.exe`，无连接地等待端口监听，然后运行 Boss #3 可见 MPC 测试。成功仍只接受引擎返回 `attack_complete`。报告写入带时间戳的 `tools\stg_lab\artifacts\engine-mpc-boss3-win-*.json`；同轮原生录像写入本地游戏的 `userdata\replay\SR_Subterrain_Reanimation_v100\analysis`，脚本会验证文件非空并打印两个路径。传入 `-RecordObservations` 可从第 0 帧归档完整观测；默认保留游戏窗口，传入 `-CloseGameWhenDone` 才会在结束后请求关闭。

也可从 PowerShell 覆盖默认参数：

```powershell
& .\run-win-boss3.ps1 `
  -LocalGameRoot 'C:\stg-win-demo' `
  -Seed 20260730 `
  -Port 24816
```

从模组根目录运行 Python/Lua 静态检查的命令如下；portable 的构建、CTest、headless 和截图命令见上方代码块：

```bash
tools/stg_lab/.venv/bin/python -m compileall -q tools/stg_lab/src/stg_lab
tools/stg_lab/.venv/bin/python -m pytest -q tools/stg_lab/tests
luac -p ../../plugins/SafetyZoneVisualizer/__init__.lua
luac -p ../../plugins/SafetyZoneVisualizer/SafetyZoneVisualizer.lua
git diff --check
```

## 原生流式策略 v1：当前严格结果

最终保留的 checkpoint 是
[`policy_native_stream_v1.pt`](../tools/stg_lab/models/policy_native_stream_v1.pt)，
SHA-256 为
`829eebe53c886e5ba53f577542938b904aad740f3a5bf04b49d61e73ab61557d`；
完整模型卡见
[`policy_native_stream_v1.md`](../tools/stg_lab/models/policy_native_stream_v1.md)。
训练档案含 25 个 episode、21,916 个决策，SHA-256 为
`1bc03ce647d34c1fb3f77ba751d50ca7b602a9e16189025819e3cf89a423c384`。

模型由全局/局部语义视觉编码器和 192 单元 GRU 组成，
`memory_size=0`、`proficiency_size=0`。输入只有最近一帧延迟后的全局/
局部六通道语义图、当前可见自机位置和 GRU 隐状态；场景/符卡编号、
绝对帧、脚本相位、记录路线、waypoint、教师风险场和外置策略记忆都不
进入模型。当前 checkpoint 的时序记忆全部由 GRU 学习。

训练采用 episode-stateful TBPTT，chunk 长度 32，并按完整 episode 平衡
优化：20 个训练 episode 每个 epoch 各贡献一个等权 optimizer step，长局
不会仅凭样本数获得更大权重。类别平衡指数为 0.75，首次开始移动的
权重为 4.0，方向变化权重为 1.5；两种变化只按同一 episode 内相邻的
`teacher_action` 计算，方向只看 `(move_x, move_y)`，不把 slow 切换当成
方向变化。episode 11-15 整局保留为验证集。

本次训练明确使用 `restore_best_validation=false`。内部验证 loss 最低的是
epoch 8（2.3124849571），最终保留 epoch 40 时为 2.5737202304。保留 epoch
40 是因为它是通过原生 Koishi 严格门槛的候选，而不是因为它是
validation-best；离线 loss 和动作准确率不能代替原生通关结果。

`contextualize-demos` 可以根据严格 provenance manifest，为每个完整 episode
附加仅含攻击身份的 one-hot token。词表包含 unknown token 和已登记的符卡/
道中身份，并由 `--scenario-vocabulary-manifest` 写入 checkpoint；其中不含
相位、坐标、动作、waypoint 或路线。这样可以让网络权重自行学习不同攻击
所需的行为，而不是新增手写策略分支。启用
`--previous-action-conditioning` 时，另一个 18 维 one-hot token 只表示上一条
真正执行的移动动作。direct-corrective DAgger 保留这条实际执行动作流，
live inference 也只在引擎确认推进后提交；其中不含未来动作、教师建议、
坐标、帧、相位、waypoint 或路线，只作为网络学习时序动力学的运动反馈。
该实验仍未通过发布门槛：一个 256
单元 GRU 候选在 915 帧严格通过 Koishi #1，但 Okuu #3 于 522 帧中弹；再
训练 20 epoch 只延长到 642 帧。一局严格完成的 Okuu #3 DAgger 数据需要
在 1272 个决策中介入 1100 次，随后续训 10 epoch 的纯模型仍于 514 帧
中弹。因此正式随附的 checkpoint 继续采用 `memory_size=0`。

四类控制结果必须分开：

| 类别 | 移动决策来源 | 证据边界 |
| --- | --- | --- |
| pure GRU | 延迟语义视觉、当前可见自机位置、GRU 隐状态 | 可作为学习模型结果 |
| visible safety | 独立的纯可见局部预测可覆盖 GRU 动作 | 仅为 hybrid 诊断，必须报告检查与介入次数 |
| DAgger teacher | 学生访问状态，精确状态 MPC 为每步打标签并可介入执行 | 仅训练数据；严格完成不等于学生通过 |
| Engine MPC | 精确对象几何/速度和可选的教师专用区域动力学记忆 | 规划器/教师结果，不是神经 checkpoint 输出 |

DAgger 目前有两种显式声明的档案契约。旧 teacher-labelled 档案在每个学生
访问状态上都以 `teacher_action` 为监督目标。当前 direct-corrective 档案则
保留学生实际执行动作和完整 recurrent context；教师介入时，实际执行标签
就是教师纠正动作，`supervision_mask` 可以让动作损失只监督这些介入/纠正
点。带 mask 的档案与旧无 mask 档案合并时，旧样本仍按全监督处理。只有
满足严格原生完成条件的局才可进入档案，但这只证明数据有效，不能把教师
介入后的击破记作学生成功。

新一轮 Okuu #3 direct-corrective 收集在 seed 20260813 和 20260815 均于
第 3,815 帧返回 `attack_complete`、death 0，但分别介入 489/1,272 次
（38.44%）和 467/1,272 次（36.71%）。报告明确设置
`pure_policy=false`、`pure_policy_success=false` 和
`pure_policy_validation_eligible=false`。v13 all-label 聚合档案含 35 个
episode、33,866 个 recurrent 决策；critical-intervention 聚合档案保留同样
的 33,866 个决策，总计 30,376 个有监督标签，并为六局 Okuu #3 保留完整
recurrent context，但在这些局中只监督 4,142 个教师介入点。

最新 pure、关闭 visible safety 的 Okuu #3 候选严格结果如下：

| 候选 | seed | 逻辑帧 | 观测 HP | 终态 / death | 结果 |
| --- | ---: | ---: | --- | --- | --- |
| v10 unweighted | 20260812 | 392 | `6000 -> 5524.5` | `player_hit` / 100 | 失败 |
| v11 corrected unique | 20260812 | 764 | `6000 -> 4924.5` | `player_hit` / 100 | 失败 |
| v11 纠正样本重复四次 | 20260812 | 431 | `6000 -> 5453` | `player_hit` / 100 | 失败 |
| v12 Okuu specialist，最终 epoch 80 | 20260812 | 726 | `6000 -> 5086` | `player_hit` / 100 | 失败 |
| v12 general，最终 epoch 30 | 20260812 | 414 | `6000 -> 5478` | `player_hit` / 100 | 失败 |
| v13 all-label，validation-best epoch 3 | 20260816 | 413 | `6000 -> 5481` | `player_hit` / 100 | 失败 |

v13 all-label 在 epoch 3 取得最低 validation loss 1.9728760589。该离线
改善和前述两局教师辅助击破都没有通过 pure native 发布门槛，因此
v10-v13 均不能发布或替换 v1。

符卡严格成功必须同时满足：`terminated=true`、
`termination_reason=attack_complete`，以及显式、有限、非布尔的数值
`final_player.death=0`。完整关卡必须改为 `stage_complete` 并满足相同的
零死亡证据。达到帧数/时间上限、仅削减部分 HP、活得更久、死亡后完成、
ghost/protected 状态、缺失或非数值 death 都是失败。以下是最终 checkpoint
已经执行的完整 pure GRU 矩阵，统一使用 history 1、观察延迟 5、expert，
且关闭 visible safety：

| 目标 | seed | 逻辑帧 | 观测 HP | 终态 / death | 严格结果 |
| --- | ---: | ---: | --- | --- | --- |
| Koishi #1 | 20260738 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Koishi #1 | 20260739 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Koishi #1 | 20260740 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Yamame #3 | 20260738 | 491 | `1800 -> 1306.25` | `player_hit` / 100 | 失败 |
| Satori #5 | 20260738 | 384 | `3200 -> 2883` | `player_hit` / 100 | 失败 |
| Okuu #3 | 20260740 | 754 | `6000 -> 5046` | `player_hit` / 100 | 失败 |
| Okuu #4 | 20260740 | 1065 | `3500 -> 2629.5` | `player_hit` / 100 | 失败 |
| Okuu EX #2 | 20260740 | 446 | `3333 -> 2834` | `player_hit` / 100 | 失败 |
| Orin #4 | 20260740 | 261 | `4500 -> 4407` | `player_hit` / 100 | 失败 |
| Stage 1 Normal | 20260740 | 543 | 道中多波 HP 不可直接比较 | `player_hit` / 100 | 失败 |

随后又在最终源码指纹
`fa891752547e10f478fbec6b4f85349e4c43061fb3788bea9014ac1f9337ac56`
上复测保留 checkpoint，并验证 checkpoint 路径/SHA-256；其余设置仍为
history 1、观察延迟 5、expert、关闭 visible safety：

| 当前源码目标 | seed | 逻辑帧 | 观测 HP | 终态 / death | 严格结果 |
| --- | ---: | ---: | --- | --- | --- |
| Koishi #1 | 20260738 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Koishi #1 | 20260739 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Koishi #1 | 20260740 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Koishi #1，held-out | 314159265 | 906 | `700 -> 0` | `attack_complete` / 0 | 通过 |
| Yamame #3，held-out | 314159265 | 493 | `1800 -> 1300.25` | `player_hit` / 100 | 失败 |
| Okuu #4，held-out | 314159265 | 1065 | `3500 -> 2629.5` | `player_hit` / 100 | 失败 |
| Stage 1 Normal，held-out | 314159265 | 543 | 道中多波 HP 不可直接比较 | `player_hit` / 100 | 失败 |

因此当前发布版只证明 Koishi #1 的 strict pure 完成：三个已知 seed 和一个
独立选择的 held-out seed 均通过，而当前源码上的跨目标探测全部失败，不
外推为更广的成功率。原三局每千帧变向 105.96-107.06 次、精确反向
10-11 次、ABA 均为 7、方向保持中位数 6 帧。严格 Koishi DAgger 教师参照
为每千帧
变向 97.46 次、ABA 0、保持中位数 9 帧，因此 checkpoint 仍显得更躁动。
这说明成功轨迹没有退化为静止，但同一张近确定性符卡的三个已知 seed 加
一个 held-out seed 仍不能声称达到真人水平。后来用于 DAgger 的 Yamame #1
和 Satori #1 没有最终 checkpoint 的 pure 运行，不能声称模型通过。与真人
尽可能相近的通用表现、跨符卡可靠性和完整关卡能力仍未达成。

熟练度由模型外的确定性执行参数模拟，不是网络条件输入。expert 为新增
反应延迟 0、最短保持 3 帧、次优动作 0%；intermediate 为 3/6/1%；novice
为 6/9/4%。在仍关闭 visible safety 时，intermediate 的 Koishi #1 seed
20260741 于 593 帧中弹（`700 -> 291.5`、death 100），novice 于 478 帧
中弹（`700 -> 342.5`、death 100）。expert 的 3/3 使用另一组 seed，不能
当作严格配对的熟练度统计。

validation-best 和专用模型也没有绕过原生门槛。Yamame #3 specialist v1
的 validation-best epoch 1 于 598 帧中弹，同模型 epoch 40 于 495 帧中弹；
recurrent size 256 的 specialist v2 validation-best epoch 19 在 seed
20260740/20260743 分别于 530/502 帧中弹，四局 death 均为 100。因此当前
specialist 和 validation-best 候选均作为部署方案否决，而不是用较好的
离线 loss 美化结果。另一个含 29 个 episode、26,234 个决策的 v5 候选从头
训练 60 epoch，validation loss 降至 2.2334204706，但在 seed
20260742/20260744 的 Koishi #1 都于 391 帧中弹（`700 -> 592.5`、death
100）；再以 `5e-5` 续训 15 epoch 也只到 436 帧。该模型在 Okuu #3 加入
可见安全层后，经 66 次介入仍于 902 帧死亡，反而短于纯模型的 1,146 帧，
因此两者都没有替换当前 3/3 checkpoint。

完整关卡仍失败。最终 pure GRU 在 Stage 1 Normal 全程保持静止并于 543
帧中弹。另行运行的 MPC teacher 在 seed 20260731 首次达到 7,199 帧
`time_limit`，death 0，仍按规则失败；seed 20260732 后续运行在 12,453
帧返回 `stage_complete`、death 0。后者只是教师证据，不是模型成功；该
超长 episode 因过度放大静止决策而从符卡训练池中移除。

## 实施阶段

1. 建立 JSONL protocol v2、`reset/step/observation` 和确定性状态哈希。
2. 实现 Stage 5 #3/#4 独立场景、碰撞和预测接口。
3. 实现时空风险场、连通区域和允许跨越较低危险区的规划器。
4. 实现由可见半径与核弹排位移学习的四相/横向流周期、跨帧安全连通域身份和动态 portal 选择；旧 SQLite 路线工具只保留为历史 standalone 回归。
5. 用 visible-v2 重建全局/局部双视野、空白冷启动、可见位移估速和五帧延迟数据。
6. 用规划器生成示范并训练 `policy_visible_v2.pt`；shield 只保留为权威状态诊断工具。
7. 在 Windows LuaSTG 后端运行带 PID、二进制和源码指纹的 53 项双进程确定性回归。

训练/验证按 episode 与 seed 分组切分，避免同一轨迹的相邻窗口同时进入两侧。每个延迟历史窗口只监督最后一个决策时刻，更早画面仅作为时序上下文；类别平衡权重只由训练组统计。

最终冻结的 CNN+GRU checkpoint 为 `policy_visible_v2.pt`，SHA-256 是 `f9815eb6fd4e0e5e35e856c836567f078f36ca1d010a4052c36ea31b2ff550e6`。40 epoch 训练使用 5,208 个 visible-v2 样本，训练集 SHA-256 为 `52be736f0743379c4fb15321b6761ff3300ed2062acf76201460492ddc91f53d`；最终内部验证动作准确率 98.3146%，训练 loss 0.0196861、验证 loss 0.0568566、risk MAE 0.0316795。训练指标 JSON SHA-256 为 `28c83647f4a798cb4feab93396ea3e404882469c5b20faf2a970cdac4c782446`。

持出集含 868 个样本，SHA-256 为 `7318ed50f5f1bba81c48af87bbcfc69f67bf5714f5a47193fd55b11e759f41d3`；visible 数据 manifest SHA-256 为 `e380f4cea85373ebaeb71f574985c6cb083a9498ed0aa1d773c6d64fd7bc14ca`。严格重放 checkpoint 后，整体教师动作一致率为 796/868（91.7051%）：#3 为 369/400（92.25%），#4 为 427/468（91.2393%）。一致率报告 SHA-256 为 `1208ed6dc57cae5a47a5ba6f8d7bcbbd90cab27af06dd3ffe7dc3cbb1f95abcd`。

## 达标条件

独立仿真至少满足：

- 固定 seed 重复运行时状态哈希完全一致；
- #3/#4 规划器在 100 个持出 seed 上存活率不低于 95%；
- 五帧观察延迟和三帧动作保持下，仅由在线可见信息选择的无 shield 可部署控制器存活率不低于 90%；
- 视觉模型对教师动作的持出一致率不低于 85%；
- 历史 #4 路线记忆基准中，第一次失败后第二次存活或最大路径风险至少降低 30%；
- 所有单元和集成测试通过。

标准独立仿真窗口为：#3 运行 600 帧（覆盖三次扩张周期），#4 运行 700 帧（覆盖星体生成、至少一整圈和持续扇形弹）。它们是存活测试，不是击破测试，因为独立后端尚未模拟自机伤害。61/69 秒完整 timer 耐久只作为额外压力测试。原版 #4 基本确定性，独立仿真用有记录的小幅相位扰动测试鲁棒性；报告必须包含唯一轨迹哈希数量，禁止用重复局扩充样本。

## v2 最终独立仿真证据（历史路线基准）

严格验收器现在要求每个文件产物绑定当前 Python 实现指纹。可部署控制器报告若使用 shield、权威状态或非在线可见 cue 会直接失败；checkpoint 和 held-out visible-v2 数据的 SHA-256 必须与实际文件匹配；规划器与控制器必须使用完全相同的不同 seed 集合。

最终产物绑定实现 SHA-256 `ba7f4d2ee9fe5bf232f264a180a51280657f007629a54fd36c3f4884ca966cb9`，规划器和可见控制器使用完全相同的 100 个不同 seed（5001-5100）：

| 证据 | 严格复算结果 | SHA-256 |
| --- | --- | --- |
| 精确状态规划器 #3 | 600 帧存活 100/100，终态哈希 100 个不同值 | `f10ce59b18329a6d71ec0d5ba20c9bc9fd4bb21d7a0af416d5dcfe40cf78bd6d` |
| 精确状态规划器 #4 | 700 帧存活 100/100，终态哈希 100 个不同值 | `6ad7b42b6977b6e8af3f854e294bbe86497290d53ad7a1c5ad636c1750fe92c8` |
| 历史延迟可见单路线控制器 #3 | 无 shield、无权威状态、在线 cue，存活 100/100 | `f29d3cf92864b1af2c8846ea53669d8769a7d4e74c6da9c2caffa85b41589069` |
| 历史延迟可见路线库控制器 #4 | 无 shield、无权威状态、在线 cue，存活 98/100 | `004e4696766f5c4a0fa1a682a21aef72787dedef98426ad04749ed10a6ca00aa` |
| 历史 #4 外置记忆基准 | 第一次在第 340 帧死亡；控制帧 138 根据延迟来源帧 133 选择 memory 2；第二次存活 700 帧 | `b57f15e26d8ee49de6895b5145f52489f8da562e307605adb481ba31ebf44b42` |
| 逐帧确定性 | 两个全新 #3/#4 环境的全部 601/701 个状态哈希及动作一致 | `dbc0cdc1b50f9e75315672580c6e4c1aecb0b670700d7a02cb45c3df50be12e4` |
| 严格独立验收 | `passed: true`、`issues: []` | `7035a896eb96f9633ca5515314d39edfd763ddcd7cdfaad673f5b7d3f595d061` |

历史只读 SQLite 记忆库 SHA-256 为 `e774d3148ba0bd0cc89d1e8f9d68db8e3bf1612a1b56fd2b61b14940d321b584`；#3 单路线 SHA-256 为 `d492d790cfb5310cb6f519a80bfefe8750a0d0f6dfcbd5271475ad785a76bf17`；#4 五路线库 SHA-256 为 `6a1c47342f0f7d1190b810e96cfacf104b76ee111dfabd6041321b273f20873a`。记忆门槛是通过“第二次存活”分支达标；两次报告的 peak risk 都是 0，因此不声明数值风险降低。

这份历史验收只证明文档定义的独立仿真门槛。规划器是精确状态教师；路线评估使用延迟可见 cue 和旧路线库，其动作不是神经 checkpoint 输出，也不是当前 Engine MPC 的记忆方案。Python 场景仍是近似，存活结果不等于原符卡击破，也不等于 AI 已在真实引擎中存活。

原生 Engine MPC 的符卡单局成功标准只有：报告同时满足 `terminated=true`、`termination_reason=attack_complete`，以及显式、有限、非布尔的数值 `final_player.death=0`。由引擎确认 Boss 被击破或符卡完整结束均可；完整关卡则必须使用 `termination_reason=stage_complete` 并提供相同的零死亡证据。达到 `max_frames`、仅延长存活时间、只削减部分 Boss HP、死亡后完成，或缺少有效 death 证据，一律不计成功。所有动作必须保持 `spell=false`，并在每个有效逻辑帧保持 `shoot=true`。射击不改变自机速度或碰撞判定，把它与躲避净空耦合只会降低输出并延长危险暴露；旧射击阈值仅保留用于兼容已有命令与风险遥测，不再控制射击。新版 live runner 报告使用 schema 3，以 `shoot_command_frames`、`shoot_command_rate` 和 `continuous_fire` 表达发送给引擎的射击命令；schema 2 中语义错误的 `unsafe_shot_frames` 仅保留为 `null` 和弃用标记，移动规划的预测碰撞另行统计。带动作前缀的运行必须单列为“前缀辅助复现”，不能混入无前缀成功率。

### 2026-08-04 Boss #3 真人行为校准

`humanlike` profile 的目标是先保留原生严格击破，再缩小与成功真人 replay
的运动差异。控制器仍只使用五帧延迟可见信息、在线运动估计和本局状态；本轮
三局均没有传入 `--region-dynamics-memory`、路线、动作前缀或 checkpoint。
`slot2.rep` 和 `slot3.rep` 仅由 replay 分析器产生报告，不会成为控制器输入。

`slot2` 在第 2579 帧撞到强制区域，Boss 余 1190.5 HP；死亡前区域净空从
`9.49 -> 4.64 -> -1.10`，而子弹净空仍为 84.58。`slot3` 在第 3236 帧
严格击破。成功真人通常保持低速和较长稳定方向段，区域换边次数与 AI 接近，
所以差异主要是局部重规划抖动，而不是安全区周期或换边逻辑错误。

| 实际 replay 指标 | 成功真人 `slot3` | AI v12 | AI v14 |
| --- | ---: | ---: | ---: |
| 帧数 / 路径 | 3236 / 4939.79 | 3281 / 6366.70 | 3291 / 6178.90 |
| 移动占比 / 低速占比 | 64.43% / 86.31% | 73.06% / 73.64% | 68.92% / 72.17% |
| 底边钳制占比 / 平均 Y | 55.72% / -195.22 | 22.25% / -192.90 | 34.82% / -195.55 |
| 移动中变向 / `>=90` / `>90` | 168 / 6 / 0 | 298 / 172 / 69 | 278 / 164 / 79 |
| 精确反向 / 低速状态切换 | 0 / 72 | 22 / 251 | 24 / 251 |
| 总净空 P10 / 区域净空 P10 | 12.17 / 12.96 | 11.17 / 11.12 | 11.56 / 11.94 |

v14 修正了 `bottom_anchor_enabled` 在区域导航中的语义：只有当自机已经位于
目标 `exterior:left/right` 连通区且模式为 `settle` 时，区域锚点才回到有效
底边；`preposition`、`evacuate`、强制穿越、gap 入口和碰撞优先级完全不变。
这使代表 seed 的路径缩短 2.95%，移动占比下降 4.14 个百分点，底边驻留增加
12.57 个百分点，并修复 v12 在 seed `20260732` 第 2630 帧的区域碰撞。

最终源码指纹为
`0a67effb8e54225e0bcc7209902cacd7068f17dc386c68bbde2394a22aac9a1e`。
三次原生 arm64 macOS headless 运行都使用 60 帧 horizon、五帧观察延迟、
持续射击、禁用 spell、无外置区域记忆，并满足唯一严格成功标准：

| seed | replay 帧 / 路径 | 终态 | replay CRC32 | 报告 SHA-256 |
| ---: | ---: | --- | --- | --- |
| 20260730 | 3291 / 6178.90 | `attack_complete`、HP 0、death 0 | `be0090d2` | `a3a69309078e83ff2646ee69fb55b3f6c8e4dc308992230e58cc8a5a98967181` |
| 20260731 | 3278 / 6223.50 | `attack_complete`、HP 0、death 0 | `f44782c7` | `3c06e2dc8a4ef0c04656777e77613506c04702c022b508becbe5744f32ce7987` |
| 20260732 | 3301 / 6582.41 | `attack_complete`、HP 0、death 0 | `3dd61d1c` | `7478675195b2ac1a6f66fb0ae36835a83225fc5f0e6b76346c8361e3cbec03bc` |

三份 replay 都是 `saved=true`、`verified=true`、射击率 100%。对应 replay
SHA-256 依次为 `f0c07634ac2aeb41a8fd49133fd0e9e2bcfe85097829bf338945edf40e913fb5`、
`b98bd4ef36bc94aecab82151fdf0d677bd9a8160119e3ac6fada9ea201006fa2` 和
`cb1fab256c947d70d38f5a170c8501911eb6237d9f4fe583bb7c3b97af0fb7cc`。
这是已执行 seed 的 `3/3`，不是未测试 seed 的统计成功率。

两个看似更平滑的消融被严格拒绝。按低速路程估算区域 deadline 的 v13 在
第 1917 帧撞区域，Boss 余 2558 HP；在每次锐角转向前强插安全中间方向的
v15 在第 2270 帧撞区域，Boss 余 1869 HP。后者代码已从最终源码删除；
60 帧内“不更差”不足以保证三帧扰动后仍进入同一未来连通区。低速 deadline
和 neutral-beat 实验开关保持默认关闭。不能用延长存活时间替代击破结果。

v14 改善了路径、移动比例、底边驻留和跨 seed 稳健性，但 `>90` 转向、精确
反向和低速切换仍明显差于真人，因此不声明达到真人水平。下一阶段应从成功
真人 replay 构建带同步可见观测的训练数据，让 recurrent policy 学习动作
持续性；仅有坐标轨迹的分析报告不能安全地反推出当时可见输入，也不能作为
录制路线记忆注入当前 MPC。

### 当前原生 Boss #3 严格结果

保留的三次 CrossOver 验证都使用 CrossOver 26.3 + DXVK 启动新的原生 LuaSTG Sub 进程，从 Spell Practice reset 开始连续 `live_mpc` 闭环。v40/v41 是五帧观察延迟的持出验证，v42 是单列的零延迟回归；三局都使用区域动力学记忆 v2、`authority_state_shield=false` 和 `spell=false`。结果均在第 3816 帧由引擎确认 Boss HP `6000 -> 0`，自机 `death=0`，且 `unsafe_shot_frames=0`：

| 报告 | seed | 观察延迟 | 严格终态 | Boss HP / 自机 | SHA-256 |
| --- | ---: | ---: | --- | --- | --- |
| v40 | 20260730 | 5 | `terminated=true`、`attack_complete`、`passed=true` | `6000 -> 0`、`death=0` | `e7577aa475ed9a9de6542fedfba8a193dca1b3d8a927e139371e22f41b2d94ef` |
| v41 | 20260731 | 5 | `terminated=true`、`attack_complete`、`passed=true` | `6000 -> 0`、`death=0` | `5cedf76c0b17028b4239f480dbd146a54cb92ead17dc898f8fc8d6fb52e981fa` |
| v42 | 20260732 | 0 | `terminated=true`、`attack_complete`、`passed=true` | `6000 -> 0`、`death=0` | `a45084f331ecd82d0fecff636bf1921c9b547f41f82a1db6846ff76d15d7e37f` |

三份报告绑定同一实现指纹 `5a81172add05549fdf1ea6d65272d26dd08afc3de6c289ab3124e9f7b2e69613` 和同一 v2 记忆 SHA-256 `dcdcddeeed840d733e144477934f217b21f6795ab9a83790d7f3813d273546f7`。五帧延迟持出验证为两个已执行 seed 的 `2/2`，零延迟回归另行通过；这里不外推为尚未执行 seed 的统计成功率。

最终区域排序和 committed-action 安全修复又在原生 macOS 无 DirectX 引擎上从 Spell Practice reset 完整运行两个五帧延迟 seed，均在第 3816 帧严格击破：

| 报告 | seed | 严格终态 | Boss HP / 自机 | 变向 / 反向 / ABA | 净空中位数 | SHA-256 |
| --- | ---: | --- | --- | --- | ---: | --- |
| `engine-mpc-boss3-safety-sort-v46-seed20260730.json` | 20260730 | `attack_complete`、`passed=true` | `6000 -> 0`、`death=0` | 487 / 20 / 20 | 8.27 | `e1a4df0fd857b3aa7cc52e6bcd6416a856f74228f34e07c26b4a19882cbe6d39` |
| `engine-mpc-boss3-safety-sort-v47-seed20260731.json` | 20260731 | `attack_complete`、`passed=true` | `6000 -> 0`、`death=0` | 502 / 10 / 30 | 8.31 | `d69d6b7fa2b88637590f5bbe5f605e1768bd7dccbd47307bdfa25fd651d50faf` |

两局均为 `unsafe_shot_frames=0`、强制 `spell=false`，实现指纹为 `8422915228d7b867ae01ffae0e2d0ae85d7ab8d1aac71e3a383c6b8a6e6d2044`。这是已执行尝试的 `2/2`，不外推未执行 seed，也不证明 Boss #4；报告仍以 `acceptance_claim=false` 正确标记为 live teacher 严格引擎证据，而不是部署模型验收。它们取代修复前的 v44/v45；后两份只保留为中间过程记录。

当前 live bridge 和 MPC 还覆盖直线及曲线激光。渐宽/渐窄的直线段和曲线
折线段使用保守的 16-32 px 圆覆盖，每个圆的半径包含准确的分段半步长；
连续采样位置会反映旋转和长度变化，不会错误地只复用激光原点位移。实现
`fab98499b72c55fb92ceb5586b58be5093df9e42b9755631e008633ceaf96f95`
在原生 headless Okuu #2、零观察延迟下的一局严格结果于 episode 第 3036
帧完成，Boss HP `4300 -> 0`、`death=0`、`unsafe_shot_frames=0`，观测峰值
为 552 发子弹和 40 条激光。规划器威胁数中位数为 302、最大值为 697。
报告 SHA-256 为
`70d5afc69faa6fbae55cef0bdd678f6fe090d977608ec4d16ec281377f85dfd2`。
这只是一局零延迟 MPC 教师的几何集成结果，不是五帧延迟或学习模型证据，
也不是成功率声明。

另有一轮完整引擎的原生 macOS OpenGL 验证在启用优化 F7 覆盖层时满足同一严格成功定义。`engine-mpc-boss3-gpu-overlay-strict-seed20260730.json` 的 SHA-256 为 `ec3f758a8a5135b33e139076bdecdb050bf1117f1e13622679a91c40e8110def`；报告在第 3816 帧写入 `terminated=true`、`termination_reason=attack_complete` 和 `passed=true`，Boss HP `6000 -> 0`、`death=0`、`unsafe_shot_frames=0`，并强制 `spell=false`。该报告的 `acceptance_claim=false`，所以它是单局 live MPC teacher 的严格原生击破证据，不是部署模型验收，也不证明 Boss #4。该运行早于当前覆盖层实现，只能作为历史版本的完整引擎集成证据，不能作为当前覆盖层 SHA-256 的实机执行证明。

桥接器已经由真实引擎启动并监听 `24816` 后，可用下列命令复跑；命令在未达到唯一成功条件时以非零状态退出：

```bash
cd tools/stg_lab
uv run stg-lab engine-mpc-play \
  --host 127.0.0.1 --port 24816 \
  --scenario 'okuu:Lunatic' --attack 3 --seed 20260730 \
  --max-frames 4200 --horizon-frames 60 --observation-delay 5 \
  --region-dynamics-memory models/region_dynamics_boss3_v2.json \
  --replay-name boss3-seed20260730 \
  --output artifacts/engine-mpc-boss3-rerun.json

for report in \
  artifacts/engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json \
  artifacts/engine-mpc-boss3-heldout-v41-seed20260731-d5-region-dynamics-v2.json \
  artifacts/engine-mpc-boss3-regression-v42-seed20260732-d0-region-dynamics-v2.json
do
  jq -e '
    .passed == true and .terminated == true and
    .termination_reason == "attack_complete" and
    .config.authority_state_shield == false and
    .config.spell_forced_off == true and
    .outcome_evidence.boss_hp_last_observed == 0 and
    .outcome_evidence.final_player.death == 0 and
    .unsafe_shot_frames == 0
  ' "$report" >/dev/null || exit 1
done
```

### 原生内容确定性回归

引擎验证还必须满足：

- 实时目录严格包含 22 个场景、53 个攻击；每项默认推进 300 个逻辑帧，这也是正式验收下限；
- 默认使用 `step_batch=1`、低速静止、不射击，逐逻辑帧请求并计算哈希；
- 每项都必须同时观察到 Boss/敌人与有效危险物；危险物可以是敌弹、不可摧毁物、激光，或脚本用作危险物的额外可碰撞敌人对象；
- 同 seed、同输入在两个全新进程中的每个逻辑帧规范化观测哈希一致；
- 不启用测试模式时，输入、渲染、随机数和正常流程没有行为变化；
- 53 个攻击窗口内无 Lua 错误、进程崩溃、NaN、对象池耗尽或协议失步；真实引擎 AI 存活率不属于该编译器范围。

严格 `engine-accept` 只接受两份各自通过的 protocol/report-schema-v2 完整报告，并强制检查：不同且非空的 `session_id`/`process_nonce`；不同的正数 OS PID；当前 Python 实现指纹；相同的引擎二进制 CRC、运行时 Lua CRC 和本地源码 SHA-256；完全相同的严格配置；精确的 53/22 数量与完整目录顺序；唯一且一致的场景、攻击、卡片序号和按目录递增的 seed；每逻辑帧一个请求；重置帧在内共 `frames+1` 个合法且非静态的 32 位小写十六进制哈希；终态哈希一致；有效内容峰值计数；测试窗口内不提前结束；错误数组为空；以及两轮所有逐帧哈希完全一致。单份 `engine-test` 即使通过也固定写入 `engine_verified: false`，只有两轮合并报告通过后才能为 true；合并产物还会保存两份输入报告的规范化 SHA-256。

当前主机上，CrossOver 26.3 配合 DXVK 已能启动真实可执行文件、加载完整 SR 资源/脚本和桥接器，并打开 TCP 监听端口。wined3d 仍在 framebuffer 创建时报 `GL_INVALID_FRAMEBUFFER_OPERATION (0x506)`；DXMT 能走到 Lscreen RenderTarget，但获取 `IDXGISurface` 时返回 `E_NOINTERFACE`。这些后端诊断本身不是验收证据。

最终 protocol-v2 DXVK 两轮使用不同的 session `engine-v2-a`/`engine-v2-b`、不同的 process nonce 和 Win32 PID 212/204。两轮引擎可执行文件 CRC32 均为 `8844e525`，18 项运行时 Lua CRC32 均与本地文件一致，两份 18 项本地 SHA-256 映射也与当前工作树一致。每轮都完整保留 22 个场景、53 个攻击，每项含 reset 在内 301 个逐帧哈希，结果均为 53/53 通过且无错误；文件 SHA-256 分别为 `ac7996a2ee92417e08deda8ff5e86d3a0937278f7bb15a772e53089275a0abeb` 和 `8d692aa4a4f79ed6de0d701d628ebbbbc64cce624f016deb12d198dec6e5b257`。合并报告对 53 个攻击的 15,953 个哈希位置全部匹配，写入 `passed: true`、`engine_verified: true`，SHA-256 为 `a2a7cca87e6e416c43b963483e450b5a1e000cdfcdbd32eadbdb16d5e19ea1d2`。旧 protocol-v1 报告仍不满足 v2 门槛；本结论边界只是 headless 符卡内容回归，不等于渲染帧对比、完整关卡通关或 AI 在真实引擎中的存活率证明。
