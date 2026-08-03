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

当前 `engine-mpc-play` 已删除动作前缀 loader、运行参数和回放分支。原生诊断报告仍保存逐帧动作、位置和绝对帧以便定位故障，但当前代码没有把报告重新解释为策略输入的入口；正式运行只能从 reset 开始由 live MPC 连续闭环控制。

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

最终同源码、连续射击的原生 A/B 使用 Okuu #3 seed `20260730`。两份报告均
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

引擎根目录的 `game/plugins/SafetyZoneVisualizer` 提供原生游戏内 F7 分级覆盖层；也可设置 `SR_SAFETY_ZONE_OVERLAY=1` 在首次游戏渲染前开启。它按当前可碰撞对象及未来 0 到 24 帧的每个逻辑帧线性投影，绘制绿色安全、黄色注意、橙色危险和红色碰撞四级区域。16 单元网格按半格对角线膨胀判定半径，因此覆盖整个 cell 而不只是中心点；红色为净空不大于 0，橙色为净空不大于 16 或同一未来层 36 范围内至少 5 个威胁，黄色为净空不大于 28 或至少 3 个威胁。净空取所有未来帧最差值，密度逐帧计数后取最大值，不再把先后经过的子弹虚构成同一时刻的密集弹幕。椭圆半径只沿增长方向外推：最近重建提供观测增长率，`GROUP_INDES` 椭圆另至少按 Boss #3 的 0.7 单位/帧增长保护预测，绝不靠假设未来缩小来把格子判安全。插件支持圆、旋转椭圆、矩形及直线激光。直线激光不再依赖通用椭圆字段：即使 THlib 对象的 `a=b=0`，插件也会按 `l1/l2/l3/w` 构造两端渐宽/渐窄的多边形并计算距离。插件只读对象几何，不修改输入、RNG、碰撞或 AI 记忆；曲线激光内部仍不由此诊断插件栅格化，实际碰撞以引擎为准。

覆盖层的保守空间索引只登记可能影响分级的投影边界内单元。独立测试针对椭圆、旋转矩形、渐宽/渐窄激光、旧 3 帧层之间穿越的快速子弹、增长中的强制位移体、格角碰撞、错时密度、边界条件和 350 发弹幕逐格对比 full scan，任何风险等级降低都会失败。当前实跑仅有 `94585/5880000` 次查询进入精确净空距离计算，即 1.61%；672 个栅格单元合并为 45 个同级渲染矩形，即 `45/672`（渲染矩形/原单元），同时保持逐格完全等价。主脚本 SHA-256 为 `b87ff9802e43345a300bca9329572917e9454e8dda62a720ef7b3f471011c4ee`。

### 原生渲染性能与指标边界

保留的同配置基准使用 LuaSTG Sub v0.21.129、原生 arm64 macOS 15.7.3、Apple M4 Max 40 核 GPU、`640x480` 可见窗口和开启的 VSync。四轮都运行 Stage 5 Boss #3、seed `20260730`，记录 1200 个有效样本、0 个无效样本，逐帧渲染，决策间隔 3、观察延迟 5；密集段定义为 `OBJ >= 300`，每轮都有 273 个密集样本，峰值为 421 OBJ。

| 渲染器 / 覆盖层 | 密集段中位 FPS | P10 FPS | 最低 FPS | 报告 SHA-256 |
| --- | ---: | ---: | ---: | --- |
| CPU 软件 / 关闭 | 30.252 | 29.390 | 28.241 | `7cfcd17ca4ac9e86d3815ca9f302a33b4a6bb704f88b78c6e7949db1d1c61a4f` |
| CPU 软件 / 优化覆盖层 | 29.159 | 28.290 | 25.463 | `b8bf286e2bc68ae6319dee67a7cc80140356b33e24101e844577db16def3d9a0` |
| OpenGL GPU/FBO / 关闭 | 59.990 | 59.867 | 58.911 | `74482a01232db58d21fa75dc51f7a9e10b8ff2f3844b3484ade9f9bf9747d126` |
| OpenGL GPU/FBO / 优化覆盖层 | 59.988 | 59.984 | 52.265 | `67f5c5335b317d11584c01e392e91ff1e9b039723498d437a05614fdd43e94b7` |

严格 GPU 加覆盖层击破报告还覆盖 3816 个有效显示样本，峰值 532 OBJ；其中 2844 个密集样本中位为 59.988 FPS、P10 为 59.492 FPS，并且最终达到 `attack_complete`。

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

完成本地游戏复制和 `.venv-win` 安装后，可直接双击 `tools\stg_lab\run-win-boss3.cmd`。它只启动现有文件，不复制、覆盖或安装游戏与 Python 环境。脚本会校验本地桥接器与当前源码一致，默认读取已纳入版本库的 `models\region_dynamics_boss3_v2.json`，设置测试环境变量，以 `setting.vsync=true` 启动 `C:\stg-win-demo\LuaSTGSub.exe`，无连接地等待端口监听，然后运行 Boss #3 可见 MPC 测试。成功仍只接受引擎返回 `attack_complete`。报告写入带时间戳的 `tools\stg_lab\artifacts\engine-mpc-boss3-win-*.json`；默认保留游戏窗口，传入 `-CloseGameWhenDone` 才会在结束后请求关闭。

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
