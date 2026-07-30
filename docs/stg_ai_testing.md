# STG Automated Testing and AI Training

## 1. Purpose

This project provides a reproducible test and training stack for LuaSTG bullet-hell content. It has two authoritative boundaries:

1. The LuaSTG bridge runs the original Lua scripts, object pool, collision system, lasers, player code, and random-number generator. It is the source of truth for game compatibility and final evaluation.
2. The standalone Python environment runs deterministic approximations without LuaSTG. It is used for fast unit tests, safety planning, imitation learning, and algorithm iteration.

The standalone environment must never be presented as proof that an original
spell works. In this repository, `engine_verified` is scoped to the strict v2
live content regression. Real-engine AI survival remains a separate evaluation
and cannot be inferred from either standalone survival or content regression.

## 2. Architecture

```text
LuaSTG / SR                          Cross-platform STG Lab
+---------------------------+       +------------------------------+
| bridge.lua                | JSONL | protocol client              |
| - deterministic reset     +<----->+ reset / step / observation   |
| - virtual key state       |       +------------------------------+
| - exact object telemetry  |                    |
| - optional no-render      |                    v
+---------------------------+       +------------------------------+
                                    | vision + state estimation    |
Standalone simulator                | risk field + region planner  |
+---------------------------+       | phase + topology memory      |
| deterministic scenarios  +------>+ policy + Engine MPC          |
| Stage 5 #3 approximation |       +------------------------------+
| Stage 5 #4 approximation |
+---------------------------+
```

The initial bridge uses LuaSocket and cjson already bundled with LuaSTG Sub. A later `simulation_only` engine target may replace the rendered executable, but it must keep the same protocol.

## 3. Protocol

Messages are newline-delimited JSON objects. Every request has an integer `id`; the response repeats it.

### Catalog and process identity

```json
{"id":1,"command":"catalog"}
```

`catalog` exposes the runtime Spell Practice manifest generated from the same
boss card tables as the menu. Its top-level `attacks` entries contain
`scenario`, one-based `attack`, raw `card_index`, and `label`; the bundled SR
manifest has 53 attacks in 22 scenarios. Test clients enumerate this live
array and do not maintain a second card-index table.

```json
{"id":2,"command":"ping"}
```

Bridge protocol v2 adds runtime provenance to `ping`. In addition to
protocol/frame/capability metadata, it returns the caller-supplied
`SR_TEST_SESSION_ID`, a bridge-generated `process_nonce`, and
`runtime_identity`. The runtime identity contains the real Win32 OS PID, the
engine executable path and CRC32, and CRC32 values computed inside LuaSTG for
the 18 SR Lua files actually loaded by the mod: its entry points, compatibility
modules, player modules, background modules, and test bridge.

The Python runner independently computes CRC32 and SHA-256 for those 18
files and rejects a runtime/local mismatch. Reports also carry an
`implementation_sha256` fingerprint over all shipped Python modules. Two-run
acceptance requires distinct session IDs, process nonces, and positive OS
PIDs, plus matching executable/runtime/local-source fingerprints. Bridge
protocol and live-engine report schema are v2; the embedded catalog schema
remains v1.
An installed Python wheel uses `STG_LAB_MOD_ROOT` for the local SR source
root, or discovers it by walking upward from the current directory.

### Reset

```json
{"id":3,"command":"reset","scenario":"okuu:Lunatic","attack":3,"seed":42,"player":"reimu_player","options":{"player_protect_frames":900}}
```

The response contains the first observation, scenario identity, effective
seed, player identity, and raw card index.
`player_protect_frames` raises the newly initialized player's protection
counter without replacing engine collision behavior. It is used by the
catalog-wide content regression, together with 99 lives, to keep the test
focused on loading and advancing attack content.

### Step

```json
{"id":4,"command":"step","action":{"move_x":-1,"move_y":1,"slow":true,"shoot":true,"spell":false},"repeat":3}
```

An observation contains the player, visible threats, enemies, lasers, frame
number, score/lives/bombs, and death/completion events. Python computes a
canonical state hash from that response. Debug telemetry may contain exact
object fields, but human-vision policies are not allowed to consume it.

`enemy_bullets`, `enemies`, `nontjt_enemies`, `indestructibles`, and `lasers`
are JSON arrays even when empty, and each `counts` field equals its array
length. Catalog arrays, nested scenario attack arrays, and bent-laser point
arrays have the same contract; `{}` is invalid where `[]` is required. Lasers
also remain in their owning group array, so `lasers` intentionally duplicates
those records.

Visibility filtering does not control episode completion. The no-enemy stop
condition queries the raw `GROUP_ENEMY` and `GROUP_NONTJT` object pools, so a
temporarily hidden or off-playfield boss does not cause a false
`attack_complete`.

When `SR_TEST_MODE` is enabled, `SR_TEST_STARTUP_ACCEPT_TIMEOUT` defaults to 30
seconds (otherwise zero). The bridge can therefore accept the first client and
activate lockstep/headless interception before the first Render/Present path.

## 4. Human-Vision Contract

The deployable policy receives only information derivable from the rendered playfield:

- a full-playfield semantic or RGB view for global pattern awareness;
- a high-resolution crop around the player for precise dodging;
- four historical frames for velocity and rotation estimation;
- configurable observation delay, normally five frames;
- configurable action hold, normally three frames;
- optional visual noise and missed detection of very small peripheral bullets.

The policy does not receive exact script timers, future projectile positions, Lua class names, hidden random state, or the teacher's risk map. Scenario identity is allowed because a human player can know the current stage and attack. Timing memories should be anchored to visible cues, not an undisclosed engine frame number.

The visible-v2 implementation enforces this boundary through `DelayedVision`:
neural inference receives only delayed global/local arrays. Cold-start history
is blank rather than filled by copying the reset observation backward in time.
Horizontal and vertical motion channels are computed from displacement between
visible observations; simulator or engine `vx`/`vy` is not exposed. Occupancy
uses `log1p` density and overlapping motion uses a deterministic weighted mean,
so raster results do not depend on object order.

`canonical_train_visible_v2.npz` and
`canonical_heldout_visible_v2.npz` are rebuilt by replaying grouped teacher
episodes through exactly this path. The manifest records four-frame history,
five-frame delay, six channels, a `48x56` global view, and a `40x40` local view
with 72-world-unit half extents. The six channels are occupancy, visible
horizontal displacement, visible vertical displacement, warning geometry,
player position, and playfield boundary.

## 5. Risk Field and Region Motion

For candidate player position `p` and future offset `h`, every threat produces a signed clearance:

```text
clearance_i(p, h) = distance(p, predicted_shape_i(h))
                    - player_radius
                    - uncertainty_margin_i(h)
```

Risk combines future collision, swept speed, local density, prediction
uncertainty, and boundary cost. A fast sparse bullet can therefore be more
dangerous than a dense group moving away from the player.

The implementation classifies each sampled risk layer into graded danger
cells. It then performs dynamic programming from grid cells in one time layer
to reachable cells in the next, checking the swept grid segment at focused or
unfocused speed. Each cell retains the best lexicographic cost:

1. minimize the maximum danger level crossed;
2. minimize integrated risk;
3. minimize travel distance.

Connected components are computed for region labels and diagnostics, but they
are not the search nodes: there is no explicit `(component, time)` graph. This
is still a player-region route rather than a projectile route. The graded-cell
objective can choose a short lower-grade danger corridor when remaining in the
current area has a worse future peak cost.

The Python Stage 5 Boss #3 scenario models moving, periodically expanding
hazards and rotating fans from the latter portion of the original spell. It
does not model the complete original card and must not be presented as a
frame-equivalent reproduction. Boss #4 is represented by a rotating emitter
and sweep that rewards advance positioning before a reactive controller could
keep up.

The approximation parameters are tied to the bundled Lua: player-center
margins are 8 left/right, 16 bottom, and 32 top; Reimu's hit radius is 0.5;
Boss #3 Lunatic uses all ten sources, expands to radius 28, and has no synthetic
warning channel; Boss #4 waits 120 frames for the star and uses two nine-shot
fans. Forecast intervals retain first, largest, and last geometry for threats
that are born and destroyed between sample endpoints.

## 6. Engine MPC Phase and Topology Memory

The current Engine MPC keeps only long-lived state that describes a pattern
learned from online-visible information and the changing topology of safe
space. For Boss #3 it takes the median visible nuclear-hazard radius, making
the estimate insensitive to object reordering, newly spawned large outliers,
and small plateau oscillations, and identifies this ordered four-phase cycle:

`expanding -> maximum_hold -> contracting -> minimum_hold`

Phase memory learns and retains only the cycle period, per-phase durations,
and expansion/contraction rates. Observation frame indices may be used
transiently to measure time between visible transitions, but an absolute
`episode_frame` is never a reusable trigger. Future portal closure is forecast
from the currently visible phase and learned durations/rates, without a fixed
frame table, Lua script timer, or hidden script phase.

Topology memory associates safe connected components across observations and
tracks the current component, target component, and dynamically selected
`portal`. The same exterior free space retains one semantic identity across
different hazard rows, while the portal geometry is recomputed from the
current visible layout. A change in phase, navigation mode, current component,
target component, or portal invalidates the old short action commitment. None
of the following is strategy memory:

- an absolute episode-frame trigger;
- a fixed world coordinate or waypoint;
- a fixed action fragment or full-episode action sequence;
- a route that is specific to one recording.

A recorded action prefix exists only for exact experiment reproduction and for
skipping an already checked opening while debugging a later segment. It is not
an Engine MPC memory or policy input. The loader verifies
`scenario/attack/seed/player` identity, no authority-state shield,
`spell=false`, action fields and frame spans, and contiguous decisions starting
at the reset frame. A prefix-assisted result establishes only closed-loop
behavior after handoff; it cannot be counted as no-prefix generalization. Such
evidence must run the current controller continuously from reset to completion.

The earlier standalone v2 SQLite single-route and route-library artifacts and
their hashes remain reproducible historical benchmarks. Because they contain
recorded routes, they do not represent current Engine MPC strategy memory and
do not establish live-engine generalization.

## 7. Model and Training

The reference policy uses a dual visual encoder:

- a global CNN processes the entire playfield;
- a local CNN processes the player crop;
- a GRU provides working memory across delayed observations;
- an optional four-value context/memory input is supported by the checkpoint configuration;
- the policy head selects one of 18 movement actions: nine directions times focused/unfocused speed.

Training proceeds in this order:

1. Generate demonstrations with the exact-state risk planner.
2. Rebuild training and held-out data through the visible-v2 contract.
3. Train the visual policy with behavior cloning and auxiliary survival-risk prediction.
4. Evaluate the raw, unshielded policy and, for legacy v2 reproducibility only, the visible-cue route controllers on held-out seeds.
5. Use the authority-state safety shield only as a diagnostic ablation; it cannot contribute to final acceptance.

Demonstrations are split by episode/seed, not by adjacent windows. Only the
last decision in a delayed history window is supervised; earlier frames are
temporal context. Class weights are calculated from training groups only.

The frozen v2 CNN+GRU checkpoint is `policy_visible_v2.pt`, SHA-256
`f9815eb6fd4e0e5e35e856c836567f078f36ca1d010a4052c36ea31b2ff550e6`.
Its 40-epoch run used 5,208 visible-v2 training samples (archive SHA-256
`52be736f0743379c4fb15321b6761ff3300ed2062acf76201460492ddc91f53d`)
and ended at 98.3146% internal validation action accuracy, 0.0196861 training
loss, 0.0568566 validation loss, and 0.0316795 risk MAE. The training metrics
JSON SHA-256 is
`28c83647f4a798cb4feab93396ea3e404882469c5b20faf2a970cdac4c782446`.

The 868-sample held-out archive SHA-256 is
`7318ed50f5f1bba81c48af87bbcfc69f67bf5714f5a47193fd55b11e759f41d3`,
and the visible dataset manifest SHA-256 is
`e380f4cea85373ebaeb71f574985c6cb083a9498ed0aa1d773c6d64fd7bc14ca`.
Strict replay obtains 796/868 overall held-out agreement (91.7051%): 369/400
for #3 (92.25%) and 427/468 for #4 (91.2393%). The agreement report SHA-256 is
`1208ed6dc57cae5a47a5ba6f8d7bcbbd90cab27af06dd3ffe7dc3cbb1f95abcd`.

## 8. Acceptance Criteria

### Standalone acceptance

- Fixed scenario and seed produce identical state hashes across repeated runs.
- Geometry, swept collision, risk grading, connectivity, path reconstruction, protocol, and memory tests all pass.
- Planner survival is at least 95% over 100 held-out seeds for both Stage 5 #3 and #4 approximations.
- With five-frame observation delay and three-frame action hold, an unshielded
  deployable controller selected only from online visible information survives
  at least 90% over the same held-out set.
- Visual-policy agreement with planner actions is at least 85% on a held-out demonstration set.
- For the legacy scripted #4 memory benchmark, the second attempt either survives or reduces maximum route risk by at least 30% after a first-attempt failure.

The canonical standalone windows are 600 frames for #3 (three expansion
cycles) and 700 frames for #4 (star spawn, more than one complete orbit, and
sustained fan fire). They are survival tests, not boss defeats: the standalone
backend does not simulate player-shot damage. Full 61/69-second timer endurance
is an optional stress test. Original #4 is nearly deterministic, so documented
small phase perturbations provide held-out robustness variants; reports include
the unique trajectory-hash count so duplicate runs cannot inflate the sample.

### Final standalone v2 evidence (legacy route benchmark)

The strict compiler now binds every file-based artifact to the current Python
implementation fingerprint. Visual reports are rejected unless they are
unshielded, use no authority state, select memory from an online visible cue,
and identify a checkpoint whose checksum matches the file. Agreement likewise
binds the visible-v2 held-out dataset and `policy_visible_v2.pt`; planner and
visual reports must use identical distinct seed sets.

The accepted artifacts use implementation SHA-256
`ba7f4d2ee9fe5bf232f264a180a51280657f007629a54fd36c3f4884ca966cb9`
and identical planner/visual seed sets 5001-5100 (100 distinct seeds):

| Evidence | Recomputed result | SHA-256 |
| --- | --- | --- |
| Exact-state planner, #3 | 100/100 survived 600 frames; 100 unique terminal hashes | `f10ce59b18329a6d71ec0d5ba20c9bc9fd4bb21d7a0af416d5dcfe40cf78bd6d` |
| Exact-state planner, #4 | 100/100 survived 700 frames; 100 unique terminal hashes | `6ad7b42b6977b6e8af3f854e294bbe86497290d53ad7a1c5ad636c1750fe92c8` |
| Legacy delayed-visible route controller, #3 | 100/100 survived; `shield: false`, no authority state, online cue | `f29d3cf92864b1af2c8846ea53669d8769a7d4e74c6da9c2caffa85b41589069` |
| Legacy delayed-visible route library, #4 | 98/100 survived; `shield: false`, no authority state, online cue | `004e4696766f5c4a0fa1a682a21aef72787dedef98426ad04749ed10a6ca00aa` |
| Legacy #4 memory benchmark | first attempt died at frame 340; memory 2 was selected at control frame 138 from delayed source frame 133; second attempt survived 700 frames | `b57f15e26d8ee49de6895b5145f52489f8da562e307605adb481ba31ebf44b42` |
| Per-frame determinism | fresh #3/#4 runs matched all 601/701 state hashes and actions | `dbc0cdc1b50f9e75315672580c6e4c1aecb0b670700d7a02cb45c3df50be12e4` |
| Strict standalone report | `passed: true`, `issues: []` | `7035a896eb96f9633ca5515314d39edfd763ddcd7cdfaad673f5b7d3f595d061` |

The legacy read-only memory database SHA-256 is
`e774d3148ba0bd0cc89d1e8f9d68db8e3bf1612a1b56fd2b61b14940d321b584`.
The #3 route SHA-256 is
`d492d790cfb5310cb6f519a80bfefe8750a0d0f6dfcbd5271475ad785a76bf17`,
and the #4 five-route library SHA-256 is
`6a1c47342f0f7d1190b810e96cfacf104b76ee111dfabd6041321b273f20873a`.
The memory criterion passed through second-attempt survival; both recorded
peak-risk values are zero, so no numeric risk-improvement claim is made.

This historical compilation establishes the documented standalone thresholds
only. The planner is an exact-state teacher, while route evaluation uses
delayed visible cues and the legacy route library; its actions are not
neural-checkpoint outputs and this is not the current Engine MPC memory design.
The Python scenarios remain approximations and these survival results are
neither original-card defeats nor live-engine AI results.

### Live Engine MPC outcome criterion

A live Engine MPC attempt is successful only when the final report has both
`terminated=true` and `termination_reason=attack_complete`. Engine-confirmed
boss defeat or full attack endurance may satisfy that condition. Reaching
`max_frames`, surviving longer, or reducing boss HP without attack completion
does not count as success. Every action keeps `spell=false`; shooting is enabled
only when predicted safety margin meets the configured threshold and is
disabled during danger. Prefix-assisted reproductions are reported separately
and never included in a no-prefix success rate.

### Engine acceptance

- The live catalog contains exactly 53 registered attacks in 22 scenarios, and
  every entry can reset and advance for at least 300 logical frames without a
  Lua error.
- The default regression uses `step_batch=1`, focused neutral movement, no
  shooting, 99 lives, and `player_protect_frames=frames_per_attack+600`.
- Every attack exposes a boss/enemy plus an active hazard. A hazard may be an
  enemy bullet, indestructible, laser, or an additional collidable enemy object
  used by the script as attack content.
- Equal seed and equal actions produce equal canonical observation hashes for
  every logical frame in two fresh runs.
- Bridge-disabled gameplay has no changed input, rendering, RNG, or stage behavior.
- The 53 attack windows complete without process crashes, NaN state,
  object-pool exhaustion, or protocol desynchronization. Real-engine AI
  survival is outside this compiler's scope.

The strict `engine-accept` compiler accepts only two individually passing
protocol/report-schema-v2 reports with nonempty, distinct session IDs and
process nonces and distinct positive OS PIDs. Each report must have the current
implementation fingerprint, a valid executable CRC32, runtime Lua CRCs that
match local files, local source SHA-256 values, and a complete retained catalog
in original order. The pair must use equal executable/runtime/local-source
fingerprints and identical configurations. Exact 53/22 counts, catalog-order
seeds, one request per logical frame, `frames+1` valid non-static hashes,
active-content peak counts, no early termination, empty errors, and identical
per-frame hash arrays remain mandatory. A single `engine-test` report always
records `engine_verified: false`; only the passing combined report may set it
true. The combined artifact also records canonical SHA-256 digests of both
input reports.

Engine acceptance requires a working Windows/MSVC or compatible LuaSTG runtime
and is reported separately from standalone results.

On the current host, CrossOver 26.3 with DXVK starts the real executable,
loads the complete SR resources/scripts and bridge, and opens the TCP listener.
The wined3d backend fails framebuffer creation with
`GL_INVALID_FRAMEBUFFER_OPERATION (0x506)`. DXMT reaches the Lscreen render
target but fails `IDXGISurface` acquisition with `E_NOINTERFACE`. These backend
findings are not themselves acceptance evidence.

The final protocol-v2 DXVK runs used distinct sessions `engine-v2-a` and
`engine-v2-b`, distinct process nonces, and Win32 PIDs 212 and 204. Both used
executable CRC32 `8844e525`; all 18 runtime Lua CRC32 values matched the local
files, and both 18-entry local SHA-256 maps matched the current tree. Each
passed all 53 attacks across 22 scenarios, retained the reset plus 300
logical-frame hashes per attack, and reported no errors. Their file SHA-256 values are
`ac7996a2ee92417e08deda8ff5e86d3a0937278f7bb15a772e53089275a0abeb` and
`8d692aa4a4f79ed6de0d701d628ebbbbc64cce624f016deb12d198dec6e5b257`.
The combined report matched all 53 attacks and 15,953 per-frame hash positions,
sets `passed: true` and `engine_verified: true`, and has SHA-256
`a2a7cca87e6e416c43b963483e450b5a1e000cdfcdbd32eadbdb16d5e19ea1d2`.
Protocol-v1 reports remain invalid. This live claim is scoped to headless Spell
Practice content regression; it is not rendered-frame comparison, a full-stage
clear, or proof of AI survival inside the real engine.

## 9. Tool Layout

```text
tools/stg_lab/
  pyproject.toml
  src/stg_lab/
    protocol.py       shared messages and action encoding
    sim.py            deterministic simulation primitives
    scenarios.py      Stage 5 #3/#4 approximations
    planning.py       risk field and time-expanded planning
    memory.py         legacy SQLite episodic-memory benchmark
    route_memory.py   legacy visible-cue route controller
    route_benchmark.py legacy single/multi-route evaluation
    vision.py         delayed global/local observations
    policy.py         PyTorch policy
    training.py       demonstrations, training, evaluation
    engine.py         JSONL client for the live LuaSTG bridge
    engine_benchmark.py live 53-attack content regression
    engine_acceptance.py strict two-process report comparison
    provenance.py     implementation and file fingerprints
    cli.py            test/train/evaluate/engine-test/engine-accept/accept commands
  tests/
compat/testing/
  bridge.lua          real-engine protocol adapter
  PROTOCOL.md         complete wire and environment contract
```

## 10. Environment

The base simulator and planner require only Python and NumPy. Neural training uses an isolated Python 3.11 or 3.12 environment with PyTorch. The system Python 3.14 is not used for model training because supported binary wheels are not guaranteed.
