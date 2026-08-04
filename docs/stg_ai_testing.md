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

### Native replay capture

An attack `reset` or final-stage `reset_stage` may include a portable
`replay_name`. The bridge then owns a private `plus.ReplayFrameWriter`, records
the final THlib `KeyState` after every logical `GetInput()`, and leaves the
global `replayWriter` untouched. This avoids corrupting THlib's private stage
record table during later transitions. The client finalizes the file with:

```json
{"id":5,"command":"save_replay","finish":false,"reason":"attack_complete"}
```

The bridge writes standard STGR v1 data to
`userdata/replay/<setting.mod>/analysis/<replay_name>.rep`, reads it back with
`plus.ReplayManager.ReadReplayInfo`, compares the saved header and serialized
initial state, reads every declared input byte, and requires the final frame to
end exactly at EOF. The result includes the file size, verified frame-byte
count, completion reason, and CRC32. A transient write, read, or verification
failure retains the in-memory writer so the save can be retried. Spell Practice
always uses `group_finish=0`; only a strictly completed final stage may use
`group_finish=1`. Capture is rejected for non-final isolated stages and for
test-only reset overrides such as ghost/collision/protection settings because
those states cannot be reproduced by the native replay alone.
This is structural verification, not a claim that a native replay-menu load has
already reproduced the run. Use a unique name when retaining prior captures;
THlib overwrites an existing file with the same name.

`engine-mpc-play --replay-name NAME` performs this lifecycle automatically and
stores the returned metadata in the JSON report as `native_replay`. Death and
frame-limit runs are saved too, with `finish=false`. A `.rep` contains the
initial serialized game state and input bytes, not MPC observations, forecasts,
or decision reasons; keep the same run's JSON report for analysis. Use
`--record-observations-from-frame 0` when full visible object snapshots are
needed in addition to the normal decision trace.

The final native arm64 macOS headless validation on 2026-08-03 used Okuu Boss
#3, seed `20260730`, five-frame observation delay, a 60-frame horizon, and the
`bullet-group-expert` profile. It reached `attack_complete` after 3363
controlled frames with boss HP 0, player death 0, and a 1.0 shoot-command rate.
The 3891-byte verified replay has 3364 frames including the neutral reset
frame, `group_finish=0`, CRC32 `5f964c53`, and SHA-256
`4d6be63144e7706de04112c2204e2960dffb78fda57cc19a4a0423a2b8aa85d2`.
An independent STGR parser reconstructed all 3364 expected bytes from the JSON
decision holds with zero mismatches; all 3363 controlled frames fired. Its
paired 1121-decision/full-observation report has SHA-256
`20822e937020f0137e3d5e064e9e663f5d9c4134426f2ac698c56db3aa268170`.
Both files are ignored local analysis artifacts rather than source-controlled
fixtures.

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

The current Engine MPC keeps only cross-episode strategy memory that describes
safe-region dynamics learned from online-visible information. For Boss #3 it
takes the median visible nuclear-hazard radius, making
the estimate insensitive to object reordering, newly spawned large outliers,
and small plateau oscillations, and identifies this ordered four-phase cycle:

`expanding -> maximum_hold -> contracting -> minimum_hold`

Region-dynamics memory schema v2 retains the radius cycle period, per-phase
durations, expansion/contraction rates, and the period of the visible lateral
flow. Observation frame indices may be used transiently to measure intervals,
but an absolute `episode_frame` is never a reusable trigger. Future portal
closure is forecast from the currently visible phase and learned
durations/rates, without a fixed frame table, Lua script timer, or hidden
script phase.

The lateral-flow fit groups the highest visible row of collidable
indestructible hazards in consecutive recorded controller-input observations.
It matches at least three visible objects and derives horizontal flow from
their position displacement; it does not consume object `dx`/`vx`, Lua class,
or script-timer fields. Schema v2 stores only the fitted flow period and this
relative rule:

```json
"lateral_flow": {
  "cycle_frames": 360.0,
  "safe_side_rule": "opposite_incoming_lateral_flow"
}
```

At runtime the controller combines the current visible flow direction with
that rule to choose the exterior region that should remain open at the next
expansion. The memory contains no absolute lateral phase, fixed side sequence,
or recorded movement path. The controller projects the currently visible
hazard rows to the start, midpoint, and end of the next expansion event, so it
can move toward an exterior region before its portal closes. This region goal
is separate from the local beam search that avoids individual bullets along
the way; a player-region route is not treated as a projectile trajectory.
Schema-v1 memory remains readable for compatibility, but without lateral-flow
data it cannot enable this next-expansion side prediction.

Per-episode working state associates safe connected components across
observations and tracks the current component, target component, and
dynamically selected `portal`. The same exterior free space retains one
semantic identity across different hazard rows, while portal geometry is
recomputed from the current visible layout. Component, portal, and short action
state is cleared by every `reset()` and is never written to a cross-episode
artifact. A change in phase, navigation mode, current component, target
component, or portal invalidates the old short action commitment. None of the
following is strategy memory:

- an absolute episode-frame trigger;
- a fixed world coordinate or waypoint;
- a fixed action fragment or full-episode action sequence;
- a route that is specific to one recording.

The current `engine-mpc-play` command has no recorded-action loader or playback
branch. Its optional native replay capture is output-only and never supplies an
action or other controller input. Diagnostic reports retain per-frame actions,
coordinates, and absolute frames for failure analysis, but current code cannot
reinterpret either a report or `.rep` as a policy input. Every formal run is
controlled continuously by live MPC from reset.

When `--region-dynamics-memory` is omitted, four stable visible-radius samples
enable an initial `0 / horizon/2 / horizon` row-flow projection without naming
an unobserved plateau. The controller then learns the radius envelope, phase
durations, rates, and cycle from the current episode. A selected exterior
component is retained through brief ambiguous geometry and cleared on reset;
an explicit opposite visible forecast replaces it. If its latest safe
departure enters the visual horizon while the direct route is blocked, actual
collision and ordinary-bullet danger stay ahead of route progress, while route
progress may temporarily outrank the normal forced-region reserve. No scenario,
attack, seed, fixed coordinate, absolute frame, side sequence, or recorded
action participates in these decisions.

Final native arm64 macOS headless validation used implementation
`0c0b25f53ab677500a830d40e0f38377151e75e2b1ed7fcd63ffa621d9c0f268`,
five frames of observation delay, a 60-frame horizon, and no memory option.
Seeds `20260730/31/32` all returned `attack_complete`, reduced HP `6000 -> 0`,
and ended with death 0 in `3327/3334/3331` controlled frames. Their JSON
SHA-256 values are respectively
`a6344535eb1eee7364ac8b4a87dc85b0a02b9d09785a00ee817dc0122ac603ba`,
`2334dd12cd4c9dc51b99be86bdccc7e122f2442473a7763b5372cffd566b0539`,
and `d6d2422c03ebe4f8807e8bce8129e5214e6da800d3a469f8426affe100c3851d`.
Every decision is `live_mpc`, every action fires with spell disabled, and all
three verified native replays are retained as ignored local analysis output.
This is 3/3 for the executed seeds only and remains teacher, not neural-policy,
evidence.

Continuous whole-run evaluation uses the separate `engine-mpc-campaign`
command. It sends one `reset_campaign`, retains one `EngineMPC` instance while
native gameplay advances through Stage 1-5, and clears scene-local geometry,
delayed frames, region/gap state, and committed plans when
`observation.campaign.stage_transition_count` advances. Campaign metadata is
used only for lifecycle boundaries and removed before MPC selection. The CLI
has no region-memory, route, checkpoint, action-prefix, reset-option, or replay
surface. Strict success requires all five ordered stages with active content,
native `campaign_complete`, finite death 0, four intermediate boundary clears,
only `live_mpc` decisions, continuous fire with spell disabled, and null
external-memory fields. THlib campaign replay remains unsupported because each
native stage needs its own serialized initial state and frame record.

Campaign reports record the Python source fingerprint at both ends of the run;
any in-run source drift makes the strict result fail. For terminal diagnosis,
the report keeps a bounded window containing the last 24 raw authority
observations and eight delayed MPC inputs from the final stage. The window is
cleared on every native stage transition and is output evidence only, not
controller memory.

The first native Lunatic diagnostic campaign at seed `20260804` completed
Stage 1 at continuous episode frame 12550, then was hit in Stage 2 at frame
20173 (Stage 2 timer 7624). It used one reset, one MPC instance, continuous
fire, no spell, and null external-memory fields. The last two decisions
forecast collisions at ETA 29 and 22, but a bullet reached the player within
six live frames, identifying delayed stationary-to-moving launch prediction as
the next failure class. `no-memory-lunatic-campaign-gap-commit-v1-seed20260804.json`
is diagnostic evidence only: Python source changed while that run was active,
before the dual-fingerprint rejection rule existed, so it is not a strict
acceptance artifact.

### 2026-08-04 memory-free campaign pause point

This section records the working state when further optimization was paused; it
does not replace the acceptance definition above. Here, memory-free still means
one `reset_campaign`, one live MPC instance, five frames of observation delay,
three-frame action holds, and no region memory, route, checkpoint, action
prefix, replay, or SQLite input. Every action must have `shoot=true` and
`spell=false`. Only all five stages in order, native
`campaign_complete=true`, and finite numeric `death=0` count as a pass.

The current general implementation includes:

- `engine-mpc-campaign` preserves the native Stage 1-5 resources and flow. It
  clears tracks, delayed inputs, region/gap state, and short plans only at real
  stage boundaries; stage metadata is not exposed to dodging decisions.
- Ordinary and region-aware beams retain first-action and spatial-cell
  diversity. The active profile uses `beam_width=512` and `beam_cell_size=8`.
  Constant-acceleration extrapolation is off by default and remains an explicit
  ablation because a local turn does not justify 60 frames of constant
  acceleration.
- Delayed-launch templates are learned only from visible stationary-to-moving
  transitions in the current run. A still-stationary bullet is withdrawn from
  prediction once its learned launch deadline has passed.
- Parallel-wave gaps, connected safe regions, boundary clearance, and committed
  gap plans are revalidated against each new visible geometry. Stage names,
  script timers, fixed frame numbers, and recorded coordinates are not inputs.
- The live runner now feeds every newly matured delayed source frame through
  the trajectory observer while selecting a beam action only once per three
  native frames. Duplicate delay-padding frames are skipped, same-source
  observation and selection share the threat cache, and campaign feeds reset
  at stage boundaries. The focused MPC/play/campaign suite passed 150 tests.

Native-engine evidence at the pause point is below. Longer survival still
counts as failure:

| Target / configuration | End frame | Strict terminal state | Result |
| --- | ---: | --- | --- |
| Okuu Lunatic #4, constant acceleration off | 4295 | `attack_complete`, death 0 | Pass |
| Koishi Normal #1, 512/8 spatial beam | 977 | `player_hit`, death 100 | Fail |
| Koishi Normal #1, corner reserve 96 / weight 0.5 | 1061 | `player_hit`, death 100 | Fail |
| Koishi Normal #1, escape-plan revalidation ablation | 1136 | `player_hit`, death 100 | Fail |
| Stage 1 Normal, earlier memory-free baseline | 12838 | `stage_complete`, death 0 | Pass, historical source |
| Stage 1 Lunatic, edge-gap v2 | 12974 | `stage_complete`, death 0 | Pass, historical source |
| Current Lunatic campaign v4 | 11073 | Stage 1 `player_hit`, death 100 | Fail, zero transitions |

The Okuu #4 report is
`no-memory-okuu4-no-accel-current-seed20260804.json`, SHA-256
`e8f932add4bae7d8fad26f4477dc97433511e82bfc18c4f76ef8dff4e94fdba6`.
The three Koishi report hashes are
`fced5708fbe9ba575eb556f79eafc6401f2671de0e22006e16f607b4e0a1c74e`,
`f872d103e12c9e741a8b3986328f759eb885cb2ac0769454ee042ee7c4ac4bef`,
and `0c550790fff653547097173d4424c9e5ce343ebf821256d124e865643319cfff`.
The campaign v4 report hash is
`4c8f59135aa03b1e793d8c1da0dbfc9a72a39d7e61230d46a3c112d4c2cfe34b`.

There is no whole-campaign pass on the current source. Earlier Normal and
Lunatic campaigns entered Stage 2 after completing Stage 1, then were hit at
total frames 21970 and 20173 respectively. Those runs establish only the older
Stage 1 behavior and cannot be combined with current-source evidence into a
claim that the first two stages pass.

Koishi #1 is a real controller failure, not a mismatch between the report and
the engine result. Six chains of visible, stationary, collidable
`nontjt_enemies` divide the playfield into wedge-shaped connected components,
while two heart-bullet sets move non-uniformly in opposite radial directions.
At source frame 995 the controller had a long collision-free plan. After the
first three-frame action, rolling replanning changed direction at frame 998
although the old remainder still revalidated collision-free and the replacement
predicted a collision. Generic escape commitment extended survival from 1061 to
1136 frames but ended in the lower-right corner, so it is not a solved result.
A 90-frame horizon failed at 1061; stronger center/vertical anchoring reached
only 1068.

Stage 1 also exposes bullets born entirely inside the observation-delay window.
One fatal bullet was absent at source frame 10511 and present by authority frame
10518. Visible history through 10511 was sufficient to infer an approximately
two-frame emission period, 60-unit relative radius, `-21.5 deg/frame` phase
rate, and about 0.37-pixel birth-position error. The worktree contains an
anonymous spawn-family prototype that derives `spawn_forecast_inferred` threats
from newly visible tracks, nearby visible anchors, relative offsets, period,
and phase. It is intended not to use class, timer, image, rotation, or raw
velocity metadata. It has not yet been validated for shuffled or reused IDs,
withdrawal after two missed emissions, nearby multiple emitters, metadata
invariance, or native Stage 1 regression, and must not be described as a
completed Stage 1 fix.

No training, parameter ablation, or LuaSTG server process was left running at
the pause point. When work resumes, the next step is to complete the prototype's
visible-data contract and unit coverage, then independently regress Stage 1,
delayed launch, Koishi #1, and Okuu #4 before restarting continuous Normal or
Lunatic campaigns.

The same implementation was then rerun visibly in the native macOS OpenGL
window at seed `20260730`, still without a memory option. It completed in 3327
controlled frames with `attack_complete`, HP `6000 -> 0`, and death 0. Its 1109
live decisions exactly match the headless run at the same seed. The overlay
reports `enabled=true`, `data_source=controller`, revision 3320, a 60-frame
horizon, and 16/20/8 margins. The report
`engine-mpc-boss3-no-memory-visible-demo-20260804-seed20260730.json` has SHA-256
`963df911ffa73f9509f97e48463ec517166ff8a8c3ffc1d2704db96c6e845bff`;
the verified replay has SHA-256
`9aa6f4afa27306660fe84f0dfafef77eace07ef1b09bb0b2c843c833bc6b76bb`
and CRC32 `c3faa620`. Per-frame overlay rendering measured 27.73 median FPS
overall, 27.35 for `OBJ >= 300`, and 14.89 minimum. Lockstep preserves test
correctness, but this visible run is not a 60-FPS performance result.

`train-region-dynamics` reads source time, visible region radius, and adjacent
visible indestructible positions from recorded controller-input observations.
Time is used only for relative intervals and visible displacement. It rejects
recorded/non-live actions, authority shields, and sources that do not force
`spell=false`, and writes provenance separately from loadable memory:

```bash
.venv/bin/stg-lab train-region-dynamics \
  --input artifacts/engine-mpc-boss3-heldout-v37-d5-memory-no-actions.json \
  --memory-output artifacts/region-dynamics-boss3-v2.json \
  --report-output artifacts/region-dynamics-boss3-v2-training.json
```

The memory SHA-256 is
`dcdcddeeed840d733e144477934f217b21f6795ab9a83790d7f3813d273546f7`.
Its loadable model is limited to radius limits, change rates, ordered phase
durations, radius-cycle length, lateral-flow cycle length, and the relative
safe-side rule above. The separate training report SHA-256 is
`95f4b3e45952f476f430158416d6f13b7617a4a5d25dbe07e522fc0107ac8e99`;
it binds source trace SHA-256
`60c1dd6bb0cfdece73d7170b3c968736479470ba87fed6e96fa68e42290134f5`,
357 radius samples, and 303 lateral-flow samples. The radius fit is `7/28`,
rates are `0.7/0.7`, phase durations are `30/30/30/90`, and the radius cycle is
180 frames. The 360-frame lateral repeat has 201 pairs, correlation 1.0, and
normalized RMSE 0; its 180-frame sign inversion has 252 pairs, correlation
1.0, and normalized RMSE 0. Shifting the entire input timeline does not change
memory or relative sample statistics.

### Parallel-wavefront gap prediction

The live Engine MPC can combine its safe-region forecast with a general gap
predictor for coherent bullet waves. Gap grouping is restricted to
`enemy_bullets`; lasers, enemies, indestructibles, and forced-displacement
geometry still participate in ordinary collision/path checks but can never
define a gap. Moving bullets are clustered by velocity direction and speed,
then split by depth along that direction into distinct wavefronts. Sorting one
wavefront along its perpendicular axis makes each adjacent bullet pair a
candidate corridor. This is inferred from current visible geometry and motion,
not from an attack identifier, script phase, fixed coordinate, or recorded
route.

Corridor width is center-space clearance, not raw center spacing. It subtracts
the two bordering bullet radii and, on both sides, the player radius, a 10-unit
safety reserve, and displacement uncertainty induced by observation delay.
Candidates are resampled at multiple future times through arrival and hold;
reordered boundaries, a future closure, or insufficient lifetime rejects the
corridor. A wavefront coverage threshold rejects isolated bullet pairs that do
not describe a meaningful barrier. Geometry is computed for every opening, but
only the active opening and up to eight region-compatible nearby candidates are
entry-certified. Certification first checks an executable three-frame-block
direct route, then uses a small diverse action beam when another threat blocks
it. Every route is checked against every threat type and must retain at least
the 4-unit emergency margin, raised to the 8-unit forced-region reserve while a
region anchor is active. This permits a necessary short transit through a
lower-risk area; the gap itself still reserves 10 units per side plus delay
uncertainty. One group's opening therefore cannot hide an intersecting wave,
laser, enemy, or forced-region object.

Gap and region anchors coexist. The region anchor describes the globally
preferred connected area or portal, while a selected gap supplies a local
corridor through an approaching wavefront; candidate ordering favors gaps
compatible with the region target. Collision prediction remains ahead of gap
entry. In `enter`, the controller emits the certified route's first action, so
it may deliberately accept lower ordinary clearance without accepting a hit;
the higher region-entry reserve still applies when both anchors are active. In
`hold`, the entire usable interval is a soft target and the controller does not
chase its exact center. If staying put already preserves the ordinary safety
target, gaps remain diagnostic `observe` targets. The active gap identity is
retained across replans and progresses through `enter`, `hold`, and `exit`
rather than switching to the newest opening every decision.

`engine-mpc-play` enables this feature by default. `--gap-prediction` makes the
setting explicit and `--no-gap-prediction` provides a deterministic ablation.
Each JSON decision contains `gap_bullet_group_count`, `gap_corridor_count`,
`gap_selected_center`, `gap_selected_width`,
`gap_selected_lifetime_frames`, and `gap_navigation_mode`; the latter is one of
`inactive`, `observe`, `enter`, `hold`, or `exit`. The report-level
`gap_prediction` object includes `enabled`, detected/selected decision counts,
per-mode counts, `maximum_bullet_group_count`, and
`maximum_corridor_count`.

Three deterministic MPC profiles now model different player capability at
processing bullet groups while holding ordinary movement scoring constant:

| Profile | Perception | Accepted opening | Search capacity |
| --- | --- | --- | --- |
| `bullet-group-novice` | 5+ bullets, 5-degree direction tolerance, 6%/0.15 speed tolerance | 18-unit side reserve, 10 usable units, 24-frame lifetime, 65% coverage | sample every 12 frames, certify 2 entries, detour beam 12 |
| `bullet-group-intermediate` | 4+ bullets, 8-degree direction tolerance, 12%/0.25 speed tolerance | 14-unit side reserve, 6 usable units, 18-frame lifetime, 55% coverage | sample every 9 frames, certify 4 entries, detour beam 24 |
| `bullet-group-expert` | 3+ bullets, 12-degree direction tolerance, 20%/0.35 speed tolerance | 10-unit side reserve, 4 usable units, 12-frame lifetime, 45% coverage | sample every 6 frames, certify 8 entries, detour beam 48 |

These are capability limits rather than random action corruption. Controlled
fixtures establish monotonic behavior: 3/4/5-bullet wavefronts are recognized
by expert only, intermediate plus expert, and all profiles; 20/12/0-degree
direction spreads produce the same ordering; and 32/40/50-unit canonical gaps
are accepted at the same three levels. `engine-mpc-play --profile` selects one
level, while repeated matrix `--profile` options compare them on identical
targets and seeds. This is MPC-teacher behavior, not evidence that the released
neural checkpoint learned these levels.

A current-source native comparison ran all three profiles on Okuu #3 seed
`20260730`. Runner configuration, region-dynamics memory, runtime identity, and
verified runtime source maps are equal; only the 15 documented bullet-group
controller fields differ. All reports bind implementation SHA-256
`394d8298f5b42c6a42d586a75ab908bac0fdb7281d6cd5fd54478a83e048d99b`.

| Profile | Strict outcome | Group behavior | Movement | Report SHA-256 |
| --- | --- | --- | --- | --- |
| novice | `attack_complete`; 3,360 frames; HP `6000 -> 0`; death 0; shoot `3360/3360` | maximum 6 groups, no acceptable corridor, all modes 0 | path 9500.6747; changes 308; reversals 26; sharp turns 122; ABA 6; median hold 9 | `8fb9d002e7b0c8a6e993edd602995cf1d5e5fbcfdd4a457fb73a8bfc37c77850` |
| intermediate | `attack_complete`; 3,360 frames; HP `6000 -> 0`; death 0; shoot `3360/3360` | 42 detected/observe decisions, 0 selected; maxima `30/3` groups/corridors | identical to novice in this episode | `dd73c9fdb36e1e0a34c4dbf4c39ffd1e30749f667953d46eeae6594d6a217c57` |
| expert | `attack_complete`; 3,363 frames; HP `6000 -> 0`; death 0; shoot `3363/3363` | 172 detected, 44 selected; observe/enter/hold/exit `119/44/0/9`; maxima `54/13` | path 9621.0025; changes 316; reversals 26; sharp turns 112; ABA 7; median hold 9 | `3370da8b2debfdb5b77e05fb7f023d3c3181e0e3894a88e0e5dde6ef8eaf0047` |

Intermediate recognition did not force unnecessary motion because ordinary MPC
was already safe when those corridors appeared. Expert entry planning changed
802 emitted actions and 749 directions across the 1,120 decisions shared with
either lower profile. The three strict clears validate graded processing and
non-regression on one seed; they do not calibrate human success rates.

A deterministic four-bullet wavefront regression verifies that the selected
anchor can affect control rather than telemetry only. Starting just outside a
natural 10-unit corridor, gap prediction changes the collision-free first move
from straight down to left toward the opening. A separate blocked-route fixture
requires and finds a safe multi-block detour. These remain synthetic integration
fixtures, not native success-rate results.

A local synthetic performance probe containing 299 bullets, 23 wavefronts, and
276 corridors measures a 2.92 ms median for geometry. Complete-decision medians
are 135.24 ms with gap prediction and 131.66 ms without it, a 3.58 ms delta.
Only the active and up to eight ordered candidates receive entry certification,
so dense geometry no longer runs a Python route search for every corridor.
These figures are not a native episode A/B, real-time throughput, or
success-rate claim.

A preceding same-source, continuous-fire native A/B used Okuu #3 seed `20260730`.
Both reports bind implementation SHA-256
`002d770a1e4d10ad98a2ce00f21796dd7deeddc931b08ab638a3e07e0bbefb86`,
have identical top-level run configuration and runtime source maps, and differ
in controller configuration only at `gap_prediction_enabled`.

| Gap prediction | Strict engine outcome | Gap telemetry | Movement diagnostics | Report SHA-256 |
| --- | --- | --- | --- | --- |
| on | `attack_complete`, `passed=true`; 3,363 frames / 1,121 decisions; Boss HP `6000 -> 0`; death 0; shoot commands `3363/3363` (1.0); predicted-collision movement plans 522 frames | 172 detected; 44 selected; observe/enter/hold/exit `119/44/0/9`; maxima `54/13` | path 9621.0025; 316 direction changes (93.9637/1,000 frames); 26 exact reversals; 112 sharp turns; 7 ABA; hold min/median/mean/max `3/9/10.6088/291` frames | `7dc328637957f0682974d97e0227475bea10f4eb79994334bea9599b76b18ea1` |
| off | `attack_complete`, `passed=true`; 3,360 frames / 1,120 decisions; Boss HP `6000 -> 0`; death 0; shoot commands `3360/3360` (1.0); predicted-collision movement plans 528 frames | disabled; all gap counts zero | path 9500.6747; 308 direction changes (91.6667/1,000 frames); 26 exact reversals; 122 sharp turns; 6 ABA; hold min/median/mean/max `3/9/10.8738/291` frames | `93f27f603bb852021ccab8f62e87285072140a1788cab510a02d69ed51c15e3a` |

Across 1,120 common decisions, direct comparison finds 802 different emitted
actions, including 749 different movement choices, 947 different planned-action
arrays, and 952 different observed decision-boundary positions. The enabled run
has one additional terminal decision; the first action and position divergence
is decision 168. Gap prediction therefore activated and changed native movement
while both sides still strictly completed. Its smoothness figures are
descriptive for this episode, not a general result. Because it recorded no
`hold` decision, this native episode does not evidence sustained corridor
holding; the deterministic regression covers `enter -> hold`. Schema 3 retires
the unsafe-shot metric, so `unsafe_shot_frames=null` is not a zero count. Because
this is one seed and both sides already pass, the pair demonstrates activation
and non-regression, not a success-rate improvement. These
`acceptance_claim=false` live MPC-teacher reports are not learned-policy
evidence.

The predictor is a general visual teacher rule, not spell-specific memory
inserted into the released neural checkpoint. The A/B reports have not yet been
converted into a gap-aware DAgger or demonstration archive, and the published
stream-v1 checkpoint has not been retrained from such an archive.

### Exact MPC prefilter equivalence and performance

`experiments/benchmark_engine_mpc_beam.py` compares the conservative
candidate-AABB threat prefilter with the unfiltered reference beam on recorded
observations from
`engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json` (SHA-256
`e7577aa475ed9a9de6542fedfba8a193dca1b3d8a927e139371e22f41b2d94ef`).
A recorded three-repeat median run produced:

| Source frame | Mode | Threats | Unfiltered median | Prefiltered median | Speedup | Exact beam output |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 995 | local beam | 364 | 0.3653 s | 0.1292 s | 2.83x | equal |
| 1292 | region planning | 193 | 0.3809 s | 0.1209 s | 3.15x | equal |

This is exact field-by-field equality, not only equal selected actions or final
states. Every candidate's action, `collided`, `collision_frames`,
`earliest_collision_frame`, floating-point `minimum_margin`, boundary penalty,
boss-alignment penalty, and complete 20-segment plan are equal. The stateful
frame-1292 replay also matched the unfiltered controller for 21 decisions after
a 60-frame warm-up, including the committed action, planned actions, and
collision fields from the source report. Five-repeat reruns on the same host
have varied within approximately 2.7x-3.4x, so these timings are benchmark
samples rather than a guaranteed lower bound.

### Clearance, direction hysteresis, and grid ablation

Live MPC still replans every three frames. Ordinary bullets use a hard 20-unit
reserve and forced-region passages use a separately scored 8-unit reserve, so a
narrow forced corridor cannot hide poor ordinary-bullet clearance. The 16-unit
danger tier ranks before the 20-unit target. Ordinary clearance keeps earning a
capped soft reward up to 48, and immediate corner clearance has a 48-unit soft
reserve; neither 48-unit preference is an impassable boundary. Beam search
accumulates costs of `3/9/6/0.75` for a direction switch, an additional exact
180-degree reversal, an `A -> B -> A` oscillation, and a same-direction speed
mode change.

A direction is normally held for 12 frames. A near collision, an 8-unit reserve
gain late in the hold, an immediate corner escape, or a deadline-driven
`evacuate` direction with at least three cost units of real boundary/boss-route
progress can release it early. `evacuate` no longer bypasses hysteresis merely
because of its mode name, and neither does `preposition`. A committed plan must
pass the same safety gate and cannot restore a stale direction after a release.
Collision state, first collision, collision-frame count, and the 16/20-unit
reserve tiers always outrank smoothness.

On 32 recorded `hold/preposition/evacuate/settle` samples, region targets
`4/6/8/12` all produced 15 collision frames and all eight crossing samples
remained reachable without a collision. Median clearances were approximately
`4.35/6.33/8.29/12.18`. Target 12 reduced crossing progress and increased final
anchor L1 error from approximately 0.87 at target 8 to 1.25, so 8 is used for
narrow forced-region passages.

An earlier v46 same-seed native comparison quantified the smoothing change.
From the earlier strict run to that smoothed run, adjacent decision direction
changes fell `835 -> 487`, exact reversals `65 -> 20`, `A -> B -> A` changes
`154 -> 20`, and consecutive nonzero displacement angles over 90 degrees
`236 -> 154`. Median predicted minimum clearance rose `1.43 -> 8.27`, decisions
at or below clearance 4 fell `815 -> 215`, and predicted collision frames fell
`2346 -> 1009`. These are 60-frame rolling predictions, not actual hits; both
runs ended with `death=0`. This evidence predates the current 12-frame
hysteresis and evacuation-release fix, so it remains a historical comparison
rather than validation of the final frozen source.

`experiments/benchmark_engine_mpc_grid.py` compares 8/12/16-unit time-layered
grids against continuous beam plans on the same recorded five-frame-delay
observations at source frames `488/1292/2102/2801/3695`. Grid layers remain
three frames apart, but each receives every logical-frame swept-threat occupancy
from its interval. Within each observation all planners are truncated to one
common shortest action horizon, then recomputed with Engine MPC's
per-logical-frame continuous circle geometry. This compares complete planners,
not only rasterization, and the grid's own levels never certify safety:

| Planner | Predicted collision plans | Collision-frame rate | Median minimum clearance | Direction changes per 60 frames | Mean time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continuous beam 20/8 | 3/5 | 9.67% | -4.47 | 6.20 | 0.120 s |
| Grid 8, center sample | 2/5 | 9.00% | 1.90 | 7.80 | 1.326 s |
| Grid 12, center sample | 3/5 | 10.00% | -3.04 | 9.20 | 0.698 s |
| Grid 16, center sample | 4/5 | 45.33% | -26.29 | 10.60 | 0.460 s |

Whole-cell half-diagonal inflation also did not win: collision-frame rates for
8/12/16-unit grids were 11.67%/15.67%/52.33%. The 8-unit center grid's two-frame
advantage (`27/300` versus `29/300`) is too small to establish a closed-loop
episode win; it takes about 11 times the mean planning time and changes direction
more often. Grids remain useful for visualization, connectivity, and global
region hints, but final local movement keeps per-frame continuous geometry,
hysteresis, and the committed-plan safety gate. The working report SHA-256 is
`ffc69082eefd2501d473981689eddd4fcfb67a0152a5723b9f099a46c7bbd901`.
This is an open-loop five-observation complete-planner ablation, not an
`attack_complete` success rate.

### 2026-08-04 Boss #3 human-behavior calibration

The `humanlike` profile first preserves strict native completion, then reduces
movement differences from a successful human replay. The controller still uses
only five-frame-delayed visible information, online motion estimates, and
episode-local state. None of the three final runs supplied region-dynamics
memory, a route, an action prefix, or a checkpoint. `slot2.rep` and `slot3.rep`
are analysis inputs only and are never read by the controller.

`slot2` hit a forced region at frame 2579 with 1190.5 boss HP remaining. Its
region clearance fell `9.49 -> 4.64 -> -1.10` while bullet clearance was still
84.58. `slot3` strictly completed at frame 3236. The successful human stays
focused and holds directions longer; region-side changes are already close to
the AI, locating the main mismatch in local replanning jitter rather than cycle
or side-selection logic.

| Executed-replay metric | successful `slot3` | AI v12 | AI v14 |
| --- | ---: | ---: | ---: |
| Frames / path | 3236 / 4939.79 | 3281 / 6366.70 | 3291 / 6178.90 |
| Moving / focused | 64.43% / 86.31% | 73.06% / 73.64% | 68.92% / 72.17% |
| Bottom-clamped / mean Y | 55.72% / -195.22 | 22.25% / -192.90 | 34.82% / -195.55 |
| Moving turns / `>=90` / `>90` | 168 / 6 / 0 | 298 / 172 / 69 | 278 / 164 / 79 |
| Exact reversals / focus-mode changes | 0 / 72 | 22 / 251 | 24 / 251 |
| All-clearance P10 / region P10 | 12.17 / 12.96 | 11.17 / 11.12 | 11.56 / 11.94 |

v14 makes `bottom_anchor_enabled` apply consistently during region navigation.
Only when the player is already in the target `exterior:left/right` component
and navigation mode is `settle` does the region anchor return to the effective
floor. `preposition`, `evacuate`, forced crossings, certified gap entry, and all
collision/deadline priorities are unchanged. On the representative seed this
reduced path by 2.95%, moving time by 4.14 percentage points, increased
bottom-clamped time by 12.57 points, and fixed v12's seed-`20260732` region hit
at frame 2630.

The final source fingerprint is
`0a67effb8e54225e0bcc7209902cacd7068f17dc386c68bbde2394a22aac9a1e`.
Three native arm64 macOS headless runs used a 60-frame horizon, five frames of
observation delay, continuous fire, forced spell-off, and no external region
memory. Each met the sole strict success criterion:

| Seed | Replay frames / path | Terminal state | Replay CRC32 | Report SHA-256 |
| ---: | ---: | --- | --- | --- |
| 20260730 | 3291 / 6178.90 | `attack_complete`, HP 0, death 0 | `be0090d2` | `a3a69309078e83ff2646ee69fb55b3f6c8e4dc308992230e58cc8a5a98967181` |
| 20260731 | 3278 / 6223.50 | `attack_complete`, HP 0, death 0 | `f44782c7` | `3c06e2dc8a4ef0c04656777e77613506c04702c022b508becbe5744f32ce7987` |
| 20260732 | 3301 / 6582.41 | `attack_complete`, HP 0, death 0 | `3dd61d1c` | `7478675195b2ac1a6f66fb0ae36835a83225fc5f0e6b76346c8361e3cbec03bc` |

All three replays report `saved=true`, `verified=true`, and 100% firing. Their
SHA-256 values are respectively
`f0c07634ac2aeb41a8fd49133fd0e9e2bcfe85097829bf338945edf40e913fb5`,
`b98bd4ef36bc94aecab82151fdf0d677bd9a8160119e3ac6fada9ea201006fa2`,
and `cb1fab256c947d70d38f5a170c8501911eb6237d9f4fe583bb7c3b97af0fb7cc`.
This is 3/3 only among executed seeds, not a statistical success claim.

Two apparently smoother ablations were strictly rejected. v13 estimated region
deadlines from focused-speed reachability and hit a region at frame 1917 with
2558 HP remaining. v15 inserted a safe intermediate direction before obtuse
turns and hit a region at frame 2270 with 1869 HP remaining. The latter code was
removed from the final source: being no worse over the current 60-frame forecast
does not prove that a three-frame perturbation reaches the same future connected
component. Focus-deadline and neutral-beat experiment switches remain disabled.
Longer survival never substitutes for attack completion.

v14 improves path, moving time, bottom dwell, and three-seed robustness, but
obtuse turns, exact reversals, and focus-mode changes remain far from the human
trace. It therefore does not claim human-level behavior. A later learned-policy
iteration should capture synchronized visible observations while replaying
successful human inputs and train action persistence in the recurrent model.
Coordinate-only analysis cannot reconstruct what was visible at each decision
and must not be injected into MPC as recorded route memory.

### Strict native Boss #3 results

The retained CrossOver evidence uses three fresh original LuaSTG Sub processes
under CrossOver 26.3 with DXVK. All passed the sole native success condition.
v40 and v41 are five-frame-delay
held-out runs; v42 is a separate zero-delay regression. All three used
`authority_state_shield=false`, forced `spell=false`, and the v2 memory above.
They reduced boss HP from 6000 to 0 without a player death, terminated at
episode frame 3816 with `termination_reason=attack_complete`, and recorded zero
unsafe-shot frames:

| Run | Seed | Observation delay | HP | Player death | Final frame | Unsafe-shot frames | Artifact SHA-256 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json` | 20260730 | 5 | `6000 -> 0` | 0 | 3816 | 0 | `e7577aa475ed9a9de6542fedfba8a193dca1b3d8a927e139371e22f41b2d94ef` |
| `engine-mpc-boss3-heldout-v41-seed20260731-d5-region-dynamics-v2.json` | 20260731 | 5 | `6000 -> 0` | 0 | 3816 | 0 | `5cedf76c0b17028b4239f480dbd146a54cb92ead17dc898f8fc8d6fb52e981fa` |
| `engine-mpc-boss3-regression-v42-seed20260732-d0-region-dynamics-v2.json` | 20260732 | 0 | `6000 -> 0` | 0 | 3816 | 0 | `a45084f331ecd82d0fecff636bf1921c9b547f41f82a1db6846ff76d15d7e37f` |

All three reports bind implementation SHA-256
`5a81172add05549fdf1ea6d65272d26dd08afc3de6c289ab3124e9f7b2e69613`.
The two delay-5 held-out attempts passed, and the zero-delay regression passed;
this is not a statistical success-rate claim or evidence for Boss #4.

The final sorting and committed-action safety fix was then run from Spell
Practice reset on the native macOS DirectX-free engine for two five-frame-delay
seeds. Both strictly defeated the attack at frame 3816:

| Run | Seed | Terminal result | HP / death | Changes / reversals / ABA | Median clearance | SHA-256 |
| --- | ---: | --- | --- | --- | ---: | --- |
| `engine-mpc-boss3-safety-sort-v46-seed20260730.json` | 20260730 | `attack_complete`, `passed=true` | `6000 -> 0` / 0 | 487 / 20 / 20 | 8.27 | `e1a4df0fd857b3aa7cc52e6bcd6416a856f74228f34e07c26b4a19882cbe6d39` |
| `engine-mpc-boss3-safety-sort-v47-seed20260731.json` | 20260731 | `attack_complete`, `passed=true` | `6000 -> 0` / 0 | 502 / 10 / 30 | 8.31 | `d69d6b7fa2b88637590f5bbe5f605e1768bd7dccbd47307bdfa25fd651d50faf` |

Both have `unsafe_shot_frames=0`, force `spell=false`, and bind implementation
SHA-256 `8422915228d7b867ae01ffae0e2d0ae85d7ab8d1aac71e3a383c6b8a6e6d2044`.
This is 2/2 among executed attempts, not an extrapolation to unexecuted seeds
or evidence for Boss #4. Their `acceptance_claim=false` correctly scopes them
as strict live-teacher engine evidence rather than deployable-model acceptance.
They supersede v44/v45, which predate the final region-sort and committed-action
safety fixes.

The live bridge and MPC now also cover straight and bent lasers. Straight
tapers and bent polyline segments use a conservative 16-32 px circle cover
whose radius includes the exact segment half-step; successive sample positions
capture rotation and changing length rather than incorrectly reusing only the
laser origin displacement. A native headless Okuu #2 run of implementation
`fab98499b72c55fb92ceb5586b58be5093df9e42b9755631e008633ceaf96f95` at zero
observation delay strictly completed at episode frame 3036 with HP `4300 -> 0`,
`death=0`, and `unsafe_shot_frames=0`, while observing peaks of 552 bullets and
40 lasers. Planner threat count was 302 at the median and 697 at the maximum.
The report SHA-256 is
`70d5afc69faa6fbae55cef0bdd678f6fe090d977608ec4d16ec281377f85dfd2`.
This is one zero-delay MPC-teacher geometry result, not delay-5 or learned-policy
evidence and not a success-rate claim.

A separate full-engine native macOS OpenGL run kept the optimized F7 overlay
enabled and met the same strict success definition. Report
`engine-mpc-boss3-gpu-overlay-strict-seed20260730.json` (SHA-256
`ec3f758a8a5135b33e139076bdecdb050bf1117f1e13622679a91c40e8110def`)
has `terminated=true`, `termination_reason=attack_complete`, and `passed=true`
at episode frame 3816. Boss HP changed from 6000 to 0, player `death=0`,
`unsafe_shot_frames=0`, and `spell=false` was forced. It is single-run live MPC
evidence with `acceptance_claim=false`, not a deployable-policy acceptance or a
Boss #4 result. This run predates the current overlay implementation; it is
historical full-engine integration evidence rather than an execution proof for
the current overlay SHA-256.

The earlier standalone v2 SQLite single-route and route-library artifacts and
their hashes remain reproducible historical benchmarks. Because they contain
recorded routes, they do not represent current Engine MPC strategy memory and
do not establish live-engine generalization.

## 7. Portable Full Engine, High-speed Test Target, and Rendering

The current LuaSTG-Sub source tree no longer has the old
complete-engine DirectX dependency. The complete executable supports Windows,
macOS, and Linux through SDL2 window/input/audio plus an OpenGL 4.1 GPU
sprite/FBO renderer, or null graphics/window/audio for uncapped headless runs.
Both paths execute the real Lua scripts, resource system, collision logic, and
test bridge. The OpenGL path batches legacy vertices and indices, caches source
textures on the GPU, renders to FBO-backed targets, and presents the main target
without a per-frame CPU-canvas upload. Supported builds compile and link no
DirectX code.

The remaining renderer boundaries are explicit: arbitrary legacy HLSL effects
are not translated to GLSL and currently fall back to visible RenderTarget
pass-through composition; model drawing and the modern MeshRenderer path are
not yet feature-equivalent. The CrossOver/DXVK results above concern the
original Windows executable and remain separate evidence. They are not a
runtime requirement for the current portable source build.

The full macOS presets configure, build, run the DirectX audit, and execute the
corresponding tests:

```bash
LUA_STG_SUB_ROOT=/path/to/LuaSTG-Sub
cd "$LUA_STG_SUB_ROOT"
cmake --workflow --preset macos-headless-release
cmake --workflow --preset macos-opengl-release
```

The repository also retains a smaller, isolated `LuaSTGPortableTest` target.
It reuses the engine's `XCollision` circle/rotated-ellipse implementation and
supports an uncapped no-video headless mode plus a simplified SDL2 software
collision/risk view. This target is useful for algorithm tests but is not the
complete macOS game runtime.

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
  --scenario pulse --frames 180 --render-every 6 \
  --screenshot build/portable-native/portable/pulse-macos.bmp
SDL_VIDEODRIVER=dummy \
build/portable-native/portable/LuaSTGPortableTest \
  --scenario orbit --frames 700 --render-every 6 \
  --screenshot build/portable-native/portable/orbit-dummy.bmp
```

The clean arm64 macOS build passed its CTest (`1/1`). The measured 100,000-frame
headless pulse run reached approximately 1,511,690 logical frames/second, and
the orbit probe with graded-risk analysis reached approximately 496,620
frames/second. The native macOS SDL2 software capture
`build/portable-native/portable/pulse-macos.bmp` and SDL dummy software capture
`build/portable-native/portable/orbit-dummy.bmp` are both nonempty `480x560`
32-bit BMPs. This target is a partial collision and algorithm-development
module. It does not execute arbitrary SR Lua and cannot replace full-engine
`attack_complete` acceptance.

`game/plugins/SafetyZoneVisualizer` adds an in-engine F7 overlay, or it can be
enabled at launch with `SR_SAFETY_ZONE_OVERLAY=1`. The 16-unit lattice now
samples each cell center instead of inflating the player radius to certify the
whole cell. During `engine-mpc-play`, each decision publishes the exact
`PredictedThreat` records used by EngineMPC, profile margins, adjusted player
bounds and radius, region-navigation state, and per-frame region-radius
envelope. The bridge publishes this once per three-frame action hold and the
overlay directly evaluates stationary samples at future frames 1 through the
configured horizon. It therefore cannot drift to a different velocity or Boss
#3 phase model. Runtime telemetry reports `data_source=controller` and the
consumed revision.

The strict shortfall boundaries are red for any signed clearance `<= 0`,
orange for ordinary clearance `0 < margin < 16`, yellow for ordinary clearance
`16 <= margin < 20`, and green at `margin >= 20`. An active forced region uses
its separate 8-unit target; `0 < region_margin < 8` is yellow and `>= 8`
satisfies that layer. Values exactly equal to 16, 20, or 8 have no shortfall.
Only worst projected clearance controls the grade; threat count and local
density never upgrade it. The 4-unit emergency margin releases movement
hysteresis rather than defining another color. Clearance above 20 still earns
a continuous MPC preference reward up to 48, which four colors cannot encode.

Published ellipses and rectangles already use EngineMPC's conservative
`max(a,b)` circle, and straight/bent lasers already use its 16--32-unit
segment-circle cover. Enemy bullets and indestructibles retain the full motion
horizon; enemies, `GROUP_NONTJT`, and lasers move for at most nine future
frames, and radius trends stop after six. The controller's actual region anchor
selects the separate 8-unit layer; without an anchor, indestructibles use the
ordinary 16/20 margins. Its learned Boss #3 7/28, 0.7-per-frame envelope is
published directly, including the conservative contraction hold.

Without controller state, the plugin retains a local read-only fallback. It
mirrors bridge visibility and the same delayed displacement, circle, laser,
motion-horizon, and radius-horizon rules, but it does not claim the controller's
learned region phase. F7 mid-episode needs a short local-history warm-up only
in this fallback mode.

The standalone test includes an independent boundary oracle, density
counterexamples, the 1..60 frame contract, the 9/60-frame motion split, the
six-frame radius cap, delayed displacement, laser-circle coverage, controller
state ingestion, region topology, and indexed/full-scan parity. Its
deterministic 350-bullet field uses 51,257 exact clearance calculations out of
14,112,000 possible calls (0.36%) and coalesces 672 cells into 179 render
rectangles. `SafetyZoneVisualizer.lua` SHA-256 is
`8f3d3941c5fa644b5391902935dacad5df3b964222d918ccf6300538ede09178`.
The overlay remains a
read-only static-field diagnostic. It does not alter input, RNG, collision, AI
actions, or memory, and it does not reproduce moving-path, boundary, Boss,
portal, gap-entry, or direction-hysteresis scores.

### Native renderer performance and metric boundaries

The current controller-fed overlay was validated on native arm64 macOS with a
visible `640x480` OpenGL window, VSync, per-frame rendering, Okuu Stage 5 Boss
#3, seed `20260730`, and the `bullet-group-expert` profile. The engine returned
`attack_complete` after 3363 controlled frames, Boss HP changed from 6000 to 0,
the player had zero deaths, and the shoot-command rate was 1.0. The final
engine telemetry proves the enabled overlay consumed controller revision 3356
with horizon 60 and margins 16/20/8. Across 3364 valid display samples, the
`OBJ >= 300` subset has 2540 samples, median 59.989 FPS, P10 59.765 FPS, and
minimum 51.364 FPS; peak object count is 504. The report is
`tools/stg_lab/artifacts/engine-mpc-boss3-controller-overlay-rendered-validation.json`,
SHA-256
`399a8f6358703e03bd7bd0ba4d3dbf81cdc65a916685e6d3b88dd4520294b4ac`.
Artifacts remain ignored working data rather than repository source.

The retained renderer benchmark predates the current 60-frame MPC-aligned
overlay and is historical performance evidence for the earlier 24-frame
whole-cell/density revision. It uses LuaSTG Sub v0.21.129 on native arm64
macOS 15.7.3, an Apple M4 Max with a 40-core GPU, a visible `640x480` window,
and VSync enabled. Every 1200-sample run uses Stage 5 Boss #3, seed `20260730`,
`render=true`, `render_every=1`, decision interval 3, observation delay 5, and
the same live MPC configuration. Dense samples are frames with `OBJ >= 300`;
all four runs have 273 dense samples, zero invalid samples, and peak at 421
objects.

| Renderer / overlay | Dense median FPS | Dense P10 FPS | Dense minimum FPS | Artifact SHA-256 |
| --- | ---: | ---: | ---: | --- |
| CPU software / off | 30.252 | 29.390 | 28.241 | `7cfcd17ca4ac9e86d3815ca9f302a33b4a6bb704f88b78c6e7949db1d1c61a4f` |
| CPU software / optimized overlay | 29.159 | 28.290 | 25.463 | `b8bf286e2bc68ae6319dee67a7cc80140356b33e24101e844577db16def3d9a0` |
| OpenGL GPU/FBO / off | 59.990 | 59.867 | 58.911 | `74482a01232db58d21fa75dc51f7a9e10b8ff2f3844b3484ade9f9bf9747d126` |
| OpenGL GPU/FBO / optimized overlay | 59.988 | 59.984 | 52.265 | `67f5c5335b317d11584c01e392e91ff1e9b039723498d437a05614fdd43e94b7` |

The historical strict GPU-plus-overlay defeat report above extends this to
3816 valid display samples, peak 532 objects, and 2844 dense samples, with
dense median 59.988 FPS and P10 59.492 FPS while still reaching
`attack_complete`. It used the earlier overlay revision.

`render_performance` is reporting and diagnosis only. Its source is
`lstg.GetFPS`, a 60-native-frame display average, plus `lstg.GetnObj`; the
report explicitly sets `reporting_only_not_controller_input=true`. These fields
do not enter the AI observation, MPC selection, or region-dynamics memory.
Display FPS describes the native render/Present cadence and is affected by
VSync. Lockstep throughput instead means logical frames or decisions advanced
per wall-clock second under Python/bridge control. It must be reported
separately and must not be called display FPS or AI inference FPS.

### Windows visible lockstep and flicker

Native LuaSTG clears its swap-chain target before calling the Lua `RenderFunc`
and presents it afterward. Returning early from `RenderFunc` does not cancel
that present, so suppressing duplicate logical frames submits black buffers
while lockstep waits for Python. The bridge now redraws the current logical
state on every visible native render pass. `--render-every` remains an accepted
protocol compatibility hint, but no longer suppresses Lua drawing; headless
mode still suppresses scene drawing. Visible Windows runs should use
`setting.vsync=true`, `SR_TEST_HEADLESS=0`, `SR_TEST_LOCKSTEP=1`, and
`--render --render-every 1`. Vsync reduces tearing but cannot repair the black
frames produced by an older bridge. A copied Windows game directory must be
updated with the current `compat/testing/bridge.lua`, and LuaSTG must be
restarted after the file is replaced.

## 8. Model and Training

### Native stream policy v1: strict status

The retained checkpoint is
[`policy_native_stream_v1.pt`](../tools/stg_lab/models/policy_native_stream_v1.pt),
SHA-256
`829eebe53c886e5ba53f577542938b904aad740f3a5bf04b49d61e73ab61557d`.
The detailed model card is
[`policy_native_stream_v1.md`](../tools/stg_lab/models/policy_native_stream_v1.md).
Its 25-episode, 21,916-decision training archive has SHA-256
`1bc03ce647d34c1fb3f77ba751d50ca7b602a9e16189025819e3cf89a423c384`.

This model uses a 192-unit GRU and dual semantic visual encoders. It has no
scenario memory or proficiency input (`memory_size=0`,
`proficiency_size=0`). Its inputs are the latest delayed global/local semantic
frames, the current visible player pose, and recurrent hidden state. Attack
identity, absolute time, script phase, recorded routes, waypoints, teacher
risk, and external memory are not model inputs.

The training run uses episode-stateful TBPTT with 32-decision chunks,
episode-balanced optimization, class-balance power 0.75, movement-onset weight
4.0, and direction-change weight 1.5. Each complete training episode has equal
optimizer weight regardless of length. Onsets and changes come only from
adjacent teacher actions inside one episode. Episode IDs 11-15 remain intact
for validation. The run uses `restore_best_validation=false`: internal
validation loss was lowest at epoch 8 (2.3124849571), while the retained epoch
40 loss was 2.5737202304. The retained candidate is selected by strict native
outcomes, not by presenting epoch 40 as validation-best.

DAgger has two declared archive contracts. Legacy teacher-labelled archives
use `teacher_action` as the target for every student-visited decision.
Direct-corrective archives preserve the student's executed actions and the
complete recurrent context; an intervened executed label is the teacher
correction, and `supervision_mask` can restrict action loss to only the
intervention/correction points. Older unmasked archives remain fully supervised
when merged with masked archives. Only a strictly completed native episode is
admitted, but its assisted completion is training-data evidence, not evidence
that the student policy completed the attack.

`contextualize-demos` optionally attaches an identity-only one-hot token to
each complete episode from its strict provenance manifest. The vocabulary
contains an unknown token and registered attack/stage identities and is stored
in the checkpoint by `--scenario-vocabulary-manifest`; it contains no phase,
coordinate, action, waypoint, or route data. This lets network weights learn
identity-dependent behavior instead of adding handwritten strategy branches.
With `--previous-action-conditioning`, a separate 18-way one-hot token contains
only the previous motor action actually executed. Direct-corrective DAgger
preserves that executed-action stream, and live inference commits it only after
the engine advances. It never contains a future action, teacher proposal,
position, frame, phase, waypoint, or route; it is motor feedback for learned
temporal dynamics.
The experiment did not pass release gates: a recurrent-256 candidate cleared
Koishi #1 at 915 frames but failed Okuu #3 at 522; 20 more epochs reached 642.
One strictly completed Okuu #3 DAgger episode required 1100/1272 teacher
interventions; a subsequent 10-epoch continuation failed its pure Okuu #3 run
at 514 frames. The shipped checkpoint therefore remains `memory_size=0`.

New corrective collections at Okuu #3 seeds 20260813 and 20260815 reached
`attack_complete` at frame 3,815 with death 0, but required 489/1,272 (38.44%)
and 467/1,272 (36.71%) teacher interventions. Their reports explicitly set
`pure_policy=false`, `pure_policy_success=false`, and
`pure_policy_validation_eligible=false`. The v13 all-label aggregate contains
35 episodes and 33,866 recurrent decisions. The critical-intervention aggregate
preserves the same 33,866 decisions, has 30,376 supervised labels overall, and
retains all recurrent context for six Okuu #3 corrective episodes while
supervising only their 4,142 intervention points.

The latest pure, unshielded Okuu #3 candidates produced these strict results:

| Candidate | Seed | Frames | HP observed | Reason / death | Result |
| --- | ---: | ---: | --- | --- | --- |
| v10 unweighted | 20260812 | 392 | `6000 -> 5524.5` | `player_hit` / 100 | fail |
| v11 corrected unique | 20260812 | 764 | `6000 -> 4924.5` | `player_hit` / 100 | fail |
| v11 corrections repeated x4 | 20260812 | 431 | `6000 -> 5453` | `player_hit` / 100 | fail |
| v12 Okuu specialist, final epoch 80 | 20260812 | 726 | `6000 -> 5086` | `player_hit` / 100 | fail |
| v12 general, final epoch 30 | 20260812 | 414 | `6000 -> 5478` | `player_hit` / 100 | fail |
| v13 all-label, validation-best epoch 3 | 20260816 | 413 | `6000 -> 5481` | `player_hit` / 100 | fail |

The v13 all-label checkpoint selected epoch 3 at validation loss
1.9728760589. Neither that offline improvement nor either assisted clear passes
the pure native release gate, so no v10-v13 candidate is publishable.

These result classes are intentionally disjoint:

| Class | Controller authority | Evidence boundary |
| --- | --- | --- |
| pure GRU | delayed visual stream and GRU hidden state | eligible learned-policy result |
| visible safety | separate visible-only local forecast may override GRU | diagnostic hybrid; intervention count must be disclosed |
| DAgger teacher | exact-state MPC labels all student-visited states and may execute | training only; not student success |
| Engine MPC | exact object state and optional teacher-only region memory | planner/teacher result; not checkpoint output |

For an attack, strict success is
`terminated=true`, `termination_reason=attack_complete`, and an explicit
finite numeric, non-Boolean `final_player.death=0`. For a complete stage, the
reason must instead be `stage_complete`. Frame/time limits, partial damage,
long survival, completion after death, ghost/protected state, and missing or
non-numeric death evidence fail. The final matrix used history 1, five-frame
observation delay, expert execution, and no visible-safety shield:

| Target | Seed | Frames | HP observed | Reason / death | Strict result |
| --- | ---: | ---: | --- | --- | --- |
| Koishi #1 | 20260738 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260739 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260740 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Yamame #3 | 20260738 | 491 | `1800 -> 1306.25` | `player_hit` / 100 | fail |
| Satori #5 | 20260738 | 384 | `3200 -> 2883` | `player_hit` / 100 | fail |
| Okuu #3 | 20260740 | 754 | `6000 -> 5046` | `player_hit` / 100 | fail |
| Okuu #4 | 20260740 | 1065 | `3500 -> 2629.5` | `player_hit` / 100 | fail |
| Okuu EX #2 | 20260740 | 446 | `3333 -> 2834` | `player_hit` / 100 | fail |
| Orin #4 | 20260740 | 261 | `4500 -> 4407` | `player_hit` / 100 | fail |
| Stage 1 Normal | 20260740 | 543 | across-wave HP is not comparable | `player_hit` / 100 | fail |

The retained checkpoint was then replayed against the final source fingerprint
`fa891752547e10f478fbec6b4f85349e4c43061fb3788bea9014ac1f9337ac56`,
with checkpoint path/SHA-256 verification and the same pure-policy settings:

| Current-source target | Seed | Frames | HP observed | Reason / death | Strict result |
| --- | ---: | ---: | --- | --- | --- |
| Koishi #1 | 20260738 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260739 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260740 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1, held out | 314159265 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Yamame #3, held out | 314159265 | 493 | `1800 -> 1300.25` | `player_hit` / 100 | fail |
| Okuu #4, held out | 314159265 | 1065 | `3500 -> 2629.5` | `player_hit` / 100 | fail |
| Stage 1 Normal, held out | 314159265 | 543 | across-wave HP is not comparable | `player_hit` / 100 | fail |

The current release therefore proves strict pure completion only for Koishi
#1: all three known seeds and one independently selected held-out seed passed,
while every current-source cross-target probe failed. The three original runs
made 105.96-107.06 direction changes per 1,000 frames,
10-11 exact reversals, 7 ABA changes, and had a 6-frame median hold. Three
near-deterministic known-seed runs plus one held-out run on the same card do not
establish a general success rate or human equivalence. The strict Koishi DAgger
teacher reference made 97.46
changes per 1,000 frames, no ABA changes, and had a 9-frame median hold; the
checkpoint is still more restless. Later Yamame #1 and Satori #1 DAgger
episodes have no pure checkpoint run and are not claimed as model successes.
Human-like general play remains unmet, as do cross-card reliability and
full-stage completion.

Proficiency is a seeded runtime transform, not network conditioning. Expert
uses 0 added delay, a 3-frame minimum hold, and 0% suboptimal choices;
intermediate uses 3/6/1%; novice uses 6/9/4%. With visible safety disabled,
intermediate Koishi #1 seed 20260741 failed at 593 frames (`700 -> 291.5`,
death 100), and novice failed at 478 (`700 -> 342.5`, death 100). The expert
result is 3/3 on different seeds; this is not a controlled proficiency study.

Offline validation-best and specialist selection did not replace the strict
gate. Yamame #3 specialist v1 validation-best epoch 1 failed at 598 frames;
its final epoch 40 failed at 495. Recurrent-size-256 specialist v2
validation-best epoch 19 failed at 530 and 502 frames on seeds 20260740 and
20260743. All four ended in `player_hit`, death 100. A 29-episode,
26,234-decision v5 candidate trained from scratch for 60 epochs lowered its
validation loss to 2.2334204706, yet failed Koishi #1 at frame 391 on seeds
20260742 and 20260744 (`700 -> 592.5`, death 100). Fifteen more low-rate epochs
reached only frame 436. Its Okuu #3 visible-safety hybrid died at frame 902
after 66 interventions, shorter than the scratch model's 1,146-frame pure run.
Specialist, validation-best, expanded-scratch, and hybrid candidates were
therefore rejected as deployment evidence.

Full-stage status is also negative for the learned policy. The final pure GRU
remained stationary and failed Stage 1 Normal at 543 frames. Separately, an
MPC-teacher Stage 1 attempt at seed 20260731 failed at a 7,199-frame time
limit with death 0; a seed-20260732 teacher rerun reached `stage_complete` at
12,453 frames with death 0. The latter is teacher evidence, not model success,
and the long episode was excluded from the spell pool after it over-weighted
stationary decisions.

### Legacy standalone visible-v2 model

The legacy standalone reference policy uses a dual visual encoder:

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

## 9. Acceptance Criteria

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

A live Engine MPC attack attempt is successful only when the final report has
`terminated=true`, exactly `termination_reason=attack_complete`, and an
explicit finite numeric, non-Boolean `final_player.death=0`. Engine-confirmed
boss defeat or full attack endurance may satisfy that condition. A complete
stage instead requires `termination_reason=stage_complete` with the same
zero-death evidence. Reaching `max_frames`, surviving longer, reducing boss HP
without attack completion, completing after death, or omitting valid death
evidence does not count as success. Every action keeps `spell=false`; shooting
stays enabled on every active logical frame. Firing does not change player
movement or collision, so tying it to forecast clearance only lowers damage and
extends attack exposure. Legacy shoot-risk and minimum-margin options remain
parseable for command/report compatibility but are reporting-only and cannot
disable firing. Live runner reports now use schema 3 with `continuous_fire` and
explicit `shoot_command_frames`/`shoot_command_rate` fields. The misleading
schema-2 `unsafe_shot_frames` value remains only as `null` with a deprecation
marker; movement-plan collision forecasts are separate diagnostics. The current
command has no recorded-action prefix loader;
externally prefix-assisted evidence would be ineligible.

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

## 10. Tool Layout

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

## 11. Environment

The base simulator and planner require only Python and NumPy. Neural training uses an isolated Python 3.11 or 3.12 environment with PyTorch. The system Python 3.14 is not used for model training because supported binary wheels are not guaranteed.
