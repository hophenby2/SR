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

The current `engine-mpc-play` command has no recorded-action loader, CLI option,
or replay branch. Diagnostic reports retain per-frame actions, coordinates, and
absolute frames for failure analysis, but current code cannot reinterpret a
report as a policy input. Every formal run is controlled continuously by live
MPC from reset.

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

Live MPC still replans every three frames. Its ordinary clearance target is now
20 instead of 12, and the forced-region target is 8 instead of 1. Beam search
accumulates costs of `3/9/6/0.75` for a direction switch, an additional exact
180-degree reversal, an `A -> B -> A` oscillation, and a same-direction speed
mode change. Clearance beyond the target receives a capped reward. A direction
is normally held for nine frames, but the hold is released for a collision in
the next 12 frames, incumbent clearance at most 4, a clearance gain of at least
6, or a true deadline-driven `evacuate`. `preposition` no longer bypasses the
hold, and a previously committed plan must pass the same gate. Collision state,
first collision, collision-frame count, and clearance shortfall always outrank
smoothness. A committed action cannot override a different direction that has
already passed an emergency or clearance-gain release inside the hold window.

On 32 recorded `hold/preposition/evacuate/settle` samples, region targets
`4/6/8/12` all produced 15 collision frames and all eight crossing samples
remained reachable without a collision. Median clearances were approximately
`4.35/6.33/8.29/12.18`. Target 12 reduced crossing progress and increased final
anchor L1 error from approximately 0.87 at target 8 to 1.25, so 8 is used for
narrow forced-region passages.

Two same-seed native closed-loop runs both strictly defeated the attack. From
the earlier strict run to the final smoothed run, adjacent decision direction
changes fell `835 -> 487`, exact reversals `65 -> 20`, `A -> B -> A` changes
`154 -> 20`, and consecutive nonzero displacement angles over 90 degrees
`236 -> 154`. Median predicted minimum clearance rose `1.43 -> 8.27`, decisions
at or below clearance 4 fell `815 -> 215`, and predicted collision frames fell
`2346 -> 1009`. These are 60-frame rolling predictions, not actual hits; both
runs ended with `death=0`.

`experiments/benchmark_engine_mpc_grid.py` compares 8/12/16-unit time-layered
grids against continuous beam plans on the same recorded five-frame-delay
observations at source frames `488/1292/2102/2801/3695`. Every output plan is
then recomputed with Engine MPC's per-logical-frame continuous circle geometry;
the grid's own levels are not accepted as proof of safety. The final v46 result
is:

| Planner | Predicted collision plans | Collision-frame rate | Median minimum clearance | Direction changes per 60 frames | Mean time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continuous beam 20/8 | 3/5 | 9.09% | -4.87 | 5.66 | 0.103 s |
| Grid 8, center sample | 3/5 | 11.00% | -8.51 | 7.40 | 0.638 s |
| Grid 12, center sample | 3/5 | 10.67% | -5.49 | 9.40 | 0.248 s |
| Grid 16, center sample | 4/5 | 47.67% | -27.67 | 10.80 | 0.085 s |

Whole-cell half-diagonal inflation also did not win: collision-frame rates for
8/12/16-unit grids were 14.33%/17.33%/52.00%. Of the 33 collision frames in
the 8-unit center variant, 22 occurred between sampled layers and 11 occurred
on sampled layers, showing combined temporal, spatial-quantization, and risk
aggregation failures. Grids remain useful for visualization, connectivity,
and global-region hints, but they do not replace continuous beam validation.
The final report SHA-256 is
`a05c36395bc13c6bc829b5694d1b556bdf9be9fb720d554f0bfaa641bc6c67d9`.
This is an open-loop plan ablation on fixed observations, not an
`attack_complete` success rate.

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

The current `/Users/happyelements/LuaSTG-Sub` source no longer has the old
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
cd /Users/happyelements/LuaSTG-Sub
cmake --workflow --preset macos-headless-release
cmake --workflow --preset macos-opengl-release
```

The repository also retains a smaller, isolated `LuaSTGPortableTest` target.
It reuses the engine's `XCollision` circle/rotated-ellipse implementation and
supports an uncapped no-video headless mode plus a simplified SDL2 software
collision/risk view. This target is useful for algorithm tests but is not the
complete macOS game runtime.

```bash
cd /Users/happyelements/LuaSTG-Sub
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
enabled at launch with `SR_SAFETY_ZONE_OVERLAY=1`. It grades safe, caution,
danger, and collision cells from current colliders and linear projections every
three frames from 0 through 24. A 16-unit cell is classified as a whole by
inflating the player radius by its half diagonal. Red is signed clearance at
most 0; orange is clearance at most 12 or at least five simultaneous nearby
threats within 36; yellow is clearance at most 24 or at least three. Clearance
takes the worst future layer, while density is counted in each layer before
taking its maximum, so threats arriving at different times are not combined
into a fictitious crowd. Circles, rotated ellipses, rectangles, and straight
lasers are supported. Straight lasers are evaluated as tapered polygons from
`l1`, `l2`, `l3`, and `w`, including THlib laser objects whose generic ellipse
fields `a=b=0`; curved-laser interiors are not rasterized.

The conservative spatial index registers only cells inside
classification-relevant projected bounds. The standalone test compares every
optimized cell with the full scan, covering ellipses, rotated rectangles,
tapered lasers, fast projections, cell-corner collisions, time-separated
density, boundary cases, and a 350-bullet field; any lower risk level fails.
It required 34,548 exact clearance calculations out of 2,116,800 (1.63%) and
coalesced 672 cells into 45 render rectangles (`45/672`, rectangles/cells)
while preserving exact per-cell parity. The implementation SHA-256 is
`a39c7a1ac49e8a6214c1af2f1df6d6ae72ea113ac7bca6bc195c83a7bcf871be`.
The overlay reads geometry only and does not alter input, RNG, collision, or AI
memory.

### Native renderer performance and metric boundaries

The retained renderer benchmark uses LuaSTG Sub v0.21.129 on native arm64
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

A live Engine MPC attempt is successful only when the final report has both
`terminated=true` and `termination_reason=attack_complete`. Engine-confirmed
boss defeat or full attack endurance may satisfy that condition. Reaching
`max_frames`, surviving longer, or reducing boss HP without attack completion
does not count as success. Every action keeps `spell=false`; shooting is enabled
only when predicted safety margin meets the configured threshold and is
disabled during danger. The current command has no recorded-action prefix
loader; externally prefix-assisted evidence would be ineligible.

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
