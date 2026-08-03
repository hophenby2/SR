# STG Lab

Standalone deterministic testing, region-risk planning, external episodic memory, and visual-policy training for SR and other LuaSTG bullet-hell projects.

The Python simulator is an algorithm-development environment. Final game claims must be verified through `compat/testing/bridge.lua` against the original LuaSTG scripts.

The current native-controller acceptance contract is narrower than the older
standalone route benchmarks retained below. Cross-episode strategy memory may
describe only safe-region dynamics; recorded actions, fixed coordinates,
absolute-frame triggers, waypoints, and full-episode routes are forbidden.
Native attack success requires `terminated=true`, exactly
`termination_reason=attack_complete`, and an explicit finite, non-Boolean
numeric `final_player.death=0`. Full-stage success uses `stage_complete` with
the same zero-death evidence.

Canonical standalone survival windows are 600 frames for Stage 5 Boss #3 and
700 frames for #4. They exercise the requested mechanics but do not claim a
boss defeat; player-shot damage is outside the standalone approximation. The
Boss #3 Python scenario approximates only the navigation mechanics of the
latter part of the original spell. It is not a full-script or frame-equivalent
model of that spell.

Set up the isolated Python 3.12 environment from this directory:

```bash
uv sync --python 3.12 --extra train
```

## Commands

`test` runs the complete pytest suite by default. Arguments following `--` are
passed to pytest unchanged:

```bash
uv run stg-lab test
uv run stg-lab test -- -k vision
```

The planner benchmark is an explicit mode and is not a replacement for the
unit suite. Its defaults are deliberately a smoke profile: one seed per
scenario, at most 600 frames, and a coarse `48 / 6 / 20` forecast horizon,
sample interval, and cell size. Every report marks this as `run_kind: smoke`
and `acceptance_claim: false`.

```bash
uv run stg-lab test --planner --scenario all \
  --output artifacts/planner.json
```

For a full-duration, higher-resolution multi-seed benchmark, request every
part explicitly. Planner jobs are independent processes; no environment
closure is sent between workers.

```bash
uv run stg-lab test --planner --scenario all --episodes 100 \
  --full-duration --workers 8 \
  --planner-horizon 120 --planner-sample-every 4 --planner-cell-size 8 \
  --output artifacts/planner-full.json
```

The JSON includes elapsed time, every episode, per-scenario and overall
survival rates, and the count of unique terminal state hashes. It remains a
standalone approximation report rather than an engine acceptance claim.
Planning is lexicographic dynamic programming over cells in successive graded
risk-grid layers: peak danger level, accumulated risk, then travel distance.
Connected components are diagnostic region labels; the planner does not build
or search an explicit `(component, time)` graph.

`train` can consume a reproducible demonstration archive directly:

```bash
uv run stg-lab train \
  --demos artifacts/demonstrations.npz \
  --checkpoint artifacts/policy.pt \
  --metrics artifacts/training_metrics.json
```

Without `--demos`, demonstrations are collected from the requested standalone
scenarios before training. `--save-demos` preserves the generated dataset.
Episode IDs are renumbered across scenarios before the grouped train/validation
split, so frames from different runs cannot leak across that boundary. The
default command is a 600-frame pipeline smoke run:

```bash
uv run stg-lab train --scenario all \
  --save-demos artifacts/demonstrations.npz
```

Use `--full-duration --episodes N` for a training corpus intended for measured
evaluation. Teacher actions use the held-action collision shield by default;
`--no-teacher-shield` disables it. Action-class balancing is also enabled by
default and can be disabled with `--no-class-balance`.

Evaluate the trained policy on held-out deterministic seeds. Evaluation uses
the same one-seed, 600-frame smoke defaults. The authority-state collision
shield is disabled by default. `--shield` enables it only for diagnostics;
shielded results are ineligible for the final deployable-controller gate.

```bash
uv run stg-lab evaluate --checkpoint artifacts/policy.pt \
  --scenario all --output artifacts/evaluation-smoke.json
uv run stg-lab evaluate --checkpoint artifacts/policy.pt \
  --scenario all --episodes 100 --full-duration --no-shield \
  --output artifacts/evaluation-raw.json
```

CPU evaluation can split the seed set across processes. Each worker loads the
checkpoint once and evaluates its assigned seeds serially; output seed order is
unchanged. Parallel evaluation does not support `--teacher-metrics`, and MPS or
CUDA evaluation remains single-process. Canonical windows differ, so run the
two scenarios separately:

```bash
uv run stg-lab evaluate --checkpoint artifacts/policy.pt \
  --scenario stage5_boss3 --episodes 100 --seed 3001 --duration-frames 600 \
  --action-hold-frames 3 \
  --device cpu --workers 4 --output artifacts/visual-boss3.json
uv run stg-lab evaluate --checkpoint artifacts/policy.pt \
  --scenario stage5_boss4 --episodes 100 --seed 3001 --duration-frames 700 \
  --action-hold-frames 3 \
  --device cpu --workers 4 --output artifacts/visual-boss4.json
```

Policy rollout does not execute the exact-state planner unless
`--teacher-metrics` is supplied. That option attaches action-agreement and risk
metrics but is substantially slower. Observation timing and raster shape are
configurable with `--decision-interval`, `--observation-delay`,
`--vision-history`, `--global-size`, `--local-size`, and `--local-extent`.

## Visible-v2 data and policy

The final visual contract is rebuilt by
`experiments/rebuild_visible_datasets.py`. It replays episode-grouped teacher
actions through the current `DelayedVision` implementation and writes:

- `artifacts/canonical_train_visible_v2.npz`
- `artifacts/canonical_heldout_visible_v2.npz`
- `artifacts/visible_dataset_v2_manifest.json`

```bash
uv run python experiments/rebuild_visible_datasets.py
uv run stg-lab train \
  --demos artifacts/canonical_train_visible_v2.npz \
  --checkpoint artifacts/policy_visible_v2.pt \
  --metrics artifacts/training_visible_v2.json
```

Cold-start history is blank rather than backfilled from the reset frame.
Motion channels come from displacement between visible observations, never
from simulator `vx`/`vy`. The standard view has four history frames, five
frames of observation delay, six semantic channels, a `48x56` global raster,
and a `40x40` player-local raster spanning 72 world units in each direction.

The frozen v2 CNN+GRU checkpoint is `artifacts/policy_visible_v2.pt` (SHA-256
`f9815eb6fd4e0e5e35e856c836567f078f36ca1d010a4052c36ea31b2ff550e6`).
It was trained for 40 epochs on 5,208 samples in
`canonical_train_visible_v2.npz` (SHA-256
`52be736f0743379c4fb15321b6761ff3300ed2062acf76201460492ddc91f53d`).
The final internal validation action accuracy is 98.3146%, with training loss
0.0196861, validation loss 0.0568566, and risk MAE 0.0316795. The training
metrics JSON SHA-256 is
`28c83647f4a798cb4feab93396ea3e404882469c5b20faf2a970cdac4c782446`.

The held-out archive has 868 samples (SHA-256
`7318ed50f5f1bba81c48af87bbcfc69f67bf5714f5a47193fd55b11e759f41d3`),
and the dataset manifest SHA-256 is
`e380f4cea85373ebaeb71f574985c6cb083a9498ed0aa1d773c6d64fd7bc14ca`.
Strict checkpoint replay agrees with 796/868 held-out teacher actions
(91.7051%): #3 is 369/400 (92.25%) and #4 is 427/468 (91.2393%). The agreement
report SHA-256 is
`1208ed6dc57cae5a47a5ba6f8d7bcbbd90cab27af06dd3ffe7dc3cbb1f95abcd`.

## Native streaming policy v1

The retained native checkpoint is
[`models/policy_native_stream_v1.pt`](models/policy_native_stream_v1.pt),
SHA-256
`829eebe53c886e5ba53f577542938b904aad740f3a5bf04b49d61e73ab61557d`.
Its complete model card is
[`models/policy_native_stream_v1.md`](models/policy_native_stream_v1.md).
It was trained on 21,916 decisions in 25 episode groups from
`native-stream-humanlike-spell-dagger-v3-partial.npz`, SHA-256
`1bc03ce647d34c1fb3f77ba751d50ca7b602a9e16189025819e3cf89a423c384`.

The stream model has `memory_size=0` and `proficiency_size=0`. It consumes only
one delayed global/local semantic observation, the current visible player
pose, and its 192-unit GRU state. Scenario/attack IDs, absolute frames, script
phases, recorded routes, waypoints, teacher risk fields, and external strategy
memory are excluded.

Training uses stateful TBPTT with 32-decision chunks and episode-balanced
optimization, so each complete training episode contributes one equally
weighted optimizer step per epoch. Action-class balance power is 0.75,
movement-onset weight is 4.0, and direction-change weight is 1.5. Onset and
direction transitions are calculated only from adjacent teacher actions in
the same episode. The run retained epoch 40 with
`restore_best_validation=false`; epoch 8 had the minimum internal validation
loss. Native strict outcomes, rather than validation loss alone, control the
release decision.

For new stream runs, `--no-scenario-memory-conditioning` now makes
`memory_size=0` automatically. Supplying a nonzero `--memory-size` with that
flag is rejected, which prevents a zero-filled training input from becoming a
scenario vector at inference. Use `--stateful-tbptt --episode-balanced` to
retain GRU state across each complete episode while giving long and short
episodes equal optimizer weight; `--movement-onset-weight` and
`--direction-change-weight` control the two teacher-transition terms.

An optional learned identity-token experiment is reproducible without adding
phase, position, action, or route memory:

```bash
uv run stg-lab contextualize-demos \
  --demos artifacts/native-stream-input.npz \
  --source-manifest artifacts/native-stream-input.manifest.json \
  --output artifacts/native-stream-context.npz \
  --manifest artifacts/native-stream-context.manifest.json
uv run stg-lab train \
  --demos artifacts/native-stream-context.npz \
  --scenario-vocabulary-manifest artifacts/native-stream-context.manifest.json \
  --checkpoint artifacts/policy-context.pt
```

Adding `--previous-action-conditioning` to `contextualize-demos` appends an
18-way one-hot token for the previous motor action actually executed. Native
direct-corrective DAgger preserves that executed action stream, and live
inference updates this token only after the engine advances the requested
frames. The token contains no future action, teacher proposal, position, frame,
phase, waypoint, or route; it is observable motor feedback for learned temporal
dynamics.

This creates a deterministic one-hot vocabulary containing an unknown token
and registered attack/stage identities. The network learns what to do with the
token; it contains no handcrafted phase logic. The vocabulary is saved in the
checkpoint and unknown attacks use the unknown token. It remains experimental:
a recurrent-256 context candidate cleared Koishi #1 at 915 frames but failed
Okuu #3 at 522 frames; 20 more epochs reached only 642 frames. Adding one
strictly completed Okuu #3 DAgger episode and fine-tuning 10 epochs then failed
at 514 frames. None replaces the shipped zero-context checkpoint.

There are now two explicitly different DAgger archive contracts. Legacy
teacher-labelled archives supervise every student-visited decision with
`teacher_action`. Direct-corrective archives instead retain the student's
executed actions and complete recurrent context; at an intervention the
executed label is the teacher correction, while `supervision_mask` selects only
those intervention/correction points for correction-only training. When a
masked archive is merged with an older unmasked archive, the older samples are
treated as fully supervised. A teacher-assisted strict clear still reports
`pure_policy=false`, `pure_policy_success=false`, and
`pure_policy_validation_eligible=false`; it is training evidence, never a pure
policy success.

The current direct-corrective Okuu #3 collections at seeds 20260813 and
20260815 reached `attack_complete` at frame 3,815 with death 0, but required
489/1,272 (38.44%) and 467/1,272 (36.71%) teacher interventions. The v13
all-label aggregate has 35 episodes and 33,866 recurrent decisions. Its
critical-intervention counterpart preserves the same 33,866 decisions, has
30,376 supervised labels overall, and keeps all recurrent context for six Okuu
#3 corrective episodes while supervising only their 4,142 intervention points.

The latest pure, unshielded Okuu #3 candidates all failed the strict native
gate:

| Candidate | Seed | Frames | HP observed | Reason / death |
| --- | ---: | ---: | --- | --- |
| v10 unweighted | 20260812 | 392 | `6000 -> 5524.5` | `player_hit` / 100 |
| v11 corrected unique | 20260812 | 764 | `6000 -> 4924.5` | `player_hit` / 100 |
| v11 corrections repeated x4 | 20260812 | 431 | `6000 -> 5453` | `player_hit` / 100 |
| v12 Okuu specialist, final epoch 80 | 20260812 | 726 | `6000 -> 5086` | `player_hit` / 100 |
| v12 general, final epoch 30 | 20260812 | 414 | `6000 -> 5478` | `player_hit` / 100 |
| v13 all-label, validation-best epoch 3 | 20260816 | 413 | `6000 -> 5481` | `player_hit` / 100 |

The v13 all-label run selected epoch 3 at validation loss
1.9728760589. That offline improvement and the assisted clears above do not
meet the release contract, so no v10-v13 candidate replaces v1.

The four controller modes must remain distinct:

| Mode | Interpretation |
| --- | --- |
| pure GRU | The checkpoint selects movement from delayed semantic vision and GRU state; eligible for model evidence |
| visible safety | A separate visible-only forecast may override the GRU; diagnostic hybrid, never a pure result |
| DAgger teacher | MPC labels student-visited states and may intervene; strict completion admits data but is not student success |
| Engine MPC | Exact-state planner/teacher, optionally with teacher-only region memory; not checkpoint output |

Strict attack success requires `terminated=true`, exactly
`termination_reason=attack_complete`, and an explicit finite, non-Boolean
numeric `final_player.death=0`. A full stage requires `stage_complete` and the
same death evidence. Time/frame limits, partial HP reduction, survival time,
post-death completion, ghost/protected state, and missing death evidence all
fail. A DAgger episode is retained only after this same check, but its NPZ
target semantics depend on the declared archive contract: legacy
teacher-labelled archives use `teacher_action` everywhere, whereas
direct-corrective archives preserve executed actions/context and may restrict
action loss with `supervision_mask`.

The complete executed pure-GRU matrix below used history 1, observation delay
5, expert execution, and `--no-visible-safety-shield`:

```bash
uv run stg-lab engine-play \
  --host 127.0.0.1 --port 24816 \
  --checkpoint models/policy_native_stream_v1.pt \
  --scenario 'koishi1:Lunatic' --attack 1 --seed 20260738 \
  --policy-scenario-key 'koishi1:Lunatic' \
  --proficiency expert --observation-delay 5 --vision-history 1 \
  --max-frames 4200 --no-visible-safety-shield \
  --output artifacts/policy-native-stream-v1-koishi.json
```

The `--policy-scenario-key` value is retained in report metadata and supplies
legacy memory only when a checkpoint has `memory_size>0`; this checkpoint has
`memory_size=0`, so the value does not enter inference.

| Target | Seed | Frames | HP observed | Reason / death | Result |
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

The retained file was also replayed against the final source fingerprint
(`implementation_sha256=fa891752547e10f478fbec6b4f85349e4c43061fb3788bea9014ac1f9337ac56`),
with checkpoint path and SHA-256 verification, history 1, observation delay 5,
expert execution, and no visible-safety intervention:

| Current-source target | Seed | Frames | HP observed | Reason / death | Result |
| --- | ---: | ---: | --- | --- | --- |
| Koishi #1 | 20260738 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260739 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260740 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1, held out | 314159265 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Yamame #3, held out | 314159265 | 493 | `1800 -> 1300.25` | `player_hit` / 100 | fail |
| Okuu #4, held out | 314159265 | 1065 | `3500 -> 2629.5` | `player_hit` / 100 | fail |
| Stage 1 Normal, held out | 314159265 | 543 | across-wave HP is not comparable | `player_hit` / 100 | fail |

Thus the current release proves strict pure completion only for Koishi #1: the
three known seeds and one independently selected held-out seed passed, while
all current-source cross-target probes failed. No broader rate is claimed. The
three original successful runs have 105.96-107.06
direction changes per 1,000 frames, 10-11 exact reversals, 7 ABA changes, and
a 6-frame median hold. The strict Koishi DAgger teacher reference has 97.46
changes per 1,000 frames, no ABA changes, and a 9-frame median hold, so the
checkpoint is still more restless. No pure run exists for DAgger-only Yamame
#1 or Satori #1, so their teacher-assisted episodes are not listed as policy
successes. General SR play and human-like behavior remain unmet: four strict
clears of one near-deterministic card do not establish human equivalence,
cross-card reliability, or full-stage competence.

Proficiency is an external execution transform, not network conditioning.
With the shield still disabled, intermediate Koishi #1 seed 20260741 failed at
593 frames (`700 -> 291.5`, death 100), and novice failed at 478 frames
(`700 -> 342.5`, death 100). Expert passed 3/3 on seeds 20260738-20260740;
these different seed sets are not a controlled skill calibration.

Validation-best specialist attempts were also rejected by native outcomes.
Yamame #3 specialist v1 (validation-best epoch 1) failed at 598 frames, its
epoch-40 variant failed at 495, and recurrent-size-256 specialist v2
(validation-best epoch 19) failed at 530/502 frames on seeds 20260740/20260743;
all ended in `player_hit`, death 100. A new 29-episode/26,234-decision v5
candidate trained from scratch for 60 epochs lowered validation loss to 2.2334,
but failed Koishi #1 at frame 391 on seeds 20260742 and 20260744, with death 100
and 592.5 HP remaining. Fifteen more epochs at `5e-5` reached only frame 436;
an Okuu #3 visible-safety hybrid died at frame 902 after 66 interventions,
shorter than the scratch model's 1,146-frame pure run. These were rejected
instead of replacing the 3/3 retained checkpoint. Full-stage policy success is
likewise absent: the final pure GRU
stayed neutral and failed Stage 1 at frame 543. A separate MPC-teacher Stage 1
attempt first failed at a 7,199-frame time limit and later completed at 12,453
frames on another seed; neither is checkpoint success, and the long episode
was excluded from the spell-policy pool.

## Live-engine regression

`engine-test` queries the live `catalog` command and resets every registered
attack instead of duplicating the Spell Practice mapping in Python. The
bundled SR manifest is 53 attacks across 22 scenarios. Defaults are 300 logical
frames per attack, `--step-batch 1`, focused neutral movement, and no shooting.
Each reset uses 99 lives and `player_protect_frames=frames_per_attack+600` so
this content-plumbing regression is not cut short by player collision.
When the wheel is run outside this source tree, set `STG_LAB_MOD_ROOT` to the
SR mod directory so runtime Lua fingerprints are compared with the intended
installation. Otherwise the tool discovers the mod root from the current
directory or one of its parents.

```bash
uv run stg-lab engine-test \
  --host 127.0.0.1 --port 24816 \
  --seed 20260729 --frames-per-attack 300 --step-batch 1 \
  --expected-attacks 53 --no-shoot \
  --output artifacts/engine-v2-a.json
```

The bridge must be launched with `SR_TEST_MODE=1`, a unique nonempty
`SR_TEST_SESSION_ID`, and normally `SR_TEST_STARTUP_ACCEPT_TIMEOUT=30`.
Startup accept waits for the first client before the initial Render/Present
path; setting the timeout to `0` keeps installation nonblocking. `ping` also
returns a bridge-generated `process_nonce` and protocol-v2 `runtime_identity`.
The latter contains the Win32 OS PID, executable path/CRC32, and CRC32 values
computed inside LuaSTG for `root.lua`, the bridge/init/spell-practice files,
and `_editor_output.lua`.

Observation object collections are strict JSON arrays even when empty, and
their `counts` values must match. An attack proves active content only after it
has both a boss/enemy and a hazard: a bullet, indestructible, laser, or an
additional collidable enemy object used as an attack hazard. Attack-complete
termination is based on the raw `GROUP_ENEMY`/`GROUP_NONTJT` pools, not the
visibility-filtered arrays, so an off-screen boss does not cause a false stop.

A successful `engine-test` artifact deliberately keeps
`engine_verified: false`; it is evidence from only one process. Launch a second
fresh engine with another session ID, produce the same complete report, then
run:

```bash
uv run stg-lab engine-accept \
  --first artifacts/engine-v2-a.json \
  --second artifacts/engine-v2-b.json \
  --expected-attacks 53 \
  --output artifacts/engine-acceptance-v2.json
```

`engine-accept` requires different session IDs and process nonces, identical
positive OS PIDs, identical strict configurations, equal executable and
runtime-source CRC fingerprints, equal local source SHA-256 maps, and a current
Python implementation fingerprint. It also requires exactly 53 attacks and 22
scenarios, at least 300 frames per attack, one request/hash per logical frame,
non-static active-content evidence, no early termination, complete catalog
order, and identical card/seed/frame metadata and per-frame hash arrays. The
bridge protocol and live-engine report schema are v2; the embedded catalog
schema remains v1. Runtime provenance covers all 18 actually loaded SR Lua
files. Only the combined report can set `engine_verified: true`.

## Current native Engine MPC and region-dynamics memory

Checkpoint-driven `engine-play --visible-safety-shield` keeps its emergency
avoidance visible-only: it reads the delayed local semantic raster and inferred
motion, never the scenario/attack identifier or script clock. `--proficiency`
also controls native execution. Novice/intermediate/expert apply respectively
6/3/0 reaction-delay frames, 9/6/3 minimum direction-hold frames, 4%/1%/0%
seeded suboptimal choices, 3/6/12-frame visual prediction, and 25%/65%/100%
seeded shield availability per decision. `--visible-safety-horizon N` is an
optional cap on that profile horizon rather than a replacement for it. Reports
include the effective horizon, probability, shield checks, and probability
skips so the selected execution level can be audited.

`engine-mpc-play` starts at the attack reset and remains under continuous live
MPC control. The command has no recorded-action prefix loader, prefix CLI
option, or replay branch. Its movement candidates and emitted actions always
set `spell=false`. Shooting remains enabled on every active logical frame:
player firing does not change movement speed or collision, so coupling it to
forecast clearance only reduces damage and extends exposure to the attack.
The legacy `--shoot-minimum-margin` option remains parseable for existing
launchers and report comparison, but is reporting-only and does not control
firing. MPC, checkpoint-policy, and DAgger reports expose `continuous_fire=true`
and explicit `shoot_command_frames`/`shoot_command_rate` fields, and mark their
legacy shoot-risk thresholds as non-controlling. These live reports now use
schema 3. The misleading schema-2 `unsafe_shot_frames` field is retained as
`null` plus a deprecation marker; movement-plan collision forecasts are reported
separately and never gate firing.

```bash
uv run stg-lab engine-mpc-play \
  --host 127.0.0.1 --port 24816 \
  --scenario 'okuu:Lunatic' --attack 3 --seed 20260730 \
  --observation-delay 5 --horizon-frames 60 \
  --gap-prediction \
  --region-dynamics-memory models/region_dynamics_boss3_v2.json \
  --output artifacts/engine-mpc-boss3.json
```

The fitted Boss #3 memory is versioned at
`models/region_dynamics_boss3_v2.json`; training commands may still write new
candidate memories and provenance reports under ignored `artifacts/` paths.

Gap prediction is enabled by default; use `--gap-prediction` to state that
choice explicitly or `--no-gap-prediction` for a same-controller ablation. It
considers only `enemy_bullets`. Bullets with similar speed and parallel velocity
are clustered, then split by longitudinal depth into moving wavefronts. Adjacent
bullets on each wavefront's perpendicular axis define candidate corridors. The
usable center-space width removes both bullet radii, the player radius on both
sides, a 10-unit safety margin, and observation-delay displacement uncertainty.
The corridor must remain open at multiple future samples, cover enough of the
playfield cross-section to represent a coherent wavefront, and admit an entry
path whose clearance against *all* threats is at least the 4-unit emergency
margin (or the 8-unit forced-region reserve while region navigation is active).
Geometry is generated for every corridor first; only the active corridor and up
to eight region-compatible, nearby candidates receive the more expensive entry
certificate. The certificate first tries an executable three-frame-block direct
route, then a small spatially diverse action beam if another threat obstructs
that route. The 4-unit lower bound applies only to transient entry; the opening
itself still reserves 10 units per side plus observation-delay uncertainty.

An active corridor keeps a stable identity across replans. It is offered beside
the region anchor, and `enter` emits the first action of its certified route;
`hold` uses the full opening as a soft interval rather than chasing its exact
center. Collision prediction remains earlier than gap entry, while the separate
route certificate permits a deliberate transit through lower ordinary-clearance
space without accepting a collision. When remaining still already preserves
the ordinary safe margin, detected corridors stay in `observe` mode and do not
pull the player across open space. Per-decision JSON records
`gap_bullet_group_count`, `gap_corridor_count`, `gap_selected_center`,
`gap_selected_width`, `gap_selected_lifetime_frames`, and
`gap_navigation_mode` (`inactive`, `observe`, `enter`, `hold`, or `exit`). The
top-level `gap_prediction` summary reports whether it was enabled, detection and
selection counts, counts for each navigation mode, and maximum group/corridor
counts.

A deterministic four-bullet wavefront regression also verifies that this is an
action input rather than telemetry only. With the player just outside a natural
10-unit corridor, the enabled controller chooses left toward the selected
opening while the otherwise identical disabled controller chooses straight
down; both candidates remain collision-free. A second regression places a
stationary obstacle on the direct route and verifies a safe multi-block detour.
These are synthetic integration fixtures, not native success-rate evidence.

A local synthetic probe with 299 bullets, 23 wavefronts, and 276 corridors now
takes a median 2.92 ms for gap geometry. The complete decision medians are
135.24 ms with gap prediction and 131.66 ms with it disabled, a 3.58 ms delta;
entry certification is explicitly capped after geometry instead of running for
all 276 corridors. These are focused algorithm-performance checks, not native
closed-loop success or real-time throughput claims.

A final same-source, continuous-fire native A/B exercised the predictor on Okuu
#3 seed `20260730`. Both runs bind implementation SHA-256
`002d770a1e4d10ad98a2ce00f21796dd7deeddc931b08ab638a3e07e0bbefb86`;
their top-level run configuration, runtime source map, and seed are identical,
and the only controller-config difference is `gap_prediction_enabled`.

| Gap prediction | Strict outcome | Gap telemetry | Path and smoothness | Report SHA-256 |
| --- | --- | --- | --- | --- |
| on | `attack_complete`, `passed=true`; 3,363 frames / 1,121 decisions; HP `6000 -> 0`; death 0; shoot commands `3363/3363` (1.0); predicted-collision movement plans 522 frames | detected 172; selected 44; observe/enter/hold/exit `119/44/0/9`; maximum groups/corridors `54/13` | path 9621.0025; changes 316 (93.9637/1,000 frames); reversals 26; sharp turns 112; ABA 7; hold min/median/mean/max `3/9/10.6088/291` | `7dc328637957f0682974d97e0227475bea10f4eb79994334bea9599b76b18ea1` |
| off | `attack_complete`, `passed=true`; 3,360 frames / 1,120 decisions; HP `6000 -> 0`; death 0; shoot commands `3360/3360` (1.0); predicted-collision movement plans 528 frames | disabled; all gap counts 0 | path 9500.6747; changes 308 (91.6667/1,000 frames); reversals 26; sharp turns 122; ABA 6; hold min/median/mean/max `3/9/10.8738/291` | `93f27f603bb852021ccab8f62e87285072140a1788cab510a02d69ed51c15e3a` |

Across 1,120 common decisions, direct comparison finds 802 different emitted
actions, including 749 different movement choices, 947 different planned-action
arrays, and 952 different observed decision-boundary positions. The enabled run
has one additional terminal decision; the first action and position divergence
is decision 168. Gap prediction therefore activated and changed native
movement while both sides still strictly cleared. The enabled run's smoothness
figures are descriptive for this episode, not a general result. It recorded no
`hold` decision, so the native episode does not evidence sustained corridor
holding; the deterministic regression covers `enter -> hold`. Schema 3 retires
the old unsafe-shot metric, so `unsafe_shot_frames=null` is not a zero count.
With one seed and both sides already successful, the pair demonstrates
activation and non-regression, not a success-rate improvement. Both reports
remain `acceptance_claim=false` live MPC-teacher evidence, not learned-policy
results.

This is a general visual teacher rule, not spell-specific memory added to the
released neural checkpoint. The A/B reports have not yet been converted into a
gap-aware DAgger or demonstration archive, and the published stream-v1
checkpoint has not been retrained from such an archive.

The sole success condition is
`terminated=true && termination_reason=attack_complete && final death=0`, with
death present as a finite numeric engine observation. The report sets `success`,
`passed`, and `episode_completed` from exactly that expression, and the command
exits nonzero otherwise. Reaching `max_frames`, surviving longer, partially
reducing boss HP, or completing with missing/nonzero death evidence is a failure.

### Strict cross-attack and full-stage matrix

`engine-mpc-matrix` runs a catalog-validated Cartesian product of targets,
seeds, and MPC profiles on one live engine connection. The `current` profile
uses clearance targets 20/8; `legacy-clearance-12-1` keeps the earlier 12/1
targets while retaining the current controller implementation. For example:

```bash
uv run stg-lab engine-mpc-matrix \
  --host 127.0.0.1 --port 24816 \
  --scenario 'okuu:Lunatic' --attack 3 --attack 4 \
  --stage 'Stage 5@Lunatic' \
  --seed 20260730 --seed 20260731 \
  --profile current --profile legacy-clearance-12-1 \
  --max-frames 9000 \
  --trace-directory artifacts/strict-matrix-traces \
  --output artifacts/strict-matrix.json
```

Use `--all-attacks` for all 53 spell-practice attacks, including catalog
entries labelled `Mid`, and `--all-stages` for the 10 complete Normal/Lunatic
Stage 1-5 entries. Full-stage reset uses the separate `reset_stage` protocol.
An empty enemy pool between stage waves never ends the episode. A non-final
stage completes only on its registered same-difficulty successor; the final
stage completes only on a menu transition. Unexpected stage changes and
`engine_exit` remain failures. Set bridge `SR_TEST_MAX_FRAMES` at least as high
as the matrix `--max-frames` for complete-stage runs.

The matrix independently recomputes every result. Attacks succeed only with
`attack_complete` and explicit numeric `death=0`; full stages require
`stage_complete` and the same zero-death evidence. A runner's own `success=true`,
survival to the frame limit, partial HP reduction, or a stage change before
active content is observed cannot count. Per-episode and
group summaries include deaths, observed boss HP reduction, frames, path
distance, adjacent direction changes, exact reversals, greater-than-90-degree
turns, ABA changes, and direction-hold durations. Optional trace files are
bound into the summary by SHA-256. This is strict native-engine evidence but
keeps `acceptance_claim=false`; it does not replace two-process deterministic
`engine-accept` evidence.

For a visible native-engine run, use `--render --render-every 1` and launch
LuaSTG with `SR_TEST_HEADLESS=0`, lockstep enabled, and `setting.vsync=true`.
The engine clears its swap-chain target before every Lua `RenderFunc` call, so
the bridge redraws even when the logical frame is waiting for Python;
`--render-every` is retained only as a protocol compatibility hint for native
runs. If the game is copied to a local Windows directory, replace that copy's
`compat/testing/bridge.lua` and restart LuaSTG before testing this behavior.
On Windows, double-click `run-win-boss3.cmd` after the local game and
`.venv-win` already exist. The launcher does not copy or install anything; it
validates those prerequisites, starts the local engine, waits for the listener
without consuming the bridge connection, and runs the strict Boss #3 test.
Use `run-win-boss3.ps1` directly when command-line parameter overrides are
needed.

The retained CrossOver strict Boss #3 evidence consists of three fresh original
LuaSTG Sub processes under CrossOver 26.3 with DXVK. v40 and v41 are
five-frame-delay held-out runs; v42 is a separate zero-delay regression. All
three used `authority_state_shield=false`, forced `spell=false`, and
region-dynamics v2. They reduced boss HP from 6000 to 0 with `death=0`, ended
at episode frame 3816 with `attack_complete`, and recorded
`unsafe_shot_frames=0`:

| Run | Seed | Delay | HP | Death | Final frame | Unsafe shots | File SHA-256 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json` | 20260730 | 5 | `6000 -> 0` | 0 | 3816 | 0 | `e7577aa475ed9a9de6542fedfba8a193dca1b3d8a927e139371e22f41b2d94ef` |
| `engine-mpc-boss3-heldout-v41-seed20260731-d5-region-dynamics-v2.json` | 20260731 | 5 | `6000 -> 0` | 0 | 3816 | 0 | `5cedf76c0b17028b4239f480dbd146a54cb92ead17dc898f8fc8d6fb52e981fa` |
| `engine-mpc-boss3-regression-v42-seed20260732-d0-region-dynamics-v2.json` | 20260732 | 0 | `6000 -> 0` | 0 | 3816 | 0 | `a45084f331ecd82d0fecff636bf1921c9b547f41f82a1db6846ff76d15d7e37f` |

All three reports bind implementation SHA-256
`5a81172add05549fdf1ea6d65272d26dd08afc3de6c289ab3124e9f7b2e69613`.
The two delay-5 held-out attempts passed, and the zero-delay regression passed;
this is not a statistical success-rate claim or Boss #4 evidence.

The current controller still replans every three frames. Ordinary bullets use
a hard 20-unit reserve, while forced-region passages use 8; they are tracked
separately so a narrow passage cannot hide poor ordinary-bullet clearance. The
16-unit danger tier ranks before the 20-unit reserve, ordinary clearance keeps
earning a soft reward up to 48, and immediate corner clearance has a 48-unit
soft reserve. The 48 values are preferences, not impassable boundaries.

A direction is normally held for 12 frames. A near collision, a material
8-unit reserve gain near the end of the hold, an immediate corner escape, or an
urgent `evacuate` plan with at least three cost units of real route progress can
release it. `evacuate` no longer disables hysteresis merely because of its mode
name. The beam also penalizes a direction change, exact reversal,
`A -> B -> A` oscillation, and same-direction speed-mode change by
`3/9/6/0.75`. Collision and reserve tiers outrank smoothing, and a committed
action cannot restore a stale direction after a safety release.

Two native macOS DirectX-free closed-loop runs started at the Spell Practice
reset with five frames of observation delay and strictly defeated Boss #3 at
frame 3816:

| Run | Seed | HP / death | Changes / reversals / ABA | Median clearance | File SHA-256 |
| --- | ---: | --- | --- | ---: | --- |
| `engine-mpc-boss3-safety-sort-v46-seed20260730.json` | 20260730 | `6000 -> 0` / 0 | 487 / 20 / 20 | 8.27 | `e1a4df0fd857b3aa7cc52e6bcd6416a856f74228f34e07c26b4a19882cbe6d39` |
| `engine-mpc-boss3-safety-sort-v47-seed20260731.json` | 20260731 | `6000 -> 0` / 0 | 502 / 10 / 30 | 8.31 | `d69d6b7fa2b88637590f5bbe5f605e1768bd7dccbd47307bdfa25fd651d50faf` |

Both reports have `terminated=true`, `termination_reason=attack_complete`,
`passed=true`, `unsafe_shot_frames=0`, and forced `spell=false`. They bind
implementation SHA-256
`8422915228d7b867ae01ffae0e2d0ae85d7ab8d1aac71e3a383c6b8a6e6d2044`.
This is 2/2 among executed attempts only; it is not evidence for unexecuted
seeds, Boss #4, or a deployable learned policy.
These reports supersede v44/v45, which predate the final region-sort and
committed-action safety fixes.

Cross-card geometry coverage now includes bridge-exported straight and bent
lasers. Tapered straight segments and bent polyline segments are represented by
a conservative 16-32 px circle cover; each radius includes the exact segment
half-step, so the reduced sample count cannot create gaps. Successive sample
positions, rather than the laser origin displacement, capture rotation and
length changes. A native headless Okuu #2 run of implementation
`fab98499b72c55fb92ceb5586b58be5093df9e42b9755631e008633ceaf96f95` at zero
observation delay strictly reached `attack_complete` at episode frame 3036,
reduced HP `4300 -> 0`, kept `death=0` and `unsafe_shot_frames=0`, and observed
up to 552 bullets and 40 lasers. Its threat count was 302 at the median and 697
at the maximum. Report SHA-256 is
`70d5afc69faa6fbae55cef0bdd678f6fe090d977608ec4d16ec281377f85dfd2`.
This is one zero-delay MPC-teacher integration result, not delay-5 or learned
policy evidence and not a success-rate claim.

`experiments/benchmark_engine_mpc_grid.py` compares the continuous beam with
8/12/16-unit time-layered grid planners on the same recorded delay-5
observations at source frames `488/1292/2102/2801/3695`. Grid layers remain
three frames apart, but each layer receives every logical-frame threat
occupancy in its interval. Within each observation all plans are truncated to
one common action horizon, then independently checked at every logical frame
with continuous circle geometry. This compares complete planners, not only
rasterization, and a grid never certifies its own safety.

| Planner | Collision-frame rate | Median minimum clearance | Changes per 60 frames | Mean time |
| --- | ---: | ---: | ---: | ---: |
| Continuous beam 20/8 | 9.67% | -4.47 | 6.20 | 0.120 s |
| Grid 8, center sample | 9.00% | 1.90 | 7.80 | 1.326 s |
| Grid 12, center sample | 10.00% | -3.04 | 9.20 | 0.698 s |
| Grid 16, center sample | 45.33% | -26.29 | 10.60 | 0.460 s |

Whole-cell half-diagonal inflation scored 11.67%/15.67%/52.33%. The 8-unit
grid's two-frame advantage (`27` versus `29` collision frames out of `300`) is
too small to establish an episode-level win, while it costs 11.0x the mean
planning time and changes direction more often. Grids therefore remain useful
for visual overlays, connected regions, and global hints; final local movement
stays with continuous geometry and hysteresis. Working report SHA-256 is
`ffc69082eefd2501d473981689eddd4fcfb67a0152a5723b9f099a46c7bbd901`.
This is an open-loop five-observation ablation, not an `attack_complete`
success rate.

A separate full-engine native macOS OpenGL run kept the optimized F7 overlay
enabled and passed the same strict condition. Artifact
`engine-mpc-boss3-gpu-overlay-strict-seed20260730.json` has SHA-256
`ec3f758a8a5135b33e139076bdecdb050bf1117f1e13622679a91c40e8110def`.
It reached episode frame 3816 with `terminated=true`,
`termination_reason=attack_complete`, `passed=true`, boss HP `6000 -> 0`,
`death=0`, `unsafe_shot_frames=0`, and forced `spell=false`. Its
`acceptance_claim=false` correctly scopes it as single-run live MPC teacher
evidence, not deployable-policy acceptance or Boss #4 evidence.
That run predates the current overlay implementation described below; it is
historical full-engine integration evidence, not an execution proof for the
current overlay SHA-256.

`train-region-dynamics` fits the allowed cross-episode strategy memory from one
or more native Engine MPC JSON reports:

```bash
uv run stg-lab train-region-dynamics \
  --input artifacts/engine-mpc-boss3-heldout-v37-d5-memory-no-actions.json \
  --memory-output artifacts/region-dynamics-boss3-v2.json \
  --report-output artifacts/region-dynamics-boss3-v2-training.json
```

The trainer's observation whitelist is strict. It consumes `source_frame`,
`region_observed_radius`, and adjacent visible positions from the highest row
of collidable indestructible hazards. It estimates lateral flow from the
position displacement of at least three matched visible objects, never from
object `dx`/`vx`, Lua class, or script-timer fields. Source frames are used
only for relative intervals and displacement, never as reusable absolute-frame
triggers. Every input must be a live Engine MPC report for the same scenario
and attack, have `authority_state_shield=false`, have `spell_forced_off=true`,
be marked `region_dynamics_training_eligible=true`, contain only
`control_source=live_mpc` decisions, and contain no enabled recorded prefix or
prefix artifact. Inputs without complete radius cycles or a repeated
lateral-flow cycle with half-cycle sign inversion are rejected.

The loadable memory contains only identity metadata plus this strategy
whitelist: ordered safe-region phase/change logic, minimum and maximum radius,
expansion and contraction rates, relative phase durations, radius-cycle
length, and this lateral-flow contract:

```json
"lateral_flow": {
  "cycle_frames": 360.0,
  "safe_side_rule": "opposite_incoming_lateral_flow"
}
```

It does not retain a lateral phase offset or fixed side sequence. Training
inputs, hashes, sample counts, and fitted-sample statistics are kept in the
separate provenance report. Neither output may encode absolute episode frames,
fixed world coordinates or waypoints, recorded actions, action fragments, or a
route tied to one run.

At runtime v2 projects the current visible hazard rows to the next expansion's
start, midpoint, and end, then chooses an exterior region expected to remain
open. That region target is separate from the local beam search around bullets.
Schema-v1 memory remains loadable for compatibility, but it cannot enable this
lateral-phase side prediction.

The v2 memory SHA-256 is
`dcdcddeeed840d733e144477934f217b21f6795ab9a83790d7f3813d273546f7`;
the training-report SHA-256 is
`95f4b3e45952f476f430158416d6f13b7617a4a5d25dbe07e522fc0107ac8e99`.
The report binds source trace SHA-256
`60c1dd6bb0cfdece73d7170b3c968736479470ba87fed6e96fa68e42290134f5`,
357 radius samples, and 303 flow samples. It fits radii `7/28`, rates
`0.7/0.7`, phase durations `30/30/30/90`, and a 180-frame radius cycle. The
360-frame flow repeat has 201 pairs, correlation 1.0, and normalized RMSE 0;
the 180-frame sign inversion has 252 pairs, correlation 1.0, and normalized
RMSE 0.

### Exact MPC prefilter benchmark

`experiments/benchmark_engine_mpc_beam.py` compares the conservative
candidate-AABB threat prefilter with the unfiltered reference beam on
observations recorded in
`engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json` (SHA-256
`e7577aa475ed9a9de6542fedfba8a193dca1b3d8a927e139371e22f41b2d94ef`).
The recorded three-repeat medians are:

| Source frame | Mode | Threats | Unfiltered | Prefiltered | Speedup | Output |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 995 | local beam | 364 | 0.3653 s | 0.1292 s | 2.83x | exact equality |
| 1292 | region planning | 193 | 0.3809 s | 0.1209 s | 3.15x | exact equality |

Equality covers every candidate action, `collided`, `collision_frames`,
`earliest_collision_frame`, exact floating-point `minimum_margin`, boundary
penalty, boss-alignment penalty, and each complete 20-segment action plan. The
frame-1292 stateful replay also matched 21 optimized and unfiltered decisions
after a 60-frame warm-up, including committed action, plan, and source-report
collision fields. Five-repeat runs on the same host vary around 2.7x-3.4x, so
the timing values are samples, not a guaranteed performance floor.

## Portable full engine, development target, and in-engine risk overlay

The current LuaSTG-Sub complete executable supports
Windows, macOS, and Linux with SDL2 window/input/audio and an OpenGL 4.1 GPU
sprite/FBO renderer, or null graphics/window/audio for uncapped headless runs.
Both execute real Lua scripts, resources, collision logic, and the test bridge.
The OpenGL renderer batches legacy vertices and indices, caches revisioned
source textures on the GPU, renders into FBO-backed targets, and presents the
main target without a per-frame CPU-canvas upload. Supported builds compile and
link no DirectX code. Arbitrary HLSL-to-GLSL translation, model drawing, and the
modern MeshRenderer path are not yet feature-equivalent; old post effects use
visible RenderTarget pass-through composition.

The CrossOver/DXVK reports above validate the original Windows executable and
remain separate evidence, not a requirement for the current source build. The
full macOS presets configure, build, audit the final binary for DirectX, and
run the corresponding tests:

```bash
LUA_STG_SUB_ROOT=/path/to/LuaSTG-Sub
cd "$LUA_STG_SUB_ROOT"
cmake --workflow --preset macos-headless-release
cmake --workflow --preset macos-opengl-release
```

The repository also provides the isolated `LuaSTGPortableTest` target for
partial algorithm development. It reuses `XCollision` circle/rotated-ellipse
checks and offers an uncapped no-video headless mode plus an SDL2 software
simplified
collision/risk view. It is not the complete macOS game runtime.

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

The `pulse` and `orbit` scenarios exercise region phases, forced displacement,
collision, risk analysis, and simplified rendering. The clean arm64 macOS build
passed CTest (`1/1`). A measured 100,000-frame headless pulse run reached
approximately 1,511,690 logical frames/second, while the orbit graded-risk
probe reached approximately 496,620 frames/second. The native macOS SDL2
software capture `build/portable-native/portable/pulse-macos.bmp` and the SDL
dummy software capture `build/portable-native/portable/orbit-dummy.bmp` are
nonempty `480x560` 32-bit BMPs. This partial runtime does not execute arbitrary SR Lua
and cannot replace native LuaSTG `attack_complete` training or acceptance.

The engine installation's `game/plugins/SafetyZoneVisualizer` plugin toggles
with `F7`, or starts enabled with `SR_SAFETY_ZONE_OVERLAY=1`. It grades safe,
caution, danger, and collision cells from current colliders and linear
projections at every logical frame from 0 through 24. Growing indestructible
ellipses use the larger of their observed nonnegative radius rate and the
Boss #3 guard of 0.7 units per frame; the overlay never assumes shrinkage will
make a future cell safe. A 16-unit cell is classified
as a whole by inflating the player radius by its half diagonal. Red is signed
clearance at most 0; orange is clearance at most 16 or at least five
simultaneous nearby threats within 36; yellow is clearance at most 28 or at
least three. Clearance takes the worst future layer, while density is counted
within each layer before taking its maximum, so threats arriving at different
times are not combined into a fictitious crowd. Circles, rotated ellipses,
rectangles, and straight lasers are supported. Straight-laser clearance uses
the tapered polygon defined by `l1`, `l2`, `l3`, and `w`, including THlib laser
objects whose generic ellipse collider reports `a=b=0`; curved-laser interiors
are not rasterized. A standalone parity test compares every optimized cell
with the original full scan across ellipses, rotated rectangles, tapered
lasers, fast projections, cell-corner collisions, time-separated density,
boundary cases, and a 350-bullet field. It performed 94,585 exact clearance
calculations out of 5,880,000 (1.61%) and coalesced 672 cells into 45 render
rectangles (`45/672`, rectangles/cells) without changing any cell's risk level.
`SafetyZoneVisualizer.lua` SHA-256 is
`b87ff9802e43345a300bca9329572917e9454e8dda62a720ef7b3f471011c4ee`.
The overlay is diagnostic and read-only: it does not change collision,
objects, input, RNG, AI actions, or memory.

### Native renderer benchmark and metric boundaries

The retained benchmark uses LuaSTG Sub v0.21.129 on native arm64 macOS 15.7.3,
an Apple M4 Max with a 40-core GPU, a visible `640x480` window, and VSync. Each
1200-sample run uses Stage 5 Boss #3, seed `20260730`, per-frame rendering,
decision interval 3, observation delay 5, and the same live MPC settings.
`OBJ >= 300` defines the dense subset; all runs have 273 dense samples, zero
invalid samples, and peak at 421 objects.

| Renderer / overlay | Dense median FPS | P10 | Minimum | Artifact SHA-256 |
| --- | ---: | ---: | ---: | --- |
| CPU software / off | 30.252 | 29.390 | 28.241 | `7cfcd17ca4ac9e86d3815ca9f302a33b4a6bb704f88b78c6e7949db1d1c61a4f` |
| CPU software / optimized overlay | 29.159 | 28.290 | 25.463 | `b8bf286e2bc68ae6319dee67a7cc80140356b33e24101e844577db16def3d9a0` |
| OpenGL GPU/FBO / off | 59.990 | 59.867 | 58.911 | `74482a01232db58d21fa75dc51f7a9e10b8ff2f3844b3484ade9f9bf9747d126` |
| OpenGL GPU/FBO / optimized overlay | 59.988 | 59.984 | 52.265 | `67f5c5335b317d11584c01e392e91ff1e9b039723498d437a05614fdd43e94b7` |

The historical strict GPU-plus-overlay defeat artifact extends this to 3816
valid display samples, peak 532 objects, and 2844 dense samples. Dense median
is 59.988 FPS and P10 is 59.492 FPS while the engine still returns
`attack_complete`; as noted above, that run used the earlier overlay revision.

`render_performance` is reporting-only diagnostics. Its exact source is the
60-native-frame `lstg.GetFPS` display average plus `lstg.GetnObj`, and its report
sets `reporting_only_not_controller_input=true`. It is removed from AI
observations and is not used by MPC or region-dynamics memory. Display FPS is
the native render/Present cadence and is affected by VSync. Lockstep throughput
is logical frames or decisions advanced per wall-clock second under
Python/bridge control; report it separately and do not label it display FPS or
AI inference FPS.

## Legacy standalone benchmark: persistent visual cues and routes

**Legacy standalone benchmark only.** The commands and artifacts in this
section retain historical reproducibility, but they store or replay recorded
routes. They are forbidden as training input, strategy memory, controller
input, or acceptance evidence for the current native `engine-mpc-play`
workflow.

`experiments/build_route_memory_v2.py` rebuilds
`artifacts/episodic_memory_v2.sqlite`, the Boss #3 single-route artifact, and
the Boss #4 route library. These are prior successful flows stored outside the
network, not routes reconstructed by an implemented death-point backtracking
algorithm.

```bash
uv run python experiments/build_route_memory_v2.py
uv run python experiments/build_memory_benchmark_v2.py
```

Boss #3 triggers its full-episode route from an online delayed semantic cue and
then indexes it only with the controller's own decision counter. Boss #4 stores
five routes. Once enough delayed occupancy is visible, it pools channels 0-2
to a semantic signature and selects the nearest stored signature, with
confidence and memory ID as deterministic tie breakers. It then follows that
route on the controller-local episode timeline. Neither controller reads the
environment frame, script timer, phase, class name, hidden RNG state, or exact
teacher risk field.

These historical v2 route evaluations must use `--no-shield`. Consequently
their reports must contain `shield: false`, `authority_state_used: false`, and
`online_visible_cue: true`. `policy_visible_v2.pt` remains the system checkpoint
identity for agreement and cross-artifact consistency; route-controller
actions are not network outputs.

```bash
uv run stg-lab evaluate-route \
  --route-artifact artifacts/route_memory_boss3_v2.json \
  --memory-database artifacts/episodic_memory_v2.sqlite --memory-id 1 \
  --checkpoint artifacts/policy_visible_v2.pt \
  --scenario stage5_boss3 --difficulty lunatic \
  --episodes 100 --seed 5001 --duration-frames 600 \
  --motor-delay-frames 0 --action-hold-frames 3 --decision-interval 3 \
  --observation-delay 5 --no-shield \
  --workers 8 --output artifacts/visual_v2_boss3_route.json
```

```bash
uv run stg-lab evaluate-route-library \
  --library-artifact artifacts/route_library_boss4_v2.json \
  --memory-database artifacts/episodic_memory_v2.sqlite \
  --checkpoint artifacts/policy_visible_v2.pt \
  --scenario stage5_boss4 --difficulty lunatic \
  --episodes 100 --seed 5001 --duration-frames 700 \
  --motor-delay-frames 0 --action-hold-frames 3 --decision-interval 3 \
  --observation-delay 5 --no-shield \
  --workers 8 --output artifacts/visual_v2_boss4_library.json
```

## Legacy standalone v2 acceptance compiler

This section documents the frozen route-based standalone acceptance compiler.
It does not accept or validate the current native Engine MPC controller, and a
passing legacy report does not satisfy the native `attack_complete` criterion.

`accept` ignores input `passed` flags and recomputes thresholds, seed-set
identity, timing configuration, checkpoint identity/SHA, memory semantics, and
per-frame determinism from raw evidence:

```bash
uv run stg-lab accept \
  --planner-artifact artifacts/planner_v2_boss3.json \
  --planner-artifact artifacts/planner_v2_boss4.json \
  --visual-artifact artifacts/visual_v2_boss3_route.json \
  --visual-artifact artifacts/visual_v2_boss4_library.json \
  --agreement-artifact artifacts/agreement_visible_v2.json \
  --memory-artifact artifacts/memory_benchmark_v2.json \
  --determinism-artifact artifacts/determinism_v2.json \
  --output artifacts/acceptance_report_v2.json
```

The compiler rejects stale implementation fingerprints, shielded/authority
visual reports, offline or hidden-state cue selection, mismatched checkpoint
or dataset checksums, unequal seed sets, duplicate evidence, and incomplete
determinism evidence. It recomputes thresholds from raw report fields rather
than trusting an input `passed` value.

CrossOver 26.3 with DXVK starts the real executable, loads the full SR
resources/scripts and bridge, and opens the TCP listener. wined3d still fails
framebuffer creation with `GL_INVALID_FRAMEBUFFER_OPERATION (0x506)`, while
DXMT reaches the Lscreen render target and fails `IDXGISurface` acquisition
with `E_NOINTERFACE`.

The frozen legacy standalone artifacts are bound to implementation SHA-256
`ba7f4d2ee9fe5bf232f264a180a51280657f007629a54fd36c3f4884ca966cb9`
and use the same 100 distinct seeds, 5001 through 5100, for planner and visual
evaluation:

| Evidence | Recomputed result | SHA-256 |
| --- | --- | --- |
| Exact-state planner, #3 | 100/100 survived for 600 frames; 100 unique terminal hashes | `f10ce59b18329a6d71ec0d5ba20c9bc9fd4bb21d7a0af416d5dcfe40cf78bd6d` |
| Exact-state planner, #4 | 100/100 survived for 700 frames; 100 unique terminal hashes | `6ad7b42b6977b6e8af3f854e294bbe86497290d53ad7a1c5ad636c1750fe92c8` |
| Delayed-visible route controller, #3 | 100/100 survived, no shield or authority state | `f29d3cf92864b1af2c8846ea53669d8769a7d4e74c6da9c2caffa85b41589069` |
| Delayed-visible route-library controller, #4 | 98/100 survived, no shield or authority state | `004e4696766f5c4a0fa1a682a21aef72787dedef98426ad04749ed10a6ca00aa` |
| #4 episodic-memory benchmark | first attempt died at frame 340; the delayed cue selected memory 2 at control frame 138 from source frame 133; second attempt survived 700 frames | `b57f15e26d8ee49de6895b5145f52489f8da562e307605adb481ba31ebf44b42` |
| Per-frame determinism | fresh #3/#4 runs matched all 601/701 state hashes and actions | `dbc0cdc1b50f9e75315672580c6e4c1aecb0b670700d7a02cb45c3df50be12e4` |
| Strict standalone compilation | `passed: true`, `issues: []` | `7035a896eb96f9633ca5515314d39edfd763ddcd7cdfaad673f5b7d3f595d061` |

The read-only SQLite memory SHA-256 is
`e774d3148ba0bd0cc89d1e8f9d68db8e3bf1612a1b56fd2b61b14940d321b584`;
the #3 route and #4 five-route library SHA-256 values are respectively
`d492d790cfb5310cb6f519a80bfefe8750a0d0f6dfcbd5271475ad785a76bf17`
and `6a1c47342f0f7d1190b810e96cfacf104b76ee111dfabd6041321b273f20873a`.
The memory gate passed because the second attempt survived; the report does
not claim a numeric risk reduction. These are standalone approximation
results, and route-controller actions are not checkpoint outputs.

### Historical live-engine content-plumbing evidence

The following artifacts verify attack registration, active content, runtime
identity, and deterministic stepping. They were produced with player
protection and neutral/no-shoot input, so `53/53` is not an AI survival or
attack-completion result and is not current native Engine MPC acceptance.

The historical protocol-v2 DXVK reports use distinct sessions, nonces, and
Win32 PIDs 212/204. Both report executable CRC32 `8844e525`; all 18 runtime Lua
CRC32 values match the local files, and both 18-entry local SHA-256 maps match
the current tree. Both passed all 53 attacks in 22 scenarios with 301 retained
hashes per attack. `engine-acceptance-v2.json` matched all 15,953 hash positions
and records `passed: true`, `engine_verified: true`:

| Live-engine evidence | Result | File SHA-256 |
| --- | --- | --- |
| `engine-v2-a.json` | 53/53 passed, PID 212 | `ac7996a2ee92417e08deda8ff5e86d3a0937278f7bb15a772e53089275a0abeb` |
| `engine-v2-b.json` | 53/53 passed, PID 204 | `8d692aa4a4f79ed6de0d701d628ebbbbc64cce624f016deb12d198dec6e5b257` |
| `engine-acceptance-v2.json` | 53/53 matched, verified | `a2a7cca87e6e416c43b963483e450b5a1e000cdfcdbd32eadbdb16d5e19ea1d2` |

Existing protocol-v1 reports are not current acceptance evidence.

Checkpoint validation does not run simulator episodes:

```bash
uv run stg-lab evaluate --checkpoint artifacts/policy_visible_v2.pt --metadata-only
```

Commands that print results use JSON so CI can consume them. Exit status is
non-zero for invalid input, missing optional dependencies, or failed pytest
runs. Use `uv run stg-lab <command> --help` for all tuning options. Standalone
reports must still be paired with `compat/testing/bridge.lua` engine results
before making a game-level acceptance claim.

See `../../docs/stg_ai_testing.md` for architecture, observation rules, and acceptance criteria.
