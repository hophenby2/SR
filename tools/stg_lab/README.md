# STG Lab

Standalone deterministic testing, region-risk planning, external episodic memory, and visual-policy training for SR and other LuaSTG bullet-hell projects.

The Python simulator is an algorithm-development environment. Final game claims must be verified through `compat/testing/bridge.lua` against the original LuaSTG scripts.

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

## Persistent visual cue memory

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

The final v2 route evaluations must use `--no-shield`. Consequently their
reports must contain `shield: false`, `authority_state_used: false`, and
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

## Strict acceptance

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

The final standalone artifacts are bound to implementation SHA-256
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

The final protocol-v2 DXVK reports use distinct sessions, nonces, and Win32
PIDs 212/204. Both report executable CRC32 `8844e525`; all 18 runtime Lua CRC32
values match the local files, and both 18-entry local SHA-256 maps match the
current tree. Both passed all 53 attacks in 22 scenarios with 301 retained
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
