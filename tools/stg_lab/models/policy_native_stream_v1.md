# Native Streaming Policy v1 Model Card

## Identity and intended use

- Checkpoint: `tools/stg_lab/models/policy_native_stream_v1.pt`
- SHA-256: `829eebe53c886e5ba53f577542938b904aad740f3a5bf04b49d61e73ab61557d`
- Source checkpoint: `artifacts/policy-native-stream-humanlike-spell-dagger-v3-epbal40.pt`
- Training archive: `artifacts/native-stream-humanlike-spell-dagger-v3-partial.npz`
- Training archive SHA-256: `1bc03ce647d34c1fb3f77ba751d50ca7b602a9e16189025819e3cf89a423c384`
- Architecture: dual visual CNN encoders, a 192-unit GRU, and an 18-action
  movement head. The checkpoint has `memory_size=0`, `proficiency_size=0`, and
  `inference_mode=stream`.

The model receives one delayed six-channel global semantic frame, one delayed
local semantic frame, the current visible player pose, and its recurrent hidden
state. It does not receive scenario or attack identity, an absolute frame,
script phase, a recorded route, waypoints, MPC risk fields, or external
strategy memory. The GRU is the only learned temporal memory in this
checkpoint.

This is a limited native-engine reference checkpoint. Its demonstrated pure
GRU scope is Koishi #1: three known expert seeds and one independently selected
held-out seed pass on the final source. It is not a generally successful SR
player, a full-stage policy, evidence that other cards pass, or a human-like
general player.

## Strict outcome contract

An attack is a success only when all of the following are true:

- the native engine returns `terminated=true`;
- `termination_reason` is exactly `attack_complete`;
- `outcome_evidence.final_player.death` is present, is a finite numeric value
  rather than a Boolean, and equals zero.

A full stage substitutes `stage_complete` for `attack_complete` and keeps the
same zero-death requirement. `max_frames`, time limits, partial HP reduction,
survival duration, a completion observed after death, ghost/protected state,
or missing/non-numeric death evidence are failures. Reports set `success`,
`passed`, and `episode_completed` from this contract. Native policy, DAgger,
and MPC runs force `spell=false`.

## Training data and optimization

The training archive contains 21,916 decisions in 25 native episode groups.
It merges the earlier strict spell archive with strict DAgger episodes for
Okuu #3, Orin #4, Koishi #1, and Yamame #3. A DAgger episode enters an archive
only after the same strict native completion check. This historical v1 archive
uses the legacy teacher-labelled contract: every stored target is the MPC
`teacher_action` at a student-visited state, and the mixed executed action is
not the supervision target. Teacher intervention therefore qualifies training
data, but does not prove that the student passed the card.

New direct-corrective archives use a different, explicit contract. They retain
the student's executed action stream and complete recurrent context. At an
intervention, the executed label is the teacher correction; a
`supervision_mask` can restrict action loss to only intervention/correction
points. If masked data is merged with older unmasked data, the legacy samples
remain fully supervised. A teacher-assisted strict clear still reports
`pure_policy=false`, `pure_policy_success=false`, and
`pure_policy_validation_eligible=false`; it cannot be counted as policy
success.

Training used:

- complete-policy initialization from the balanced20 checkpoint
  `d52b4c37407ac823e2d81c8e9e5c369cc4399c8c4b941de981463d6666c7b4a6`;
- stateful truncated backpropagation through time (TBPTT), chunk length 32;
- episode-balanced optimization: each of the 20 training episodes contributes
  one equally weighted optimizer step per epoch, independent of episode
  length;
- action-class balance power 0.75, movement-onset weight 4.0, and
  direction-change weight 1.5;
- onset/change detection only between adjacent teacher actions in the same
  episode, with direction defined by `(move_x, move_y)` independently of slow
  mode;
- 10% horizontal reflection, auxiliary risk weight 0.1, learning rate
  `2e-4`, and 40 epochs on MPS;
- episode IDs 11-15 reserved intact for validation.

The run deliberately used `restore_best_validation=false`. Epoch 8 had the
lowest internal validation loss (2.3124849571), while the retained epoch 40 had
validation loss 2.5737202304 and action accuracy 0.3373737374. Epoch 40 is
retained because it was the candidate that passed the strict native Koishi
gate, not because it was validation-best. Internal loss or action accuracy is
not a substitute for native completion.

## Controller boundaries

| Mode | State used to choose movement | What a success means |
| --- | --- | --- |
| Pure GRU | Delayed semantic vision, current visible player pose, GRU hidden state | Eligible model evidence when the strict native outcome contract passes |
| Visible safety | A separate visible-only local forecast may override the GRU action | Diagnostic hybrid evidence; must report checks/interventions and never be relabelled pure GRU |
| DAgger teacher | The student visits states while exact-state MPC labels every decision and may intervene | Training-data collection only; strict completion admits labels but is not student success |
| Engine MPC | Exact object geometry/velocity and optional teacher-only region-dynamics memory | Planner/teacher evidence only; not neural checkpoint output |

The final pure matrix below used `--no-visible-safety-shield`. Consequently all
visible-safety intervention counts are zero. No shielded result is used to
cover a pure-policy failure.

## Strict native pure-GRU matrix

All attacks are Lunatic, use history 1, observation delay 5, expert execution,
and no visible-safety shield unless stated otherwise.

| Target | Seed | Logical frames | Boss HP observed | Final reason / death | Strict result |
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
| Stage 1 Normal | 20260740 | 543 | not comparable across waves | `player_hit` / 100 | fail; no `stage_complete` |

The only positive pure-GRU target in that matrix is Koishi #1, 3/3 among the
three executed expert seeds. The pattern is close to deterministic across
these seeds, so 3/3 must not be extrapolated to a broader success rate. No
pure-GRU run was made for the later DAgger-only Yamame #1 or Satori #1
episodes, and their teacher-assisted completion is not a model result.

The three report SHA-256 values are, in seed order,
`ba9bee9918f4069cadd76cbaeff55ea92ebb70913e9c59e97daf09cce2bdf796`,
`ef519d61bf186993348dda08468a8444a11d7a0a35bf2e6680cac61d12e42a8e`,
and `5de7f591b6f2ba6a4a9f955ab3afa01cbfc1e21f6bef0a6ead28fb59be47953a`.
A serial seed-20260738 rerun after the executed-action runtime-state fix
reproduced the same 906-frame action sequence and strict clear with zero
visible-safety checks or interventions. It binds candidate implementation SHA-256
`93b53ee3a0bea3fcbd05fb17857b8f4e2ba968ff980adbaac1f647b697a48016`;
its report SHA-256 is
`3405c7b3fd8ec51e23bf6318dc11ac9fd19e83c7f792b38dc03778492ab3d71a`.
Later scenario-conditioning and laser-observation work changed the complete
source-tree fingerprint, so this report is retained as strict checkpoint-path
evidence rather than described as a final-source reproduction.

The retained file was subsequently replayed on final implementation SHA-256
`fa891752547e10f478fbec6b4f85349e4c43061fb3788bea9014ac1f9337ac56`.
The reports verify the checkpoint path and SHA-256, use history 1, observation
delay 5, expert execution, and no visible-safety intervention:

| Current-source target | Seed | Logical frames | Boss HP observed | Final reason / death | Strict result |
| --- | ---: | ---: | --- | --- | --- |
| Koishi #1 | 20260738 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260739 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1 | 20260740 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Koishi #1, held out | 314159265 | 906 | `700 -> 0` | `attack_complete` / 0 | pass |
| Yamame #3, held out | 314159265 | 493 | `1800 -> 1300.25` | `player_hit` / 100 | fail |
| Okuu #4, held out | 314159265 | 1065 | `3500 -> 2629.5` | `player_hit` / 100 | fail |
| Stage 1 Normal, held out | 314159265 | 543 | not comparable across waves | `player_hit` / 100 | fail; no `stage_complete` |

These reports are stored under the ignored `artifacts/final-source-v1-*`
paths. They prove only Koishi #1 among the tested targets. The held-out Koishi
clear strengthens source reproducibility but does not establish a cross-card
success rate or human equivalence.

The three successful Koishi runs made 105.96, 107.06, and 107.06 direction
changes per 1,000 frames. Their exact reversals were 10, 11, and 10; each had
7 ABA changes and a 6-frame median direction hold. The strict Koishi DAgger
teacher reference made 97.46 changes per 1,000 frames, no ABA changes, and had
a 9-frame median hold. The model is in the same broad movement-frequency range
but is measurably more restless; three runs on one card remain insufficient to
claim human equivalence.

## Proficiency profiles

Proficiency is an external execution transform; it is not a model input. The
visible-safety horizon/probability values below are inert in these pure runs
because the shield was disabled.

| Profile | Extra reaction delay | Minimum direction hold | Seeded suboptimal actions | Visible-safety horizon / availability | Koishi #1 strict result |
| --- | ---: | ---: | ---: | --- | --- |
| expert | 0 frames | 3 frames | 0% | 12 frames / 100% | 3/3, seeds 20260738-20260740 |
| intermediate | 3 frames | 6 frames | 1% | 6 frames / 65% | 0/1; seed 20260741, hit at 593 frames, HP `700 -> 291.5`, death 100 |
| novice | 6 frames | 9 frames | 4% | 3 frames / 25% | 0/1; seed 20260741, hit at 478 frames, HP `700 -> 342.5`, death 100 |

These are executed outcomes, not calibrated population skill distributions.
In particular, intermediate and novice currently model degraded control but
do not complete the reference card.

## Rejected candidates and full-stage evidence

Validation-best and specialist checkpoints were not accepted on offline loss:

| Candidate | Selection | Native Yamame #3 result |
| --- | --- | --- |
| specialist v1 | minimum complete-episode validation loss, epoch 1 | seed 20260740: hit at 598 frames, death 100 |
| specialist v1 final | epoch 40 | seed 20260740: hit at 495 frames, death 100 |
| specialist v2, recurrent size 256 | minimum complete-episode validation loss, epoch 19 | seed 20260740: hit at 530 frames; seed 20260743: hit at 502 frames; both death 100 |
| expanded v5 from scratch | 29 episodes, 26,234 decisions, final epoch 60 | seeds 20260742/20260744: hit at 391 frames, HP `700 -> 592.5`, death 100 |
| expanded v5 continued | another 15 epochs at `5e-5` | Koishi: hit at 436 frames, HP `700 -> 458.5`; Okuu #3: hit at 856 frames, HP `6000 -> 4959`; seed 20260742, death 100 |
| expanded v5 factorized loss | validation-best epoch 20 after 30 epochs; separate direction/speed/consistency losses | Koishi seed 20260743: hit at 436 frames, HP `700 -> 502`, death 100 |
| identity-conditioned v5 | 10-entry attack identity one-hot, fresh 256-unit GRU, final epoch 20 | Koishi seed 20260745: strict clear at 915 frames; Okuu #3 seed 20260745: hit at 522 frames, HP `6000 -> 5287`, death 100 |
| identity-conditioned v6 | v5 continued through epoch 40, then 10 low-rate epochs with one strict Okuu #3 DAgger episode | Okuu #3 seed 20260748: hit at 514 frames, HP `6000 -> 5363.5`, death 100 |
| expanded v5 plus visible safety | horizon 12, margin 8 | Okuu #3 seed 20260742: hit at 902 frames after 66 interventions, death 100; pure v5 reached 1,146 frames |

The specialist approach and validation-best selection are therefore rejected
as deployment evidence. The expanded scratch run reduced validation loss from
2.8278408229 to 2.2334204706 but regressed the retained checkpoint's Koishi
clear. Low-rate continuation, a factorized direction/speed/consistency loss,
and the visible-only safety hybrid also failed, so adding more cards and
selecting by offline loss did not produce a release candidate. The factorized
checkpoint SHA-256 was
`4e4ca71c9c481b0780a8ffa20c2d841655c777a28050771945b996e63b1c3286`.
This does not prove that specialization or broader data cannot work; it records
that the implemented candidates failed the strict native gate.

The identity-conditioned experiment supplied only the registered attack token;
it still excluded coordinates, absolute frames, script phase, actions,
waypoints, and routes. Its v5 epoch-20 checkpoint SHA-256 was
`d4393aa80035ec8a865ced6f72607eeebf5a932a3b05c554d65447f0053f5d88`.
Continuing to epoch 40 raised validation loss from `2.2220275734` to
`2.3351919582`. A native Okuu #3 DAgger run then cleared at 3,815 frames with
zero deaths, but required 1,100 teacher interventions and therefore counted
only as training data. The resulting v6 checkpoint SHA-256 was
`124e3629c73902d26cfd84494cbfd8947927ad1f9c4cfbec05c6c02c4d2b0918`;
its pure Okuu #3 failure was earlier than the unconditioned expanded-v5 run.
The retained release model remains the context-free checkpoint above.

### Corrective v10-v13 evidence

The direct-corrective Okuu #3 collections at seeds 20260813 and 20260815
strictly reached `attack_complete` at frame 3,815 with death 0, but used
489/1,272 (38.44%) and 467/1,272 (36.71%) teacher interventions. They are
teacher-assisted training episodes, not pure-policy clears. The v13 all-label
aggregate contains 35 episodes and 33,866 recurrent decisions. The matching
critical-intervention aggregate keeps the same 33,866 decisions, contains
30,376 supervised labels overall, and retains full recurrent context for six
Okuu #3 corrective episodes while supervising only their 4,142 intervention
points.

Every latest pure, unshielded Okuu #3 candidate failed:

| Candidate | Seed | Logical frames | Boss HP observed | Final reason / death |
| --- | ---: | ---: | --- | --- |
| v10 unweighted | 20260812 | 392 | `6000 -> 5524.5` | `player_hit` / 100 |
| v11 corrected unique | 20260812 | 764 | `6000 -> 4924.5` | `player_hit` / 100 |
| v11 corrections repeated x4 | 20260812 | 431 | `6000 -> 5453` | `player_hit` / 100 |
| v12 Okuu specialist, final epoch 80 | 20260812 | 726 | `6000 -> 5086` | `player_hit` / 100 |
| v12 general, final epoch 30 | 20260812 | 414 | `6000 -> 5478` | `player_hit` / 100 |
| v13 all-label, validation-best epoch 3 | 20260816 | 413 | `6000 -> 5481` | `player_hit` / 100 |

The v13 all-label training run selected epoch 3 at validation loss
1.9728760589. Offline validation improved, but native completion did not; none
of v10-v13 is a release candidate.

### v14-v25 Okuu #3 strict-gate record

The 21 recorded `pure-v14` through `pure-v25` matrix reports contain 42
executed Okuu #3 episodes. Every episode is marked pure policy, has zero visible
safety interventions, and ended in `player_hit` with `death=100`; strict and
pure-policy successes are both `0/42`. Logical lifetime ranges from 412 to 774
frames and last observed Boss HP from 4,966.0 to 5,480.5, never
`attack_complete`. The table below lists only checkpoints that have a native
matrix. Each result cell is `frames / last Boss HP`; its terminal result is
`player_hit / 100` in every case.
The corresponding `policy-native-stream-v14` through `v25` metrics files are
epoch arrays; values below are taken from the epoch selected by each matrix's
checkpoint metadata, not inferred from the filename.

| Version / matrix variant | Selected epoch / validation loss in corresponding metrics | Executed seed results |
| --- | --- | --- |
| v14 critical weighted3 | e11 / 2.498090 | 20260812: `523 / 5317.5`; 20260816: `620 / 5179.0` |
| v15 critical factorized | e11 / 2.518296 | 20260812: `551 / 5259.0`; 20260816: `652 / 5093.0` |
| v16 factorized r512 | e17 / 2.723818 | 20260821: `412 / 5480.5`; 314159265: `546 / 5278.5` |
| v17 action dropout 0.30 | e4 / 3.001734 | 20260821: `548 / 5304.5`; 314159265: `414 / 5475.5` |
| v18 identity r512 | e12 / 3.098788 | 20260821: `413 / 5479.0`; 314159265: `763 / 4966.0` |
| v19 Okuu vision | e21 / 4.135544 | 20260821: `413 / 5479.0`; 314159265: `517 / 5327.5` |
| v20 clean vision | e54 / 3.845654 | 20260821: `418 / 5473.5`; 314159265: `679 / 5078.5` |
| v21 clean chunk steps | final e30 / 4.142337 (minimum was e1 / 3.792430) | 20260821: `434 / 5446.5`; 314159265: `421 / 5470.0` |
| v22 critical e5/e10 | e5 / 4.852544; e10 / 4.830507 | 20260821: `548 / 5288.5`, `550 / 5257.0`; 20260825: `548 / 5287.0`, `774 / 4969.5` |
| v22 hard-general e5/e10/e20 | e5 / 4.546795; e10 / 4.502049; e20 / 4.445973 | 20260821: `437 / 5460.5`, `554 / 5272.0`, `544 / 5271.0`; 20260825: `719 / 5117.5`, `543 / 5289.5`, `420 / 5474.0` |
| v23 replay e3/e5/e10 | e3 / 4.574768; e5 / 4.548106; e10 / 4.500945 | seeds 20260821/20260825: `419-752 / 5078.5-5475.5` |
| v24 replay + future visual 0.05, e3/e5/e10 | e3 / 4.592971; e5 / 4.566108; e10 / 4.518486 | exact same paired native outcomes and movement as v23 |
| v25 replay + future visual 0.5, e5/e10 | e5 / 4.728051; e10 / 4.676230 | exact same paired native outcomes and movement as v23 e5/e10 |

The future-visual comparison used the same replay archive, seeds 20260821 and
20260825, and horizons 20/40/80 decisions. v23 has
`future_visual_loss_weight=0`; v24 uses `0.05`; v25 uses `0.5`. For paired e5
and e10 checkpoints, both nonzero weights emitted the same movement action at
every native decision as the corresponding v23 replay control. The stored
summaries consequently match exactly in frames, last HP, path distance, and
all smoothness fields: e5 is `419 / 5475.5 / 110.0` and
`420 / 5474.0 / 108.0`; e10 is `420 / 5474.5 / 111.3033556098` and
`752 / 5078.5 / 1208.4384987675` for the two seeds. v24 e3 likewise matches
v23 e3 at `419 / 5475.5` and `421 / 5467.0`; v25 e3 was not run.

The e5/e10 metric files record 25,732 train and 3,676 validation future labels.
Validation future-visual loss is 0.3607637490/0.3520529895 for v24 e5/e10 and
0.3605689421/0.3517647910 for v25 e5/e10. These auxiliary losses did not change
the tested native action sequence or produce a strict completion. No v14-v25
checkpoint replaces the published v1 model.

The current parallel-wave gap logic is still an Engine MPC planner/teacher
feature. Current-source novice/intermediate/expert bullet-group profiles bind
implementation SHA-256
`394d8298f5b42c6a42d586a75ab908bac0fdb7281d6cd5fd54478a83e048d99b`
and all strictly complete Okuu #3 seed 20260730 with continuous fire and zero
deaths. Novice accepts no corridor; intermediate records 42 observe decisions
but does not enter; expert records 172 detections and 44 certified entries,
changing 749 movement directions from either lower profile. Their report
SHA-256 values are respectively
`8fb9d002e7b0c8a6e993edd602995cf1d5e5fbcfdd4a457fb73a8bfc37c77850`,
`dd73c9fdb36e1e0a34c4dbf4c39ffd1e30749f667953d46eeae6594d6a217c57`,
and `3370da8b2debfdb5b77e05fb7f023d3c3181e0e3894a88e0e5dde6ef8eaf0047`.
This single seed demonstrates graded teacher processing and non-regression, not
human success-rate calibration or learned-policy evidence.
These reports are not training archives: no gap-aware DAgger or demonstration
archive has yet been generated, and the GRU has not been retrained from one.
Therefore the v14-v25 results neither train nor validate learned gap-entry
behavior.

The scratch-v5 cross-card results were Yamame #3 at 596 frames/1,367 HP,
Satori #5 at 331/2,898, Okuu #3 at 1,146/4,530, and Okuu EX #2 at
586/2,888.5; every run ended in `player_hit`, death 100. Okuu #3 and Okuu EX
survived longer than the retained model, while Koishi and Satori regressed, so
this was a trade rather than a broader clear. Its movement rates spanned about
21-90 direction changes per 1,000 frames: the failure is not merely rapid
oscillation or stationary collapse, but unreliable selection of a long-term
safe region from visual history.

Stage 1 also remains unresolved. A full-stage MPC-teacher attempt at seed
20260731 reached a 7,199-frame `time_limit` with death 0 and failed. A later
teacher attempt at seed 20260732 reached `stage_complete` after 12,453 frames
with death 0, but it is MPC-teacher data, not pure-GRU evidence. The long stage
episode was excluded from the spell-policy training pool because its duration
overweighted stationary decisions. The final pure GRU then failed Stage 1
Normal at 543 logical frames with `player_hit`, death 100, and no movement.

## Limitations

- Strict pure success has been shown only for one card: three known expert
  seeds and one independently selected held-out seed on the final source.
- Every other executed target in the matrix failed; there is no full-stage
  pure success.
- Intermediate and novice profiles do not yet complete Koishi #1.
- The checkpoint has learned recurrent state but no external strategy memory,
  scenario conditioning, or proficiency conditioning.
- DAgger and MPC reports must remain labelled teacher-assisted/planner runs.
  Their successes cannot be pooled with pure-GRU outcomes.
- Cross-card and full-stage behavior remains far below the requested
  human-like general-player target.
