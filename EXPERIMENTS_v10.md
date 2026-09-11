# Contextual TD7 experiment history — V10

## v10.8.0 — Sweepable headless training launches

- `run_headless.py` forwards everything after a bare `--` to the Python
  controller's own CLI, so every `main.py` training option becomes a sweep
  axis. Previously `run_ud7_stage1.py` and `run_ud7_best_stage2.py` held a
  fixed argument list and only the input DB and episode count could vary.
- The overrides are appended after each runner's defaults, so a swept value
  wins while the UD7 contract (UBOC 5, reward Q, SALE/LAP, random eviction,
  replay capacity, stage horizon) stays the default for anything not swept.
- Reject the options the launcher owns — `--port`/`--ports`/`--num-sim`,
  `--mode`/`--stage`, `--sim-end-time`, `--device`, `--wandb`/`--no-wandb`,
  `--episode-summary-path`, `--checkpoint-root`, `--resume-checkpoint`,
  `--load-stage1-policy` — naming the launcher flag that sets each one.
  `--sim-end-time` matters most: the native simulator takes its horizon from
  `--end-time`, and a second value would desynchronize the two.
- `--device` now reaches training as well as inference, replacing the
  hardcoded CUDA in both training runners. The default stays `cuda`.
- `--note` sets the run tag that names the W&B run and the checkpoint root,
  so sweep configurations started in the same minute stay distinguishable.
  Omitting it keeps each runner's existing tag (`episodebest`, `besttat`,
  `headless`), so an unswept launch is byte-identical to v10.7.0.
- Record `note` and `train_args` in `--check` output and in each run's
  `run.json`, and append the overrides to the W&B run description, so a
  finished run states the parameters it used.
- Validation: 16 headless tests and 27 subtests pass, plus the full 444-test
  suite. `run_ud7_stage1.py --check` resolves swept `--seed`, `--batch-size`,
  `--warmup-steps` and `--exploration-noise-std` into the runtime config while
  keeping the UD7 defaults, and an unknown override is rejected by `main.py`
  before any process starts. No simulator rollout was run for this change.

## v10.7.0 — Log episode final TAT and simulation acceleration in W&B

- Add sparse `eval/final_tat` (seconds) at completed episode boundaries and
  `runtime/acceleration_rate` (simulated seconds / wall seconds) at the existing
  W&B logging cadence. The TAT source is the last active simulator packet;
  the separate result-DB evaluation window is unchanged.
- Measure acceleration from the first active packet of each episode, excluding
  GUI connection/model-loading waits. Include simulator execution, policy/learner
  work and TCP waits while the episode runs; reset the clock for every episode.
- Keep terminal-only metrics sparse in CSV exports: scan history without a
  required-key intersection, then project the configured export columns.
- Delegate implementation and regression tests to gpt-5.6-luna; root reviews
  metric semantics, export/step behavior, documentation and PR integration.
- Preserve checkpoint/reward/simulation behavior and the running training
  process. New logging applies when a Python controller starts with this code.
- Validation: 62 runtime/headless tests and 26 subtests pass. Cover sparse
  terminal logging, duplicate/Reset-only ends, invalid/failed/early boundaries,
  first-packet timing and episode/reconnection resets, and CSV row preservation.
  Both training and actor-inference schemas contain the new keys; export has
  221 columns. No online W&B run or simulator rollout was started for this change.

## v10.6.5 — Package the validated headless release for GitHub

- Branch from contextual-td7-v9 as headless, including the UD7 prerequisites,
  native host sources, launchers, evaluation/checkpoint workflow and reports.
- Include previously ignored GUI handshake and evaluation-window regression
  tests. Keep checkpoints, generated binaries, vendor DLLs, input/results DBs,
  decompiled vendor sources and live logs outside the new commit.
- Add explicit UD7 ensemble metric export guards alongside the shared W&B
  schema, and document the original licensed simulator prerequisites for a clone.
- Remove three stale patch-version assertions from checkpoint/reward tests;
  retain their major-version rejection and reward compatibility checks.
- Simulation and learner behavior are unchanged from the validated release.
  The in-progress v10.6.3 training is left running with its existing runtime.
- Validation: native host rebuild and GUI .NET 4.8 metadata comparison pass;
  Stage 1 50-episode argument dry run and 219-column W&B export schema pass.
  Targeted Python suites cover 286 tests and 105 subtests: the initial run had
  only three stale version assertions; the affected 59 tests and 40 subtests
  all pass after the correction. Existing GUI packet/DB parity evidence is
  documented in HEADLESS_REVALIDATION.md; no new full GUI rollout is claimed.

## v10.6.4 — Revalidate and document the final headless implementation

- User requests a second verification and a step-by-step account of actual
  decompilation tools, analysis, implementation and evidence. Delegate technical
  documentation to gpt-5.6-sol and consistency edits to gpt-5.6-luna; root owns
  the code/evidence checks. No changes to simulation or learning semantics.
- Recheck immutable 2,000-second GUI/fixed-headless DBs and recorded wire data,
  strictly decompile the four managed targets again, verify framework metadata
  and the transport patch, and run a short fresh production-host wire rollout.
  Save new evidence under results/parity_recheck without overwriting originals.
- Existing Stage 1 run remains v10.6.3; no training restart or episode monitoring.
- Fresh 120 s production-host rollout completed in 32.15 s with TAT 74.4:
  `results/parity_recheck/rollouts/run_0911_1825_v10.6.4_b_rl_0.0-1.0_Q_headless`.
  All 120 comparable GUI frames and 121 prior fixed-host frames match; complete
  payload SHA-256 values match their indices. The preserved GUI final payload
  remains incomplete and is explicitly excluded from complete-frame comparison.
- Preserved 2,000 s DBs retain their hashes, equal 43-column schemas and exactly
  equal values for all 8,921 commands at the common completion cutoff. This is
  DB revalidation, not a fresh full 2,000 s rollout. Six artifact identities match.
- Four strict managed decompilations succeeded with outputs identical to prior
  strict results; TCP 131-method audit identifies only the supervisor change.
  Framework positive/targetless negative checks passed; 13 headless tests passed.
- Updated implementation and usage/comparison/time/Linux documents; new
  HEADLESS_REVALIDATION.md and results/parity_recheck/revalidation.json preserve
  evidence, tool/source hashes, scope limits and actual model delegation.
- Validate the documented decompiler compilation with its required netstandard
  and System.Reflection.Metadata references. Allowlist final root reports and
  the build-required VerifyFramework.cs in .gitignore; large raw artifacts stay
  local. Check local documentation link targets before delivery.

## v10.6.3 — Restart Stage 1 for 50 episodes with GUI-compatible runtime

- User requests a fresh 50-episode Stage 1 run after v10.6.2 fixed native
  framework compatibility. Stop the previous 100-episode batch and its waiting
  finalizer, preserving its existing checkpoints/results and state snapshot.
- New learner and normalizers start fresh: reward Q, UBOC five critics, SALE/LAP,
  random replay eviction, CUDA, original input DB, 2,000 simulation seconds per
  episode, port 9100. Existing hyperparameters remain unchanged.
- Keep one checkpoint per episode and the lowest eligible full-horizon terminal
  TAT checkpoint. After 50 episodes, the existing finalizer audits selection and
  runs a 2,000-second deterministic inference. No per-episode assistant polling.
- Run and checkpoint directories come from EXP_META/_make_run_name. The new
  runtime is built with the GUI's .NET 4.8 target and checked at build/startup.
- Started `results/headless/run_0911_1740_v10.6.3_b_rl_0.0-1.0_Q_headless`;
  fresh checkpoint root ends in `run_0911_1740_v10.6.3_b_rl_0.0-1.0_Q_episodebest`.
  Actual training PID 27692, listener 28616 on 9100. Native 4.8 target, TCP
  handshake, advancing simulation and fresh warmup/replay were verified.
  Finalizer successfully waits for this exact process identity. Previous state
  and checkpoint pointer are archived under results/stage1_restart_v10.6.3.

## v10.6.2 — Match GUI framework semantics for idle-vehicle selection

- The headless entry EXE lacked the GUI's TargetFrameworkAttribute (.NET 4.8).
  Legacy sort compatibility reordered tied candidates in Scheduler.GetFastsOHTNPath,
  causing different idle OHT destinations at 28.3994059321392 seconds.
- The user's actual GUI raw capture matches the old host through 28 seconds;
  OHT 53135/53137 destinations diverge at 29 seconds, followed by policy costs
  and job data. A target-attribute-only private probe matches all 120 complete
  captured frames (0..119), ignoring only cost-packet wall-clock timestamps.
- Add the matching assembly attribute, startup assertion/log, and a build-time
  metadata comparison against the original GUI. Old targetless EXE rejected;
  fixed EXE accepted. No original engine/model/C++/GUI binaries changed.
- Original C# engine/model/GUI/private transport full decompilation succeeds with
  strict assembly resolution after framework reference-path repair; C++ source
  recovery is not claimed. 131 transport methods differ only in the known
  supervisor Sleep(5) patch. Hypotheses involving A* initialization/animation/
  GUI model constructor did not explain the discrepancy; GUI shape probe had
  window-handle errors and is not counted as full GUI equivalence validation.
- EXP_META/generated runs, native event probe, wire decoders/comparators and
  minimum sort reproduction live under results/parity_fix. Production 2,000 s
  fixed episode54 inference completed in 497.34 s with TAT 167.7, matching GUI.
  At 07:32:59 cutoff, all 43 COMMAND_LOG fields of all 8,921 commands are exactly
  equal (mean 167.71560721892166, reroutes 5,099). Headless flushes 95 additional
  completed commands through 07:33:19; this does not alter matched-row equality.
- Python headless suite: 13 tests passed. Native build and compatibility checks
  passed. HEADLESS_PARITY_FIX.md records exact evidence and remaining scope;
  LINUX_SIMULATION_PLAN.md covers the user's Linux multi-parameter server goal.
- Existing running native runtimes remain unchanged; new launches use the fix.
  Historical pre-fix scores require reevaluation before comparison to GUI;
  full 45,000 s evaluation parity is outside this 2,000 s verification.

## v10.6.1 — Audit the real episode54 GUI/headless discrepancy

- User reports GUI TAT 167.7. Actual process command, W&B output and terminal
  JSONL identify episode54/108050, not the episode57 training checkpoint.
- Preserve and compare the completed GUI st1_gui_1.db with the headless
  episode54 result at matching completion cutoffs. Inspect native parameters
  and early trajectory metrics. No production policy/native changes.
- Diagnostic artifacts and conclusions are stored under results/episode54.

Result: GUI packet 167.7 s versus headless 163.0 s. At the shared completed-time
cutoff 07:32:59, native means are 167.715607 and 162.974547 s, so final flush
timing is insufficient to explain the 4.741060 s gap. A production-host 300 s
scalar trace first differs in sampled route diagnostics at 29 s and sampled
policy mean/std at 49 s. Native RESULT_SPEC tables match. The GUI's actual
Python batch automatically sets AI2_0 and rerouting=true; both routing toggles
are hidden by Visibility.Never. Remove misleading manual-toggle instructions.
No production physics/algorithm change; root cause remains unresolved.
Reproducible read-only DB comparison: results/episode54/audit_gui.py.

## v10.6.0 — Correct GUI launch guidance and distinguish the meeting evaluation window

- Native GUI log shows an empty pre-load TCP handshake followed by ordinary
  RunSimulation(72000), which does not refresh Python topology. Correct the
  instructions to use Python TCP/IP Open Files folder batch execution, with
  a dedicated single-input folder. Do not change the installed native DLLs.
- Permit idle waiting at a command boundary before an episode starts, retaining
  the bounded timeout for incomplete packets and stalled active episodes.
  Reject active frames without model initialization before policy execution.
- Explain the user-provided meeting contract: 12 hours evaluated plus 30 minutes
  for completion, separate from the current 20-hour input and legacy 10-hour
  evaluate.py window. Add explicit evaluation window / baseline DB selection;
  never silently apply the legacy 174.4236-second baseline to a different window.
- Preserve the running training, the fixed 45000-second Stage 2 horizon and
  episode54 checkpoint identity. Verification recorded after the checks below.

Validation: 100 tests and 68 subtests passed (existing SyntaxWarning), including
wire-level empty handshake -> four timeout intervals -> populated model reset,
partial-packet timeout and fail-before-policy checks. A real 30-second native
inference completed in 11.9 seconds. After adding separate meeting_eval_* CSV
fields, all 18 affected evaluation/headless tests and 15 subtests passed.
The existing completed baseline scores 176.73236276351494 s over 210804 commands
on 07-19, versus 176.10184197466072 s on legacy 09-19. The twelve-hour reference
has not been equated with the historical GUI baseline. No GUI full rollout
claim is made. Original simulator DLLs and the active training remain unchanged.

## v10.5.2 — Evaluate the requested 108050-step Stage 1 policy

- Select immutable episode 54 / step 108050, terminal training TAT 161.9 s,
  from run_0910_2228_v10.2.0_b_rl_0.0-1.0_Q_episodebest. Pin and verify its
  SHA-256 af984e9e3a3858d49fe5f512b2007b659acfc19c5baa8f7c229b9dba1276d7f3.
- Run one deterministic 2000-second CUDA headless inference on port 9101,
  with saved action strength and frozen normalizers. Keep training on 9100.
- Provide a GUI controller command on port 9102 with the same checkpoint,
  horizon and runtime options, plus distinct result metadata for comparison.
- No policy, network, reward, native dynamics or training configuration change.
  Results and GUI instructions are recorded in results/episode54/README.md.

Validation/result: full 2000-second deterministic inference completed in
472.14 seconds. Terminal packet TAT 163.0 s; native completed-command mean
163.01164695479164 s over 9047 rows. No early termination, DB quick_check=ok,
and pinned/runtime checkpoint hashes match. GUI launch config differs only
in port and summary path. 72 tests and 55 subtests passed (existing warning).
GUI rollout remains for the user to run; this is not the 09-19 final-window score.

## v10.5.1 — Audit actual input time coverage and GPU preprocessing

- Inspect the encrypted input through the installed simulator's existing,
  licensed load-only path. Measure stored model dates separately from actual
  command timestamps, Python's 45000-second horizon and evaluate.py's window.
- Record a bounded offline GPU preprocessing benchmark under results/gpu_audit;
  no running learner, production native dynamics or policy behavior is changed.
- Time coverage findings and validation are recorded in INPUT_TIME_AUDIT.md.
- Current input contains 347308 commands spanning 72001 seconds; model dates
  span 72000 seconds. Preserve the existing 45000-second episode and evaluate.py
  window, and log model/effective dates separately in the native host.
- Replace only the scalar float finiteness check in observation construction
  with math.isfinite. Synthetic real-topology benchmark: identical arrays,
  27.29 to 16.45 ms per build. GPU gather remains an unapplied candidate.
- Completed v10.4.2 baseline scores 176.10184197466072 seconds on the existing
  evaluate.py window versus its hardcoded 174.4236; no GUI parity claim.

Validation: 104 tests and 58 subtests passed; one pre-existing SyntaxWarning.
Native build and licensed load-only succeeded, printing the actual 72000-second
model duration. Existing running training was not restarted or monitored.

## v10.5.0 — Make policy evaluation reproducible and expose a cost-strength experiment

- Pin the inference checkpoint and optional Stage 1 prefix once per run, so
  later training saves cannot replace a policy between its rollouts. Record
  source paths, saved stage and SHA-256 identities in run.json.
- Reject Stage 2 checkpoints without an explicit matching Stage 1 prefix
  before starting the native process. The existing runtime verifies prefix
  provenance against the saved Stage 2 SHA-256.
- Optional --rl-cost-lambda (finite 0..1, inference only) passes an explicit
  cost-strength calibration to the runtime. Omission retains the saved value.
  This deliberately changes the full rollout, including a Stage 1 prefix;
  ordinary checkpoint comparisons should omit the override.
- No network, reward, topology, replay or native dynamics change. The full
  baseline already running under v10.4.2 is left intact.

Validation: 91 tests and 66 subtests passed (one pre-existing SyntaxWarning).
Two real native baseline episodes reused one executable and advanced Python
episode IDs 1 to 2; both DBs and summaries completed. Runtime preparation was
0.781 seconds. The original evaluate.py TAT and the new final-window helper
were exactly equal on a 175298-command real evaluation window. A real Stage 1
checkpoint was pinned and read with its SHA-256 unchanged; an explicit lambda
0.45 survived checkpoint config restoration while preserving UBOC/5 critics.


## v10.4.2 — Validate sequential reuse and calibrate the full-horizon baseline

- Run two 30-second baseline episodes to verify one prepared native runtime,
  one persistent Python server, independent native resets, distinct result DBs,
  complete episode summaries, and absent final-window verdicts for short runs.
- Start a 45000-second baseline rollout on port 9101, retaining the existing
  training on 9100. It will write evaluate.py-compatible final TAT and coverage
  without assistant polling of the 100-episode training run.
- Preserve native simulation behavior: GUI-default and direct GUI-loader
  diagnostics do not justify changing the physical engine or event loop.
- The saved v9.0.0 Stage 2 policy was identified as schedule step 220000
  (global step 232000), with its matching Stage 1 prefix. Identity with the
  user's reported 4.2% run remains unconfirmed; no performance claim is made.


## v10.4.1 — Compare against the installed GUI model-loading method

- Diagnostic-only reference: invoke the installed MainFrame.LoadSimModel method
  without constructing a form, after the same licensed native initialization.
  Compare the first 300 seconds against the hand-extracted headless loader.
- The previous default-initialization experiment was identical at all 2001
  TAT, queue, waiting, policy-mean and route-correlation samples. Connecting
  TCP before model load also reproduced the original first-300 trajectory.
- The unmodified native transport was tested separately over the first 300
  seconds; its trace is retained to assess the private supervisor sleep.
- No production simulator dynamics have been changed on these hypotheses.


## v10.4.0 — Reuse isolated runtime binaries and score the final window

- Build the native host once directly into each run's runtime directory;
  reuse it for sequential episodes. No per-episode executable copying and
  no shared-bin replacement that could affect a currently running batch.
  Python remains alive; each native process still starts with fresh singletons.
- Record runtime preparation time and SHA-256 identities in run.json.
- Add optional --stage1-policy for correct Stage 2 actor inference with a
  frozen prefix, and default its horizon to 45000 seconds.
- Score evaluate.py's strict activation window automatically in evaluation.csv.
  Baseline is 174.4236 seconds, target is 165.70242 seconds; short rollouts
  receive no final target verdict. Preserve command coverage alongside TAT.
- Isolated next diagnostic: connect Python before loading the native model,
  matching the GUI user's TCP-on-before-Open-Files sequence. Production C#
  behavior is unchanged until a controlled comparison establishes a fix.


## v10.3.5 — Reproduce GUI default initialization and establish the final evaluation contract

- Isolated native diagnostic: restore GUI AI20 parameter copy/apply/setting event,
  animation interval 0.1, Python flag before model loading, Astar penalty before
  loading, and acceleration flag timing after initial line cost. This is an
  A/B hypothesis, not a confirmed fix. The running Stage 1 host is unchanged.
- GUI comparison must distinguish initial 2000-second TAT from evaluate.py's
  09:00–19:00 activation-window metric (baseline 2.90706 minutes).
- Investigate run-local immutable runtime reuse and checkpoint selection toward
  a 5% improvement, preserving per-episode native process reset.


## v10.3.4 — Diagnose the same-policy GUI/headless discrepancy

- Investigate GUI run `o9s1ssl5` using its W&B history, native log, and local
  result provenance. The apparent `ud7_st2_1.db` candidate was overwritten
  after that run and is excluded from causal comparison.
- Use an isolated diagnostic host (only native INFO logging and assembly/flag
  reporting added) and an evaluation-only scalar tick recorder. Original host,
  engine, policy and the running training process remain unchanged. Diagnostic
  sources and results live under `results/gui_compare`.
- Compare the fixed original checkpoint with the GUI's frozen Stage 1 prefix;
  record confirmed differences and distinguish them from unproven hypotheses.

## v10.3.3 — Compare the GUI training run and evaluate the original UD7 policy

- Inspect the user-provided GUI W&B run `szclabdk` and compare its configuration,
  episode TAT and available trajectories with the existing headless artifacts.
- Pin `PythonCode/v10.0.0_stage1_policy_ud7.pt` (runtime step 117,000,
  episode metadata 59, applied action scale 1.0) and evaluate one 2,000-second
  inference on port 9101. Existing training and its finalizer remain running;
  no repeated monitoring of the 100-episode batch is resumed.
- This patch records an experiment and comparison report without changing the
  simulator or policy behavior. Evaluation completed at 2,000 simulation seconds
  in 502.65 wall seconds, TAT **168.2 s**. Native COMMAND_LOG contains 9,030
  completed commands with mean TOTAL_TIME **168.201089922 s**. Integrity, hashes,
  fixed action scale/normalizers, no learning/replay and successful exit passed.
- GUI run `szclabdk` is training and stops during episode 59, not a complete
  frozen evaluation. Its 11,732 history rows were downloaded. Episodes 6–50
  have median/mean TAT 166.9/169.496 s versus headless 166.7/168.944 s; GUI
  last samples are at 1999 s, whereas headless terminal summaries are at 2000 s.
- Found the exact original model hash in GUI Stage 2 run `o9s1ssl5`. Its first
  frozen Stage 1 prefix has TAT **164.4 s at 1999 s**, with scale 1.0 and zero
  learner updates. The new 168.2 s evaluation is 3.8 s higher, while 4.6 s
  lower than headless episode 47 inference. Warmup state differences and
  2000-vs-2001 episode observation counts prevent claiming numerical parity.
  The source of the residual gap is not established.
- `GUI_HEADLESS_COMPARISON.md` records sources, settings, horizon caveats and
  charts. Checkpoint/variant suites passed **59 tests and 40 subtests** with
  one existing regex escape warning; the three version assertions were updated.

## v10.3.2 — Isolated episode 47 evaluation and headless implementation audit

- Pin the explicitly requested episode 47 checkpoint (step 94,043, training
  TAT 162.5 s) and evaluate one complete 2,000-second episode on port 9101.
  Keep the existing training on port 9100 and its unattended finalizer running;
  discontinue assistant polling of the 100-episode training batch as requested.
- Document the GUI-to-headless initialization and event-loop mapping, shared
  native assemblies, private transport change, isolation and validation limits
  in `HEADLESS_IMPLEMENTATION_REPORT.md`. No simulator or learner behavior is
  changed in this patch. The live training retains its loaded v10.2.0 runtime.
- The isolated evaluation completed at 2,000 simulation seconds in 499.08 wall
  seconds with TAT **172.8 s**, compared with **162.5 s** during training.
  Its native DB contains 8,944 completed commands with mean TOTAL_TIME
  172.825241838 s, matching the rounded TCP metric. Source/pin/input hashes,
  frozen normalizers, action scale 1.0, no learner updates/replay, native DB
  integrity and successful exit were verified. Results and provenance are in
  `results/episode47/result.json`; this does not claim GUI numerical parity.
- A read-only comparison of the original transport and the actual evaluation
  copy checked 131 method bodies. Only the supervisor's inserted Sleep(5)
  instructions differ. Audit source/output and file hashes accompany the report.
- Validation: checkpoint, variant, headless and finalizer suites passed
  **80 tests and 55 subtests** with PYTHONPATH=PythonCode. Updated three release
  version assertions. The initial invocation without PYTHONPATH failed test
  collection; rerunning with the repository module path resolved it.

## v10.3.1 — Keep automated inference within native Windows path limits

- A real full-horizon inference validation was launched from the current
  episode 8 best (TAT 167.4 s, saved action scale 0.3019800903603665) while
  the existing 100-episode training batch remained running. The first deeply
  nested output placed the native EXE at 253 characters and its config at
  260; CLR exited with -1 before any native log output. The same pinned model
  and input start successfully with a shorter output root.
- Final evaluation now uses a short deterministic `results/eval_<runhash>/`
  root, linked from the training run's final report. Added preflight checks
  for native input, result, working DB and EXE config paths so oversized paths
  produce an actionable error before starting Python. No model, reward,
  observation or learning changes.
- Existing training continues with v10.2.0. The live validation uses the
  already loaded v10.3.0 runtime and reports fixed saved action strength,
  zero replay collection and disabled learner updates. The waiting finalizer
  is replaced after tests so final evaluation uses the corrected path.
- Validation: finalizer, headless, checkpoint and variant suites passed
  **80 tests and 55 subtests** (one existing regex-escape warning). Only the
  verified waiting finalizer was restarted; the original training process
  and the live inference validation continued running.
- Real validation completed: the fixed episode 8 policy completed 2,000
  simulation seconds at **166.2 s TAT** (its exploratory training TAT was
  167.4 s). Native wall time including batch handling was 495.84 seconds
  while training ran concurrently. Result DB integrity, successful process
  exit, checkpoint/input hashes and disabled learning/replay were verified.
  This is an interim validation; final best selection still waits for the
  full 100-episode batch and is evaluated separately afterward.
- The real inference summary also exposed an overly strict verifier check:
  standalone actor inference reports runtime `stage: null`; Stage 1 identity
  belongs to the pinned checkpoint payload. The verifier now accepts an unset
  inference stage or explicit Stage 1 while continuing to reject Stage 2;
  fixtures now model the real standalone inference summary.
- Independent native-data cross-check: inference `COMMAND_LOG` contains
  9,015 completed commands with AVG(TOTAL_TIME) = 166.217790682 s, matching
  the TCP/JSONL TAT of 166.2 s after rounding. All first ten completed training
  episodes also match their native command averages to the reported 0.1 s.
  The detailed audit is saved alongside the live training artifacts.

## v10.3.0 — Complete Stage 1 selection and inference after the running batch

- Added `finalize_ud7_stage1.py`, a one-time follower for the already running
  authorized 100-episode batch. It waits on the original Windows process
  handle and verifies its creation time, so a recycled PID or stale state
  cannot cause it to follow an unrelated process. It does not restart or
  change the current training process. An OS file lock prevents duplicate
  followers for the same run.
- After successful training completion, audits the requested episode count,
  input hash, every episode metric/checkpoint pair and checkpoint hash, every
  native result DB/log, and recomputes the lowest eligible TAT with earliest
  episode winning ties. A partial batch, warmup promotion, corrupted artifact
  or inconsistent best selection is an error.
- Pins the verified policy, checks its actual payload (UD7, trained weights,
  populated frozen normalizers, saved action scale and provenance), and starts
  a full 2000-second deterministic inference through the existing headless
  launcher. The launcher fills EXP_META before initialization; no metrics or
  learning semantics change. Existing v10.2.0 training remains running with
  the code it already loaded; subsequent inference uses the current runtime.
- `finalizer.json` records the follower phase and PID. On successful inference,
  `final_stage1/` contains the pinned checkpoint, selection provenance,
  episode ranking, result JSON, report, and native inference results. A failed
  or early-stopped inference does not become a completed report. The agent
  must still inspect these artifacts before declaring the thread goal complete.
- This is a checkpoint-compatible workflow addition. Read-only `--audit-only`
  refuses incomplete training runs. Tests cover complete and incomplete runs,
  lowest-TAT/tie selection, warmup exclusion, missing/corrupted artifacts,
  input identity and native error logs.
- Validation: finalizer, headless, checkpoint and variant suites passed
  (76 tests, 53 subtests); added inference acceptance/rejection checks also
  pass (10 finalizer tests, 4 subtests). The real episode 6 checkpoint hash
  and payload passed validation: 1,900 learner updates, frozen populated
  normalizers, saved action scale 0.09105548385103059. Its provisional TAT
  is 173.4 seconds; final selection still waits for all 100 episodes.

## v10.2.1 — Keep the saved action strength fixed during checkpoint inference

- While the authorized 100-episode fresh Stage 1 run was live, reviewed the
  follow-up inference path. Region-mode inference re-evaluated the curriculum
  with warmup bypass and advancing evaluation ticks instead of using the
  checkpoint's `applied_action_scale`. Early checkpoints could therefore be
  evaluated with a different action strength from the selected artifact.
- For checkpoint-backed actor inference, the region action scale now comes
  from the restored learner's saved scale. It remains fixed across ticks and
  episode resets. Existing training curricula, fixed residual modes and the
  Stage 2 frozen Stage 1 prefix are unchanged. No new W&B keys or checkpoint
  fields; this corrects the fixed-policy inference contract.
- The active training process continues with its already loaded v10.2.0 code
  and checkpoints. It was not interrupted or restarted. The subsequent
  inference process uses v10.2.1, which accepts those v10.2.0 checkpoints.
- Regression reproduced before the fix: a real saved/loaded test checkpoint
  with scale 0.73 was evaluated at scale 1.0. The test now checks the restored
  scale on actual evaluation ticks and after changing the runtime counter and
  resetting the episode, while retaining actor weights, zero noise, no replay
  insertion and no learner updates.
- Validation: training runtime, checkpoint, variants and headless suites:
  **113 passed, 56 subtests passed** in 20.35 seconds. The pre-existing regex
  escape warning remains. Live training PID 23424 continued advancing normally.

## v10.2.0 — Headless Pinokio execution and automated episode evaluation

- Added a .NET Framework 4.8 x64 console host around the installed simulator
  engine/model assemblies. It accepts input DB, result folder/name, TCP port
  and horizon, loads the original exported DB, initializes the model without
  forms/shapes, and executes the native event loop. Original GUI binaries
  remain unchanged. Native engine license checks are retained, including an
  early preflight check before creating a run.
- The repository contains the algorithm DLL source but not the full GUI or
  engine source. `HeadlessLoader.cs` follows the installed LoadSimModel
  sequence; Python execution follows MainFrame_PythonRun's event loop and
  existing initialization/reset/terminal protocol. GUI default Euclidean
  penalty-off, AI2.0/rerouting and acceleration settings are reproduced.
  Arbitrary custom GUI settings and long-run numerical equivalence have not
  been validated. No reward, observation, network or checkpoint contract changes.
- Found a busy-spin loop in PServer_Python.StartingServerLocal. Build uses
  Mono.Cecil to add a 5ms supervisory-loop wait to a private transport copy.
  The installed original SHA-256 and downloaded build archive are pinned.
  No per-tick sleep or changes to routing/protocol payloads. The engine still
  computes discrete events on CPU; CUDA remains the RL device.
- `run_headless.py` owns Python and simulator subprocesses. It accepts one
  DB or a sorted DB directory, cycles inputs, isolates episode runtimes and
  native logs, rejects port conflicts/duplicate output names, preserves the
  learner across simulator reconnects, waits for terminal checkpoint/metrics,
  flushes native completion/aggregate results and checks SQLite integrity.
  Failures stop the batch; result/checkpoint files from completed episodes
  are retained. Successful temporary work is deleted within the unique run
  directory only; `--keep-work` retains it. A stop-file closes idle Python
  cleanly; interrupt/failure cleanup targets owned children only.
- Stage 1 and Stage 2 launchers accept port, JSONL summary path and no-W&B
  overrides. Training horizons are fixed at 2000/45000. Inference resolves
  the fresh Stage 1 best unless a checkpoint is explicit. Best selection and
  Stage 2 pinning still require the v10.1.0 eligibility/hash checks.
- `episode_summary_path` is launch-controlled on checkpoint resume.
  Terminal metrics use the last active observation and distinguish full
  horizon, early termination and training failure. No W&B metric additions;
  `episodes.jsonl`, `evaluation.csv`, `run.json` are local artifacts with EXP_META.
- Entry points and examples: `start_ud7_stage1_headless.cmd`,
  `start_ud7_stage2_headless.cmd`, `HEADLESS.md`. Existing Stage 1/Stage 2
  `.cmd` entry points now delegate to these; Python-only server scripts
  remain available for manual GUI use.
- Real local verification on AICC_Input_260403.db (4999 rails, 1004 OHT):
  two consecutive 60-second TCP episodes; then two consecutive 120-second
  TCP episodes, each TAT 75.1 and full horizon, with reconnect and clean
  Python shutdown. A native-only 600-second run completed in 28.71 engine
  wall seconds (32.77 including startup/result checks).
- A fresh CUDA Stage 1 warmup episode ran 2000 simulator seconds, 2,229,937
  native events, 378.44 engine wall seconds (383.78 total), and saved
  `episode_000001_step_000002000.pt` plus provenance. Last packet time=1999,
  TAT=171.5. Correctly ineligible for best: warmup still active, normalizers
  not frozen and no learner updates. This verifies episode checkpoint I/O,
  not trained-policy quality. Source warmup run:
  `results/headless/run_0910_2217_v10.2.0_b_rl_0.0-1.0_Q_headless`.
- Targeted tests cover port ownership, output-path restrictions and cleanup,
  terminal deduplication, nonfinite metrics, incomplete JSONL writes, native
  caught-error detection, graceful headless termination, GUI TCP reuse, and
  the existing training/checkpoint/protocol suites. No simulator GUI was
  opened during headless verification. Full suite: **419 passed, 204 subtests
  passed** in 53.64 seconds; two pre-existing test warnings (regex escape and
  tensor-to-scalar conversion).
- Final checks: normal 30-second native run succeeded with the license/state
  preflight checks enabled. An intentional one-second batch timeout returned
  failure, recorded the error, closed its port and left no headless children.
  Existing Stage 1 `.cmd --check` resolves the headless preset without starting
  a run. The original transport DLL hash remains unchanged.

## v10.1.0 — Fresh Stage 1 with episode checkpoints and lowest endpoint TAT

- User reports poor performance from step 95000 and requests fresh Stage 1.
  `run_ud7_stage1.py` loads neither a checkpoint nor prior normalizers. It
  creates an isolated directory using `_make_run_name` and fills EXP_META
  before runtime/W&B initialization. Reward Q, UBOC 5, SALE/LAP, random
  100000-step replay and existing warmup/exploration schedules are retained.
- Single-simulator Stage 1 saves its full compatible checkpoint on terminal
  v=1, with Reset as a fallback before cached episode diagnostics are cleared.
  The PClient reset handshake has already consumed new simulator data, so
  selection uses cached last-active-packet TAT, not reset PClient fields.
  Duplicate terminal/reset notifications are idempotent.
- Each episode snapshot has immutable episode/step naming and JSON provenance.
  All observed episode ends are saved, including warmup and early termination.
  Best selection requires the full horizon (last SimTime >= end-1), no forced
  termination, the entire episode after warmup, populated frozen normalizers,
  positive learner update count and finite positive TAT. Ties retain the first
  winner. `best_tat/checkpoint.pt` is a byte-identical copy of the selected
  episode snapshot, with SHA-256 in `selection.json`.
- Checkpoints retain encoder/actor/critic, SALE, optimizers and normalizers;
  replay remains excluded under the existing contract. Writes use temporary
  files and rename. A failed episode save stops training rather than silently
  dropping the model. Existing latest/periodic cadence is preserved.
- `run_ud7_best_stage2.py` now resolves `ud7_stage1_current.json`, refuses to
  start without an eligible best, pins a per-launch policy copy and verifies
  its SHA-256. It no longer contains a 95000 checkpoint fallback. The existing
  2000-tick frozen prefix and Stage 2 warm start are unchanged.
- Double-click entry points: `start_ud7_stage1.cmd`, `start_ud7_stage2_best.cmd`.
  Operator steps and metric limitations are documented in `UD7_START.md`.
- No W&B metric keys or reward/network contracts change; metadata is stored
  with artifacts and printed to the console. Compatible feature addition:
  v10.1.0; prior v10 checkpoints remain compatible.
- Validation: episode eligibility/promotion, duplicate termination, reset
  fallback, failed saves, and loading the saved best as a frozen Stage 2
  warm-start policy; existing runtime and TCP protocol regression suites.
  Fresh-launch `--check` validates the Stage 1 preset without opening TCP or
  starting training. Actual Pinokio GUI execution is left to the operator.

## v10.0.1 — Select the retained UD7 policy nearest the best Stage 1 episode

- Reviewed local Stage 1 W&B history `szclabdk`, checkpoint cadence, UBOC
  target aggregation, actor ensemble averaging, and Stage 2 prefix handling.
- The old `PythonCode/v10.0.0_stage1_policy_ud7.pt` is byte-identical to
  Stage 1 `latest/checkpoint.pt` (SHA-256
  `3dfd5fd8a308540a065934fc9a46abf382252c01f79af8408b1e69e80eeaa0d3`).
  Static pickle opcode inspection identifies runtime step 117000. It was
  not a best-TAT selection.
- Lowest completed-episode last logged TAT: **162.5 s**, episode 47,
  runtime step **94000**, simulator time **1999**. There is no retained
  step-94000 checkpoint: periodic snapshots are every 5000 steps and the
  latest snapshot is overwritten every 1000 steps. The unfinished final
  episode (59) is excluded. History is in `ud7_stage1_history.json`.
- Select **step_95000.pt**, as discussed with the user: 1000 updates after
  the best episode versus 4000 updates before it for step_90000.pt.
  This is a proximity-based candidate, not an inference-validated optimum.
  Step 95000's TAT 157.4 s was at simulator time 999 and must not be ranked
  against end-of-episode TAT. SHA-256:
  `d2f8cdbab2c5571250a68cc247b7f12ea4816cd4ce17c289b4773d2b6f43fd31`.
- New launcher: `PythonCode/run_ud7_best_stage2.py`; `--check` validates
  the configuration without deserializing a checkpoint or opening a server.
  The launcher fills EXP_META before initialization; existing `_make_run_name`
  and W&B config/notes handling remain in use. No metric schema changes.
- Each episode runs frozen, noise-free Stage 1 inference for 2000 ticks,
  then UD7 Stage 2 until simulator time 45000. Policy warm start is enabled;
  critic, optimizers and replay are fresh. UBOC N=5, beta=1/sqrt(pi), LAP,
  random eviction, 200000-step replay and flat exploration std 0.05 remain.
  New output root is `checkpoints/ud7_best_saved_stage2`.
- Review caveat: existing Stage 2 curriculum starts at scale 0.05 after the
  frozen prefix (saved scale 1.0), increasing to 1.0 over 20000 Stage 2 ticks.
  Warm-started weights do not imply identical applied costs at this boundary.
  This pre-existing behavior is preserved; it can affect apparent performance.
- Validation: target/network tests passed (40 tests); learner tests passed
  (23 tests). Initial combined learner import failed because the tests need
  discovery with `tests` on the import path; corrected discovery passed.
  Launcher `--check` passed with the selected step-95000 source.
- Runtime status: checkpoint loading/inference has NOT been performed in
  this review. Automatic approval review rejected `weights_only=False`
  because pickle can execute code. Restricted `weights_only=True` rejected
  the stored NumPy reconstruction type. Existing simulator/training processes
  have not been stopped or replaced. Claude's launch issue is not diagnosed.
- Checkpoint/network/reward contracts are unchanged; v10.0.0 artifacts remain
  compatible with this patch release.

## Experiments — where UBOC's benefit can come from in this task, and where it cannot

Follow-up to the run above. The goal here is performance, not an A/B, so the
question is which UD7 knob has leverage. Two candidate causes of the tiny
`sigma` were tested and both came back negative.

### The shared SALE projection is not the cause

This project's ensemble critic diverges from the UD7 reference in one place:
`EnsembleQNet._build_q_net` gives every critic its own `s_input_layer`, while
`ContextualEnsembleCritic` shares a single `task_sa_projection` across all
heads, so the heads see identical features. That looked like the obvious
explanation for the heads collapsing in function space.

It is not. Training the real learner twice on the same replay for 3,000
updates on the GPU, once shipped and once with a per-head projection built as
the reference does:

| | mean `sigma` (last 1k) | mean penalty / \|Q\| | final pair L2 |
| --- | ---: | ---: | ---: |
| shared projection (shipped) | 0.01018 | 0.0882% | 19.53 |
| per-head projection (UD7 ref) | 0.01021 | 0.0884% | 19.52 |

Indistinguishable. The shared first layer is not what limits diversity, so
the shipped architecture stays and no MAJOR network change is warranted.

### Summing N losses does not inflate the shared encoder's step

The contextual encoder is optimized by the critic objective, so summing five
per-head losses instead of two raises its raw gradient. `grad/encoder_norm`
against its clip of 1.0, over the logged history:

| run | median | p95 | % at or above the clip |
| --- | ---: | ---: | ---: |
| `yy01yrkc` uboc5 | 5.84 | 11.45 | 93.8% |
| `lvjsupx0` v9.3.1 | 21.08 | 48.14 | 98.8% |
| `4i7zh684` v9.3.1 | 4.61 | 10.11 | 94.5% |

The encoder gradient is clipped in 94-99% of updates in every run, baselines
included, so its effective step is the clip norm and the extra loss terms do
not change it. `grad/critic_norm` medians are 1.97 (uboc5) against 1.99
(`4i7zh684`) at a clip of 10.0, so nothing is inflated there either. No
rescaling is needed.

### Averaging the heads barely denoises the policy gradient

If the uncertainty penalty is negligible, the remaining mechanism is UD7
Eq. 26: the actor maximizes the ensemble mean, which should cut the policy
gradient's noise. Measured on an eight-critic learner after 2,500 updates,
taking `dQ_i/da` per head over a 1,024-sample batch:

- pairwise correlation of `dQ_i/da` across heads: **mean +0.937**, min +0.72

For pairwise correlation `rho`, averaging N heads leaves
`rho + (1 - rho)/N` of one head's variance. At `rho` = 0.937 that is 95% for
N = 5 and 93.7% in the limit, so ensemble averaging buys about a 5% reduction
and raising N cannot buy more. Compute grows linearly in N for that.

### Conclusion for tuning

Neither of UD7's two mechanisms has room in this task. The penalty is 0.03% of
`|Q|` past the first thousand updates, and the policy-gradient averaging
recovers 5%. Within the method there is no knob that changes this: N moves the
penalty only through `c4(N)` (0.940 at N=5 to 0.973 at N=10, a 3.5% change)
and moves the gradient noise by the formula above, while `beta` is the only
real multiplier and would need to be roughly 11x the paper's value to make the
correction 1% of `|Q|`, far outside the validated ablation range.

So UD7 is run at the paper's defaults, N = 5 and `beta` = 1/sqrt(pi), and it
should be expected to land close to TD7 here rather than ahead of it. If
performance is the goal, the leverage in this system is in the reward, replay,
exploration and curriculum settings, not in the critic aggregation.

Settings chosen for the production runs: `uboc` N = 5 in **both** stages,
random replay eviction in both, and LAP left on in Stage 2 as UD7 specifies —
note the v9.3.1 Stage 2 baselines ran with LAP off, so that is a deliberate
difference from them, not an inherited default.

## Experiments — first UD7 run was not a comparison

Run `yy01yrkc` (v10.0.0, `uboc` N=5, reward Q) looked nothing like the
v9.3.1 runs it was set beside, `lvjsupx0` and `4i7zh684`. The cause was the
launch, not UBOC. Diffing the three W&B configs shows fifteen differing keys,
eleven of which have nothing to do with the critic:

| setting | v9.3.1 baseline | `yy01yrkc` |
| --- | --- | --- |
| `stage` | 2 | none |
| `load_stage1_policy_path` | `v9.0.0_stage1_policy.pt` | none |
| `stage1_policy_warm_start` | True | False |
| `state_normalizer_warmup_bypass` | True | False |
| `lap_enabled` | False | True |
| `replay_capacity_env_steps` | 200,000 | 100,000 |
| `replay_eviction_mode` | random | fifo |
| `exploration_noise_std` | 0.05 flat | 0.1 -> 0.02 |

The baselines are Stage 2 runs resuming a trained Stage 1 prefix with its
normalizers, so they open at `env/tat` ~165 on an already-loaded system.
`yy01yrkc` was a from-scratch run: 10,000 warm-up steps on an empty system,
then learning from a random initialization, opening at `env/tat` 69.5. Two
different experiments.

### The implementation checks out on real data

Over the 1,380 logged learner rows of `yy01yrkc`:

- `critic/target_uboc_penalty_mean` equals `beta * critic/target_ensemble_std_mean`
  to a maximum absolute residual of **7.2e-09**, which is float32 noise.
- `critic/target_vs_min_gap_mean` is negative in **zero** rows: the aggregate
  never falls below the ensemble minimum, as `mean - beta*std` cannot.
- The ensemble stayed diverse. `critic/parameter_pair_l2_mean` rose 18.5 ->
  25.2, and `critic/q_grad_norm_min` never reached zero, so no head stopped
  learning or duplicated another.
- `sigma` decayed 0.122 -> 0.008 as the critics converged, which is the
  self-scaling behaviour the method predicts rather than a fixed penalty.
- `numeric/learner_finite_ratio` bottomed at 0.99999994. That is the float32
  value immediately below 1.0, the rounding of averaging ~11k ones; a single
  genuine non-finite element among them would read 0.99991, and the critic's
  own `_require_finite` would have raised first.

### The UBOC lever is small in this task

`critic/target_uboc_penalty_mean` against `|critic/target_q_mean|`:

| learner updates | \|target Q\| | sigma | beta*sigma | penalty / \|Q\| |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.50 | 0.1222 | 0.0689 | **13.9%** |
| 1,538 | 2.46 | 0.0069 | 0.0039 | 0.16% |
| 6,128 | 7.85 | 0.0056 | 0.0032 | 0.04% |
| 13,798 | 17.14 | 0.0100 | 0.0056 | 0.03% |

The correction does real work for roughly the first thousand updates and is
negligible after that. This is not specific to UBOC: the v9.3.1 baselines'
own `critic/q_abs_diff_mean` runs 0.029 -> 0.007 and 0.061 -> 0.018, the same
order as this run's 0.074 -> 0.012, so clipped double-Q's own min-versus-mean
gap is equally negligible there. The critics agree in function space within
about a thousand updates even though their parameters stay far apart, so the
aggregation rule has little room to change anything past that point. Expect a
small effect and size the comparison accordingly; the paper's gains come from
tasks where early-stage critic disagreement persists.

### A/B protocol

The v9 Stage 1 artifact cannot seed a v10 run: the major version gate refuses
it. Its policy side is in fact compatible — `online_actor`, `online_encoder`
and `sale_online` have identical keys and shapes against a fresh v10 learner,
and only `online_critic` was re-keyed, which a Stage 1 load never reads
because Stage 2 keeps a fresh critic — but rather than relax the gate, Stage 1
is retrained under v10.

One Stage 1 artifact is shared by both Stage 2 arms so the arms differ only in
the critic rule, matching how `v9.0.0_stage1_policy.pt` was reused by both
v9.3.1 baselines. The shared prefix is trained with `cdq` N=2, the provenance
the v9 prefix had.

Note that the v9 Stage 1 artifact's own `runtime_config` shows Stage 1 was run
with LAP **on**, a 100,000 FIFO replay and annealed 0.1 -> 0.02 exploration
noise — the current defaults — while Stage 2 uses LAP off, a 200,000 random
replay and flat 0.05 noise. The two stages do not share replay or exploration
settings.

Checkpoint trees, distinct per arm:

```
checkpoints/ctx_td7q2_v10.0.0_s1_l1_a0_g1_..._efifo_...    Stage 1, shared
checkpoints/ctx_td7q2_v10.0.0_s1_l0_a0_g2_..._erandom_...  Stage 2, TD7 arm
checkpoints/ctx_ud7q5_v10.0.0_s1_l0_a0_g2_..._erandom_...  Stage 2, UD7 arm
```

UBOC is left at the paper's defaults (N = 5, `beta` = 1/sqrt(pi)) for this
comparison, so that "no measurable difference in this task" is a result rather
than a consequence of retuning.

## v10.0.0 — UD7: UBOC critic ensemble replaces clipped double-Q

### Purpose

Add the UD7 methodology of Kim, Jeong and Han, *Provable generalization of
clipped double Q-learning for variance reduction and sample efficiency*
(Neurocomputing 673, 2026), to the contextual TD7 learner. UD7 keeps every
TD7 component this project already runs — SALE, LAP, decoupled encoders,
delayed policy updates, target policy smoothing, value clipping — and changes
only how the per-critic next-state values are collapsed into one Bellman
target.

Clipped double-Q reads two critics and takes the minimum. Because the estimate
rests on two samples, its variance is high exactly when the critics are worst
learned, which is early training. Uncertainty-based overestimation correction
(UBOC) reads N critics and takes the ensemble mean less a penalty proportional
to the ensemble's own spread:

```
mu(s',a')    = mean_i Q_target_i(s',a')
sigma(s',a') = unbiased std_i Q_target_i(s',a')     (ddof = 1)
target       = r + gamma * (1 - done) * clip(mu - beta * sigma)
```

The penalty is large while the critics disagree and vanishes as they converge,
so the correction is self-scaling rather than a fixed heuristic.

### beta is 1/sqrt(pi), not 1/sqrt(N)

`beta = 1/sqrt(pi) = 0.5641896`, independent of N. For two i.i.d. normal
samples `E[min(X, Y)] = mu - sigma/sqrt(pi)`, which is the order statistic
that Theorem 4.1 matches; the UD7 reference implementation
(`github.com/jangwonkim-cocel/UD7`, `ud7.py`) hard-codes the same constant.
The paper's PDF renders the symbol as a broken glyph, so `1/sqrt(N)` is an
easy misreading — it is wrong, and at N = 5 it would over-penalize by 26%.

### The finite-N expectation sits above the clipped double-Q one

Theorem 4.1 states an exact equality between `E[min(X, Y)]` and
`E[mu_hat - beta * sigma_hat]`. That holds in the limit, not at finite N: the
unbiased estimator is of the *variance*, and its square root underestimates
sigma by the factor `c4(N) = sqrt(2/(N-1)) * Gamma(N/2) / Gamma((N-1)/2)`. So
the implemented target's expectation is `mu - beta * c4(N) * sigma`:

| N | c4(N) | effective penalty | shortfall vs CDQ |
| ---: | ---: | ---: | ---: |
| 2 | 0.79789 | 0.450158 sigma | 20.21% |
| 3 | 0.88623 | 0.500000 sigma | 11.38% |
| 5 | 0.93999 | 0.530330 sigma | **6.00%** |
| 10 | 0.97266 | 0.548764 sigma | 2.73% |
| 25 | 0.98964 | 0.558345 sigma | 1.04% |
| 100 | 0.99748 | 0.562767 sigma | 0.25% |

At the shipped N = 5 the UD7 target is about 6% less pessimistic than clipped
double-Q in expectation. Monte Carlo over 300k draws at mu = 1, sigma = 2
matches the closed form to within 0.0013 at every N in the table. This is a
property of the estimator the reference implementation uses, not a deviation
from it — recorded here because it means UD7 is not a pure variance reduction
at small N, it also relaxes pessimism slightly.

Theorem 4.2 holds as stated and at every size: the variance gap against
clipped double-Q is strictly positive for all N >= 2 (2.736 -> 2.461 at N = 2,
0.944 at N = 5, 0.471 at N = 10 for sigma = 2), monotonically decreasing in N,
and bounded by `sigma^2 (1 - 1/pi) = 2.727` as Corollary 4.3 predicts.

At N = 3 the effective penalty is exactly `0.5 sigma`, since
`beta * c4(3) = (1/sqrt(pi)) * (sqrt(pi)/2)`.

### Both rules stay selectable

`critic_target_mode` picks the aggregation and `num_critics` sizes the
ensemble. Defaults are the UD7 ones, `uboc` with N = 5; the TD7 baseline is
`--critic-target-mode cdq --num-critics 2`. `cdq` with any other N is
rejected at config validation rather than silently becoming a min-over-N rule
that is neither algorithm.

`--uboc-beta` exposes the paper's Section 7.2 ablation
(`1/(2*sqrt(pi))`, `1/sqrt(pi)`, `2/sqrt(pi)`). `--num-critics` exposes
Section 7.1.

### Value clipping moved to the aggregate

TD7's value clip bounded each target critic before the minimum. The minimum
commutes with a shared clamp, so that was equivalent to clamping the
aggregate — but the UBOC mean and spread do not commute with it: clamping
per critic compresses the spread and silently shrinks the uncertainty penalty.
The clip is now applied to the aggregated `[B, 1]` value, matching both the
TD7 and UD7 reference implementations, and `bellman_target` takes that
aggregate instead of a critic pair.

### Critic and policy losses

The critic loss is one Huber per member, summed over the ensemble. Both
reference implementations do this (`LAP_huber` sums over the critic axis),
and it makes the gradient each head receives independent of N. For N = 2 it
is bit-identical to the previous `q1_loss + q2_loss`. The LAP priority stays
the worst per-sample error across members, which also matches at N = 2.

The policy gradient now reads the ensemble mean (UD7 Eq. 26, and the same in
the TD7 reference), where this project previously read `q1` alone.
`actor_q_aggregation` defaults to `mean` under `uboc` and to `q1` under `cdq`,
so the prior v9 baseline is reproduced exactly and either factor can be
isolated: `--critic-target-mode uboc --actor-q-aggregation q1` changes only
the target rule.

### Network and checkpoint layout

`ContextualTwinCritic` becomes `ContextualEnsembleCritic`: `q1`/`q2` become
`q_nets`, an `nn.ModuleList` of N heads over the same shared SALE projection,
each constructed independently. `TwinCriticOutput(q1, q2)` becomes
`EnsembleCriticOutput(q)` with `q` shaped `[B, N]`.

Every member regresses the same target, so independent initialization is the
only thing keeping them apart — and a collapsed ensemble would drive the UBOC
penalty to zero without any other symptom. The old twin-symmetry guards
generalize to the closest pair: `critic/parameter_l2_distance` and
`critic/parameter_max_abs_diff` are now the minimum over all pairs (identical
to before at N = 2), `critic/parameter_pair_l2_mean` is the average pair, and
saving or resuming refuses on a zero.

MAJOR: the critic state dict is re-keyed, so v9 checkpoints do not load. The
version gate already refuses across majors. Resume additionally guards
`critic_target_mode` and `uboc_beta`, since the same weights mean a different
value function under a different aggregation, and `num_critics` is caught by
the existing `network_config` comparison. None of the four new fields is
launch-controlled: a resumed run takes them from the checkpoint.

### Diagnostics

`critic/q1_mean`, `critic/q2_mean`, `critic/q1_loss`, `critic/q2_loss`,
`critic/q1_grad_norm`, `critic/q2_grad_norm` and `critic/q_abs_diff_*` keep
their meaning as the first two members, so existing dashboards and the reward
diagnostic schema are unchanged. `critic/q_min` and `critic/q_max` now span
the whole ensemble.

New, of which six reach W&B:

| key | W&B | meaning |
| --- | :-: | --- |
| `critic/target_ensemble_std_mean` | yes | sigma(s',a'), the uncertainty itself |
| `critic/target_uboc_penalty_mean` | yes | beta * sigma, the correction applied |
| `critic/target_vs_min_gap_mean` | yes | how far the target sits above the ensemble minimum |
| `critic/q_ensemble_std_mean` | yes | online member disagreement on (s, a) |
| `critic/q_grad_norm_min` | yes | catches a member that stopped learning |
| `critic/parameter_pair_l2_mean` | yes | ensemble diversity over time |
| `critic/target_ensemble_mean`, `critic/target_ensemble_std_max` | no | target aggregate detail |
| `critic/q_ensemble_mean`, `critic/q_spread_mean` | no | online aggregate detail |
| `critic/q_loss_mean`, `critic/q_grad_norm_mean` | no | per-member averages |
| `critic/num_critics`, `critic/uboc_enabled`, `critic/uboc_beta`, `critic/actor_reads_ensemble_mean` | no | run identity |

`num_critics`, `critic_target_mode`, `uboc_beta` and `actor_q_aggregation`
also go into the W&B run config, and the algorithm variant string gains a
`_uboc5` / `_cdq2` suffix so checkpoint directories and run names separate the
two arms.

`critic/q_mean_delta_100` and `critic/q_mean_delta_1000` now track
`critic/q_ensemble_mean` rather than the average of the first two members.

### Checkpoint directory names

`checkpoint_variant` previously produced the same directory name for a UBOC
run and a clipped double-Q run, so two different algorithms would have shared
one checkpoint tree. The compact name now carries both: `ctx_ud7q5_...` and
`ctx_td7q2_...`. Enumerating all 576 valid runtime combinations gives 576
distinct names.

Two tags were abbreviated in the same style as the existing `ffres` action
tag, to hold the name inside the Windows path budget: `random_rail` ->
`rrail`, `snapshot` -> `snap`. The name is shorter than v9's despite the added
ensemble tag and the longer version string — 76 for the default arm, and the
worst valid combination drops from 84 to 81. The 80-character assertion in
`test_default_checkpoint_root_isolated_by_runtime_variant` covers five
configurations, all of which are now 76 or 77; the handful of extreme
combinations that exceed it (`random_rail` with random eviction and a
seven-digit curriculum end step) already did so before this change.

### Cost

Section 7.5 of the paper reports roughly linear O(N) growth in update cost.
Here only the N critic heads multiply — the contextual encoder, the SALE
encoders and the shared critic projection are computed once per update
regardless of N — so the growth is well below N/2. Measured on CPU over 60
updates at batch 256 with the small test network (`context_dim` 32,
`hidden_dim` 64, SALE 64), which understates the critic's share of a real
run:

| arm | ms/update | `timing/critic_update_ms` |
| --- | ---: | ---: |
| `cdq` N=2 | 28.3 | 5.7 |
| `uboc` N=2 | 29.2 | 6.0 |
| `uboc` N=3 | 29.9 | 6.7 |
| `uboc` N=5 | 31.5 | 8.3 |

Measured again on the GTX 1080 Ti at the production network width
(`context_dim` 128, `hidden_dim` 256, SALE 256, batch 1024), median of 80
updates after 15 warm-up updates:

| arm | ms/update | `timing/critic_update_ms` | peak GPU |
| --- | ---: | ---: | ---: |
| `cdq` N=2 | 61.6 | 7.8 | 132 MiB |
| `uboc` N=5 | 66.3 | 11.4 | 147 MiB |

N=5 costs 7.5% more per update and 45% more inside the critic stage; GPU
memory is negligible either way. The encoder and SALE dominate the update, so
the O(N) growth in the critic does not reach the step time — well short of
the N/2 a naive reading of Section 7.5 would suggest.

### Run identity

`EXP_META["note"]` is now `ud7` and its `description` states the UBOC target,
replacing the v9 Reward Q text. The generated run name carries the arm, so the
two arms never share one: `..._ud7_uboc5_rQ_...` against `..._ud7_cdq2_rQ_...`.
The W&B run config gains `num_critics`, `critic_target_mode`, `uboc_beta` and
`actor_q_aggregation` — the last recording the *resolved* value (`mean` or
`q1`) rather than the `auto` sentinel, with the configured value kept
alongside as `actor_q_aggregation_configured`.

The runtime summary prints a `critic` section, so a launch says which
algorithm it is running before it waits for the simulator:

```
  critic:
    algorithm    : UD7 (UBOC)
    aggregation  : uboc
    critics      : 5
    target rule  : mean - 0.5641896 * std  (beta = 1/sqrt(pi))
    policy reads : ensemble mean
    variant      : contextual_td7_sale_lap_v6_compact_actor_critic_tat_uboc5
```

### GPU environment

`.venv` at the repo root (git-ignored), Python 3.14, **torch 2.14.0+cu126**.
The CUDA index matters: this host's GTX 1080 Ti is Pascal (`sm_61`), and the
cu128 and later builds dropped Pascal kernels, so a cu128 wheel would install
and import cleanly and then fail at the first kernel launch with "no kernel
image is available for execution on the device". The cu126 2.14.0 arch list is
`sm_50, 60, 61, 70, 75, 80, 86, 90`; matmul, Linear/LayerNorm/SiLU with
backward, Embedding gather and `var(unbiased=True)` were all executed on the
device to confirm. Driver 566.36 (CUDA 12.7) is well above the cu126 minimum.

Pins are recorded in `PythonCode/requirements-gpu.txt`, which also carries the
two install commands and the reason for cu126.

At `--replay-capacity-env-steps 100000` the packed CPU replay reserves an
estimated 11.74 GiB of host RAM; this host has 63.9 GiB, so it fits.

### Verification

`python -m pytest PythonCode/tests -q` — **404 passed, 186 subtests**, on
torch 2.14.0+cu126 with the GPU present (the one CUDA-gated test that is
skipped on a CPU-only install now runs). Beyond the suite:

- No reference to a removed symbol (`ContextualTwinCritic`,
  `TwinCriticOutput`, `q1_value`, `twin_parameter_diagnostics`,
  `config.algorithm_variant`, the four-argument `bellman_target`) remains
  anywhere in `PythonCode/`, and every intra-project import across all 98
  modules resolves to a name its target module actually defines.
- The tables above for `c4(N)`, the variance gap and the Corollary 4.3 bound
  are measured, reproduced against the closed forms to within 0.0013.
- The `cdq` arm reproduces v9 numerically. Summing one Huber per member is
  bit-identical to the old `q1_loss + q2_loss` at N = 2; under MSE the two
  differ by 2.4e-07, which is reduction order, not semantics. The
  max-over-members priority and the aggregate value clip are both
  bit-identical to the old two-critic code.
- Both arms run 60 consecutive learner updates with every diagnostic finite.
  `critic/target_vs_min_gap_mean` is exactly 0 under `cdq` and positive under
  `uboc`, and `critic/target_uboc_penalty_mean` equals beta times the
  reported spread.

Tripwires that pin the version string were moved to `v10.0.0`:
`test_contextual_checkpoint.py` (two) and `test_contextual_variants.py` (one).

A `--mode training` launch was taken as far as it goes without a simulator:
both arms build their networks on CUDA, print the summary, and bind the
listener. `python -c "import oht_routing.utils.export_wandb_run"` passes its
schema self-checks and all six new W&B keys appear in `EXPORT_COLUMNS`
(219 columns, schema `v10.0.0`).

Not verified here: no simulator-in-the-loop run, so nothing about learning
behaviour. The first real launch should confirm that
`critic/target_ensemble_std_mean` decays as the ensemble converges — if it
stays flat, the heads are not diversifying and `critic/parameter_pair_l2_mean`
will show it — and that `critic/target_vs_min_gap_mean` stays small relative
to `critic/target_q_mean`.
