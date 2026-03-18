# Agent 01 Runtime Architecture Report

## Scope

This report focuses on the runtime architecture for evolving the current PI05 inference path from a synchronous control loop into a robust real-time deployment path. The goal is not to produce the final solution document, but to compare viable runtime directions and recommend the one that best fits the current repository state.

## What Exists Today

### Current custom runner is synchronous

`my_devs/train/pi/so101/run_pi05_infer.py` is a single-loop controller:

1. read observation from robot,
2. preprocess observation,
3. call `predict_action(...)`,
4. convert and send one action,
5. sleep to maintain target FPS.

This matters because `predict_action(...)` in `src/lerobot/utils/control_utils.py` uses `policy.select_action(...)`, which is the single-step policy path.

### PI05 already supports chunk prediction, but not on the path currently used

`src/lerobot/policies/pi05/modeling_pi05.py` exposes:

- `select_action()`: compatible with the current synchronous runner,
- `predict_action_chunk()`: the chunk-oriented path needed for real-time queue scheduling,
- RTC is explicitly blocked on `select_action()` and must go through `predict_action_chunk()`.

So the current runtime is not only synchronous; it is also structurally unable to use RTC without being rewritten around chunk scheduling.

### The repository already has a generic async runtime skeleton

`src/lerobot/async_inference/` already provides:

- `PolicyServer`
- `RobotClient`
- chunk-based action streaming,
- separate action receive/control loop threads,
- queue-threshold based observation refresh.

This is the most important existing building block. It already solves the core decoupling problem: execute current actions while precomputing the next chunk.

## Runtime Options

## Option A: Incremental local dual-thread chunk runner

### Description

Keep everything local inside a new PI05-specific script under `my_devs/train/pi/so101/`, but refactor runtime around two threads:

- thread 1: observation capture + chunk inference,
- thread 2: action dequeue + robot execution.

The local runner would directly instantiate the policy, processors, and robot, then call `policy.predict_action_chunk(...)` and manage its own queue.

### Advantages

- Lowest conceptual overhead for the user who already runs one script.
- Easiest path to preserve current custom robot/camera wiring and exact processor usage.
- Straight path to PI05-specific instrumentation because the script owns all runtime state.
- Fastest route for first milestone if the goal is "make the existing script smoother" rather than "build a reusable service."

### Disadvantages

- Reimplements concurrency, queueing, chunk thresholding, and timing logic that already exist in `src/lerobot/async_inference/`.
- Increases long-term divergence between custom PI05 runtime and repository-native async runtime.
- Likely grows into an ad hoc framework if RTC, profiling, warmup, fallback modes, and debug tooling are added later.
- Higher maintenance risk because future fixes to generic async runtime will not automatically benefit this script.

### Assessment

Good as a prototype or fallback path. Weak as the primary architecture because it duplicates the most failure-prone part of the system: real-time queue orchestration.

## Option B: Reuse `async_inference` directly as the primary runtime

### Description

Adopt `src/lerobot/async_inference/policy_server.py` and `src/lerobot/async_inference/robot_client.py` as the main execution architecture for PI05. Wrap them with PI05-specific launch scripts or config helpers under `my_devs/`, instead of building a separate queueing runtime.

### Advantages

- Best reuse of existing concurrency and chunk-based runtime logic.
- Already aligned with the server/client design the repository uses for real-time chunk execution.
- Conceptually matches PI05 chunk inference much better than the current single-step loop.
- Cleaner separation between robot IO timing and model inference timing.
- Better long-term operability if policy process and robot process need to be isolated later.

### Disadvantages

- Current async stack is generic, not PI05-ergonomic.
- It does not preserve the exact custom processor-loading workflow from `run_pi05_infer.py` out of the box.
- Debugging becomes two-process or multi-thread debugging, which is harder than single-script local debugging.
- It solves async chunk execution, but not by itself the PI05-specific RTC runtime state management.
- There is integration work around robot feature mapping, checkpoint-specific pre/post processors, and task-specific launch ergonomics.

### Assessment

Architecturally strongest for the long term, but too abrupt as a first landing if the team is still validating PI05 on real hardware through a custom script.

## Option C: Hybrid approach

### Description

Use the repository async architecture as the target runtime model, but introduce it through a PI05-specific adapter layer and a staged rollout:

1. build a local chunked PI05 runtime adapter that uses the same processor and robot conventions as `run_pi05_infer.py`,
2. shape that adapter to match `async_inference` semantics,
3. then either embed or call into `async_inference` components with minimal PI05-specific glue,
4. add RTC on top only after chunked execution is stable.

This is not "write two separate systems." The key is to implement PI05-specific runtime pieces once in reusable helpers, while staging adoption to reduce deployment risk.

### Advantages

- Preserves the working knowledge and exact checkpoint assumptions of the current script.
- Avoids a big-bang migration from single-step local control to generic async service mode.
- Lets the team validate chunked execution locally before adding network/process boundaries and RTC.
- Keeps the end-state aligned with existing async modules instead of drifting into a one-off script.
- Best debugging posture: start in a local process, then move the same chunk orchestration model into async runtime.

### Disadvantages

- Requires discipline to avoid accidentally building a permanent second runtime.
- More design work up front because boundaries must be made explicit.
- If poorly executed, the hybrid path can create temporary abstraction layers that outlive their usefulness.

### Assessment

Best fit for this repository right now. It balances reuse and risk better than either extreme.

## Recommendation

Recommend **Option C: hybrid approach with async_inference as the target runtime model**.

More specifically:

- do **not** keep the current synchronous `select_action()` path as the main future direction,
- do **not** jump straight to treating `robot_client.py` and `policy_server.py` as a drop-in replacement for the existing custom runner,
- instead build a **PI05 chunk-runtime adapter** under `my_devs/` that:
  - preserves the exact checkpoint processor loading pattern from `run_pi05_infer.py`,
  - moves execution onto `predict_action_chunk()`,
  - introduces explicit action queue management and timing metrics,
  - is intentionally shaped so its core logic can later be hosted by or integrated with `async_inference`.

This recommendation is the best tradeoff across:

- code reuse: because the final runtime still converges on existing async architecture,
- risk: because migration happens in stages rather than a hard switch,
- latency: because chunk prediction and execution are decoupled early,
- operability: because local single-machine validation remains possible,
- debugging: because the first step keeps state visible inside one PI05-oriented runtime entry point.

## Why not recommend pure local dual-thread as the final direction

The local dual-thread runner is attractive because it is easy to start, but it pushes the repo toward a second independent runtime framework. The queueing logic, chunk thresholds, action aggregation, and timing semantics are precisely the parts that should be shared, not re-invented per policy.

If a local runner is built, it should be treated as a transitional PI05 adapter or thin wrapper, not as the final architecture.

## Why not recommend direct async_inference adoption as the first step

The current working script contains important PI05-specific knowledge:

- exact checkpoint processor loading,
- custom robot feature construction,
- custom camera topology and SO101 deployment assumptions,
- a simpler local debugging surface.

If the team jumps straight into generic async runtime without carrying those assumptions forward explicitly, the likely outcome is a long integration loop where "architecture is cleaner" but robot bring-up regresses.

## Likely Code Touch Points

### New code under `my_devs/`

- Add a PI05 real-time runtime module under `my_devs/train/pi/so101/` or a nearby `my_devs/pi05_runtime/` area.
- Split current `run_pi05_infer.py` responsibilities into:
  - environment and checkpoint setup,
  - observation preprocessing adapter,
  - chunk inference worker,
  - action queue / execution loop,
  - logging and timing instrumentation.

### Existing script updates

- Refactor `my_devs/train/pi/so101/run_pi05_infer.py` from monolithic synchronous loop into a thin launcher or keep it as a baseline reference script.

### Reuse and extension points in `src/`

- `src/lerobot/utils/control_utils.py`
  - current `predict_action()` is single-step oriented; likely keep as-is for baseline, not as the main PI05 real-time API.
- `src/lerobot/policies/pi05/modeling_pi05.py`
  - no core modeling change appears required for chunk runtime itself,
  - but runtime callers must switch to `predict_action_chunk()`.
- `src/lerobot/async_inference/`
  - likely needs PI05-oriented integration glue rather than architectural rewrite,
  - especially around how the custom script currently loads processors and maps robot observations/actions.

## Suggested Rollout Stages

### Stage 0: Preserve baseline

- Keep the existing synchronous script runnable as a known-good fallback.
- Treat its current performance and behavior as the regression baseline.

### Stage 1: Local chunk runtime without RTC

- Introduce a local PI05 chunk runner.
- Use `predict_action_chunk()`.
- Add queue depth metrics, chunk latency, action starvation counters, and basic warmup handling.
- Validate that behavior on robot is at least as stable as the synchronous baseline.

### Stage 2: Align with async runtime boundaries

- Move queue semantics and thread roles to match `async_inference` concepts:
  - observation sender / inference worker,
  - action receiver / executor,
  - chunk threshold triggering.
- Prefer extracting reusable helpers instead of copying logic.

### Stage 3: PI05 async deployment path

- Add PI05-specific launcher scripts or wrappers that run the async client/server path with repo-compatible config.
- Ensure the custom checkpoint processor loading path is supported cleanly.

### Stage 4: RTC integration

- Only after chunk runtime is stable, add RTC-specific state flow and queue replacement semantics.
- RTC should be layered on top of a stable chunk runner, not introduced while the base runtime is still changing.

## Risks

### Architecture drift risk

If the hybrid approach is not constrained, the temporary PI05 adapter can become a permanent alternative runtime. This should be controlled by explicitly designing the adapter around async-compatible boundaries.

### Processor mismatch risk

The current script loads saved pre/post processors directly from checkpoint files. Any new runtime that skips or abstracts this incorrectly can silently change inference behavior.

### Debugging complexity risk

Moving too fast into full async client/server mode will make failures harder to localize. This is why staged adoption is preferable.

### False performance confidence

The user log shows the script is running, but not whether end-to-end control cadence is stable under chunked execution, or whether action starvation exists. The new runtime must log queue starvation and chunk freshness explicitly.

## Validation Priorities

The runtime architecture decision should be validated against:

- action queue never empty under nominal load,
- end-to-end cadence stability at target FPS,
- no regression in robot behavior relative to baseline synchronous script,
- clean startup, warmup, and shutdown behavior,
- reproducible local debugging before server/client deployment.

## Bottom Line

The best runtime direction is a **hybrid migration**:

- near-term: PI05-specific local chunk runtime adapter,
- target architecture: reuse `async_inference` semantics and components,
- later layer: RTC after chunk execution is proven stable.

That gives the team a realistic path from "the script works" to "the runtime is engineered."
