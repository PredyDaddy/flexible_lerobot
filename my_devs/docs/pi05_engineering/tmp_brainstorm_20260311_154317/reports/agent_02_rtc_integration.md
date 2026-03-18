# Agent 02 RTC Integration Report

## Scope

This report focuses only on the RTC-specific integration gap for the current PI05 real-robot inference path.

## 1. Current Gap: `select_action` to RTC-capable `predict_action_chunk`

Today `my_devs/train/pi/so101/run_pi05_infer.py` calls `predict_action(...)`, and that path ends in
`policy.select_action(...)` from `src/lerobot/utils/control_utils.py`.

That is the wrong runtime path for RTC:

- `PI05Policy.select_action()` is single-step oriented.
- `PI05Policy.select_action()` explicitly asserts that RTC is not supported on this path.
- RTC is only available when the runtime calls `PI05Policy.predict_action_chunk(...)`.

So the required architectural shift is:

1. stop asking the policy for one action at a time,
2. ask for an action chunk,
3. keep a runtime action queue outside the policy,
4. feed RTC-specific kwargs into chunk inference:
   - `prev_chunk_left_over`
   - `inference_delay`
   - optionally `execution_horizon`

Without this shift, PI05 can run, but RTC cannot actually be exercised.

## 2. Required Runtime State And Queue Ownership

RTC needs more runtime state than the current synchronous loop owns.

### Required state

- current processed action queue for robot execution,
- original unprocessed action queue for RTC leftover computation,
- current action consumption index,
- `action_index_before_inference`,
- measured or derived `real_delay`,
- `prev_chunk_left_over`,
- current chunk size and execution horizon,
- timestamps and per-chunk latency metrics.

### Queue ownership model

The queue cannot stay implicit inside `select_action()`. It must become an explicit runtime object.

The correct ownership pattern is close to `src/lerobot/policies/rtc/action_queue.py`:

- queue owns both original actions and processed actions,
- runtime pops processed actions for robot execution,
- runtime computes leftover original actions for the next RTC call,
- when RTC is enabled, new chunk merge semantics are replacement-oriented,
- when RTC is disabled, new chunk merge semantics can fall back to append-oriented behavior.

This ownership should live in the real-time runner, not inside the model.

## 3. Major Failure Modes On Real Robot

### Delay mismatch

If `inference_delay` is derived incorrectly, the next chunk will be aligned to the wrong prefix.
That can cause visible discontinuities or sudden motion corrections at chunk boundaries.

### Queue drift

If queue consumption and inference timing are measured in different places without a single owner,
`prev_chunk_left_over` can become stale or off by one or more actions.

### Stale chunk overwrite

If a newly finished chunk is merged after too much of the previous queue has already been consumed,
replacement can wipe out still-valid future actions or insert actions that are already outdated.

### RTC / non-RTC mode confusion

If the runtime mixes RTC-enabled replacement behavior with non-RTC append behavior under one code path
without an explicit mode switch, queue semantics become non-deterministic and debugging becomes difficult.

### Weak observability

If the runtime only logs FPS and step count, it will be hard to prove whether RTC improved continuity
or whether the robot is simply masking queue starvation and stale chunk corrections.

## 4. Safe Staged Integration Recommendation

### Stage 1: chunk runner without RTC

First replace `select_action()` with `predict_action_chunk()` and introduce an explicit queue,
but keep RTC disabled. Validate:

- queue does not starve under nominal load,
- chunk latency is measured,
- execution cadence is stable,
- robot behavior is no worse than the current synchronous baseline.

### Stage 2: RTC state plumbing

Add:

- `ActionQueue`-style ownership,
- `action_index_before_inference`,
- `real_delay`,
- `prev_chunk_left_over`,
- per-chunk logs for queue depth, delay, leftover length, and chunk replacement.

Do this before enabling RTC guidance.

### Stage 3: enable RTC on real robot conservatively

Enable RTC only after the chunk runner is stable and observable.
Use conservative settings first, with the ability to disable RTC instantly and fall back to plain chunking.

### Stage 4: promote to async or more complex runtime later

Only after local chunk + RTC behavior is trustworthy should the team consider layering the same semantics
into a broader async client/server runtime.

## Bottom Line

RTC integration is not a small flag flip. It requires a runtime rewrite from single-step inference to
chunk scheduling with explicit queue ownership, delay accounting, and leftover tracking.

The safe path is:

1. local chunk runner first,
2. RTC state plumbing second,
3. RTC enablement third,
4. broader async/runtime generalization later.
