# Agent 03 Async Server Client Report

## Scope

This report focuses on whether `src/lerobot/async_inference/` should be reused for PI05 real-time deployment, what already matches the user's needs, what gaps remain for RTC and debugging, and how this path should be positioned relative to a custom local runner.

## Bottom-Line Recommendation

`async_inference` should be treated as a **Phase 2 deployment backend**, not the first implementation target for this repo's PI05 real-time upgrade.

Recommended positioning:

1. **Primary runtime first**: build a local in-process PI05 chunk runner in `my_devs/` that directly uses `predict_action_chunk()` and RTC-aware queue management.
2. **Async path second**: once the local chunk/RTC logic is stable, either:
   - wrap `src/lerobot/async_inference/` for PI05-only usage, or
   - add a thin PI05 specialization layer on top of it.
3. **Do not make generic `async_inference` the first milestone** for this user flow, because it solves concurrency well but currently misses the PI05-specific runtime semantics that matter most here.

The reasoning is simple: the current user already has a working local script, and the next gap is not "can we separate processes?" but "can we move to chunked execution with optional RTC and enough observability to trust it on a real robot?"

## What Already Fits PI05 Well

### 1. Policy type support is already wired

`src/lerobot/async_inference/constants.py` includes `pi05` in `SUPPORTED_POLICIES`, so the generic client/server flow already admits PI05 without policy-type hacks.

### 2. The async server already uses chunk inference

`PolicyServer._get_action_chunk()` calls:

```python
chunk = self.policy.predict_action_chunk(observation)
```

This is a strong architectural match, because PI05 RTC is only available on the chunk path, not on `select_action()`.

### 3. The runtime model is appropriate for real-time control

The async architecture already separates:

- action execution on the robot client,
- chunk generation on the policy server,
- observation enqueueing with `Queue(maxsize=1)`,
- refill triggering via `chunk_size_threshold`.

This is a sound pattern for reducing idle frames and decoupling execution cadence from model latency.

### 4. The client already carries task text in the observation payload

`RobotClient.control_loop_observation()` injects:

```python
raw_observation["task"] = task
```

That matters for PI05, since the user's current script relies on language-conditioned inference.

### 5. Dependency packaging already exists

`pyproject.toml` already defines:

- `grpcio-dep = ["grpcio==1.73.1", ...]`
- `async = ["lerobot[grpcio-dep]", "matplotlib>=3.10.3,<4.0.0"]`

So the dependency story is not ideal, but it is at least explicit and reusable.

## Gaps That Matter for PI05

### 1. Current async path is chunked, but not RTC-aware

The largest functional gap is that `PolicyServer` calls:

```python
self.policy.predict_action_chunk(observation)
```

with no PI05 RTC runtime kwargs.

PI05's chunk API is designed to accept runtime kwargs such as:

- `prev_chunk_left_over`
- `inference_delay`

Those are the inputs needed for RTC guidance. Without them, the generic async path can only provide "plain chunked asynchronous execution", not "RTC-enhanced chunk continuity".

This is the core reason I do **not** recommend `async_inference` as the first target if the repo's real goal is "PI05 real-time + RTC".

### 2. The client queue semantics do not match RTC semantics

`RobotClient` currently uses:

- Python `Queue`
- timestamp/timestep-based aggregation
- `_aggregate_action_queues(...)`

This is useful for generic chunk overlap handling, but it is materially different from `src/lerobot/policies/rtc/action_queue.py`, which tracks:

- original actions,
- processed actions,
- leftover prefix,
- real delay,
- replace vs append semantics.

RTC needs queue ownership to be explicit and deterministic. The current async client is aggregation-oriented; PI05 RTC is replacement-oriented.

### 3. Delay accounting is not wired the way RTC needs

RTC does not just need "some latency logs". It needs a runtime estimate of how many actions were consumed during inference, then uses that delay to align the next chunk.

The generic async client/server currently tracks:

- receive/send timing,
- queue sizes,
- observation timestamps,
- network latency.

What it does **not** expose is the PI05/RTC control primitive:

- `action_index_before_inference`
- `real_delay`
- `prev_chunk_left_over`
- merge policy driven by RTC mode

This is not a small omission. It changes the queue contract.

### 4. Model-loading parity may diverge from the working script

The user's current script loads PI05 with:

```python
policy_class.from_pretrained(str(policy_path), strict=False)
```

and the log already shows missing keys / warnings.

`PolicyServer.SendPolicyInstructions()` currently does:

```python
self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
```

with no visible `strict=False` override.

If the current checkpoint only runs reliably with the custom non-strict load path, then a stock async server may fail to match the known-good local runner. That is a practical integration risk.

### 5. Debugging gets harder because the control path is split

The current local script is easy to reason about:

- one process,
- one main loop,
- one log stream,
- direct access to robot timing.

The async path introduces:

- gRPC serialization and transport,
- split logs between client and server,
- separate failure domains,
- separate clocks,
- more places where queue drift can hide.

For a mature deployment pipeline this is acceptable. For the first PI05 real-time refactor on a physical robot, it slows iteration.

## Operational Cost Assessment

### Cost 1. Extra dependency and environment surface

To use the async path cleanly in `lerobot_flex`, the environment must include `grpcio` and related extras. This is manageable, but it is still one more moving part compared with an in-process runner.

### Cost 2. Two-process mental model

The operator has to think in terms of:

- client process: cameras, robot I/O, queue refill decisions,
- server process: model load, preprocessing, inference, action chunk generation.

This is a better scaling model, but it is heavier for day-to-day robot debugging.

### Cost 3. More brittle launch ergonomics

The user's current entry point is one command:

`python3 my_devs/train/pi/so101/run_pi05_infer.py`

The async path turns that into:

1. start policy server,
2. start robot client,
3. ensure ports and dependencies are correct,
4. inspect two log streams.

This is a meaningful operations downgrade unless a wrapper layer is added.

### Cost 4. Integration drift from the custom script

The current custom script explicitly:

- resolves repo root,
- ensures offline tokenizer availability,
- uses checkpoint-local processors,
- uses a known-good robot/camera setup,
- tolerates current checkpoint load quirks.

Generic async modules do not yet preserve all of that behavior by construction. They can be adapted, but that adaptation is work.

## Where Async Inference Is Still Valuable

I do not recommend discarding it. I recommend **repositioning** it.

It is valuable for:

1. **Later process separation**: when the team wants model execution and robot control to be separable.
2. **Remote or service-style deployment**: when the policy should run on a different process or machine.
3. **Stress testing concurrency**: once PI05 chunk + RTC logic exists locally, async_inference becomes a natural execution backend candidate.
4. **Generalized multi-policy runtime reuse**: if the repo later wants a common serving layer across PI05, PI0, SmolVLA, Groot, and others.

That is why I suggest Phase 2 rather than "never".

## Suggested Repo Strategy

### Recommended near-term strategy

Build a **PI05-specific local real-time runner** in `my_devs/` first.

That runner should:

1. keep robot I/O local and simple,
2. switch from `select_action()` to `predict_action_chunk()`,
3. use RTC-capable queue semantics,
4. log chunk timing, queue depth, consumed delay, and leftover handling,
5. prove stable rollout on the real robot before adding transport complexity.

### Recommended async strategy after that

Once the local runner stabilizes, create a PI05-specific wrapper layer around `async_inference`, rather than asking users to call the generic modules directly.

Concretely, I would expect:

1. a PI05 server launcher in `my_devs/train/pi/so101/`,
2. a PI05 robot client launcher in `my_devs/train/pi/so101/`,
3. an adaptation layer that adds:
   - model load parity with the current script,
   - PI05-specific processor/task handling,
   - RTC-aware queue and delay semantics if async RTC is pursued.

## Code Touch Points If Async Becomes Phase 2

If the repo later extends async support for PI05 seriously, likely touch points are:

- `src/lerobot/async_inference/policy_server.py`
  - add PI05-compatible runtime kwargs into `predict_action_chunk(...)`
  - preserve checkpoint loading behavior used by the current working script
- `src/lerobot/async_inference/robot_client.py`
  - replace or extend the current queue logic with RTC-aware semantics for PI05 mode
  - expose per-chunk delay metrics and queue state
- `src/lerobot/async_inference/configs.py`
  - add PI05/RTC runtime config fields if this becomes a first-class mode
- `my_devs/train/pi/so101/`
  - add PI05-specific launch wrappers so the user does not have to assemble generic draccus commands by hand

## Rollout View

### Good first use of async_inference

Use it first as a **non-RTC async baseline** after the local chunk runner works:

1. PI05 local sync single-step baseline
2. PI05 local chunked baseline
3. PI05 local chunked + RTC
4. PI05 async chunked without RTC
5. PI05 async chunked + RTC, only if there is a concrete need for process separation

This ordering minimizes moving pieces per stage.

### Bad first use of async_inference

Do **not** start by simultaneously introducing:

- chunking,
- RTC,
- queue redesign,
- gRPC split,
- launcher refactor,
- new debugging surface.

That bundles too many failure modes into one milestone.

## Final Recommendation

For this repo and this user's current status, `src/lerobot/async_inference/` is best treated as a **reusable concurrency substrate**, not the immediate primary runtime.

The primary runtime should be a local PI05 real-time runner that gets the policy semantics right first. After that, async_inference can be elevated into a supported PI05 backend by adding:

- checkpoint-load parity,
- RTC-aware queue/delay plumbing,
- PI05-focused wrappers and observability.

That sequencing is lower-risk, easier to debug on the real robot, and more likely to converge quickly.
