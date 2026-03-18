# PI05 Async Inference + RTC Brainstorm Context

## User Intent

The user confirmed that `python3 my_devs/train/pi/so101/run_pi05_infer.py` can already run and continuously infer on the real robot, and now wants a structured technical proposal for the next engineering phase.

Requested process:

1. Create a temporary folder under `my_devs/docs/pi05_engineering/`.
2. Spawn multiple agents to brainstorm independently.
3. Require each agent to write a work report inside this folder.
4. The main agent acts as architect, converges the reports into a final technical proposal.
5. Spawn an independent agent to write a concrete implementation plan document, focused on which code should be modified.
6. Spawn multiple agents again to independently review the final technical proposal and plan document.

## Observed Runtime Log

The user shared the following facts from `my_devs/train/pi/so101/run_pi05_infer.py`:

- The model loads successfully from a trained checkpoint.
- The script keeps running and logs:
  - `Step 30 | elapsed=3.70s`
  - `Step 60 | elapsed=4.78s`
  - `Step 90 | elapsed=5.79s`
  - `Step 120 | elapsed=6.88s`
  - `Step 150 | elapsed=7.90s`
  - `Step 180 | elapsed=8.95s`
  - `Step 210 | elapsed=10.04s`
  - `Step 240 | elapsed=11.04s`
- There is one missing key during load:
  - `model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`
- There are warnings around vision embedding keys.

## Relevant Local Context

- Existing engineering note:
  - `my_devs/docs/engineering/ASYNC_INFERENCE_AND_RTC.md`
- Current working directory for this topic:
  - `my_devs/docs/pi05_engineering/`
- Current PI05 inference script:
  - `my_devs/train/pi/so101/run_pi05_infer.py`
- Relevant policy implementation:
  - `src/lerobot/policies/pi05/modeling_pi05.py`
- Relevant RTC modules:
  - `src/lerobot/policies/rtc/`
- Relevant async inference modules:
  - `src/lerobot/async_inference/`

## Architecture Facts Already Verified

### Current `run_pi05_infer.py`

- The script is a synchronous single-loop controller.
- Its main loop does:
  1. `robot.get_observation()`
  2. preprocess observation
  3. `predict_action(...)`
  4. `robot.send_action(...)`
  5. sleep to maintain FPS
- `predict_action(...)` in `src/lerobot/utils/control_utils.py` calls `policy.select_action(...)`.
- This means the current inference path is single-step and does not expose chunk scheduling directly.

### PI05 and RTC

- `PI05Policy.select_action()` explicitly asserts RTC is not supported on the single-step path.
- `PI05Policy.predict_action_chunk()` is the RTC-capable path.
- RTC support in PI05 requires passing runtime kwargs such as:
  - `prev_chunk_left_over`
  - `inference_delay`
- `src/lerobot/policies/rtc/action_queue.py` provides the queue semantics needed for RTC-enabled chunk replacement and leftover tracking.

### Async Inference

- `src/lerobot/async_inference/` already supports `pi05` as a policy type.
- The async path is based on a `PolicyServer` and `RobotClient`.
- The client/server flow is chunk-oriented and is conceptually a better fit for `predict_action_chunk()`.
- Existing async modules are general-purpose, but the current custom script `run_pi05_infer.py` does not use them.

## Main Design Question

What is the best engineering path to evolve the current working PI05 synchronous inference script into a real-time deployment path that supports:

- more stable control cadence,
- chunked execution,
- optional RTC,
- debuggability and rollout safety,
- and an implementation path that fits this repository's current structure?

## Deliverable Expectations

Each brainstorming report should:

1. stay grounded in the current repository state,
2. explicitly distinguish:
   - what already exists,
   - what can be reused,
   - what must be newly implemented,
3. produce a recommendation, not only open-ended ideas,
4. call out risks and validation strategy.

The final architect proposal should converge toward a practical implementation direction instead of preserving all branches equally.
