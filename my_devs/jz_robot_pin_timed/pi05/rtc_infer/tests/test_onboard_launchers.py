from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RTC_INFER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RTC_INFER_DIR.parents[3]
SERVER_SCRIPT = RTC_INFER_DIR / "run_onboard_policy_server.sh"
CLIENT_SCRIPT = RTC_INFER_DIR / "run_onboard_robot_client.sh"
PROFILE_SCRIPTS = {
    "all_170": (
        RTC_INFER_DIR / "run_onboard_policy_server_all_170.sh",
        RTC_INFER_DIR / "run_onboard_robot_client_all_170.sh",
        "JZ_PI05_ALL_170_047320_CONFIRMED",
        "all_170_047320",
        "047320",
    ),
    "all_200": (
        RTC_INFER_DIR / "run_onboard_policy_server_all_200.sh",
        RTC_INFER_DIR / "run_onboard_robot_client_all_200.sh",
        "JZ_PI05_ALL_200_007320_CONFIRMED",
        "all_200_007320",
        "007320",
    ),
}
RUN_NAME = "pi05_jz_robot_pin_timed_curated_42eps_20260713_e15_b8_20260714_202432"
ARMED_ENV_VARS = (
    "JZ_ROBOT_PIN_ARMED",
    "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT",
    "JZ_POLICY_INFERENCE_ARMED",
)


def _run_script(
    script: Path,
    *,
    env: dict[str, str],
    args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _server_env(**updates: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "POLICY_PATH",
        "CONFIG_ONLY",
        "CHECK_POLICY_LOAD",
        "PRINT_COMMAND_ONLY",
        "REQUIRE_COMPLETE_STEP",
        "SERVER_HOST",
        "SERVER_PORT",
        "SERVER_AUTH_TOKEN",
        "JZ_PI05_SERVER_AUTH_TOKEN",
        "JZ_PI05_INTERMEDIATE_010470_CONFIRMED",
        "JZ_PI05_ALL_170_047320_CONFIRMED",
        "JZ_PI05_ALL_200_007320_CONFIRMED",
        "TOKENIZER_PATH",
    ):
        env.pop(name, None)
    env.update(
        {
            "CONDA_PYTHON": sys.executable,
            "PRINT_COMMAND_ONLY": "true",
        }
    )
    env.update(updates)
    return env


def _client_env(**updates: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "ONBOARD_MODE",
        "ONBOARD_CHECKPOINT",
        "ONBOARD_SENSOR_FPS",
        "ONBOARD_CONTROL_FPS",
        "ONBOARD_RUN_TIME_S",
        "MODE",
        "EXECUTION",
        "CONFIG_ONLY",
        "HEALTH_ONLY",
        "CONNECT_SMOKE",
        "INFERENCE_SMOKE",
        "SERVER_URL",
        "SERVER_AUTH_TOKEN",
        "JZ_PI05_SERVER_AUTH_TOKEN",
        "SENSOR_FPS",
        "CONTROL_FPS",
        "RUN_TIME_S",
        "TASK",
        "JZ_PI05_SINGLE_STEP_ARMED_PASSED",
        "JZ_PI05_INTERMEDIATE_010470_CONFIRMED",
        "JZ_PI05_ALL_170_047320_CONFIRMED",
        "JZ_PI05_ALL_200_007320_CONFIRMED",
        "JZ_PI05_EXPECTED_CHECKPOINT_STEP",
        "JZ_PI05_EXPECTED_CONFIGURED_STEPS",
        "JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT",
        "JZ_PI05_EXPECTED_CHECKPOINT_PATH",
        "JZ_PI05_EXPECTED_COMPLETE_STEP",
        "JZ_PI05_DISABLE_JOINT_DELTA_CHECKS",
        "I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED",
    ):
        env.pop(name, None)
    env.update(
        {
            "CONDA_PYTHON": sys.executable,
            "PRINT_COMMAND_ONLY": "true",
            "RUN_STAMP": "pytest_onboard",
            **dict.fromkeys(ARMED_ENV_VARS, "1"),
        }
    )
    env.update(updates)
    return env


def _output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def _make_final_checkpoint(
    tmp_path: Path,
    *,
    configured_steps: int = 15705,
    training_step: int = 15705,
) -> Path:
    policy_path = tmp_path / RUN_NAME / "checkpoints" / "015705" / "pretrained_model"
    policy_path.mkdir(parents=True)
    (policy_path / "train_config.json").write_text(
        json.dumps({"steps": configured_steps}),
        encoding="utf-8",
    )
    training_state = policy_path.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text(
        json.dumps({"step": training_step}),
        encoding="utf-8",
    )
    return policy_path


def _make_intermediate_checkpoint(
    tmp_path: Path,
    *,
    configured_steps: int = 15705,
    training_step: int = 10470,
) -> Path:
    policy_path = tmp_path / RUN_NAME / "checkpoints" / "010470" / "pretrained_model"
    policy_path.mkdir(parents=True)
    (policy_path / "train_config.json").write_text(
        json.dumps({"steps": configured_steps}),
        encoding="utf-8",
    )
    training_state = policy_path.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text(
        json.dumps({"step": training_step}),
        encoding="utf-8",
    )
    return policy_path


@pytest.mark.parametrize(
    "script",
    [
        SERVER_SCRIPT,
        CLIENT_SCRIPT,
        *(script for profile in PROFILE_SCRIPTS.values() for script in profile[:2]),
    ],
)
def test_onboard_launchers_are_executable_and_have_valid_bash_syntax(script: Path) -> None:
    assert script.is_file()
    assert os.access(script, os.X_OK)

    completed = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, _output(completed)


def test_server_default_is_exact_final_e15_loopback_command() -> None:
    completed = _run_script(SERVER_SCRIPT, env=_server_env())
    output = _output(completed)
    expected_policy = REPO_ROOT / RUN_NAME / "checkpoints" / "015705" / "pretrained_model"
    expected_tokenizer = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"

    assert completed.returncode == 0, output
    assert f"--policy-path={expected_policy}" in output
    assert f"--tokenizer-path={expected_tokenizer}" in output
    assert "--host=127.0.0.1" in output
    assert "--port=8088" in output
    assert "--device=cuda" in output
    assert "--require-complete-step=true" in output
    assert "nothing was started" in output


def test_server_rejects_unconfirmed_010470_checkpoint(tmp_path: Path) -> None:
    policy_path = _make_intermediate_checkpoint(tmp_path)
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(POLICY_PATH=str(policy_path)),
    )

    assert completed.returncode == 2
    assert "JZ_PI05_INTERMEDIATE_010470_CONFIRMED=1" in completed.stderr


def test_server_rejects_last_even_with_intermediate_confirmation(tmp_path: Path) -> None:
    policy_path = tmp_path / RUN_NAME / "checkpoints" / "last" / "pretrained_model"
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(
            POLICY_PATH=str(policy_path),
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
        ),
    )

    assert completed.returncode == 2
    assert "must select" in completed.stderr


def test_server_accepts_only_explicitly_confirmed_010470_without_starting(tmp_path: Path) -> None:
    policy_path = _make_intermediate_checkpoint(tmp_path)
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(
            POLICY_PATH=str(policy_path),
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
        ),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert "checkpoint_mode=intermediate_010470" in output
    assert "checkpoint_step=10470 configured_steps=15705" in output
    assert f"--policy-path={policy_path}" in output
    assert "--require-complete-step=false" in output
    assert "nothing was started" in output


@pytest.mark.parametrize(
    ("configured_steps", "training_step", "message"),
    [
        (10470, 10470, "train_config.steps must equal 15705"),
        (15705, 15705, "training_state.step must equal 10470"),
    ],
)
def test_confirmed_010470_still_requires_exact_metadata(
    tmp_path: Path,
    configured_steps: int,
    training_step: int,
    message: str,
) -> None:
    policy_path = _make_intermediate_checkpoint(
        tmp_path,
        configured_steps=configured_steps,
        training_step=training_step,
    )
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(
            POLICY_PATH=str(policy_path),
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
        ),
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_confirmed_010470_rejects_complete_step_true_override(tmp_path: Path) -> None:
    policy_path = _make_intermediate_checkpoint(tmp_path)
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(
            POLICY_PATH=str(policy_path),
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
            REQUIRE_COMPLETE_STEP="true",
        ),
    )

    assert completed.returncode == 2
    assert "requires REQUIRE_COMPLETE_STEP=false" in completed.stderr


@pytest.mark.parametrize(
    ("configured_steps", "training_step", "message"),
    [
        (10470, 15705, "train_config.steps must equal 15705"),
        (15705, 10470, "training_state.step must equal 15705"),
    ],
)
def test_server_validates_both_final_step_metadata_files(
    tmp_path: Path,
    configured_steps: int,
    training_step: int,
    message: str,
) -> None:
    policy_path = _make_final_checkpoint(
        tmp_path,
        configured_steps=configured_steps,
        training_step=training_step,
    )
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(POLICY_PATH=str(policy_path)),
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_server_accepts_matching_final_step_metadata_without_starting(tmp_path: Path) -> None:
    policy_path = _make_final_checkpoint(tmp_path)
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(POLICY_PATH=str(policy_path)),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert f"--policy-path={policy_path}" in output
    assert "checkpoint_mode=final_015705" in output
    assert "checkpoint_step=15705 configured_steps=15705" in output
    assert "nothing was started" in output


@pytest.mark.parametrize(
    ("relative_path", "contents", "message"),
    [
        ("train_config.json", "not-json", "cannot read final-checkpoint metadata"),
        ("train_config.json", "[]", "must be a JSON object"),
        ("../training_state/training_step.json", "{}", "training_state.step must equal 15705"),
    ],
)
def test_server_rejects_malformed_or_missing_final_metadata_values(
    tmp_path: Path,
    relative_path: str,
    contents: str,
    message: str,
) -> None:
    policy_path = _make_final_checkpoint(tmp_path)
    (policy_path / relative_path).resolve().write_text(contents, encoding="utf-8")
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(POLICY_PATH=str(policy_path)),
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_server_rejects_final_checkpoint_path_that_is_not_a_directory(tmp_path: Path) -> None:
    policy_path = tmp_path / RUN_NAME / "checkpoints" / "015705" / "pretrained_model"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("not a checkpoint directory", encoding="utf-8")
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(POLICY_PATH=str(policy_path)),
    )

    assert completed.returncode == 2
    assert "POLICY_PATH is not a directory" in completed.stderr


def test_server_complete_step_requirement_cannot_be_disabled() -> None:
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(REQUIRE_COMPLETE_STEP="false"),
    )

    assert completed.returncode == 2
    assert "requires REQUIRE_COMPLETE_STEP=true" in completed.stderr


def test_server_rejects_config_only_combined_with_policy_load_check() -> None:
    completed = _run_script(
        SERVER_SCRIPT,
        env=_server_env(CONFIG_ONLY="true", CHECK_POLICY_LOAD="true"),
    )

    assert completed.returncode == 2
    assert "mutually exclusive" in completed.stderr


def test_non_loopback_server_requires_token_and_redacts_it() -> None:
    missing_token = _run_script(
        SERVER_SCRIPT,
        env=_server_env(SERVER_HOST="0.0.0.0"),
    )
    assert missing_token.returncode == 2
    assert "non-loopback SERVER_HOST requires" in missing_token.stderr

    with_token = _run_script(
        SERVER_SCRIPT,
        env=_server_env(SERVER_HOST="0.0.0.0", SERVER_AUTH_TOKEN="server-test-secret"),
    )
    output = _output(with_token)
    assert with_token.returncode == 0, output
    assert "--host=0.0.0.0" in output
    assert "auth=enabled" in output
    assert "server-test-secret" not in output


def test_client_default_is_fixed_low_rate_armed_single_step() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            MODE="rtc",
            EXECUTION="dry_run",
            SENSOR_FPS="99",
            CONTROL_FPS="99",
            RUN_TIME_S="99",
            TASK="wrong task",
            CONFIG_ONLY="true",
            HEALTH_ONLY="true",
            CONNECT_SMOKE="true",
            INFERENCE_SMOKE="true",
        ),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert "REAL ROBOT ARMED mode=single_step" in output
    assert "checkpoint=final_015705" in output
    assert "joint_delta_checks=enabled" in output
    assert "task=jz robot pin timed vr teleoperation" in output
    assert "--mode=single_step" in output
    assert "--execution=armed" in output
    assert "--sensor-fps=5" in output
    assert "--control-fps=5" in output
    assert "--run-time-s=1" in output
    assert "--empty-queue-strategy=stop" in output
    assert "--config-only=false" in output
    assert "--health-only=false" in output
    assert "--connect-smoke=false" in output
    assert "--inference-smoke=false" in output
    assert "nothing was started" in output


def test_client_joint_delta_bypass_requires_both_explicit_confirmations() -> None:
    missing_ack = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(JZ_PI05_DISABLE_JOINT_DELTA_CHECKS="1"),
    )
    assert missing_ack.returncode == 2
    assert "I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1" in missing_ack.stderr

    missing_selection = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED="1"),
    )
    assert missing_selection.returncode == 2
    assert "requires JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1" in missing_selection.stderr


def test_client_joint_delta_bypass_is_visible_without_starting() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            ONBOARD_CHECKPOINT="010470",
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
            JZ_PI05_DISABLE_JOINT_DELTA_CHECKS="1",
            I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED="1",
        ),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert "joint_delta_checks=disabled" in output
    assert "nothing was started" in output


def test_client_010470_requires_checkpoint_selection_and_confirmation() -> None:
    missing_confirmation = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(ONBOARD_CHECKPOINT="010470"),
    )
    assert missing_confirmation.returncode == 2
    assert "JZ_PI05_INTERMEDIATE_010470_CONFIRMED=1" in missing_confirmation.stderr

    missing_selection = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1"),
    )
    assert missing_selection.returncode == 2
    assert "requires ONBOARD_CHECKPOINT=010470" in missing_selection.stderr


def test_client_accepts_explicitly_confirmed_010470_without_starting() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            ONBOARD_CHECKPOINT="010470",
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
            ONBOARD_SENSOR_FPS="3",
            ONBOARD_CONTROL_FPS="4",
            ONBOARD_RUN_TIME_S="12",
        ),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert "REAL ROBOT ARMED mode=single_step" in output
    assert "checkpoint=intermediate_010470" in output
    assert "--mode=single_step" in output
    assert "--execution=armed" in output
    assert "--sensor-fps=3" in output
    assert "--control-fps=4" in output
    assert "--run-time-s=12" in output
    assert "nothing was started" in output


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ONBOARD_SENSOR_FPS", "0"),
        ("ONBOARD_SENSOR_FPS", "21"),
        ("ONBOARD_CONTROL_FPS", "not-an-integer"),
    ],
)
def test_client_rejects_invalid_explicit_onboard_fps(name: str, value: str) -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            ONBOARD_CHECKPOINT="010470",
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
            **{name: value},
        ),
    )

    assert completed.returncode == 2
    assert f"{name} must be an integer in 1..20" in completed.stderr


@pytest.mark.parametrize("value", ["0", "301", "1.5", "not-an-integer"])
def test_client_rejects_invalid_explicit_onboard_run_time(value: str) -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            ONBOARD_CHECKPOINT="010470",
            JZ_PI05_INTERMEDIATE_010470_CONFIRMED="1",
            ONBOARD_RUN_TIME_S=value,
        ),
    )

    assert completed.returncode == 2
    assert "ONBOARD_RUN_TIME_S must be an integer in 1..300" in completed.stderr


def test_client_rejects_invalid_onboard_mode() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(ONBOARD_MODE="dry_run"),
    )

    assert completed.returncode == 2
    assert "ONBOARD_MODE must be one of [single_step async_single_step rtc]" in completed.stderr


@pytest.mark.parametrize("missing_name", ARMED_ENV_VARS)
def test_client_requires_every_armed_confirmation(missing_name: str) -> None:
    env = _client_env()
    env.pop(missing_name)
    completed = _run_script(CLIENT_SCRIPT, env=env)

    assert completed.returncode == 2
    assert missing_name in completed.stderr


def test_rtc_requires_separate_single_step_confirmation() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(ONBOARD_MODE="rtc"),
    )

    assert completed.returncode == 2
    assert "JZ_PI05_SINGLE_STEP_ARMED_PASSED=1" in completed.stderr


def test_async_single_step_uses_async_launcher_configuration() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            ONBOARD_MODE="async_single_step",
            JZ_PI05_SINGLE_STEP_ARMED_PASSED="1",
            ONBOARD_SENSOR_FPS="17",
            ONBOARD_CONTROL_FPS="19",
            ONBOARD_RUN_TIME_S="23",
        ),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert "REAL ROBOT ARMED mode=async_single_step" in output
    assert "--mode=async_single_step" in output
    assert "--sensor-fps=17" in output
    assert "--control-fps=19" in output
    assert "--run-time-s=23" in output


@pytest.mark.parametrize(
    ("server_script", "client_script", "confirmation", "profile", "checkpoint_dir"),
    PROFILE_SCRIPTS.values(),
)
def test_new_weight_wrappers_lock_profile_and_only_print(
    server_script: Path,
    client_script: Path,
    confirmation: str,
    profile: str,
    checkpoint_dir: str,
) -> None:
    missing_confirmation = _run_script(server_script, env=_server_env())
    assert missing_confirmation.returncode == 2
    assert confirmation in missing_confirmation.stderr

    server = _run_script(server_script, env=_server_env(**{confirmation: "1"}))
    server_output = _output(server)
    assert server.returncode == 0, server_output
    assert f"checkpoint_mode={profile}" in server_output
    assert f"/checkpoints/{checkpoint_dir}/pretrained_model" in server_output
    assert "--require-complete-step=false" in server_output
    assert "nothing was started" in server_output

    client = _run_script(
        client_script,
        env=_client_env(
            **{
                confirmation: "1",
                "ONBOARD_MODE": "rtc",
                "JZ_PI05_SINGLE_STEP_ARMED_PASSED": "1",
            }
        ),
    )
    client_output = _output(client)
    assert client.returncode == 0, client_output
    assert f"checkpoint={profile}" in client_output
    assert "--mode=rtc" in client_output
    assert "nothing was started" in client_output


def test_rtc_uses_fixed_short_20_hz_queue_configuration() -> None:
    completed = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            ONBOARD_MODE="rtc",
            JZ_PI05_SINGLE_STEP_ARMED_PASSED="1",
            SENSOR_FPS="1",
            CONTROL_FPS="1",
            RUN_TIME_S="1",
        ),
    )
    output = _output(completed)

    assert completed.returncode == 0, output
    assert "REAL ROBOT ARMED mode=rtc" in output
    assert "--mode=rtc" in output
    assert "--execution=armed" in output
    assert "--sensor-fps=20" in output
    assert "--control-fps=20" in output
    assert "--run-time-s=10" in output
    assert "--queue-low-watermark=30" in output
    assert "--max-queue-size=50" in output
    assert "--rtc-execution-horizon=10" in output
    assert "nothing was started" in output


def test_non_loopback_client_requires_token_and_redacts_it() -> None:
    missing_token = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(SERVER_URL="http://192.168.50.10:8088"),
    )
    assert missing_token.returncode == 2
    assert "non-loopback SERVER_URL requires" in missing_token.stderr

    with_token = _run_script(
        CLIENT_SCRIPT,
        env=_client_env(
            SERVER_URL="http://192.168.50.10:8088",
            SERVER_AUTH_TOKEN="client-test-secret",
        ),
    )
    output = _output(with_token)
    assert with_token.returncode == 0, output
    assert "--server-url=http://192.168.50.10:8088" in output
    assert "auth=enabled" in output
    assert "client-test-secret" not in output


@pytest.mark.parametrize("script", [SERVER_SCRIPT, CLIENT_SCRIPT])
def test_onboard_launchers_refuse_cli_overrides(script: Path) -> None:
    env = _server_env() if script == SERVER_SCRIPT else _client_env()
    completed = _run_script(script, env=env, args=("--mode=rtc",))

    assert completed.returncode == 2
    assert "accepts no CLI arguments" in completed.stderr


def test_onboard_scripts_have_no_machine_placeholders_or_angle_limit_flags() -> None:
    combined = SERVER_SCRIPT.read_text(encoding="utf-8") + CLIENT_SCRIPT.read_text(encoding="utf-8")

    for placeholder in ("<HIGH3_IP>", "<TOKEN>", "CHANGE_ME", "TODO_SERVER_IP"):
        assert placeholder not in combined
    assert "max_initial_joint_delta" not in combined
    assert "max_joint_step" not in combined
