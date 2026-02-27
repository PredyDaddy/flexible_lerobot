async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const txt = await res.text();
  let data = null;
  try {
    data = txt ? JSON.parse(txt) : null;
  } catch {
    data = { raw: txt };
  }
  if (!res.ok) {
    const msg = data?.detail ? data.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function el(id) {
  return document.getElementById(id);
}

function setLogs(id, lines) {
  const node = el(id);
  node.textContent = (lines || []).slice(-200).join("\n");
  node.scrollTop = node.scrollHeight;
}

let SETTINGS = null;

async function playServerSound(name, block = false) {
  try {
    await api("/api/sound/play", { method: "POST", body: JSON.stringify({ name, block }) });
  } catch (e) {
    console.warn("server sound failed:", name, e);
  }
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function showOverlayCountdown(sec) {
  const overlay = el("overlay");
  const cd = el("countdown");
  overlay.classList.remove("hidden");
  overlay.setAttribute("aria-hidden", "false");
  cd.textContent = String(sec);
}

function hideOverlayCountdown() {
  const overlay = el("overlay");
  overlay.classList.add("hidden");
  overlay.setAttribute("aria-hidden", "true");
}

async function runCountdown(sec) {
  for (let i = sec; i >= 1; i--) {
    showOverlayCountdown(i);
    await wait(1000);
  }
  hideOverlayCountdown();
}

async function loadSettings() {
  SETTINGS = await api("/api/settings");
  const d = SETTINGS?.defaults || {};
  if (typeof d.dataset_dir === "string") el("rec_dataset_dir").value = d.dataset_dir;
  if (typeof d.lerobot_output_dir === "string") {
    el("conv_output_dir").value = d.lerobot_output_dir;
  }
  if (typeof d.task_name === "string") {
    el("rec_task_name").value = d.task_name;
  }
  if (typeof d.dataset_dir === "string" && typeof d.task_name === "string") {
    const inputDatasetDir = `${d.dataset_dir.replace(/\/+$/, "")}/${d.task_name}`;
    el("conv_input_dataset_dir").value = inputDatasetDir;
  } else if (typeof d.dataset_dir === "string") {
    el("conv_input_dataset_dir").value = d.dataset_dir;
  }
  if (typeof d.repo_id === "string") el("conv_repo_id").value = d.repo_id;
  if (d.num_episodes != null) el("rec_num_episodes").value = String(d.num_episodes);
  if (d.reset_time_s != null) el("rec_reset_time_s").value = String(d.reset_time_s);
}

async function refreshConfigs(desired) {
  const configs = await api("/api/configs");
  const sel = el("rec_config");
  sel.innerHTML = "";
  for (const name of configs) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
  if (desired && configs.includes(desired)) sel.value = desired;
  else if (configs.includes("default.yaml")) sel.value = "default.yaml";
}

async function pollStatus() {
  try {
    const s = await api("/api/status");
    const rec = s.record;
    const conv = s.convert;
    const recText = rec.running ? "运行中" : "已停止";
    const convText = conv.running ? "运行中" : "已停止";
    el("status").textContent = `录制：${recText}    转换：${convText}`;
    setLogs("rec_logs", rec.logs);
    setLogs("conv_logs", conv.logs);

    el("rec_start").disabled = rec.running;
    el("rec_stop").disabled = !rec.running;
    el("conv_start").disabled = conv.running;
    el("conv_stop").disabled = !conv.running;
  } catch (e) {
    el("status").textContent = `状态获取失败：${e.message}`;
  }
}

async function startRecord() {
  // Play on robot PC (server-side), not in browser.
  await playServerSound("ready", false);
  await runCountdown(3);

  const instruction = el("rec_instruction").value.trim();
  if (!instruction) {
    throw new Error("instruction 不能为空。VLA 录制必须填写“指令 / 任务”。");
  }

  const body = {
    dataset_dir: el("rec_dataset_dir").value.trim() || null,
    config_name: el("rec_config").value || null,
    task_name: el("rec_task_name").value.trim() || null,
    episode_name: el("rec_episode_name").value.trim() || null,
    num_episodes: el("rec_num_episodes").value ? Number(el("rec_num_episodes").value) : 1,
    reset_time_s: el("rec_reset_time_s").value ? Number(el("rec_reset_time_s").value) : null,
    fps: el("rec_fps").value ? Number(el("rec_fps").value) : null,
    max_frames: el("rec_max_frames").value ? Number(el("rec_max_frames").value) : null,
    use_depth: el("rec_use_depth").checked,
    instruction: instruction,
  };
  const resp = await api("/api/record/start", { method: "POST", body: JSON.stringify(body) });
  el("rec_out").textContent = `输出文件：${resp.out_hdf5}`;
}

async function stopRecord() {
  await api("/api/record/stop", { method: "POST", body: "{}" });
}

async function startConvert() {
  const inputDatasetDir = el("conv_input_dataset_dir").value.trim();
  const outputDir = el("conv_output_dir").value.trim() || null;
  const repoId = el("conv_repo_id").value.trim();
  const fps = el("conv_fps").value ? Number(el("conv_fps").value) : null;
  const useVideos = el("conv_use_videos").checked;
  const swapRb = el("conv_swap_rb").checked;
  const instructionInput = el("conv_instruction").value;
  const instruction = instructionInput.trim() || null;

  if (!inputDatasetDir) throw new Error("任务目录不能为空");
  if (!repoId) throw new Error("repo_id 不能为空");

  const body = {
    input_dataset_dir: inputDatasetDir,
    output_dir: outputDir,
    repo_id: repoId,
    fps: fps,
    use_videos: useVideos,
    swap_rb: swapRb,
    instruction: instruction,
    no_base: false,
  };
  await api("/api/convert_dataset/start", { method: "POST", body: JSON.stringify(body) });
}

async function stopConvert() {
  await api("/api/convert/stop", { method: "POST", body: "{}" });
}

function wire() {
  el("rec_start").addEventListener("click", () => startRecord().catch((e) => alert(`错误：${e.message}`)));
  el("rec_stop").addEventListener("click", () => stopRecord().catch((e) => alert(`错误：${e.message}`)));
  el("conv_start").addEventListener("click", () => startConvert().catch((e) => alert(`错误：${e.message}`)));
  el("conv_stop").addEventListener("click", () => stopConvert().catch((e) => alert(`错误：${e.message}`)));
}

async function main() {
  wire();
  await loadSettings();
  await refreshConfigs(SETTINGS?.defaults?.config_name);
  await pollStatus();
  setInterval(pollStatus, 1200);
}

main().catch((e) => {
  el("status").textContent = `初始化失败：${e.message}`;
});
