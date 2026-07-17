"use strict";

const translations = {
  zh: {
    pageTitle: "JZ Timed 数采系统",
    appTitle: "数采系统",
    systemStatusAria: "系统状态",
    statusVisualization: "可视化",
    statusOperation: "任务",
    statusControl: "控制",
    checking: "检查中",
    languageToggleLabel: "English",
    languageToggleTitle: "Switch to English",
    tabsAria: "数采系统视图",
    tabVisualization: "可视化",
    tabRecord: "录制",
    tabReplay: "回放",
    tabLogs: "运行日志",
    flexibleEpisodeCount: "自由设置采集条数",
    visualizationTitle: "整机可视化",
    stopped: "已停止",
    waitingStart: "等待启动",
    meshcatAddress: "Meshcat 地址",
    targetAction: "目标动作",
    publishMode: "发布模式",
    start: "启动",
    stop: "停止",
    openPage: "打开页面",
    openMeshcatTitle: "在新窗口打开 Meshcat",
    recordTitle: "正式数据录制",
    standby: "待命",
    recordRootLabel: "完整数据集保存路径",
    suggestPath: "生成默认时间戳路径",
    recordPathRequired: "必须填写绝对路径，目标目录不能已经存在",
    episodeCountLabel: "本次采集条数",
    episodeCountHint: "可填写任意正整数；全部采集完成后统一编码",
    episodeCountInvalid: "采集条数必须是正整数",
    episodeCountValue: "{value} 条",
    recordParamsAria: "录制固定参数",
    count: "数量",
    episodeDuration: "单条时长",
    frameRate: "帧率",
    video: "视频",
    advanced: "高级参数",
    episodeDurationSeconds: "单条时长（秒）",
    resetWaitSeconds: "复位等待（秒）",
    recordFps: "录制 FPS",
    videoCrf: "视频 CRF",
    stateAdvanceWaitSeconds: "State 推进等待（秒）",
    cameraStateSkewMs: "相机/State 对齐上限（毫秒）",
    initialDeltaRad: "首帧角度差（rad）",
    stepDeltaRad: "单步角度差（rad）",
    output: "输出",
    noPathSelected: "尚未选择路径",
    startRecord: "开始录制",
    replayTitle: "数据回放",
    discoveredDatasetPath: "已发现的数据集",
    customDatasetPath: "自定义数据集绝对路径",
    loadDatasetPath: "加载并校验",
    customDatasetHint: "可以加载 tests/outputs 之外、格式一致的 timed 数据集",
    customPathRequired: "请输入自定义数据集的绝对路径",
    customPathAbsolute: "自定义数据集路径必须以 / 开头",
    customDatasetLoaded: "校验通过：{episodes} 条 · {frames} 帧 · {fps} FPS",
    customDatasetReady: "自定义回放数据集已加载",
    loadingDatasets: "正在读取数据集…",
    refreshDatasets: "刷新数据集",
    selectTimedDataset: "请选择一个 timed 数据集",
    replayEpisode: "回放条目",
    selectDatasetFirst: "先选择数据集",
    replayFps: "回放 FPS",
    sounds: "声音",
    enabled: "开启",
    disabled: "关闭",
    selection: "选择",
    noDatasetSelected: "尚未选择数据集",
    startReplay: "开始回放",
    runtimeLogs: "运行日志",
    clearView: "清空视图",
    noCurrentTask: "当前没有任务",
    waitingTask: "等待任务启动",
    recentCommand: "最近生成的命令",
    noCommand: "尚未生成命令",
    confirmStart: "确认开始",
    confirmSiteStatus: "请确认现场状态。",
    safetyConfirmation: "Orin armed executor 已启动，急停可用，机器人工作区安全",
    cancel: "取消",
    confirmExecute: "确认并执行",
    requestFailed: "请求失败：{status}",
    secondsValue: "{value} 秒",
    pathMustStartSlash: "路径必须以 / 开头",
    willCreateDataset: "将创建新的 {episodes}-episode 数据集，完成后统一编码",
    chooseEpisode: "请选择条目",
    noReplayDatasets: "没有发现可回放的数据集",
    chooseDatasetPath: "选择数据集路径",
    datasetOption: "{name} · {episodes} 条",
    episodeOption: "第 {number} 条（episode {index}）",
    datasetMeta: "{path} · {frames} 帧 · {fps} FPS",
    visualizationStarted: "可视化进程已启动",
    visualizationStopped: "可视化已停止",
    fillRecordPath: "请填写录制保存路径",
    startRecordConfirm: "开始录制 {episodes} 条数据",
    outputPath: "输出路径：{path} · 采集条数：{episodes}",
    recordStarted: "录制任务已启动",
    chooseReplayDataset: "请选择回放数据集",
    chooseReplayEpisode: "请选择回放条目",
    startReplayConfirm: "开始回放 episode {episode}",
    replayStarted: "回放任务已启动",
    taskStopped: "任务已停止",
    noRunningTask: "当前没有运行中的任务",
    running: "运行中",
    meshcatPublishing: "Meshcat 发布中",
    record: "录制",
    replay: "回放",
    recording: "录制中",
    replaying: "回放中",
    mock: "模拟",
    unlocked: "已解锁",
    locked: "未解锁",
    mockBanner: "模拟模式 · 所有按钮仅生成模拟进程，不执行机器人命令",
    blockedBanner: "控制未解锁 · 录制和回放按钮不可用",
    armedBanner: "现场控制已解锁 · 录制和回放会发送 armed UDP command",
    startedAt: "{kind} · 启动于 {time}",
    ended: "{kind} · 已结束",
    connectionFailed: "连接失败",
    datasetsRefreshed: "数据集列表已刷新",
    logsCleared: "日志视图已清空；新任务启动后恢复",
    controlTokenPrompt: "请输入数采系统控制令牌",
    initializationFailed: "初始化失败：{message}",
    serviceConnectionFailed: "服务连接失败",
  },
  en: {
    pageTitle: "JZ Timed Data Collection System",
    appTitle: "Data Collection System",
    systemStatusAria: "System status",
    statusVisualization: "Visualization",
    statusOperation: "Operation",
    statusControl: "Control",
    checking: "Checking",
    languageToggleLabel: "中文",
    languageToggleTitle: "Switch to Chinese",
    tabsAria: "Collection system views",
    tabVisualization: "Visualization",
    tabRecord: "Record",
    tabReplay: "Replay",
    tabLogs: "Logs",
    flexibleEpisodeCount: "FLEXIBLE EPISODE COUNT",
    visualizationTitle: "Robot Visualization",
    stopped: "Stopped",
    waitingStart: "Waiting to start",
    meshcatAddress: "Meshcat URL",
    targetAction: "Target action",
    publishMode: "Publish mode",
    start: "Start",
    stop: "Stop",
    openPage: "Open page",
    openMeshcatTitle: "Open Meshcat in a new window",
    recordTitle: "Production Data Recording",
    standby: "Standby",
    recordRootLabel: "Full dataset output path",
    suggestPath: "Generate a timestamped default path",
    recordPathRequired: "Enter an absolute path; the target directory must not already exist",
    episodeCountLabel: "Episodes to collect",
    episodeCountHint: "Enter any positive integer; encoding starts after all episodes are collected",
    episodeCountInvalid: "The episode count must be a positive integer",
    episodeCountValue: "{value} episodes",
    recordParamsAria: "Fixed recording parameters",
    count: "Count",
    episodeDuration: "Episode duration",
    frameRate: "Frame rate",
    video: "Video",
    advanced: "Advanced parameters",
    episodeDurationSeconds: "Episode duration (seconds)",
    resetWaitSeconds: "Reset wait (seconds)",
    recordFps: "Recording FPS",
    videoCrf: "Video CRF",
    stateAdvanceWaitSeconds: "State advance wait (seconds)",
    cameraStateSkewMs: "Camera/state skew limit (ms)",
    initialDeltaRad: "Initial joint delta (rad)",
    stepDeltaRad: "Per-step joint delta (rad)",
    output: "Output",
    noPathSelected: "No path selected",
    startRecord: "Start recording",
    replayTitle: "Dataset Replay",
    discoveredDatasetPath: "Discovered datasets",
    customDatasetPath: "Custom absolute dataset path",
    loadDatasetPath: "Load and validate",
    customDatasetHint: "Load any compatible timed dataset, including paths outside tests/outputs",
    customPathRequired: "Enter an absolute custom dataset path",
    customPathAbsolute: "The custom dataset path must start with /",
    customDatasetLoaded: "Validated: {episodes} episodes · {frames} frames · {fps} FPS",
    customDatasetReady: "Custom replay dataset loaded",
    loadingDatasets: "Loading datasets…",
    refreshDatasets: "Refresh datasets",
    selectTimedDataset: "Select a timed dataset",
    replayEpisode: "Replay episode",
    selectDatasetFirst: "Select a dataset first",
    replayFps: "Replay FPS",
    sounds: "Sounds",
    enabled: "Enabled",
    disabled: "Disabled",
    selection: "Selection",
    noDatasetSelected: "No dataset selected",
    startReplay: "Start replay",
    runtimeLogs: "Runtime Logs",
    clearView: "Clear view",
    noCurrentTask: "No current operation",
    waitingTask: "Waiting for an operation to start",
    recentCommand: "Most recent generated command",
    noCommand: "No command generated",
    confirmStart: "Confirm start",
    confirmSiteStatus: "Confirm the on-site status.",
    safetyConfirmation: "The Orin armed executor is running, the emergency stop is available, and the robot workspace is safe",
    cancel: "Cancel",
    confirmExecute: "Confirm and execute",
    requestFailed: "Request failed: {status}",
    secondsValue: "{value} sec",
    pathMustStartSlash: "The path must start with /",
    willCreateDataset: "A new {episodes}-episode dataset will be created and encoded when complete",
    chooseEpisode: "select an episode",
    noReplayDatasets: "No replayable datasets found",
    chooseDatasetPath: "Select a dataset path",
    datasetOption: "{name} · {episodes} episodes",
    episodeOption: "Episode {number} (index {index})",
    datasetMeta: "{path} · {frames} frames · {fps} FPS",
    visualizationStarted: "Visualization process started",
    visualizationStopped: "Visualization stopped",
    fillRecordPath: "Enter the recording output path",
    startRecordConfirm: "Start recording {episodes} episodes",
    outputPath: "Output path: {path} · Episodes: {episodes}",
    recordStarted: "Recording operation started",
    chooseReplayDataset: "Select a replay dataset",
    chooseReplayEpisode: "Select an episode to replay",
    startReplayConfirm: "Start replaying episode {episode}",
    replayStarted: "Replay operation started",
    taskStopped: "Operation stopped",
    noRunningTask: "No operation is currently running",
    running: "Running",
    meshcatPublishing: "Publishing to Meshcat",
    record: "Record",
    replay: "Replay",
    recording: "Recording",
    replaying: "Replaying",
    mock: "Mock",
    unlocked: "Unlocked",
    locked: "Locked",
    mockBanner: "Mock mode · Buttons only create mock processes; no robot commands are sent",
    blockedBanner: "Control locked · Recording and replay controls are disabled",
    armedBanner: "On-site control unlocked · Recording and replay send armed UDP commands",
    startedAt: "{kind} · started at {time}",
    ended: "{kind} · finished",
    connectionFailed: "Connection failed",
    datasetsRefreshed: "Dataset list refreshed",
    logsCleared: "Log view cleared; it will resume when a new operation starts",
    controlTokenPrompt: "Enter the data collection system control token",
    initializationFailed: "Initialization failed: {message}",
    serviceConnectionFailed: "Service connection failed",
  },
};

function initialLanguage() {
  try {
    return window.localStorage.getItem("jz_web_language") === "en" ? "en" : "zh";
  } catch (_error) {
    return "zh";
  }
}

const state = {
  bootstrap: null,
  datasets: [],
  status: null,
  selectedDataset: null,
  customDatasetActive: false,
  confirmAction: null,
  confirmContext: null,
  logViewCleared: false,
  language: initialLanguage(),
};

function t(key, values) {
  const dictionary = translations[state.language] || translations.zh;
  const fallback = translations.zh[key] || key;
  return String(dictionary[key] || fallback).replace(/\{(\w+)\}/g, function (_match, name) {
    return values && values[name] !== undefined ? String(values[name]) : "{" + name + "}";
  });
}

function byId(id) {
  return document.getElementById(id);
}

function applyTranslations() {
  document.documentElement.lang = state.language === "en" ? "en" : "zh-CN";
  document.title = t("pageTitle");
  document.querySelectorAll("[data-i18n]").forEach(function (element) {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-title]").forEach(function (element) {
    element.title = t(element.dataset.i18nTitle);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach(function (element) {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });

  if (state.bootstrap) updateRecordPreview();
  if (state.datasets.length || state.bootstrap) {
    const selectedPath = byId("replay-dataset").value;
    const selectedEpisode = byId("replay-episode").value;
    renderDatasetOptions(selectedPath, selectedEpisode);
  }
  if (state.status) renderStatus(state.status);
  if (state.confirmContext) renderConfirmation();
  if (state.logViewCleared) byId("runtime-logs").textContent = t("logsCleared");
}

function setLanguage(language) {
  state.language = language === "en" ? "en" : "zh";
  try {
    window.localStorage.setItem("jz_web_language", state.language);
  } catch (_error) {
    // The switch still works for the current page when storage is unavailable.
  }
  applyTranslations();
}

function controlHeaders() {
  const headers = { "Content-Type": "application/json" };
  const token = window.localStorage.getItem("jz_web_control_token");
  if (token) headers["X-JZ-Control-Token"] = token;
  return headers;
}

async function api(path, options) {
  const request = options || {};
  request.headers = Object.assign({}, controlHeaders(), request.headers || {});
  const response = await fetch(path, request);
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) throw new Error(payload.error || t("requestFailed", { status: response.status }));
  return payload;
}

function post(path, body) {
  return api(path, { method: "POST", body: JSON.stringify(body || {}) });
}

function toast(message, kind) {
  const item = document.createElement("div");
  item.className = "toast " + (kind || "info");
  item.textContent = message;
  byId("toast-region").appendChild(item);
  window.setTimeout(function () {
    item.classList.add("leaving");
    window.setTimeout(function () { item.remove(); }, 180);
  }, 3600);
}

function selectTab(name) {
  document.querySelectorAll(".tab").forEach(function (button) {
    const active = button.dataset.tab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach(function (panel) {
    panel.classList.toggle("active", panel.id === "tab-" + name);
  });
}

function setDot(id, status) {
  byId(id).className = "status-dot " + status;
}

function setBadge(id, text, status) {
  const badge = byId(id);
  badge.textContent = text;
  badge.className = "state-badge " + status;
}

function formatTime(timestamp) {
  if (!timestamp) return "—";
  const locale = state.language === "en" ? "en-GB" : "zh-CN";
  return new Date(timestamp * 1000).toLocaleTimeString(locale, { hour12: false });
}

function recordOptions() {
  return {
    NUM_EPISODES: byId("record-episodes").value,
    EPISODE_TIME_S: byId("record-episode-time").value,
    RESET_TIME_S: byId("record-reset-time").value,
    RECORD_FPS: byId("record-fps").value,
    VIDEO_CRF: byId("record-crf").value,
    STATE_ADVANCE_TIMEOUT_S: byId("record-state-timeout").value,
    MAX_CAMERA_STATE_RECEIVE_SKEW_MS: byId("record-camera-skew").value,
    MAX_INITIAL_JOINT_DELTA_RAD: byId("record-initial-delta").value,
    MAX_JOINT_STEP_RAD: byId("record-step-delta").value,
  };
}

function replayOptions() {
  return {
    REPLAY_FPS: byId("replay-fps").value,
    MAX_INITIAL_JOINT_DELTA_RAD: byId("replay-initial-delta").value,
    MAX_JOINT_STEP_RAD: byId("replay-step-delta").value,
    PLAY_SOUNDS: byId("replay-sounds").value,
  };
}

function selectedEpisodeCount() {
  const raw = byId("record-episodes").value.trim();
  const value = Number(raw);
  return raw !== "" && Number.isInteger(value) && value > 0 ? value : null;
}

function updateRecordPreview() {
  const path = byId("record-root").value.trim();
  const episodeCount = selectedEpisodeCount();
  byId("record-output-preview").textContent = path || t("noPathSelected");
  byId("record-summary-count").textContent = episodeCount === null
    ? "—"
    : t("episodeCountValue", { value: episodeCount });
  byId("record-summary-time").textContent = t("secondsValue", {
    value: byId("record-episode-time").value || "10",
  });
  byId("record-summary-fps").textContent = (byId("record-fps").value || "20") + " FPS";
  byId("record-summary-video").textContent = "H.264 / CRF" + (byId("record-crf").value || "18");
  const episodeStatus = byId("record-episodes-status");
  if (episodeCount === null) {
    episodeStatus.textContent = t("episodeCountInvalid");
    episodeStatus.className = "field-status error";
  } else {
    episodeStatus.textContent = t("episodeCountHint");
    episodeStatus.className = "field-status ok";
  }
  const pathStatus = byId("record-path-status");
  if (!path) {
    pathStatus.textContent = t("recordPathRequired");
    pathStatus.className = "field-status";
  } else if (!path.startsWith("/")) {
    pathStatus.textContent = t("pathMustStartSlash");
    pathStatus.className = "field-status error";
  } else {
    pathStatus.textContent = t("willCreateDataset", { episodes: episodeCount || "—" });
    pathStatus.className = "field-status ok";
  }
}

function updateReplayPreview() {
  const dataset = state.selectedDataset;
  const episode = byId("replay-episode").value;
  if (!dataset) {
    byId("replay-selection-preview").textContent = t("noDatasetSelected");
    return;
  }
  byId("replay-selection-preview").textContent =
    dataset.name + (episode === "" ? " · " + t("chooseEpisode") : " · episode " + episode);
}

function applyDefaults() {
  const bootstrap = state.bootstrap;
  const record = bootstrap.record_defaults;
  const replay = bootstrap.replay_defaults;
  byId("record-root").value = bootstrap.suggested_record_root;
  byId("record-episodes").value = record.NUM_EPISODES;
  byId("record-episode-time").value = record.EPISODE_TIME_S;
  byId("record-reset-time").value = record.RESET_TIME_S;
  byId("record-fps").value = record.RECORD_FPS;
  byId("record-crf").value = record.VIDEO_CRF;
  byId("record-state-timeout").value = record.STATE_ADVANCE_TIMEOUT_S;
  byId("record-camera-skew").value = record.MAX_CAMERA_STATE_RECEIVE_SKEW_MS;
  byId("record-initial-delta").value = record.MAX_INITIAL_JOINT_DELTA_RAD;
  byId("record-step-delta").value = record.MAX_JOINT_STEP_RAD;
  byId("replay-fps").value = replay.REPLAY_FPS;
  byId("replay-initial-delta").value = replay.MAX_INITIAL_JOINT_DELTA_RAD;
  byId("replay-step-delta").value = replay.MAX_JOINT_STEP_RAD;
  byId("replay-sounds").value = replay.PLAY_SOUNDS;
  byId("visual-url").href = bootstrap.visualization_url;
  byId("visual-url").textContent = bootstrap.visualization_url.replace(/^https?:\/\//, "");
  updateRecordPreview();
}

async function loadDatasets(preserveSelection) {
  const selected = preserveSelection ? byId("replay-dataset").value : "";
  const selectedEpisode = preserveSelection ? byId("replay-episode").value : "";
  const response = await api("/api/datasets");
  state.datasets = response.datasets;
  renderDatasetOptions(selected, selectedEpisode);
}

function renderDatasetOptions(selectedPath, selectedEpisode) {
  const select = byId("replay-dataset");
  const customDataset = state.customDatasetActive ? state.selectedDataset : null;
  select.innerHTML = "";
  if (!state.datasets.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = t("noReplayDatasets");
    select.appendChild(option);
    select.disabled = true;
  } else {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = t("chooseDatasetPath");
    select.appendChild(empty);
    state.datasets.forEach(function (dataset) {
      const option = document.createElement("option");
      option.value = dataset.path;
      option.textContent = t("datasetOption", {
        name: dataset.name,
        episodes: dataset.total_episodes,
      });
      option.title = dataset.path;
      select.appendChild(option);
    });
    select.disabled = false;
  }

  if (selectedPath && state.datasets.some(function (item) { return item.path === selectedPath; })) {
    select.value = selectedPath;
    state.customDatasetActive = false;
    state.selectedDataset = state.datasets.find(function (item) {
      return item.path === selectedPath;
    }) || null;
    populateEpisodes(state.selectedDataset, selectedEpisode);
    renderCustomDatasetStatus(null);
  } else if (customDataset) {
    select.value = "";
    state.selectedDataset = customDataset;
    state.customDatasetActive = true;
    populateEpisodes(customDataset, selectedEpisode);
    renderCustomDatasetStatus(customDataset);
  } else {
    state.selectedDataset = null;
    state.customDatasetActive = false;
    populateEpisodes(null);
    renderCustomDatasetStatus(null);
  }
}

function renderCustomDatasetStatus(dataset) {
  const status = byId("replay-custom-status");
  if (!dataset) {
    status.textContent = t("customDatasetHint");
    status.className = "field-status";
    return;
  }
  status.textContent = t("customDatasetLoaded", {
    episodes: dataset.total_episodes,
    frames: dataset.total_frames,
    fps: dataset.fps,
  });
  status.className = "field-status ok";
}

async function loadCustomDataset() {
  const input = byId("replay-custom-root");
  const path = input.value.trim();
  const status = byId("replay-custom-status");
  if (!path) {
    status.textContent = t("customPathRequired");
    status.className = "field-status error";
    input.focus();
    return;
  }
  if (!path.startsWith("/")) {
    status.textContent = t("customPathAbsolute");
    status.className = "field-status error";
    input.focus();
    return;
  }

  try {
    const dataset = await api("/api/dataset?path=" + encodeURIComponent(path));
    state.selectedDataset = dataset;
    state.customDatasetActive = true;
    input.value = dataset.path;
    byId("replay-dataset").value = "";
    populateEpisodes(dataset, "");
    renderCustomDatasetStatus(dataset);
    if (state.status) renderStatus(state.status);
    toast(t("customDatasetReady"), "success");
  } catch (error) {
    status.textContent = error.message;
    status.className = "field-status error";
    toast(error.message, "error");
  }
}

function populateEpisodes(dataset, selectedEpisode) {
  const select = byId("replay-episode");
  select.innerHTML = "";
  if (!dataset) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = t("selectDatasetFirst");
    select.appendChild(option);
    select.disabled = true;
    byId("replay-dataset-meta").textContent = t("selectTimedDataset");
    byId("replay-dataset-meta").className = "field-status";
    updateReplayPreview();
    return;
  }
  for (let index = 0; index < dataset.total_episodes; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = t("episodeOption", { number: index + 1, index: index });
    select.appendChild(option);
  }
  select.disabled = false;
  if (selectedEpisode !== undefined && selectedEpisode !== "" &&
      Number(selectedEpisode) >= 0 && Number(selectedEpisode) < dataset.total_episodes) {
    select.value = String(selectedEpisode);
  }
  byId("replay-dataset-meta").textContent = t("datasetMeta", {
    path: dataset.path,
    frames: dataset.total_frames,
    fps: dataset.fps,
  });
  byId("replay-dataset-meta").className = "field-status ok mono";
  updateReplayPreview();
}

function onDatasetChanged() {
  const path = byId("replay-dataset").value;
  state.customDatasetActive = false;
  state.selectedDataset = state.datasets.find(function (item) { return item.path === path; }) || null;
  renderCustomDatasetStatus(null);
  populateEpisodes(state.selectedDataset, "");
}

function renderConfirmation() {
  const context = state.confirmContext;
  if (!context) return;
  byId("confirm-title").textContent = t(context.titleKey, context.titleValues);
  byId("confirm-message").textContent = context.messageKey
    ? t(context.messageKey, context.messageValues)
    : context.message;
}

function openConfirmation(titleKey, titleValues, messageKey, messageValues, message, action) {
  state.confirmAction = action;
  state.confirmContext = { titleKey, titleValues, messageKey, messageValues, message };
  renderConfirmation();
  byId("confirm-checkbox").checked = false;
  byId("confirm-submit").disabled = true;
  byId("confirm-dialog").showModal();
}

async function startVisualization() {
  const response = await post("/api/visualization/start", {});
  toast(t("visualizationStarted"), "success");
  byId("visual-url").href = response.url;
  await pollStatus();
}

async function stopVisualization() {
  await post("/api/visualization/stop", {});
  toast(t("visualizationStopped"), "success");
  await pollStatus();
}

function requestRecordStart() {
  const path = byId("record-root").value.trim();
  const episodeCount = selectedEpisodeCount();
  if (!path) {
    toast(t("fillRecordPath"), "error");
    byId("record-root").focus();
    return;
  }
  if (episodeCount === null) {
    toast(t("episodeCountInvalid"), "error");
    byId("record-episodes").focus();
    return;
  }
  openConfirmation(
    "startRecordConfirm",
    { episodes: episodeCount },
    "outputPath",
    { path: path, episodes: episodeCount },
    "",
    async function () {
      const response = await post("/api/record/start", {
        dataset_root: path,
        confirmed: true,
        options: recordOptions(),
      });
      toast(t("recordStarted"), "success");
      byId("record-output-preview").textContent = response.dataset_root;
      selectTab("logs");
      await pollStatus();
    },
  );
}

function requestReplayStart() {
  const dataset = state.selectedDataset;
  const episode = byId("replay-episode").value;
  if (!dataset) {
    toast(t("chooseReplayDataset"), "error");
    return;
  }
  if (episode === "") {
    toast(t("chooseReplayEpisode"), "error");
    return;
  }
  openConfirmation("startReplayConfirm", { episode: episode }, null, {}, dataset.path, async function () {
    await post("/api/replay/start", {
      dataset_root: dataset.path,
      episode: Number(episode),
      confirmed: true,
      options: replayOptions(),
    });
    toast(t("replayStarted"), "success");
    selectTab("logs");
    await pollStatus();
  });
}

async function stopOperation() {
  const response = await post("/api/operation/stop", {});
  toast(response.stopped ? t("taskStopped") : t("noRunningTask"), "success");
  await pollStatus();
}

function renderStatus(status) {
  state.status = status;
  const visual = status.visualization;
  const operation = status.operation;
  const armed = status.armed_actions_enabled || status.mock_commands;

  setDot("visual-dot", visual.running ? "ok" : "idle");
  byId("visual-status").textContent = visual.running ? t("running") : t("stopped");
  setBadge("visual-state-badge", visual.running ? t("running") : t("stopped"),
    visual.running ? "ok" : "idle");
  byId("viewport-state").textContent = visual.running ? t("meshcatPublishing") : t("waitingStart");
  byId("visual-start").disabled = visual.running;
  byId("visual-stop").disabled = !visual.running;

  const kindName = operation.kind === "record" ? t("record") :
    operation.kind === "replay" ? t("replay") : t("standby");
  const runningKind = operation.kind === "record" ? t("recording") :
    operation.kind === "replay" ? t("replaying") : t("running");
  setDot("operation-dot", operation.running ? "busy" : "idle");
  byId("operation-status").textContent = operation.running ? runningKind : t("standby");
  setBadge("record-state-badge", operation.running && operation.kind === "record" ? t("recording") : t("standby"),
    operation.running && operation.kind === "record" ? "busy" : "idle");
  setBadge("replay-state-badge", operation.running && operation.kind === "replay" ? t("replaying") : t("standby"),
    operation.running && operation.kind === "replay" ? "busy" : "idle");

  setDot("armed-dot", armed ? "ok" : "blocked");
  byId("armed-status").textContent = status.mock_commands ? t("mock") : armed ? t("unlocked") : t("locked");
  byId("record-start").disabled =
    operation.running || !armed || !visual.running || selectedEpisodeCount() === null;
  byId("replay-start").disabled =
    operation.running || !armed || !state.selectedDataset || byId("replay-episode").value === "";
  byId("record-stop").disabled = !(operation.running && operation.kind === "record");
  byId("replay-stop").disabled = !(operation.running && operation.kind === "replay");

  const banner = byId("mode-banner");
  if (status.mock_commands) {
    banner.textContent = t("mockBanner");
    banner.className = "mode-banner mock";
  } else if (!status.armed_actions_enabled) {
    banner.textContent = t("blockedBanner");
    banner.className = "mode-banner blocked";
  } else {
    banner.textContent = t("armedBanner");
    banner.className = "mode-banner armed";
  }

  byId("log-job-name").textContent = operation.running
    ? t("startedAt", { kind: kindName, time: formatTime(operation.started_at) })
    : operation.kind ? t("ended", { kind: kindName }) : t("noCurrentTask");
  byId("log-return-code").textContent =
    "returncode " + (operation.returncode === null ? "—" : String(operation.returncode));
  if (!state.logViewCleared) {
    byId("runtime-logs").textContent = operation.logs.length ? operation.logs.join("\n") : t("waitingTask");
    byId("runtime-logs").scrollTop = byId("runtime-logs").scrollHeight;
  }
  byId("command-output").textContent = status.last_command_preview || t("noCommand");
}

async function pollStatus() {
  try {
    renderStatus(await api("/api/status"));
  } catch (_error) {
    setDot("operation-dot", "blocked");
    byId("operation-status").textContent = t("connectionFailed");
  }
}

function wireEvents() {
  byId("language-toggle").addEventListener("click", function () {
    setLanguage(state.language === "zh" ? "en" : "zh");
  });
  document.querySelectorAll(".tab").forEach(function (button) {
    button.addEventListener("click", function () { selectTab(button.dataset.tab); });
  });
  byId("visual-start").addEventListener("click", function () {
    startVisualization().catch(function (error) { toast(error.message, "error"); });
  });
  byId("visual-stop").addEventListener("click", function () {
    stopVisualization().catch(function (error) { toast(error.message, "error"); });
  });
  byId("visual-open").addEventListener("click", function () {
    window.open(state.bootstrap.visualization_url, "_blank", "noopener");
  });
  byId("record-suggest").addEventListener("click", function () {
    const episodeCount = selectedEpisodeCount();
    if (episodeCount === null) {
      toast(t("episodeCountInvalid"), "error");
      byId("record-episodes").focus();
      return;
    }
    const base = state.bootstrap.outputs_root;
    const now = new Date();
    const pad = function (value) { return String(value).padStart(2, "0"); };
    const stamp = now.getFullYear() + pad(now.getMonth() + 1) + pad(now.getDate()) + "_" +
      pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds());
    byId("record-root").value = base + "/jz_robot_pin_timed_real_" + episodeCount + "eps_" + stamp;
    updateRecordPreview();
  });
  byId("record-root").addEventListener("input", updateRecordPreview);
  byId("record-episodes").addEventListener("input", function () {
    const episodeCount = selectedEpisodeCount();
    const rootInput = byId("record-root");
    if (episodeCount !== null) {
      rootInput.value = rootInput.value.replace(
        /\/jz_robot_pin_timed_real_\d+eps_(\d{8}_\d{6})$/,
        "/jz_robot_pin_timed_real_" + episodeCount + "eps_$1",
      );
    }
    updateRecordPreview();
  });
  ["record-episode-time", "record-fps", "record-crf"].forEach(function (id) {
    byId(id).addEventListener("input", updateRecordPreview);
  });
  byId("datasets-refresh").addEventListener("click", function () {
    loadDatasets(true).then(function () {
      toast(t("datasetsRefreshed"), "success");
    }).catch(function (error) { toast(error.message, "error"); });
  });
  byId("replay-custom-load").addEventListener("click", loadCustomDataset);
  byId("replay-custom-root").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      loadCustomDataset();
    }
  });
  byId("replay-dataset").addEventListener("change", onDatasetChanged);
  byId("replay-episode").addEventListener("change", updateReplayPreview);
  byId("record-start").addEventListener("click", requestRecordStart);
  byId("replay-start").addEventListener("click", requestReplayStart);
  byId("record-stop").addEventListener("click", function () {
    stopOperation().catch(function (error) { toast(error.message, "error"); });
  });
  byId("replay-stop").addEventListener("click", function () {
    stopOperation().catch(function (error) { toast(error.message, "error"); });
  });
  byId("confirm-checkbox").addEventListener("change", function () {
    byId("confirm-submit").disabled = !byId("confirm-checkbox").checked;
  });
  byId("confirm-submit").addEventListener("click", async function () {
    const action = state.confirmAction;
    byId("confirm-dialog").close();
    state.confirmAction = null;
    state.confirmContext = null;
    try {
      if (action) await action();
    } catch (error) {
      toast(error.message, "error");
    }
  });
  byId("confirm-cancel").addEventListener("click", function () {
    state.confirmAction = null;
    state.confirmContext = null;
  });
  byId("confirm-dialog").addEventListener("close", function () {
    if (!byId("confirm-dialog").open) {
      state.confirmAction = null;
      state.confirmContext = null;
    }
  });
  byId("logs-clear-view").addEventListener("click", function () {
    state.logViewCleared = true;
    byId("runtime-logs").textContent = t("logsCleared");
  });
}

async function main() {
  wireEvents();
  applyTranslations();
  state.bootstrap = await api("/api/bootstrap");
  if (state.bootstrap.control_token_required && !window.localStorage.getItem("jz_web_control_token")) {
    const token = window.prompt(t("controlTokenPrompt"));
    if (token) window.localStorage.setItem("jz_web_control_token", token);
  }
  applyDefaults();
  await loadDatasets(false);
  const requestedTab = new URLSearchParams(window.location.search).get("tab");
  if (["visualization", "record", "replay", "logs"].includes(requestedTab)) selectTab(requestedTab);
  await pollStatus();
  window.setInterval(function () {
    pollStatus();
    if (state.status && state.status.operation.running) state.logViewCleared = false;
  }, 1000);
}

main().catch(function (error) {
  toast(t("initializationFailed", { message: error.message }), "error");
  byId("mode-banner").textContent = t("serviceConnectionFailed");
  byId("mode-banner").className = "mode-banner blocked";
});
