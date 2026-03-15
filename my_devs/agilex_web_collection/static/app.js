(() => {
  "use strict";

  const DATASET_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
  const REPO_PREFIX_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
  const DEFAULT_REPO_PREFIX = "dummy";
  const PRECHECK_DEBOUNCE_MS = 420;
  const POLL_INTERVAL_MS = 2500;
  const RECENT_JOB_LIMIT = 6;
  const EMPTY_LOG_RUNNING = "等待日志输出...";
  const EMPTY_LOG_DONE = "该任务当前没有可读日志。";

  const state = {
    runtime: null,
    recentJobs: [],
    activeJob: null,
    lastJob: null,
    resume: false,
    lastManualReset: "0",
    activeTemplate: "",
    preflight: null,
    preflightKey: "",
    preflightError: "",
    preflightPending: false,
    preflightTimer: 0,
    preflightSeq: 0,
    formSeeded: false,
    refreshing: false,
    isStarting: false,
    isStopping: false,
    pollTimer: 0,
    pollInFlight: false,
    pollError: "",
    pollTick: 0,
    logJobId: "",
    logCursor: 0,
    logLines: [],
    bannerTimer: 0,
    terminalNoticeKey: "",
  };

  const refs = {};

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    cacheRefs();
    bindEvents();
    void initialize();
  });

  function cacheRefs() {
    const ids = [
      "feedbackBanner",
      "confirmOverlay",
      "confirmCommand",
      "confirmOutputDir",
      "confirmRepoId",
      "confirmMode",
      "confirmTaskText",
      "confirmAckArea",
      "confirmAckDevices",
      "confirmAckDataset",
      "confirmCloseBtn",
      "confirmCancelBtn",
      "confirmStartBtn",
      "refreshButton",
      "heroEnv",
      "heroService",
      "preflightButton",
      "runtimeScriptPath",
      "runtimeCondaEnv",
      "runtimeOutputRoot",
      "runtimeRepoId",
      "runtimeTargetDir",
      "preflightDatasetStatus",
      "preflightExpectedTime",
      "preflightBadge",
      "preflightMessage",
      "preflightWarnings",
      "formLockState",
      "recordForm",
      "recordFieldset",
      "repoPrefix",
      "repoPrefixHint",
      "datasetName",
      "datasetHint",
      "episodeTime",
      "numEpisodes",
      "fps",
      "resetTime",
      "resetHint",
      "resumeToggle",
      "resumeValue",
      "resumeHint",
      "singleTaskText",
      "formIssues",
      "commandPreview",
      "repoPreview",
      "outputPreview",
      "modePreview",
      "taskPreview",
      "durationPreview",
      "startButton",
      "resetButton",
      "actionHint",
      "runtimeBadge",
      "stopButton",
      "runtimeEmpty",
      "runtimePanel",
      "runtimeJobId",
      "runtimePhase",
      "runtimeStartedAt",
      "runtimeProgress",
      "runtimeOutputDir",
      "runtimeLogPath",
      "runtimeTaskText",
      "runtimeLogs",
      "resultBadge",
      "resultTitle",
      "resultSummary",
      "resultRepoId",
      "resultOutputDir",
      "resultTaskText",
      "nextSteps",
      "recentJobs",
      "historyHint",
    ];

    for (const id of ids) {
      refs[id] = document.getElementById(id);
    }

    refs.templateButtons = Array.from(document.querySelectorAll(".template-card"));
  }

  function bindEvents() {
    refs.recordForm.addEventListener("submit", (event) => {
      event.preventDefault();
    });

    refs.repoPrefix.addEventListener("input", handleFormInput);
    refs.datasetName.addEventListener("input", handleFormInput);
    refs.episodeTime.addEventListener("input", handleFormInput);
    refs.numEpisodes.addEventListener("input", handleNumEpisodesInput);
    refs.fps.addEventListener("input", handleFormInput);
    refs.resetTime.addEventListener("input", handleResetTimeInput);
    refs.singleTaskText.addEventListener("input", handleFormInput);
    refs.resumeToggle.addEventListener("click", handleResumeToggle);
    refs.preflightButton.addEventListener("click", () => {
      void requestCurrentPreflight({ immediate: true, announceErrors: true });
    });
    refs.refreshButton.addEventListener("click", () => {
      void refreshPage({ announce: true });
    });
    refs.resetButton.addEventListener("click", () => {
      applyDefaults({ clearDatasetName: true });
    });
    refs.startButton.addEventListener("click", () => {
      void handleStartIntent();
    });
    refs.stopButton.addEventListener("click", () => {
      void handleStop();
    });
    refs.confirmCloseBtn.addEventListener("click", closeConfirmOverlay);
    refs.confirmCancelBtn.addEventListener("click", closeConfirmOverlay);
    refs.confirmStartBtn.addEventListener("click", () => {
      void handleConfirmedStart();
    });
    refs.confirmOverlay.addEventListener("click", (event) => {
      if (event.target === refs.confirmOverlay) {
        closeConfirmOverlay();
      }
    });

    for (const id of ["confirmAckArea", "confirmAckDevices", "confirmAckDataset"]) {
      refs[id].addEventListener("change", syncConfirmButtonState);
    }

    for (const button of refs.templateButtons) {
      button.addEventListener("click", () => {
        applyTemplate(button.dataset.template || "");
      });
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !refs.confirmOverlay.classList.contains("hidden")) {
        closeConfirmOverlay();
      }
    });
  }

  async function initialize() {
    try {
      await refreshPage({ announce: false });
      startPolling();
    } catch (error) {
      refs.heroService.textContent = "初始化失败";
      refs.runtimeLogs.textContent = EMPTY_LOG_DONE;
      showBanner(`初始化失败：${error.message}`, "danger", { persist: true });
    }
  }

  async function refreshPage({ announce }) {
    if (state.refreshing) {
      return;
    }

    state.refreshing = true;
    render();

    try {
      const [runtime, activeJob, recentJobs] = await Promise.all([
        api("/api/runtime"),
        api("/api/jobs/active"),
        api(`/api/jobs?limit=${RECENT_JOB_LIMIT}`),
      ]);

      state.runtime = runtime;
      state.recentJobs = Array.isArray(recentJobs) ? recentJobs : [];
      state.pollError = "";

      if (!state.formSeeded) {
        seedFormFromRuntime();
      }

      if (activeJob && activeJob.state?.job_id) {
        const detail = await fetchJobDetail(activeJob.state.job_id);
        setActiveJob(detail);
        setLastJob(detail);
        await loadLogsForDisplayedJob({ resetIfChanged: true });
      } else {
        state.activeJob = null;
        const latestJobId = state.recentJobs[0]?.job_id || state.lastJob?.state?.job_id || "";

        if (latestJobId) {
          const detail =
            state.lastJob && state.lastJob.state?.job_id === latestJobId ? state.lastJob : await fetchJobDetail(latestJobId);
          setLastJob(detail);
          await loadLogsForDisplayedJob({ resetIfChanged: true });
        } else {
          state.lastJob = null;
          clearLogs();
        }
      }

      if (announce) {
        showBanner("页面状态已刷新，并按后端返回恢复当前视图。", "info");
      }
    } finally {
      state.refreshing = false;
      render();
      void requestCurrentPreflight({ immediate: true, announceErrors: false });
    }
  }

  function seedFormFromRuntime() {
    if (!state.runtime) {
      return;
    }

    const defaults = state.runtime.defaults || {};
    refs.repoPrefix.value = normalizeRepoPrefix(defaults.repo_prefix);
    refs.datasetName.placeholder = defaults.dataset_name_placeholder || refs.datasetName.placeholder;
    refs.episodeTime.value = formatInputNumber(defaults.episode_time_s ?? 8);
    refs.numEpisodes.value = String(defaults.num_episodes ?? 1);
    refs.fps.value = String(defaults.fps ?? 10);
    refs.resetTime.value = formatInputNumber(defaults.reset_time_s ?? 0);
    refs.singleTaskText.value = normalizeTaskText(defaults.single_task_text);
    state.lastManualReset = formatInputNumber(defaults.reset_time_s ?? 0);
    setResume(Boolean(defaults.resume));
    state.formSeeded = true;
    syncResetLock();
    render();
  }

  function applyDefaults({ clearDatasetName }) {
    if (!state.runtime) {
      return;
    }

    const defaults = state.runtime.defaults || {};
    refs.repoPrefix.value = normalizeRepoPrefix(defaults.repo_prefix);
    refs.datasetName.value = clearDatasetName ? "" : refs.datasetName.value.trim();
    refs.episodeTime.value = formatInputNumber(defaults.episode_time_s ?? 8);
    refs.numEpisodes.value = String(defaults.num_episodes ?? 1);
    refs.fps.value = String(defaults.fps ?? 10);
    refs.resetTime.value = formatInputNumber(defaults.reset_time_s ?? 0);
    refs.singleTaskText.value = normalizeTaskText(defaults.single_task_text);
    state.lastManualReset = formatInputNumber(defaults.reset_time_s ?? 0);
    setResume(Boolean(defaults.resume));
    state.activeTemplate = "";
    syncResetLock();
    clearCurrentPreflight();
    render();
    void requestCurrentPreflight({ immediate: true, announceErrors: false });
  }

  function applyTemplate(templateName) {
    if (!state.runtime) {
      return;
    }

    const defaults = state.runtime.defaults || {};
    const fallbackName = buildSuggestedDatasetName(templateName || "agilex_record");
    const recentName = state.lastJob?.request?.dataset_name || state.recentJobs[0]?.dataset_name || fallbackName;
    const recentPrefix = getRepoPrefixFromJob(state.lastJob) || getRepoPrefixFromSummary(state.recentJobs[0]);
    const defaultRepoPrefix = normalizeRepoPrefix(defaults.repo_prefix);
    const defaultTaskText = normalizeTaskText(defaults.single_task_text);
    const currentTaskText = refs.singleTaskText.value.trim();

    if (templateName === "first_trial") {
      refs.repoPrefix.value = refs.repoPrefix.value.trim() || defaultRepoPrefix;
      refs.datasetName.value = refs.datasetName.value.trim() || fallbackName;
      refs.episodeTime.value = "8";
      refs.numEpisodes.value = "1";
      refs.fps.value = "30";
      refs.resetTime.value = "0";
      state.lastManualReset = "0";
      setResume(false);
      refs.singleTaskText.value = currentTaskText || defaultTaskText;
    } else if (templateName === "script_defaults") {
      refs.repoPrefix.value = defaultRepoPrefix;
      refs.datasetName.value = "";
      refs.episodeTime.value = formatInputNumber(defaults.episode_time_s ?? 8);
      refs.numEpisodes.value = String(defaults.num_episodes ?? 1);
      refs.fps.value = String(defaults.fps ?? 10);
      refs.resetTime.value = formatInputNumber(defaults.reset_time_s ?? 0);
      state.lastManualReset = formatInputNumber(defaults.reset_time_s ?? 0);
      setResume(Boolean(defaults.resume));
      refs.singleTaskText.value = defaultTaskText;
    } else if (templateName === "continue_append") {
      refs.repoPrefix.value = refs.repoPrefix.value.trim() || recentPrefix || defaultRepoPrefix;
      refs.datasetName.value = refs.datasetName.value.trim() || recentName;
      refs.episodeTime.value = "8";
      refs.numEpisodes.value = "1";
      refs.fps.value = "30";
      refs.resetTime.value = "0";
      state.lastManualReset = "0";
      setResume(true);
      refs.singleTaskText.value = currentTaskText || defaultTaskText;
    } else {
      return;
    }

    state.activeTemplate = templateName;
    syncResetLock();
    clearCurrentPreflight();
    render();
    void requestCurrentPreflight({ immediate: true, announceErrors: false });
  }

  function handleFormInput() {
    clearCurrentPreflightIfStale();
    render();
    void requestCurrentPreflight({ immediate: false, announceErrors: false });
  }

  function handleNumEpisodesInput() {
    syncResetLock();
    handleFormInput();
  }

  function handleResetTimeInput() {
    if (!refs.resetTime.disabled && refs.resetTime.value.trim()) {
      state.lastManualReset = refs.resetTime.value.trim();
    }
    handleFormInput();
  }

  function handleResumeToggle() {
    setResume(!state.resume);
    clearCurrentPreflightIfStale();
    render();
    void requestCurrentPreflight({ immediate: false, announceErrors: false });
  }

  function setResume(value) {
    state.resume = Boolean(value);
    refs.resumeToggle.setAttribute("aria-checked", String(state.resume));
    refs.resumeValue.textContent = String(state.resume);
  }

  function syncResetLock() {
    const numEpisodes = Number.parseInt(refs.numEpisodes.value, 10);
    const locked = Number.isInteger(numEpisodes) && numEpisodes === 1;

    if (locked) {
      const currentValue = refs.resetTime.value.trim();
      if (currentValue && currentValue !== "0") {
        state.lastManualReset = currentValue;
      }
      refs.resetTime.value = "0";
      refs.resetTime.disabled = true;
      refs.resetHint.textContent = "单段录制时固定为 0s，本页已锁定。";
      return;
    }

    refs.resetTime.disabled = false;
    refs.resetHint.textContent = "多段录制时用于段间等待。";

    if (!refs.resetTime.value.trim()) {
      refs.resetTime.value = state.lastManualReset || "0";
    }
  }

  async function handleStartIntent() {
    const current = getCurrentFormState();
    if (current.issues.length > 0) {
      render();
      showBanner(current.issues[0], "warning");
      return;
    }

    const preflight = await ensureFreshPreflight(current);
    if (!preflight) {
      return;
    }

    if (!preflight.ok) {
      render();
      showBanner(preflight.conflicts[0] || "预检查未通过。", "warning");
      return;
    }

    refs.confirmCommand.textContent = preflight.command_text || buildLocalCommand(current.draft);
    refs.confirmRepoId.textContent = preflight.repo_id || getRepoIdFromPayload(current.payload) || "等待 repo_prefix/dataset_name";
    refs.confirmOutputDir.textContent = preflight.dataset_dir || getDatasetDirFromPayload(current.payload);
    refs.confirmMode.textContent = buildModeDescription(current.payload, preflight);
    refs.confirmTaskText.textContent = current.payload.single_task_text;
    openConfirmOverlay();
  }

  async function handleConfirmedStart() {
    const current = getCurrentFormState();
    if (current.issues.length > 0 || !current.payload) {
      closeConfirmOverlay();
      render();
      return;
    }

    state.isStarting = true;
    render();

    try {
      const detail = await api("/api/jobs", {
        method: "POST",
        body: JSON.stringify(current.payload),
      });

      state.terminalNoticeKey = "";
      closeConfirmOverlay();
      setActiveJob(detail);
      setLastJob(detail);
      clearLogs();
      await loadLogsForDisplayedJob({ resetIfChanged: true });
      await refreshRecentJobs({ hydrateLatestWhenIdle: false });
      render();
      showBanner(`任务已创建：${detail.state.job_id}`, "success");
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        await refreshPage({ announce: false });
      }
      showBanner(`开始失败：${error.message}`, "danger", { persist: true });
    } finally {
      state.isStarting = false;
      render();
    }
  }

  async function handleStop() {
    const job = state.activeJob;
    if (!job?.state?.job_id) {
      return;
    }

    state.isStopping = true;
    render();

    try {
      const response = await api(`/api/jobs/${encodeURIComponent(job.state.job_id)}/stop`, {
        method: "POST",
      });

      const detail = response.job || (await fetchJobDetail(job.state.job_id));
      if (detail.state.active) {
        setActiveJob(detail);
      } else {
        state.activeJob = null;
      }
      setLastJob(detail);
      await loadLogsForDisplayedJob({ resetIfChanged: false });
      await refreshRecentJobs({ hydrateLatestWhenIdle: false });
      render();
      showBanner(response.message || "已发送停止请求。", "warning", { persist: true });
    } catch (error) {
      showBanner(`停止失败：${error.message}`, "danger", { persist: true });
    } finally {
      state.isStopping = false;
      render();
    }
  }

  async function requestCurrentPreflight({ immediate, announceErrors }) {
    const current = getCurrentFormState();
    if (!current.payload) {
      state.preflightPending = false;
      if (state.preflightKey && current.key !== state.preflightKey) {
        clearCurrentPreflight();
      }
      render();
      return null;
    }

    return schedulePreflight(current, { immediate, announceErrors });
  }

  async function ensureFreshPreflight(current) {
    const freshPreflight = getFreshPreflight(current.key);
    if (freshPreflight && !state.preflightPending) {
      return freshPreflight;
    }

    return schedulePreflight(current, { immediate: true, announceErrors: true });
  }

  function schedulePreflight(current, { immediate, announceErrors }) {
    window.clearTimeout(state.preflightTimer);

    return new Promise((resolve) => {
      const delay = immediate ? 0 : PRECHECK_DEBOUNCE_MS;
      state.preflightPending = true;
      render();
      state.preflightTimer = window.setTimeout(async () => {
        const result = await runPreflight(current, { announceErrors });
        resolve(result);
      }, delay);
    });
  }

  async function runPreflight(current, { announceErrors }) {
    const seq = ++state.preflightSeq;
    state.preflightPending = true;
    state.preflightError = "";
    render();

    try {
      const preflight = await api("/api/preflight", {
        method: "POST",
        body: JSON.stringify(current.payload),
      });

      if (seq !== state.preflightSeq) {
        return preflight;
      }

      state.preflight = preflight;
      state.preflightKey = current.key;
      return preflight;
    } catch (error) {
      if (seq !== state.preflightSeq) {
        return null;
      }

      state.preflight = null;
      state.preflightKey = "";
      state.preflightError = error.message;

      if (announceErrors) {
        showBanner(`预检查失败：${error.message}`, "danger", { persist: true });
      }
      return null;
    } finally {
      if (seq === state.preflightSeq) {
        state.preflightPending = false;
        render();
      }
    }
  }

  function clearCurrentPreflight() {
    window.clearTimeout(state.preflightTimer);
    state.preflight = null;
    state.preflightKey = "";
    state.preflightError = "";
    state.preflightPending = false;
  }

  function clearCurrentPreflightIfStale() {
    const current = getCurrentFormState();
    if (current.key !== state.preflightKey) {
      state.preflight = null;
      state.preflightKey = "";
      state.preflightError = "";
    }
  }

  function getFreshPreflight(currentKey) {
    if (!currentKey || currentKey !== state.preflightKey) {
      return null;
    }
    return state.preflight;
  }

  function getCurrentFormState() {
    const draft = {
      repo_prefix: refs.repoPrefix.value.trim(),
      dataset_name: refs.datasetName.value.trim(),
      episode_time_s: refs.episodeTime.value.trim(),
      num_episodes: refs.numEpisodes.value.trim(),
      fps: refs.fps.value.trim(),
      reset_time_s: refs.resetTime.value.trim(),
      resume: state.resume,
      single_task_text: refs.singleTaskText.value.trim(),
    };

    const issues = [];
    const effectiveRepoPrefix = normalizeRepoPrefix(draft.repo_prefix);

    if (draft.repo_prefix && !REPO_PREFIX_RE.test(draft.repo_prefix)) {
      issues.push("repo_prefix 只能使用字母、数字、下划线和中划线，且必须以字母或数字开头，不能包含 /。");
    }

    if (!draft.dataset_name) {
      issues.push("dataset_name 不能为空。");
    } else if (!DATASET_NAME_RE.test(draft.dataset_name)) {
      issues.push("dataset_name 只能使用字母、数字、下划线和中划线，且必须以字母或数字开头，不能包含 /。");
    }

    if (!draft.single_task_text) {
      issues.push("single_task_text 不能为空。");
    }

    const episodeTime = parseFiniteNumber(draft.episode_time_s);
    if (draft.episode_time_s === "") {
      issues.push("episode_time_s 不能为空。");
    } else if (!Number.isFinite(episodeTime) || episodeTime <= 0 || episodeTime > 3600) {
      issues.push("episode_time_s 必须在 0 到 3600 秒之间。");
    }

    const numEpisodes = parseInteger(draft.num_episodes);
    if (draft.num_episodes === "") {
      issues.push("num_episodes 不能为空。");
    } else if (!Number.isInteger(numEpisodes) || numEpisodes < 1 || numEpisodes > 1000) {
      issues.push("num_episodes 必须是 1 到 1000 的整数。");
    }

    const fps = parseInteger(draft.fps);
    if (draft.fps === "") {
      issues.push("fps 不能为空。");
    } else if (!Number.isInteger(fps) || fps < 1 || fps > 120) {
      issues.push("fps 必须是 1 到 120 的整数。");
    }

    let resetTime = 0;
    if (numEpisodes === 1) {
      resetTime = 0;
    } else {
      resetTime = parseFiniteNumber(draft.reset_time_s);
      if (draft.reset_time_s === "") {
        issues.push("多段采集时 reset_time_s 不能为空。");
      } else if (!Number.isFinite(resetTime) || resetTime < 0 || resetTime > 600) {
        issues.push("reset_time_s 必须在 0 到 600 秒之间。");
      }
    }

    const payload =
      issues.length === 0
        ? {
            repo_prefix: effectiveRepoPrefix,
            dataset_name: draft.dataset_name,
            episode_time_s: episodeTime,
            num_episodes: numEpisodes,
            fps: fps,
            reset_time_s: resetTime,
            resume: draft.resume,
            single_task_text: draft.single_task_text,
          }
        : null;

    return {
      draft,
      issues,
      payload,
      key: payload ? JSON.stringify(payload) : "",
    };
  }

  function openConfirmOverlay() {
    refs.confirmOverlay.classList.remove("hidden");
    refs.confirmOverlay.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    resetConfirmChecks();
  }

  function closeConfirmOverlay() {
    refs.confirmOverlay.classList.add("hidden");
    refs.confirmOverlay.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    resetConfirmChecks();
  }

  function resetConfirmChecks() {
    refs.confirmAckArea.checked = false;
    refs.confirmAckDevices.checked = false;
    refs.confirmAckDataset.checked = false;
    syncConfirmButtonState();
  }

  function syncConfirmButtonState() {
    const enabled = refs.confirmAckArea.checked && refs.confirmAckDevices.checked && refs.confirmAckDataset.checked;
    refs.confirmStartBtn.disabled = !enabled || state.isStarting;
  }

  function startPolling() {
    if (state.pollTimer) {
      return;
    }

    state.pollTimer = window.setInterval(() => {
      void pollJobs();
    }, POLL_INTERVAL_MS);
  }

  async function pollJobs() {
    if (state.pollInFlight || state.refreshing) {
      return;
    }

    state.pollInFlight = true;

    try {
      const activeJob = await api("/api/jobs/active");
      state.pollError = "";

      if (activeJob?.state?.job_id) {
        const detail = await fetchJobDetail(activeJob.state.job_id);
        const previousActiveId = state.activeJob?.state?.job_id || "";
        setActiveJob(detail);
        setLastJob(detail);
        await loadLogsForDisplayedJob({ resetIfChanged: previousActiveId !== detail.state.job_id });
      } else if (state.activeJob?.state?.job_id) {
        const detail = await fetchJobDetail(state.activeJob.state.job_id);
        state.activeJob = detail.state.active ? detail : null;
        setLastJob(detail);
        await loadLogsForDisplayedJob({ resetIfChanged: false });
        maybeNotifyTerminal(detail);
      } else if (!state.lastJob && state.recentJobs[0]?.job_id) {
        const detail = await fetchJobDetail(state.recentJobs[0].job_id);
        setLastJob(detail);
        await loadLogsForDisplayedJob({ resetIfChanged: true });
      }

      state.pollTick += 1;
      if (state.pollTick % 3 === 0) {
        await refreshRecentJobs({ hydrateLatestWhenIdle: false });
      }
    } catch (error) {
      state.pollError = error.message;
    } finally {
      state.pollInFlight = false;
      render();
    }
  }

  async function refreshRecentJobs({ hydrateLatestWhenIdle }) {
    const jobs = await api(`/api/jobs?limit=${RECENT_JOB_LIMIT}`);
    state.recentJobs = Array.isArray(jobs) ? jobs : [];

    if (!state.activeJob && hydrateLatestWhenIdle) {
      const latestJobId = state.recentJobs[0]?.job_id || "";
      if (latestJobId) {
        const detail = await fetchJobDetail(latestJobId);
        setLastJob(detail);
        await loadLogsForDisplayedJob({ resetIfChanged: true });
      } else {
        state.lastJob = null;
        clearLogs();
      }
    }
  }

  async function fetchJobDetail(jobId) {
    return api(`/api/jobs/${encodeURIComponent(jobId)}`);
  }

  function setActiveJob(detail) {
    state.activeJob = detail;
  }

  function setLastJob(detail) {
    state.lastJob = detail;
  }

  function getDisplayedJob() {
    return state.activeJob || state.lastJob || null;
  }

  async function loadLogsForDisplayedJob({ resetIfChanged }) {
    const job = getDisplayedJob();
    if (!job?.state?.job_id) {
      clearLogs();
      return;
    }

    const jobId = job.state.job_id;
    if (resetIfChanged && state.logJobId !== jobId) {
      clearLogs();
    }

    if (state.logJobId !== jobId) {
      state.logJobId = jobId;
      state.logCursor = 0;
      state.logLines = [];
    }

    const limitBytes = state.logCursor === 0 ? 32768 : 16384;
    const logs = await api(
      `/api/jobs/${encodeURIComponent(jobId)}/logs?cursor=${state.logCursor}&limit_bytes=${limitBytes}`
    );

    if (state.logJobId !== jobId) {
      return;
    }

    if (state.logCursor === 0 && logs.truncated) {
      state.logLines.push("[日志过长，仅展示最近一段输出]");
    }

    if (Array.isArray(logs.lines) && logs.lines.length > 0) {
      state.logLines = [...state.logLines, ...logs.lines].slice(-500);
    } else if (state.logLines.length === 0) {
      state.logLines = [job.state.active ? EMPTY_LOG_RUNNING : EMPTY_LOG_DONE];
    }

    state.logCursor = Number.isFinite(logs.next_cursor) ? logs.next_cursor : state.logCursor;
  }

  function clearLogs() {
    state.logJobId = "";
    state.logCursor = 0;
    state.logLines = [];
  }

  function maybeNotifyTerminal(detail) {
    if (detail.state.active) {
      return;
    }

    const key = `${detail.state.job_id}:${detail.state.status}:${detail.state.finished_at || ""}`;
    if (key === state.terminalNoticeKey) {
      return;
    }

    state.terminalNoticeKey = key;
    const repoId = getRepoIdFromJob(detail) || detail.request?.dataset_name || detail.state.job_id;
    if (detail.state.status === "succeeded") {
      showBanner(`任务完成：${repoId}`, "success");
    } else if (detail.state.status === "stopped") {
      showBanner(`任务已停止：${repoId}`, "warning", { persist: true });
    } else if (detail.state.status === "failed") {
      showBanner(`任务失败：${detail.state.status_message}`, "danger", { persist: true });
    }
  }

  function render() {
    const current = getCurrentFormState();
    const currentPreflight = getFreshPreflight(current.key);
    const displayedJob = getDisplayedJob();

    syncTemplateState();
    syncConfirmButtonState();
    renderHero(currentPreflight);
    renderPreflight(current, currentPreflight);
    renderForm(current, currentPreflight);
    renderPreview(current, currentPreflight);
    renderRuntime(displayedJob);
    renderResult(displayedJob);
    renderRecentJobs();
  }

  function renderHero(currentPreflight) {
    if (!state.runtime) {
      refs.heroEnv.textContent = "读取中";
      refs.heroService.textContent = "初始化中";
      return;
    }

    const runtimeEnv = state.runtime.conda_env || "lerobot_flex";
    const currentEnv = state.runtime.current_conda_env;
    refs.heroEnv.textContent =
      currentEnv && currentEnv !== runtimeEnv ? `${runtimeEnv} (当前 ${currentEnv})` : currentEnv || runtimeEnv;

    if (state.refreshing) {
      refs.heroService.textContent = "刷新中";
    } else if (state.activeJob?.state?.status) {
      refs.heroService.textContent = formatStatus(state.activeJob.state.status);
    } else if (state.preflightPending) {
      refs.heroService.textContent = "预检查中";
    } else if (state.pollError) {
      refs.heroService.textContent = "轮询异常";
    } else if (state.preflightError) {
      refs.heroService.textContent = "预检查异常";
    } else if (currentEnv && currentEnv !== runtimeEnv) {
      refs.heroService.textContent = "环境需确认";
    } else if (currentPreflight?.ok) {
      refs.heroService.textContent = "已就绪";
    } else {
      refs.heroService.textContent = "待命";
    }
  }

  function renderPreflight(current, currentPreflight) {
    refs.runtimeScriptPath.textContent = state.runtime?.script_path || "读取中";
    refs.runtimeOutputRoot.textContent = state.runtime?.output_root || "读取中";
    refs.runtimeCondaEnv.textContent = state.runtime?.conda_env || "lerobot_flex";
    refs.runtimeRepoId.textContent =
      (current.payload && getRepoIdFromPayload(current.payload)) || buildRepoIdFromDraft(current.draft) || "等待 repo_prefix/dataset_name";
    refs.runtimeTargetDir.textContent = hasDatasetDraft(current.draft) ? getDatasetDirFromDraft(current.draft) : "等待 dataset_name";
    refs.repoPrefixHint.textContent = "默认值为 dummy。只填单段前缀，不要包含 /。";
    refs.datasetHint.textContent = "仅允许叶子名字，不能包含 /；最终会和 repo_prefix 组合成 repo_prefix/dataset_name。";

    refs.resumeHint.textContent = buildResumeHint(current, currentPreflight);

    if (currentPreflight) {
      refs.preflightExpectedTime.textContent = formatDuration(currentPreflight.estimated_duration_s);
      refs.preflightDatasetStatus.textContent = currentPreflight.dataset_exists ? "目录已存在" : "目录将创建";
    } else if (current.payload) {
      refs.preflightExpectedTime.textContent = formatDuration(estimateDuration(current.payload));
      refs.preflightDatasetStatus.textContent = state.preflightPending ? "检查中" : "待检查";
    } else {
      refs.preflightExpectedTime.textContent = "-";
      refs.preflightDatasetStatus.textContent = current.draft.dataset_name ? "参数待修正" : "等待数据集名称";
    }

    if (!state.runtime) {
      setTone(refs.preflightBadge, "neutral", "等待运行时");
      refs.preflightMessage.textContent = "正在读取后端运行环境。";
    } else if (current.issues.length > 0) {
      setTone(refs.preflightBadge, "warning", "参数待修正");
      refs.preflightMessage.textContent = current.issues[0];
    } else if (state.preflightPending) {
      setTone(refs.preflightBadge, "info", "预检查中");
      refs.preflightMessage.textContent = "正在将 7 个脚本参数和 repo_prefix 发给后端做预检查。";
    } else if (state.preflightError) {
      setTone(refs.preflightBadge, "danger", "请求失败");
      refs.preflightMessage.textContent = `预检查请求失败：${state.preflightError}`;
    } else if (currentPreflight?.ok) {
      setTone(refs.preflightBadge, "success", "可以开始");
      refs.preflightMessage.textContent = "后端预检查通过，命令和输出目录已对齐。";
    } else if (currentPreflight && !currentPreflight.ok) {
      setTone(refs.preflightBadge, "danger", "不可开始");
      refs.preflightMessage.textContent = currentPreflight.conflicts[0] || "后端拒绝了当前参数组合。";
    } else {
      setTone(refs.preflightBadge, "neutral", "等待输入");
      refs.preflightMessage.textContent = "输入合法参数后会自动执行预检查。";
    }

    const warningItems = [];
    if (current.issues.length > 0) {
      warningItems.push(...current.issues);
    } else if (state.preflightError) {
      warningItems.push(`预检查请求失败：${state.preflightError}`);
    } else if (currentPreflight) {
      for (const message of currentPreflight.conflicts || []) {
        warningItems.push(`冲突：${message}`);
      }
      for (const message of currentPreflight.warnings || []) {
        warningItems.push(`提醒：${message}`);
      }
      for (const message of currentPreflight.notes || []) {
        warningItems.push(`说明：${message}`);
      }
    }

    renderList(refs.preflightWarnings, warningItems, "当前没有额外警告。");
  }

  function renderForm(current, currentPreflight) {
    renderList(refs.formIssues, current.issues, "当前参数没有本地校验问题。");

    if (state.isStarting) {
      setTone(refs.formLockState, "info", "提交中");
    } else if (state.activeJob?.state?.active) {
      setTone(refs.formLockState, "warning", "单任务模式");
    } else if (state.preflightPending) {
      setTone(refs.formLockState, "info", "预检查中");
    } else if (currentPreflight?.ok) {
      setTone(refs.formLockState, "success", "表单可开始");
    } else {
      setTone(refs.formLockState, "neutral", "表单可编辑");
    }

    const formLocked = state.isStarting || Boolean(state.activeJob?.state?.active);
    refs.recordFieldset.disabled = formLocked;
    refs.preflightButton.disabled = !state.runtime || formLocked || !current.payload;
    refs.refreshButton.disabled = state.refreshing;
  }

  function renderPreview(current, currentPreflight) {
    refs.repoPreview.textContent =
      (current.payload && getRepoIdFromPayload(current.payload)) || buildRepoIdFromDraft(current.draft) || "等待 repo_prefix/dataset_name";
    refs.outputPreview.textContent = hasDatasetDraft(current.draft) ? getDatasetDirFromDraft(current.draft) : "等待 dataset_name";

    refs.commandPreview.textContent = currentPreflight?.command_text || buildLocalCommand(current.draft);
    refs.taskPreview.textContent = current.payload?.single_task_text || "等待填写任务语义";
    refs.modePreview.textContent = current.payload
      ? buildModeDescription(current.payload, currentPreflight)
      : "填写 7 个脚本参数后生成精确预览；repo_prefix 留空时默认使用 dummy。";
    refs.durationPreview.textContent = current.payload
      ? formatDuration(currentPreflight?.estimated_duration_s ?? estimateDuration(current.payload))
      : "-";

    const canStart =
      Boolean(current.payload) &&
      Boolean(currentPreflight) &&
      currentPreflight.ok &&
      !state.preflightPending &&
      !state.isStarting &&
      !state.activeJob?.state?.active &&
      !state.refreshing;

    refs.startButton.disabled = !canStart;

    if (!state.runtime) {
      refs.actionHint.textContent = "正在读取运行时信息。";
    } else if (state.isStarting) {
      refs.actionHint.textContent = "正在向后端提交开始请求。";
    } else if (state.activeJob?.state?.active) {
      refs.actionHint.textContent = "当前已有活跃任务，表单已锁定。";
    } else if (current.issues.length > 0) {
      refs.actionHint.textContent = current.issues[0];
    } else if (state.preflightPending) {
      refs.actionHint.textContent = "正在执行预检查...";
    } else if (state.preflightError) {
      refs.actionHint.textContent = `预检查失败：${state.preflightError}`;
    } else if (currentPreflight && !currentPreflight.ok) {
      refs.actionHint.textContent = currentPreflight.conflicts[0] || "预检查未通过。";
    } else if (currentPreflight?.ok) {
      refs.actionHint.textContent = "预检查通过，可以开始采集。";
    } else {
      refs.actionHint.textContent = "开始按钮会在预检查通过后点亮。";
    }
  }

  function renderRuntime(job) {
    if (!job) {
      setTone(refs.runtimeBadge, "neutral", "空闲");
      refs.stopButton.disabled = true;
      refs.runtimeEmpty.classList.remove("hidden");
      refs.runtimePanel.classList.add("hidden");
      refs.runtimeLogs.textContent = EMPTY_LOG_DONE;
      return;
    }

    setTone(refs.runtimeBadge, toneForStatus(job.state.status), formatStatus(job.state.status));
    refs.stopButton.disabled = !job.state.active || !job.state.can_stop || state.isStopping;
    refs.runtimeEmpty.classList.add("hidden");
    refs.runtimePanel.classList.remove("hidden");

    refs.runtimeJobId.textContent = job.state.job_id || "-";
    refs.runtimePhase.textContent = `${formatPhase(job.state.phase)} · ${job.state.status_message || "等待状态更新"}`;
    refs.runtimeStartedAt.textContent = formatDateTime(job.state.started_at || job.state.created_at);
    refs.runtimeProgress.textContent = buildProgressText(job);
    refs.runtimeRepoId.textContent = getRepoIdFromJob(job) || refs.runtimeRepoId.textContent;
    refs.runtimeOutputDir.textContent = job.request?.dataset_dir || "-";
    refs.runtimeLogPath.textContent = job.request?.artifacts?.log_path || "-";
    refs.runtimeTaskText.textContent = getTaskTextFromJob(job);
    refs.runtimeLogs.textContent = state.logLines.length > 0 ? state.logLines.join("\n") : EMPTY_LOG_DONE;
    refs.runtimeLogs.scrollTop = refs.runtimeLogs.scrollHeight;
  }

  function renderResult(job) {
    if (!job) {
      setTone(refs.resultBadge, "neutral", "等待任务");
      refs.resultTitle.textContent = "还没有采集结果";
      refs.resultSummary.textContent = "完成一次任务后，这里会显示成功、失败或停止摘要。";
      refs.resultRepoId.textContent = "-";
      refs.resultOutputDir.textContent = "-";
      refs.resultTaskText.textContent = "-";
      renderList(refs.nextSteps, [], "先完成预检查，再开始采集。");
      return;
    }

    setTone(refs.resultBadge, toneForStatus(job.state.status), formatStatus(job.state.status));
    const repoId = getRepoIdFromJob(job) || job.request?.dataset_name || job.state.job_id;
    refs.resultRepoId.textContent = repoId;
    refs.resultOutputDir.textContent = job.request?.dataset_dir || "-";
    refs.resultTaskText.textContent = getTaskTextFromJob(job);

    if (job.state.active) {
      refs.resultTitle.textContent = "任务正在进行中";
      refs.resultSummary.textContent = `${repoId} 正在运行。${job.state.status_message || ""}`;
    } else if (job.state.status === "succeeded") {
      refs.resultTitle.textContent = "采集完成";
      refs.resultSummary.textContent = `${repoId} 已完成采集，可转入结果检查。`;
    } else if (job.state.status === "stopped") {
      refs.resultTitle.textContent = "任务已停止";
      refs.resultSummary.textContent = "停止请求已落到后端，建议确认当前目录内数据的完整性。";
    } else if (job.state.status === "failed") {
      refs.resultTitle.textContent = "任务失败";
      refs.resultSummary.textContent = job.state.status_message || "请先查看最近日志定位失败阶段。";
    } else {
      refs.resultTitle.textContent = "等待后端收口";
      refs.resultSummary.textContent = job.state.status_message || "后端正在更新最终结果。";
    }

    renderList(refs.nextSteps, buildNextSteps(job), "先完成预检查，再开始采集。");
  }

  function renderRecentJobs() {
    refs.historyHint.textContent = `读取自 GET /api/jobs?limit=${RECENT_JOB_LIMIT}，展示最近 ${RECENT_JOB_LIMIT} 条任务。`;

    if (!Array.isArray(state.recentJobs) || state.recentJobs.length === 0) {
      refs.recentJobs.innerHTML = "";
      const empty = document.createElement("li");
      empty.className = "history-empty";
      empty.textContent = "最近还没有可展示的任务。";
      refs.recentJobs.appendChild(empty);
      return;
    }

    refs.recentJobs.innerHTML = "";
    for (const job of state.recentJobs) {
      const item = document.createElement("li");

      const main = document.createElement("div");
      main.className = "history-main";

      const title = document.createElement("div");
      title.className = "history-title";

      const name = document.createElement("strong");
      name.textContent = getRepoIdFromSummary(job) || job.dataset_name || job.job_id;
      title.appendChild(name);

      const badge = document.createElement("span");
      badge.className = "status-badge";
      setTone(badge, toneForStatus(job.status), formatStatus(job.status));
      title.appendChild(badge);
      main.appendChild(title);

      const meta = document.createElement("div");
      meta.className = "history-meta";
      meta.textContent = `${job.job_id} · ${formatDateTime(job.started_at || job.created_at)} · ${job.dataset_dir}`;
      main.appendChild(meta);

      const detail = document.createElement("div");
      detail.className = "history-meta";
      detail.textContent = `${formatPhase(job.phase)} · ${job.status_message || "无额外说明"}`;
      main.appendChild(detail);

      const side = document.createElement("div");
      side.className = "history-meta";
      side.textContent = formatDuration(job.estimated_duration_s);

      item.appendChild(main);
      item.appendChild(side);
      refs.recentJobs.appendChild(item);
    }
  }

  function renderList(node, items, fallbackText) {
    node.innerHTML = "";
    const values = Array.isArray(items) ? items.filter(Boolean) : [];
    const source = values.length > 0 ? values : [fallbackText];
    for (const value of source) {
      const item = document.createElement("li");
      item.textContent = value;
      node.appendChild(item);
    }
  }

  function syncTemplateState() {
    for (const button of refs.templateButtons) {
      button.dataset.active = button.dataset.template === state.activeTemplate ? "true" : "false";
    }
  }

  function buildResumeHint(current, currentPreflight) {
    if (!current.draft.dataset_name) {
      return "先填写数据集名称，再决定是否续录。";
    }

    if (!currentPreflight) {
      return state.resume ? "开启后会尝试向已有目录追加。" : "关闭时要求目标目录不存在。";
    }

    if (state.resume && currentPreflight.dataset_exists) {
      return "目录已存在，将以续录模式追加。";
    }
    if (state.resume && !currentPreflight.dataset_exists) {
      return "目录不存在，resume=true 会被拒绝。";
    }
    if (!state.resume && currentPreflight.dataset_exists) {
      return "目录已存在；如需补录，请打开 resume。";
    }
    return "目录不存在，将创建新目录。";
  }

  function buildLocalCommand(draft) {
    const command = [
      "bash",
      state.runtime?.script_path || "<record.sh>",
      draft.dataset_name || "<dataset_name>",
      draft.episode_time_s || "<episode_time_s>",
      draft.num_episodes || "<num_episodes>",
      draft.fps || "<fps>",
      draft.num_episodes === "1" ? "0" : draft.reset_time_s || "<reset_time_s>",
      draft.resume ? "true" : "false",
      draft.single_task_text || "<single_task_text>",
    ];
    return command.map(shellQuote).join(" ");
  }

  function buildModeDescription(payload, currentPreflight) {
    const duration = currentPreflight?.estimated_duration_s ?? estimateDuration(payload);
    const mode = payload.resume ? "续录现有目录" : "创建新目录";
    const resetText = payload.num_episodes === 1 ? "reset 固定 0s" : `段间 reset ${formatNumber(payload.reset_time_s)}s`;
    return `${mode} · ${payload.num_episodes} 段，每段 ${formatNumber(payload.episode_time_s)}s · ${payload.fps} FPS · ${resetText} · 预计 ${formatDuration(duration)}`;
  }

  function getTaskTextFromJob(job) {
    const normalized = job?.request?.normalized_request?.single_task_text;
    if (typeof normalized === "string" && normalized.trim()) {
      return normalized.trim();
    }

    const raw = job?.request?.request?.single_task_text;
    if (typeof raw === "string" && raw.trim()) {
      return raw.trim();
    }

    return "未记录任务语义";
  }

  function buildProgressText(job) {
    const segments = [];

    if (job.state.current_episode && job.state.total_episodes) {
      segments.push(`episode ${job.state.current_episode}/${job.state.total_episodes}`);
    } else if (job.state.total_episodes) {
      segments.push(`总共 ${job.state.total_episodes} 段`);
    }

    segments.push(`已运行 ${formatElapsed(job.state.started_at || job.state.created_at, job.state.finished_at)}`);

    if (job.state.returncode !== null && job.state.returncode !== undefined) {
      segments.push(`returncode=${job.state.returncode}`);
    }

    return segments.join(" · ");
  }

  function buildNextSteps(job) {
    if (job.state.active && job.state.status === "stop_requested") {
      return [
        "停止信号已经发出，等待后端进程组完全退出。",
        "不要用关闭浏览器代替停止任务，后台进程不会因此自动结束。",
      ];
    }

    if (job.state.active) {
      return [
        "保持页面开启以持续查看日志和阶段变化。",
        "如需中止，请使用“停止采集”而不是直接关闭页面。",
      ];
    }

    if (job.state.status === "succeeded") {
      return [
        "先检查结果目录中的数据是否完整，再开始下一轮采集。",
        "如需补录，保持同名 repo_prefix 和 dataset_name，并打开 resume。",
        "下一次开始前重新做一次现场安全确认。",
      ];
    }

    if (job.state.status === "stopped") {
      return [
        "确认当前目录里的片段是否满足保留条件。",
        "再次开始前重新检查现场状态和 reset 策略。",
      ];
    }

    if (job.state.status === "failed") {
      return [
        "先查看最近日志和返回码，定位失败发生在 booting、recording 还是 finalizing。",
        "确认 record.sh、输出目录权限和设备状态后再重试。",
        "如果目录已部分生成，再决定是否切换为 resume。",
      ];
    }

    return ["等待后端完成最后一次状态收口。"];
  }

  function hasDatasetDraft(draft) {
    return Boolean(asTrimmedText(draft?.dataset_name));
  }

  function buildRepoId(repoPrefix, datasetName) {
    const normalizedDatasetName = asTrimmedText(datasetName);
    if (!normalizedDatasetName) {
      return "";
    }
    return `${normalizeRepoPrefix(repoPrefix)}/${normalizedDatasetName}`;
  }

  function buildRepoIdFromDraft(draft) {
    return buildRepoId(draft?.repo_prefix, draft?.dataset_name);
  }

  function getRepoIdFromPayload(payload) {
    return buildRepoId(payload?.repo_prefix, payload?.dataset_name);
  }

  function getDatasetDirFromDraft(draft) {
    return getDatasetDir(buildRepoIdFromDraft(draft));
  }

  function getDatasetDirFromPayload(payload) {
    return getDatasetDir(getRepoIdFromPayload(payload));
  }

  function getDatasetDir(repoId) {
    if (!repoId) {
      return "等待 dataset_name";
    }

    const datasetRoot = asTrimmedText(state.runtime?.dataset_root || state.runtime?.output_root);
    if (!datasetRoot) {
      return `outputs/${repoId}`;
    }
    return `${datasetRoot.replace(/\/+$/, "")}/${repoId}`;
  }

  function getRepoIdFromJob(job) {
    if (!job) {
      return "";
    }

    const snapshot = job.request || {};
    const repoPrefix = firstNonEmptyText(
      snapshot.repo_prefix,
      snapshot.normalized_request?.repo_prefix,
      snapshot.request?.repo_prefix,
      getRepoPrefixFromDatasetDir(snapshot.dataset_dir)
    );
    const datasetName = firstNonEmptyText(
      snapshot.dataset_name,
      snapshot.normalized_request?.dataset_name,
      snapshot.request?.dataset_name,
      getDatasetNameFromDatasetDir(snapshot.dataset_dir)
    );

    if (repoPrefix && datasetName) {
      return `${repoPrefix}/${datasetName}`;
    }
    return getRepoIdFromDatasetDir(snapshot.dataset_dir);
  }

  function getRepoPrefixFromJob(job) {
    return getRepoPrefixFromRepoId(getRepoIdFromJob(job));
  }

  function getRepoIdFromSummary(job) {
    if (!job) {
      return "";
    }

    const repoPrefix = firstNonEmptyText(job.repo_prefix, getRepoPrefixFromDatasetDir(job.dataset_dir));
    const datasetName = firstNonEmptyText(job.dataset_name, getDatasetNameFromDatasetDir(job.dataset_dir));

    if (repoPrefix && datasetName) {
      return `${repoPrefix}/${datasetName}`;
    }
    return getRepoIdFromDatasetDir(job.dataset_dir);
  }

  function getRepoPrefixFromSummary(job) {
    return getRepoPrefixFromRepoId(getRepoIdFromSummary(job));
  }

  function getRepoIdFromDatasetDir(datasetDir) {
    const normalizedDir = normalizePath(datasetDir);
    if (!normalizedDir) {
      return "";
    }

    const datasetRoot = normalizePath(state.runtime?.dataset_root || state.runtime?.output_root);
    if (datasetRoot && normalizedDir.startsWith(`${datasetRoot}/`)) {
      return normalizedDir.slice(datasetRoot.length + 1);
    }

    const outputsMarker = "/outputs/";
    const markerIndex = normalizedDir.lastIndexOf(outputsMarker);
    if (markerIndex >= 0) {
      return normalizedDir.slice(markerIndex + outputsMarker.length);
    }

    const parts = normalizedDir.split("/").filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
    }

    return "";
  }

  function getRepoPrefixFromDatasetDir(datasetDir) {
    return getRepoPrefixFromRepoId(getRepoIdFromDatasetDir(datasetDir));
  }

  function getDatasetNameFromDatasetDir(datasetDir) {
    const repoId = getRepoIdFromDatasetDir(datasetDir);
    if (!repoId) {
      return "";
    }

    const parts = repoId.split("/");
    return parts[parts.length - 1] || "";
  }

  function getRepoPrefixFromRepoId(repoId) {
    const normalizedRepoId = asTrimmedText(repoId);
    if (!normalizedRepoId || !normalizedRepoId.includes("/")) {
      return "";
    }
    return normalizedRepoId.split("/")[0] || "";
  }

  function normalizeRepoPrefix(value, fallback = DEFAULT_REPO_PREFIX) {
    const trimmed = asTrimmedText(value);
    if (!trimmed) {
      return fallback;
    }
    return REPO_PREFIX_RE.test(trimmed) ? trimmed : fallback;
  }

  function normalizePath(value) {
    const trimmed = asTrimmedText(value);
    if (!trimmed) {
      return "";
    }
    return trimmed.replace(/\\/g, "/").replace(/\/+$/, "");
  }

  function asTrimmedText(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function firstNonEmptyText(...values) {
    for (const value of values) {
      const text = asTrimmedText(value);
      if (text) {
        return text;
      }
    }
    return "";
  }

  function estimateDuration(payload) {
    const resets = Math.max(0, payload.num_episodes - 1);
    return (payload.episode_time_s * payload.num_episodes) + (payload.reset_time_s * resets);
  }

  function formatDateTime(value) {
    if (!value) {
      return "-";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(date);
  }

  function formatElapsed(startValue, endValue) {
    const start = new Date(startValue || "");
    const end = endValue ? new Date(endValue) : new Date();
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
      return "-";
    }
    return formatDuration(Math.max(0, (end.getTime() - start.getTime()) / 1000));
  }

  function formatDuration(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) {
      return "-";
    }

    if (value < 60) {
      return `${formatNumber(value)}s`;
    }

    const rounded = Math.round(value);
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const secs = rounded % 60;
    const parts = [];

    if (hours > 0) {
      parts.push(`${hours}h`);
    }
    if (minutes > 0) {
      parts.push(`${minutes}m`);
    }
    if (secs > 0 || parts.length === 0) {
      parts.push(`${secs}s`);
    }

    return parts.join(" ");
  }

  function formatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "-";
    }
    return Number.isInteger(number) ? String(number) : number.toFixed(1).replace(/\.0$/, "");
  }

  function formatInputNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "";
    }
    return Number.isInteger(number) ? String(number) : String(number);
  }

  function formatStatus(status) {
    const labels = {
      created: "已创建",
      validating: "校验中",
      rejected: "已拒绝",
      starting: "启动中",
      running: "运行中",
      stop_requested: "停止中",
      stopped: "已停止",
      succeeded: "已完成",
      failed: "失败",
    };
    return labels[status] || status || "未知状态";
  }

  function formatPhase(phase) {
    const labels = {
      preflight: "预检查",
      booting: "启动",
      recording: "采集中",
      resetting: "重置中",
      saving: "保存中",
      finalizing: "收尾中",
    };
    return labels[phase] || phase || "未知阶段";
  }

  function toneForStatus(status) {
    if (status === "running" || status === "starting") {
      return "info";
    }
    if (status === "succeeded") {
      return "success";
    }
    if (status === "failed" || status === "rejected") {
      return "danger";
    }
    if (status === "stopped" || status === "stop_requested") {
      return "warning";
    }
    return "neutral";
  }

  function setTone(node, tone, text) {
    node.dataset.tone = tone;
    node.textContent = text;
  }

  function parseFiniteNumber(value) {
    if (value === "" || value === null || value === undefined) {
      return Number.NaN;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : Number.NaN;
  }

  function parseInteger(value) {
    if (value === "" || value === null || value === undefined) {
      return Number.NaN;
    }
    const parsed = Number(value);
    return Number.isInteger(parsed) ? parsed : Number.NaN;
  }

  function buildSuggestedDatasetName(prefix) {
    const stamp = new Date();
    const parts = [
      stamp.getFullYear(),
      String(stamp.getMonth() + 1).padStart(2, "0"),
      String(stamp.getDate()).padStart(2, "0"),
      String(stamp.getHours()).padStart(2, "0"),
      String(stamp.getMinutes()).padStart(2, "0"),
    ];
    return `${prefix.replace(/[^A-Za-z0-9_-]/g, "_")}_${parts.join("")}`;
  }

  function normalizeTaskText(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function shellQuote(value) {
    const text = String(value);
    if (/^[A-Za-z0-9_./:=+-]+$/.test(text)) {
      return text;
    }
    return `'${text.replace(/'/g, `'\\''`)}'`;
  }

  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (options.body !== undefined) {
      headers["Content-Type"] = headers["Content-Type"] || "application/json";
    }

    const response = await fetch(path, {
      method: options.method || "GET",
      headers,
      body: options.body,
    });

    const rawText = await response.text();
    let payload = null;

    if (rawText) {
      try {
        payload = JSON.parse(rawText);
      } catch {
        payload = rawText;
      }
    }

    if (!response.ok) {
      throw new ApiError(extractApiMessage(payload, response.status), response.status, payload);
    }

    return payload;
  }

  function extractApiMessage(payload, status) {
    if (typeof payload === "string" && payload.trim()) {
      return payload;
    }

    if (payload && typeof payload === "object") {
      if (payload.error && typeof payload.error === "object") {
        if (typeof payload.error.message === "string") {
          return payload.error.message;
        }
        if (payload.error.details && typeof payload.error.details === "object") {
          if (typeof payload.error.details.message === "string") {
            return payload.error.details.message;
          }
          if (Array.isArray(payload.error.details.conflicts) && payload.error.details.conflicts.length > 0) {
            return payload.error.details.conflicts[0];
          }
        }
      }
      if (typeof payload.message === "string") {
        return payload.message;
      }
      if (typeof payload.detail === "string") {
        return payload.detail;
      }
      if (payload.detail && typeof payload.detail === "object") {
        if (typeof payload.detail.message === "string") {
          return payload.detail.message;
        }
        if (Array.isArray(payload.detail.conflicts) && payload.detail.conflicts.length > 0) {
          return payload.detail.conflicts[0];
        }
      }
      if (Array.isArray(payload.conflicts) && payload.conflicts.length > 0) {
        return payload.conflicts[0];
      }
    }

    return `HTTP ${status}`;
  }

  function showBanner(message, tone, options = {}) {
    window.clearTimeout(state.bannerTimer);
    refs.feedbackBanner.textContent = message;
    refs.feedbackBanner.dataset.tone = tone;
    refs.feedbackBanner.classList.remove("hidden");

    if (options.persist) {
      return;
    }

    state.bannerTimer = window.setTimeout(() => {
      refs.feedbackBanner.classList.add("hidden");
    }, 4200);
  }
})();
