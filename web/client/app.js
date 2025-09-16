const locationInfo = window.location;
const HTTP_BASE = `${locationInfo.protocol}//${locationInfo.host}`;
const WS_BASE = `${locationInfo.protocol === "https:" ? "wss" : "ws"}://${locationInfo.host}`;
const VIDEO_URL = `${HTTP_BASE}/video`;

const elements = {
  video: document.getElementById("video"),
  streamStatus: document.getElementById("streamStatus"),
  emo: document.getElementById("emo"),
  conf: document.getElementById("conf"),
  sr: document.getElementById("sr"),
  reply: document.getElementById("reply"),
  player: document.getElementById("player"),
  netstat: document.getElementById("netstat"),
  visionStatus: document.getElementById("visionStatus"),
  audioStatus: document.getElementById("audioStatus"),
  textStatusLine: document.getElementById("textStatusLine"),
  fusionWeights: document.getElementById("fusionWeights"),
  fusionLabel: document.getElementById("fusionLabel"),
  userText: document.getElementById("userText"),
  sendTextBtn: document.getElementById("sendText"),
  clearTextBtn: document.getElementById("clearText"),
  textStatus: document.getElementById("textStatus"),
  turnId: document.getElementById("turnId"),
  triggerList: document.getElementById("triggerList"),
  strategyLabel: document.getElementById("strategyLabel"),
  styleLabel: document.getElementById("styleLabel"),
  clueList: document.getElementById("clueList"),
  sourcesList: document.getElementById("sourcesList"),
  visemeFps: document.getElementById("visemeFps"),
  visemeBars: document.getElementById("visemeBars"),
  timelineList: document.getElementById("timelineList"),
  dialogStatus: document.getElementById("dialogStatus"),
  riskBadge: document.getElementById("riskBadge"),
};

const triggerMap = {
  warm_start: "首次启动",
  safety_word: "安全词",
  text_update: "文本更新",
  emotion_shift: "情绪变化",
};

const state = {
  ws: null,
  reconnectTimer: null,
  reconnectAttempts: 0,
  lastWav: null,
  textDirty: false,
};

function updateStreamStatus(status, text) {
  if (!elements.streamStatus) return;
  elements.streamStatus.dataset.state = status;
  if (typeof text === "string") {
    elements.streamStatus.textContent = text;
  }
}

function setStatus(text) {
  elements.netstat.textContent = text || "";
}

function fmtConf(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const num = Number(value);
  if (num > 1.0) {
    return `${num.toFixed(0)}%`;
  }
  return `${(num * 100).toFixed(0)}%`;
}

function translateTriggers(triggers) {
  if (!Array.isArray(triggers) || triggers.length === 0) {
    return [];
  }
  return triggers.map((t) => triggerMap[t] || t);
}

function localiseStrategy(strategy) {
  if (!strategy) return "-";
  switch (strategy) {
    case "retrieval":
      return "检索策略";
    case "fallback":
      return "规则模板";
    case "emergency":
      return "应急守护";
    case "idle":
      return "等待触发";
    default:
      return strategy;
  }
}

function updateVision(mod) {
  if (!mod) {
    elements.visionStatus.textContent = "无视觉数据";
    return;
  }
  const emo = mod.emo ?? "-";
  const conf = fmtConf(mod.conf);
  elements.visionStatus.textContent = `${emo} · 置信度 ${conf}`;
}

function updateAudio(mod) {
  if (!mod || !mod.probs) {
    elements.audioStatus.textContent = "等待音频数据";
    return;
  }
  const feats = mod.features || {};
  const parts = [];
  if (feats.rms !== undefined) parts.push(`rms ${Number(feats.rms).toFixed(2)}`);
  if (feats.zcr !== undefined) parts.push(`zcr ${Number(feats.zcr).toFixed(2)}`);
  if (feats.centroid !== undefined) parts.push(`centroid ${Math.round(Number(feats.centroid))}`);
  const featureText = parts.length ? parts.join(" · ") : "特征不足";
  elements.audioStatus.textContent = `${mod.emo ?? "-"} · ${featureText}`;
}

function updateTextMod(mod) {
  if (!mod || !mod.probs) {
    elements.textStatusLine.textContent = "尚未输入文本";
    return;
  }
  const base = `${mod.emo ?? "-"} · 置信度 ${fmtConf(mod.conf)}`;
  const matched = mod.matched && typeof mod.matched === "object"
    ? Object.entries(mod.matched)
        .filter(([, v]) => Number(v) > 0)
        .map(([k, v]) => `${k}:${v}`)
    : [];
  const riskFlag = mod.risk ? " ⚠️风险词" : "";
  if (matched.length) {
    elements.textStatusLine.textContent = `${base} · 关键词 ${matched.join("，")}${riskFlag}`;
  } else {
    elements.textStatusLine.textContent = `${base}${riskFlag}`;
  }
}

function updateFusionInfo(fusion) {
  if (!fusion) {
    elements.fusionWeights.textContent = "融合权重：-";
    elements.fusionLabel.textContent = "";
    return;
  }
  const weights = fusion.weights
    ? Object.entries(fusion.weights)
        .map(([k, v]) => `${k}:${Number(v).toFixed(2)}`)
        .join("， ")
    : "-";
  elements.fusionWeights.textContent = `融合权重：${weights}`;
  if (fusion.emo) {
    elements.fusionLabel.textContent = `当前融合：${fusion.emo} (${fmtConf(fusion.conf)})`;
  } else {
    elements.fusionLabel.textContent = "";
  }
}

function renderClues(clues) {
  elements.clueList.innerHTML = "";
  if (!Array.isArray(clues) || clues.length === 0) {
    const span = document.createElement("div");
    span.className = "empty-note";
    span.textContent = "暂无检索线索";
    elements.clueList.appendChild(span);
    return;
  }
  clues.slice(0, 6).forEach((clue, idx) => {
    const chip = document.createElement("span");
    chip.className = idx === 0 ? "chip accent" : "chip";
    chip.textContent = clue;
    elements.clueList.appendChild(chip);
  });
}

function renderSources(sources) {
  elements.sourcesList.innerHTML = "";
  if (!Array.isArray(sources) || sources.length === 0) {
    const div = document.createElement("div");
    div.className = "empty-note";
    div.textContent = "暂无命中条目，使用规则模板回复。";
    elements.sourcesList.appendChild(div);
    return;
  }
  sources.slice(0, 3).forEach((src) => {
    const item = document.createElement("div");
    item.className = "source-item";

    const title = document.createElement("div");
    title.className = "title";
    title.textContent = src.title || "策略片段";
    item.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "meta";
    const score = Number(src.score ?? 0).toFixed(2);
    const keywords = Array.isArray(src.keywords) && src.keywords.length
      ? src.keywords.join("、")
      : "无关键词";
    const parts = [`score ${score}`, `关键词 ${keywords}`];
    if (src.category) parts.push(String(src.category));
    meta.textContent = parts.join(" · ");
    item.appendChild(meta);

    if (src.summary) {
      const summary = document.createElement("div");
      summary.className = "summary";
      summary.textContent = src.summary;
      item.appendChild(summary);
    }

    elements.sourcesList.appendChild(item);
  });
}

function renderVisemes(viseme) {
  elements.visemeBars.innerHTML = "";
  elements.visemeBars.classList.remove("empty");
  if (!viseme || !Array.isArray(viseme.energy) || viseme.energy.length === 0) {
    elements.visemeBars.classList.add("empty");
    elements.visemeBars.textContent = "等待音频...";
    elements.visemeFps.textContent = "-";
    return;
  }
  elements.visemeFps.textContent = viseme.fps ?? "-";
  viseme.energy.slice(0, 60).forEach((value) => {
    const span = document.createElement("span");
    const level = Math.max(0, Math.min(1, Number(value) || 0));
    span.style.height = `${6 + level * 48}px`;
    elements.visemeBars.appendChild(span);
  });
}

function formatTime(ts) {
  if (!ts) return "-";
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleTimeString("zh-CN", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function renderTimeline(history) {
  elements.timelineList.innerHTML = "";
  if (!Array.isArray(history) || history.length === 0) {
    const empty = document.createElement("div");
    empty.className = "timeline-empty";
    empty.textContent = "暂无对话记录";
    elements.timelineList.appendChild(empty);
    return;
  }

  const frag = document.createDocumentFragment();
  history.forEach((turn) => {
    const item = document.createElement("div");
    item.className = "timeline-item";
    if (turn?.assistant?.risk) item.classList.add("risk");

    const head = document.createElement("div");
    head.className = "head";
    const left = document.createElement("span");
    left.textContent = `#${turn?.turn ?? 0}`;
    const right = document.createElement("span");
    right.textContent = formatTime(turn?.ts);
    head.append(left, right);
    item.appendChild(head);

    const body = document.createElement("div");
    body.className = "body";

    const userRow = document.createElement("div");
    userRow.className = "role";
    const userLabel = document.createElement("strong");
    userLabel.textContent = "来访者";
    const userContent = document.createElement("span");
    userContent.textContent = turn?.user?.text || "(未提供文本)";
    userRow.append(userLabel, userContent);

    const botRow = document.createElement("div");
    botRow.className = "role";
    const botLabel = document.createElement("strong");
    botLabel.textContent = "数字人";
    const botContent = document.createElement("span");
    botContent.textContent = turn?.assistant?.text || "";
    botRow.append(botLabel, botContent);

    const meta = document.createElement("div");
    meta.className = "meta";
    const triggers = Array.isArray(turn?.assistant?.triggers)
      ? translateTriggers(turn.assistant.triggers).join(" / ")
      : "-";
    const emoLabel = turn?.user?.emo ? `情绪：${turn.user.emo}` : "情绪：-";
    const strategyText = `策略：${localiseStrategy(turn?.assistant?.strategy)}`;
    meta.textContent = `${emoLabel} · ${strategyText} · 触发：${triggers}`;
    if (turn?.assistant?.risk) {
      meta.textContent += " · ⚠️ 应急";
    }

    body.append(userRow, botRow, meta);
    item.appendChild(body);
    frag.appendChild(item);
  });

  elements.timelineList.appendChild(frag);
}

async function fetchAudio(wav) {
  if (!wav) return;
  const url = `${HTTP_BASE}/audio/${encodeURIComponent(wav)}`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`状态 ${res.status}`);
    const blob = await res.blob();
    const objUrl = URL.createObjectURL(blob);
    try {
      elements.player.pause();
    } catch (err) {
      console.warn("无法暂停播放器", err);
    }
    elements.player.currentTime = 0;
    elements.player.src = objUrl;
    await elements.player.play().catch(() => {});
  } catch (err) {
    console.error("获取音频失败", err);
    setStatus(`已连接，但 /audio 返回错误`);
  }
}

function applyDialogPayload(dialog, fallbackTurnId) {
  const safeDialog = dialog || {};
  const triggers = translateTriggers(safeDialog.triggers);
  const triggerText = triggers.length ? triggers.join(" / ") : "-";
  elements.triggerList.textContent = triggerText;

  const turnId = safeDialog.turn_id ?? fallbackTurnId ?? 0;
  elements.turnId.textContent = turnId ? `#${turnId}` : "-";
  elements.strategyLabel.textContent = localiseStrategy(safeDialog.strategy);
  elements.styleLabel.textContent = safeDialog.style || "-";

  elements.dialogStatus.textContent = safeDialog.strategy
    ? `策略：${localiseStrategy(safeDialog.strategy)} · 触发：${triggerText}`
    : "";
  elements.dialogStatus.classList.toggle("risk", Boolean(safeDialog.risk));

  if (safeDialog.risk) {
    elements.riskBadge.textContent = "⚠️ 应急策略已触发";
    elements.riskBadge.style.display = "block";
  } else {
    elements.riskBadge.textContent = "";
    elements.riskBadge.style.display = "none";
  }

  renderClues(safeDialog.clues);
  renderSources(safeDialog.sources);
  renderTimeline(safeDialog.history);
}

function handlePayload(js) {
  elements.emo.textContent = js.emo ?? "-";
  elements.conf.textContent = fmtConf(js.conf);
  elements.sr.textContent = js.sr ?? 22050;
  elements.reply.textContent = js.reply || "";

  updateVision(js.modalities?.vision);
  updateAudio(js.modalities?.audio);
  updateTextMod(js.modalities?.text);
  updateFusionInfo(js.fusion);

  if (!state.textDirty && typeof js.user_text === "string") {
    elements.userText.value = js.user_text;
  }

  applyDialogPayload(js.dialog, js.turn_id);
  renderVisemes(js.visemes);

  if (js.wav && js.wav !== state.lastWav) {
    state.lastWav = js.wav;
    fetchAudio(js.wav);
  }
}

function handleMessage(event) {
  try {
    const data = JSON.parse(event.data);
    handlePayload(data);
  } catch (err) {
    console.error("无法解析消息", err);
  }
}

function clearReconnectTimer() {
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
}

function scheduleReconnect() {
  clearReconnectTimer();
  state.reconnectAttempts += 1;
  const delay = Math.min(1000 * 2 ** (state.reconnectAttempts - 1), 10000);
  updateStreamStatus("connecting", `尝试重连（第${state.reconnectAttempts}次）…`);
  state.reconnectTimer = window.setTimeout(() => {
    connectWebSocket();
  }, delay);
}

function connectWebSocket() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch (err) {
      console.warn("关闭旧的 WebSocket 失败", err);
    }
  }

  updateStreamStatus("connecting", "正在连接服务…");
  setStatus("WS 连接中…");

  const ws = new WebSocket(`${WS_BASE}/ws`);
  state.ws = ws;

  ws.addEventListener("open", () => {
    clearReconnectTimer();
    state.reconnectAttempts = 0;
    updateStreamStatus("ok", "视频来自 /video（MJPEG）");
    setStatus("WS 已连接");
    syncInitialText();
  });

  ws.addEventListener("message", handleMessage);

  ws.addEventListener("close", () => {
    if (state.ws === ws) {
      state.ws = null;
    }
    setStatus("WS 已关闭");
    scheduleReconnect();
  });

  ws.addEventListener("error", () => {
    setStatus("WS 连接错误");
    updateStreamStatus("error", "WebSocket 通道异常");
  });
}

async function pushText(text) {
  try {
    const res = await fetch(`${HTTP_BASE}/user-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (res.ok) {
      elements.textStatus.textContent = text ? "文本已发送" : "文本已清空";
      state.textDirty = false;
    } else {
      elements.textStatus.textContent = `提交失败: ${res.status}`;
    }
  } catch (err) {
    console.error("提交文本失败", err);
    elements.textStatus.textContent = "提交失败";
  }
}

async function syncInitialText() {
  try {
    const res = await fetch(`${HTTP_BASE}/user-text`);
    if (!res.ok) return;
    const data = await res.json();
    if (!state.textDirty && typeof data.text === "string") {
      elements.userText.value = data.text;
    }
  } catch (err) {
    console.warn("无法获取初始文本", err);
  }
}

async function refreshHealth() {
  try {
    const res = await fetch(`${HTTP_BASE}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    if (data.camera_opened) {
      updateStreamStatus(
        "ok",
        `视频 ${data.video_size?.join("x") || ""} @ ${data.fps_target || "?"}fps`
      );
    } else {
      updateStreamStatus("connecting", "摄像头未就绪，等待采集...");
    }
  } catch (err) {
    updateStreamStatus("error", "无法连接后端健康检查");
  }
}

function bindEvents() {
  elements.video.src = VIDEO_URL;
  elements.sendTextBtn.addEventListener("click", () => pushText(elements.userText.value));
  elements.clearTextBtn.addEventListener("click", () => {
    elements.userText.value = "";
    pushText("");
  });
  elements.userText.addEventListener("input", () => {
    state.textDirty = true;
    elements.textStatus.textContent = "";
  });
}

function init() {
  bindEvents();
  connectWebSocket();
  refreshHealth();
  window.setInterval(refreshHealth, 15000);
}

window.addEventListener("load", init);
