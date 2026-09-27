"use strict";

const ui = {
  status: document.getElementById("status"),
  sessionMeta: document.getElementById("session-meta"),
  slot: document.getElementById("controlled-slot"),
  stepFrames: document.getElementById("step-frames"),
  speed: document.getElementById("playback-speed"),
  reset: document.getElementById("reset"),
  step: document.getElementById("step"),
  play: document.getElementById("play"),
  frame: document.getElementById("game-frame"),
  frameNumber: document.getElementById("frame-number"),
  progressIndex: document.getElementById("progress-index"),
  geometry: document.getElementById("geometry"),
  history: document.getElementById("history-grid"),
  groups: document.getElementById("observation-groups"),
  observationCount: document.getElementById("observation-count"),
  valueTemplate: document.getElementById("value-row-template"),
};

let current = null;
let playing = false;
let busy = false;
let playTimer = null;

function setStatus(text, error = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function runAction(path, body) {
  if (busy) return;
  busy = true;
  for (const element of [ui.reset, ui.step, ui.slot]) element.disabled = true;
  try {
    const payload = await request(path, { method: "POST", body: JSON.stringify(body) });
    render(payload);
    setStatus(playing ? "Playing" : "Ready");
  } catch (error) {
    setStatus(error.message, true);
    stopPlaying();
  } finally {
    busy = false;
    for (const element of [ui.reset, ui.step, ui.slot]) element.disabled = false;
  }
}

function formatValue(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0 || value === 1 || value === -1) return value.toFixed(0);
  return value.toFixed(5);
}

function renderObservation(snapshot) {
  ui.groups.replaceChildren();
  for (const [groupName, entries] of Object.entries(snapshot.observation.groups)) {
    const section = document.createElement("section");
    section.className = "observation-group";
    const heading = document.createElement("h3");
    heading.textContent = groupName;
    section.append(heading);
    for (const entry of entries) {
      const row = ui.valueTemplate.content.firstElementChild.cloneNode(true);
      row.querySelector("code").textContent = entry.name;
      row.querySelector("output").textContent = formatValue(entry.value);
      section.append(row);
    }
    ui.groups.append(section);
  }
  ui.observationCount.textContent = `${snapshot.observation.values.length} values`;
}

function setupCanvas(canvas) {
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function drawArrow(context, from, to, color, label, dashed = false) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  if (length < 1) return;
  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2;
  context.setLineDash(dashed ? [6, 5] : []);
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
  context.setLineDash([]);
  const angle = Math.atan2(dy, dx);
  context.beginPath();
  context.moveTo(to.x, to.y);
  context.lineTo(to.x - 9 * Math.cos(angle - .45), to.y - 9 * Math.sin(angle - .45));
  context.lineTo(to.x - 9 * Math.cos(angle + .45), to.y - 9 * Math.sin(angle + .45));
  context.closePath();
  context.fill();
  if (label) {
    context.font = "11px ui-monospace, monospace";
    context.fillText(label, to.x + 6, to.y - 6);
  }
  context.restore();
}

function renderGeometry(snapshot) {
  const { context, width, height } = setupCanvas(ui.geometry);
  const geometry = snapshot.geometry;
  const points = [
    ...geometry.centerline,
    ...geometry.lookahead_targets.map(target => target.point),
    geometry.pickup.point,
    ...snapshot.racers.map(racer => racer.local),
  ];
  const maxMagnitude = Math.max(3, ...points.flatMap(point => [Math.abs(point.right), Math.abs(point.forward)]));
  const scale = Math.min(width, height) * .42 / (maxMagnitude * 1.08);
  const origin = { x: width / 2, y: height / 2 };
  const screen = point => ({ x: origin.x + point.right * scale, y: origin.y - point.forward * scale });

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#081218";
  context.fillRect(0, 0, width, height);

  const gridStep = Math.max(1, Math.pow(2, Math.ceil(Math.log2(maxMagnitude / 6))));
  context.strokeStyle = "rgba(116, 157, 171, .14)";
  context.lineWidth = 1;
  for (let value = -maxMagnitude; value <= maxMagnitude; value += gridStep) {
    const x = origin.x + value * scale;
    const y = origin.y - value * scale;
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }

  context.strokeStyle = "#4ad7e8";
  context.lineWidth = 3;
  context.beginPath();
  geometry.centerline.forEach((point, index) => {
    const p = screen(point);
    if (index === 0) context.moveTo(p.x, p.y); else context.lineTo(p.x, p.y);
  });
  context.stroke();

  const projection = screen(geometry.projected_point);
  context.fillStyle = "#ffc857";
  context.beginPath(); context.arc(projection.x, projection.y, 5, 0, Math.PI * 2); context.fill();
  drawArrow(context, projection, origin, "#ffc857", `lateral ${formatValue(geometry.lateral_offset)}`);
  const tangent = geometry.track_tangent;
  drawArrow(context, projection, {
    x: projection.x + tangent.right * 55,
    y: projection.y - tangent.forward * 55,
  }, "#ffc857", "tangent");

  const targetColors = { short: "#64a8ff", medium: "#be95ff", long: "#ff9f68" };
  for (const target of geometry.lookahead_targets) {
    const point = screen(target.point);
    const values = target.observation;
    drawArrow(context, origin, point, targetColors[target.label], target.label);
    context.fillStyle = targetColors[target.label];
    context.beginPath(); context.arc(point.x, point.y, 4, 0, Math.PI * 2); context.fill();
  }

  const nearbyCallouts = [];
  for (const racer of snapshot.racers) {
    const point = screen(racer.local);
    const color = racer.controlled ? "#6ce59b" : "#be95ff";
    context.fillStyle = color;
    context.beginPath(); context.arc(point.x, point.y, racer.controlled ? 7 : 5, 0, Math.PI * 2); context.fill();
    drawArrow(context, point, {
      x: point.x + racer.heading.right * 34,
      y: point.y - racer.heading.forward * 34,
    }, color, "");
    if (racer.controlled) {
      context.fillStyle = color;
      context.font = "11px ui-monospace, monospace";
      context.fillText("controlled", point.x + 8, point.y - 12);
    } else {
      nearbyCallouts.push({ racer, point });
    }
  }

  drawArrow(context, origin, { x: origin.x, y: origin.y - 62 }, "#6ce59b", "forward");
  drawArrow(context, origin, { x: origin.x + 62, y: origin.y }, "#4ad7e8", "right");

  const pickup = screen(geometry.pickup.point);
  drawArrow(context, origin, pickup, geometry.pickup.available ? "#ff7485" : "#777f84",
    "pickup", true);
  context.save();
  context.translate(pickup.x, pickup.y); context.rotate(Math.PI / 4);
  context.fillStyle = geometry.pickup.available ? "#ff7485" : "#777f84";
  context.fillRect(-6, -6, 12, 12); context.restore();

  const calloutWidth = Math.min(285, width - 24);
  const calloutHeight = 16 + nearbyCallouts.length * 29;
  const calloutX = 12;
  const calloutY = height - calloutHeight - 31;
  context.fillStyle = "rgba(4, 12, 16, .84)";
  context.fillRect(calloutX, calloutY, calloutWidth, calloutHeight);
  nearbyCallouts.forEach(({ racer, point }, index) => {
    const rowY = calloutY + 18 + index * 29;
    context.strokeStyle = "rgba(190, 149, 255, .55)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(point.x, point.y);
    context.lineTo(calloutX + calloutWidth, rowY - 4);
    context.stroke();
    context.fillStyle = "#be95ff";
    context.font = "10px ui-monospace, monospace";
    context.fillText(racer.label, calloutX + 8, rowY - 5);
    context.fillStyle = "#d8c9ff";
    context.fillText(
      `f ${formatValue(racer.observation.forward)}  r ${formatValue(racer.observation.right)}  d ${formatValue(racer.observation.distance)}  Δp ${formatValue(racer.observation.relative_progress)}`,
      calloutX + 8, rowY + 7,
    );
  });

  context.fillStyle = "#8faab2";
  context.font = "11px ui-monospace, monospace";
  context.fillText(`grid ${gridStep} local unit${gridStep === 1 ? "" : "s"}`, 12, height - 14);
  context.fillText(`reference ${geometry.progress_index} + ${geometry.segment_fraction.toFixed(3)}`, 12, 18);

  const metrics = [
    ["heading error sin/cos", `${formatValue(geometry.heading_error.sin)} / ${formatValue(geometry.heading_error.cos)}`],
    ...geometry.lookahead_targets.map(target => [
      `${target.label} target f/r`,
      `${formatValue(target.observation.forward)} / ${formatValue(target.observation.right)}`,
    ]),
    ["curvature med/long", `${formatValue(geometry.curvature.medium)} / ${formatValue(geometry.curvature.long)}`],
    ["pickup Δp / lateral error", `${formatValue(geometry.pickup.progress_distance)} / ${formatValue(geometry.pickup.lateral_error)}`],
  ];
  const boxWidth = Math.min(245, width - 24);
  const boxX = width - boxWidth - 12;
  const boxY = 12;
  context.fillStyle = "rgba(4, 12, 16, .84)";
  context.fillRect(boxX, boxY, boxWidth, 18 + metrics.length * 17);
  context.font = "10px ui-monospace, monospace";
  metrics.forEach((metric, index) => {
    context.fillStyle = "#8faab2";
    context.fillText(metric[0], boxX + 8, boxY + 15 + index * 17);
    context.fillStyle = "#e8f4f6";
    context.textAlign = "right";
    context.fillText(metric[1], boxX + boxWidth - 8, boxY + 15 + index * 17);
    context.textAlign = "left";
  });
}

function renderHistory(snapshot) {
  const existing = new Map([...ui.history.querySelectorAll(".history-item")].map(item => [item.dataset.name, item]));
  for (const [name, values] of Object.entries(snapshot.history)) {
    let item = existing.get(name);
    if (!item) {
      item = document.createElement("div");
      item.className = "history-item";
      item.dataset.name = name;
      item.innerHTML = `<div class="history-label"><span></span><output></output></div><canvas></canvas>`;
      item.querySelector("span").textContent = name;
      ui.history.append(item);
    }
    item.querySelector("output").textContent = values.length ? formatValue(values.at(-1)) : "—";
    const canvas = item.querySelector("canvas");
    const { context, width, height } = setupCanvas(canvas);
    context.clearRect(0, 0, width, height);
    if (values.length < 2) continue;
    let minimum = Math.min(...values), maximum = Math.max(...values);
    if (minimum === maximum) { minimum -= .01; maximum += .01; }
    const x = index => 2 + index * (width - 4) / Math.max(1, values.length - 1);
    const y = value => height - 3 - (value - minimum) * (height - 6) / (maximum - minimum);
    context.strokeStyle = "rgba(143, 170, 178, .25)";
    context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
    context.strokeStyle = "#4ad7e8";
    context.lineWidth = 1.5;
    context.beginPath();
    values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value)));
    context.stroke();
  }
}

function render(snapshot) {
  current = snapshot;
  ui.slot.value = String(snapshot.controlled_slot);
  ui.frame.src = snapshot.frame_url;
  ui.frameNumber.textContent = `Frame ${snapshot.raw_frame}`;
  ui.progressIndex.textContent = `Reference ${snapshot.geometry.progress_index}`;
  const policy = snapshot.metadata.policy_enabled
    ? ` · model driving slot ${snapshot.metadata.policy_slot} · action [${snapshot.metadata.policy_action.join(", ")}]`
    : " · no-input stepping";
  ui.sessionMeta.textContent = `${snapshot.track} · viewing slot ${snapshot.controlled_slot}${policy} · ${snapshot.state_name}`;
  renderObservation(snapshot);
  renderGeometry(snapshot);
  renderHistory(snapshot);
}

function stopPlaying() {
  playing = false;
  clearTimeout(playTimer);
  playTimer = null;
  ui.play.textContent = "Play";
  ui.play.setAttribute("aria-pressed", "false");
}

async function playbackLoop() {
  if (!playing) return;
  const frames = Math.max(1, Math.min(600, Number(ui.stepFrames.value) || 1));
  await runAction("/api/step", { frames });
  if (!playing) return;
  const speed = Math.max(.25, Number(ui.speed.value) || 1);
  playTimer = setTimeout(playbackLoop, 250 / speed);
}

ui.reset.addEventListener("click", () => runAction("/api/reset", { controlled_slot: Number(ui.slot.value) }));
ui.step.addEventListener("click", () => runAction("/api/step", { frames: Number(ui.stepFrames.value) || 1 }));
ui.slot.addEventListener("change", () => runAction("/api/controlled-slot", { slot: Number(ui.slot.value) }));
ui.play.addEventListener("click", () => {
  if (playing) { stopPlaying(); setStatus("Ready"); return; }
  playing = true;
  ui.play.textContent = "Pause";
  ui.play.setAttribute("aria-pressed", "true");
  setStatus("Playing");
  playbackLoop();
});

new ResizeObserver(() => {
  if (!current) return;
  renderGeometry(current);
  renderHistory(current);
}).observe(document.querySelector("main"));

request("/api/snapshot")
  .then(snapshot => { render(snapshot); setStatus("Ready"); })
  .catch(error => setStatus(error.message, true));
