// Photo booth front end: camera, analysis loop, round flow and overlays.

const EXPECTED_SERVER_VERSION = 'party-7';
const DEFAULT_ANALYSIS_INTERVAL_MS = 150;
const ANALYSIS_WIDTH = 640;   // default frame width sent for analysis
const PHOTO_WIDTH = 1280;     // width of the saved photo
const RANDOM_KIND = 'random';
const REVEAL_MS = 2200;       // new-challenge animation in the camera
const ROUND = {
  holdMs: 3000,     // challenge must stay met this long (the 3-2-1 countdown); set from the settings
  graceMs: 800,     // short detection gaps during the countdown are forgiven
};
const MENU_GROUPS = [   // dropdown sections by challenge family
  ['face', 'Look & Style'], ['expression', 'Grimassen'], ['hand', 'Hände'],
  ['drawing', 'Kreativ'], ['free', 'Kreativ'], ['debug', 'Debug'],
];
const ACCENT = '#ff6a13';
const INK = '#141414';
const BOX_COLORS = {
  success: ACCENT, muted: '#f4f4f4', lavender: '#9a9a9a', peach: ACCENT,
  blue: '#2f6fe0', red: '#e0243a', green: '#1f9d55', yellow: '#f2c200', white: '#ffffff', black: INK,
  pink: '#f25cb5', grey: '#9a9a9a', skeleton: ACCENT, joint: '#ffffff', tip: ACCENT,
};
const LIGHT_LABEL_TEXT = new Set(['black', 'blue', 'red', 'green']);   // dark fills need white text

const $ = selector => document.querySelector(selector);
const video = $('#video');
const canvas = $('#canvas');
const context = canvas.getContext('2d');
const boxesCanvas = $('#bounding-boxes');
const boxesContext = boxesCanvas.getContext('2d');
const cameraCard = $('.camera-card');
const challengeCard = $('.challenge-card');
const progress = $('#progress');
const countdownText = $('#countdown');
const stateChip = $('#state-chip');
const dot = $('#detection-dot');
const detectionLabel = $('#detection-label');
const challengeSelect = $('#challenge-select');
const debugPanel = $('#debug-panel');
const debugDetails = $('#debug-details');
const airClearButton = $('#air-clear');
const peopleProgress = $('#people-progress');
const drawGuide = $('#draw-guide');
const penStatus = $('#pen-status');

const sounds = new SoundBoard();
const airDrawing = new AirDrawing($('#air-canvas'));
airDrawing.onPenState = owned => {
  penStatus.textContent = owned ? 'Stift belegt – die malende Hand hat ihn' : 'Stift frei – Zeigefinger zeigen zum Übernehmen';
  penStatus.classList.toggle('owned', owned);
};
const countdown = new Countdown($('#countdown-overlay'), $('#countdown-number'), seconds => sounds.tick(seconds));
const photoViewer = new PhotoViewer($('#photo-viewer'));
const settingsPanel = new SettingsPanel($('#settings-panel'), applyTiming);
const solvedCounter = new SolvedCounter($('#solved-count'));

let publicShareUrl = null;
let challenge = null;
let selectedKind = RANDOM_KIND;
let analyzing = false;
// Round flow: searching -> countdown -> capturing -> viewing (full-screen photo) -> next challenge
const round = { phase: 'searching', countdownStartedAt: null, lastMetAt: 0, manual: false };

// ------------------------------------------------------------------ startup
// Warns when the page talks to an older server process than this app.js expects.
async function checkServerVersion() {
  const health = await fetch('/api/health').then(response => response.ok ? response.json() : {}).catch(() => ({}));
  if (health.version === EXPECTED_SERVER_VERSION) return true;
  showProblem('Server veraltet', 'Alter Server läuft noch – alle Python-Fenster schließen und neu starten');
  return false;
}

async function start() {
  try {
    if (!await checkServerVersion()) return;
    await fillChallengeMenu();
    await settingsPanel.load();
    await loadShareUrl();
    await loadNextChallenge();
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
    });
    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    updateStatus('Bereit', challenge.ready);
    scheduleNextAnalysis();
    requestAnimationFrame(tickRound);
  } catch (error) {
    showProblem('Kamera fehlt', 'Bitte Kamerazugriff erlauben und Seite neu laden');
    console.error(error);
  }
}

async function fillChallengeMenu() {
  const catalogue = (await (await fetch('/api/challenges')).json()).challenges;
  challengeSelect.add(new Option('Zufall – alle Challenges', RANDOM_KIND));
  const groups = new Map();
  for (const [family, label] of MENU_GROUPS) {
    if (!groups.has(label)) groups.set(label, document.createElement('optgroup'));
    groups.get(label).label = label;
  }
  for (const item of catalogue) {
    const label = (MENU_GROUPS.find(([family]) => family === item.family) || MENU_GROUPS[0])[1];
    groups.get(label).append(new Option(item.title, item.id));
  }
  for (const group of groups.values()) if (group.children.length) challengeSelect.append(group);
  challengeSelect.disabled = false;
}

// ---------------------------------------------------------------- challenge
async function loadNextChallenge() {
  const params = new URLSearchParams();
  if (selectedKind !== RANDOM_KIND) params.set('only', selectedKind);
  if (challenge) params.set('avoid', challenge.id);
  challenge = await (await fetch(`/api/challenge?${params}`)).json();
  airDrawing.stop();   // a new challenge starts with an empty canvas
  resetRound();
  applyChallenge();
  revealChallenge();
}

function applyChallenge() {
  $('#challenge-category').textContent = challenge.category || challenge.title;
  $('#challenge-title').textContent = challenge.title;
  $('#challenge-description').textContent = challenge.description;
  challengeCard.classList.remove('fresh');
  void challengeCard.offsetWidth;   // restart the entry animation
  challengeCard.classList.add('fresh');

  const autoCapture = isAutoCapture();
  // Debug and drawing modes show the whole camera frame (no cropping): nothing is hidden at the edges.
  cameraCard.classList.toggle('full-frame', Boolean(challenge.showAnalysedFrame || challenge.airDraw));
  cameraCard.classList.toggle('air-draw', Boolean(challenge.airDraw));
  // Free modes never take a photo on their own, so the countdown would only confuse.
  $('#countdown-bar').hidden = !autoCapture;
  $('#skip-challenge').hidden = !autoCapture && !challenge.stencilMode;
  // Challenge photos are taken automatically (they are the proof for the bar); free modes and the
  // stencil challenge (drawing has no natural end) are started by the guests.
  $('#manual-capture').hidden = !challenge.manualCapture;
  $('#manual-capture-label').textContent = challenge.stencilMode ? 'Masken aufsetzen' : 'Foto aufnehmen';
  airClearButton.hidden = !challenge.airDraw;
  renderPeopleProgress(autoCapture || challenge.stencilMode ? { met: 0, required: challenge.required } : null);
  drawGuide.hidden = !challenge.airDraw;
  $('#guide-erase').textContent = challenge.stencilMode
    ? 'löschen (über einer Schablone nur diese)'
    : 'alles löschen';
  if (challenge.airDraw) {
    if (!airDrawing.active) {
      airDrawing.fitResolution(cameraCard.clientWidth * (window.devicePixelRatio || 1), challenge.analysisWidth || ANALYSIS_WIDTH);
      airDrawing.useStencils(challenge.stencilMode ? challenge.required : 0);
    }
    airDrawing.start();
  } else {
    airDrawing.stop();
  }
  updateStatus('Bereit', challenge.ready);
}

// The new challenge is shown over the camera for a moment.
function revealChallenge() {
  const overlay = $('#challenge-reveal');
  $('#reveal-title').textContent = challenge.title;
  overlay.hidden = false;
  overlay.style.animation = 'none';
  void overlay.offsetWidth;   // restart the animation
  overlay.style.animation = '';
  sounds.whoosh();
  clearTimeout(revealChallenge.hideTimer);
  revealChallenge.hideTimer = setTimeout(() => { overlay.hidden = true; }, REVEAL_MS);
}

function isAutoCapture() {
  return challenge && challenge.autoCapture !== false && !challenge.stencilMode;
}

challengeSelect.addEventListener('change', async () => {
  selectedKind = challengeSelect.value;
  challengeSelect.blur();   // keyboard shortcuts should not change the selection afterwards
  await loadNextChallenge();
});

// ----------------------------------------------------------------- analysis
// Analyses run one after another; the pause between them depends on the challenge.
function scheduleNextAnalysis() {
  const interval = (challenge && challenge.analysisIntervalMs) || DEFAULT_ANALYSIS_INTERVAL_MS;
  setTimeout(async () => {
    await analyzeFrame();
    scheduleNextAnalysis();
  }, interval);
}

// Mirrored JPEG of the current camera frame. Photos may include the air drawing and a stamp.
function frameBlob(quality = .82, withDrawing = false, maxWidth = ANALYSIS_WIDTH, stamp = null) {
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  context.save(); context.scale(-1, 1); context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height); context.restore();
  if (withDrawing) airDrawing.compositeInto(context, canvas.width, canvas.height);
  if (stamp) drawStamp(context, canvas.width, canvas.height, stamp);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
}

async function analyzeFrame() {
  if (analyzing || !video.videoWidth || !challenge || round.phase === 'capturing' || round.phase === 'viewing') return;
  analyzing = true;
  const analysedChallenge = challenge;
  try {
    const sampleTime = performance.now();
    const body = await frameBlob(.72, false, challenge.analysisWidth || ANALYSIS_WIDTH);
    // Debug views show the analysed frame itself, so image and overlay always match.
    const snapshot = challenge.showAnalysedFrame ? await createImageBitmap(canvas) : null;
    const response = await fetch(`/api/analyze?challenge=${encodeURIComponent(challenge.id)}`, { method: 'POST', body, headers: { 'Content-Type': 'image/jpeg' } });
    const data = await response.json().catch(() => ({}));
    if (analysedChallenge !== challenge) return;  // the challenge changed while waiting
    if (!response.ok) {
      // The server answered, but the analysis failed: show its reason.
      showProblem('Fehler', data.error || `Analyse fehlgeschlagen (HTTP ${response.status})`);
      console.error('Analyse fehlgeschlagen:', data.error);
      return;
    }
    try {
      updateDetection(data, snapshot, sampleTime);
    } catch (error) {
      // Rendering failed although the server answered: say so instead of blaming the connection.
      showProblem('Fehler', `Anzeigefehler: ${error.message}`);
      console.error(error);
    }
  } catch (error) {
    showProblem('Offline', 'Erkennungsdienst nicht erreichbar – läuft der Python-Server?');
    console.error(error);
  } finally { analyzing = false; }
}

function updateDetection(data, snapshot = null, sampleTime = performance.now()) {
  drawBoxes(data, snapshot);
  drawLines(data);
  drawCircles(data);
  drawFrameCaption(data, snapshot);
  if (challenge.airDraw) airDrawing.update(data.pointer, data.frameWidth, data.frameHeight, sampleTime, data.faces);
  updateDebugPanel(data.debug);
  if (challenge.stencilMode) {
    showStencilStatus();
    return;
  }
  if (!isAutoCapture()) {
    updateStatus('Bereit', data.statusText || challenge.waiting);
    return;
  }
  renderPeopleProgress(data.progress);
  registerResult(Boolean(data.complete));
  cameraCard.classList.toggle('challenge-met', round.phase === 'countdown');
  const counting = round.phase === 'countdown';
  updateStatus(counting ? challenge.active : 'Bereit', counting ? 'Bleibt so – gleich wird fotografiert' : (data.statusText || challenge.waiting));
}

// Stencil challenge: progress of the drawing phase (the photo is started with the button).
function showStencilStatus() {
  if (round.phase !== 'searching') return;
  const required = challenge.required;
  const drawn = airDrawing.drawnStencilCount();
  const faces = airDrawing.visibleFaceCount();
  let label = `${drawn}/${required} Schablonen bemalt`;
  if (drawn >= required && faces >= required) label = 'Fertig? „Masken aufsetzen“ drücken';
  else if (drawn >= required) label += ` – holt noch ${required - faces} Person${required - faces > 1 ? 'en' : ''} ins Bild!`;
  else label += ' – malt die leeren Schablonen aus';
  renderPeopleProgress({ met: Math.min(drawn, required), required });
  updateStatus('Bereit', label);
}

function stencilChallengeDone() {
  return airDrawing.completedCount() >= challenge.required;
}

// ------------------------------------------------------------- round flow
function resetRound() {
  round.phase = 'searching';
  round.manual = false;
  round.countdownStartedAt = null;
  round.lastMetAt = 0;
  airDrawing.showMasks(false);
  if (challenge) drawGuide.hidden = !challenge.airDraw;
  countdown.hide();
  cameraCard.classList.remove('challenge-met');
  progress.style.width = '0%';
  countdownText.textContent = `${Math.round(ROUND.holdMs / 1000)} s`;
}

// One analysis result: start the countdown when met, keep it running through short gaps.
function registerResult(met) {
  const now = performance.now();
  if (met) round.lastMetAt = now;
  if (round.phase === 'searching' && met) {
    round.phase = 'countdown';
    round.countdownStartedAt = now;
  }
}

// Runs every animation frame so the countdown is smooth, independent of the analysis rate.
function tickRound() {
  if (round.phase === 'countdown') {
    const now = performance.now();
    if (!round.manual && now - round.lastMetAt > ROUND.graceMs) {
      resetRound();
    } else {
      const remaining = ROUND.holdMs - (now - round.countdownStartedAt);
      progress.style.width = `${Math.min(100, 100 - remaining / ROUND.holdMs * 100)}%`;
      countdownText.textContent = `${Math.max(0, remaining / 1000).toFixed(1)} s`;
      $('#challenge-reveal').hidden = true;   // the countdown has priority over the title card
      if (remaining > 0) countdown.show(remaining);
      else finishRound();
    }
  }
  requestAnimationFrame(tickRound);
}

async function finishRound() {
  round.phase = 'capturing';
  countdown.hide();
  cameraCard.classList.remove('challenge-met');
  const completed = challenge.stencilMode ? stencilChallengeDone() : true;
  const photo = await capturePhoto(completed);
  if (!photo) { resetRound(); return; }
  if (completed) solvedCounter.increment();
  showPhoto(photo, completed, loadNextChallenge);
}

// Full-screen photo; afterwards the photo is deleted and the booth continues.
function showPhoto(photo, completed, afterClose) {
  round.phase = 'viewing';
  updateStatus(completed ? 'Geschafft!' : 'Foto', 'Foto wird gleich wieder gelöscht', false);
  if (completed) sounds.fanfare();
  photoViewer.open(photo, { completed, shareUrl: publicShareUrl }, afterClose);
}

async function manualCapture() {
  if (round.phase === 'capturing' || round.phase === 'viewing' || $('#manual-capture').hidden) return;
  if (challenge.stencilMode) {
    startMaskCountdown();
    return;
  }
  round.phase = 'capturing';
  const photo = await capturePhoto(false);
  if (!photo) { resetRound(); return; }
  showPhoto(photo, false, async () => { resetRound(); applyChallenge(); });
}

// Stencils disappear, the masks sit on the real faces, and a normal 3-2-1 countdown runs.
function startMaskCountdown() {
  if (round.phase !== 'searching') return;
  airDrawing.showMasks(true);
  round.phase = 'countdown';
  round.manual = true;   // no detection has to stay "met" during this countdown
  round.countdownStartedAt = performance.now();
  cameraCard.classList.add('challenge-met');
  drawGuide.hidden = true;   // nothing to draw now, the guide would only cover faces
  updateStatus('Masken auf', 'Bleibt so – gleich wird fotografiert', false);
}

// Settings changed (or loaded): the countdown length comes from the server settings.
function applyTiming(values) {
  ROUND.holdMs = values.countdown_seconds * 1000;
  if (round.phase === 'searching') countdownText.textContent = `${Math.round(ROUND.holdMs / 1000)} s`;
}

function skipChallenge() {
  if (round.phase === 'searching' || round.phase === 'countdown') loadNextChallenge();
}

// ------------------------------------------------------------------ photos
async function capturePhoto(completed = false) {
  if (!video.videoWidth) return null;
  try {
    const stamp = { title: challenge.title, completed, time: new Date() };
    const body = await frameBlob(.92, true, PHOTO_WIDTH, stamp);
    const response = await fetch('/api/captures', { method: 'POST', body, headers: { 'Content-Type': 'image/jpeg' } });
    if (!response.ok) throw new Error('Speichern fehlgeschlagen');
    const photo = await response.json();
    photo.title = challenge.title;
    sounds.shutter();
    $('#flash').classList.add('show');
    setTimeout(() => $('#flash').classList.remove('show'), 550);
    return photo;
  } catch (error) {
    showProblem('Fehler', 'Das Foto konnte nicht gespeichert werden.');
    return null;
  }
}

// Banner burnt into the photo: which challenge was solved and when – the proof shown at the bar.
function drawStamp(context, width, height, { title, completed, time }) {
  const scale = width / 1280;
  const barHeight = Math.round(84 * scale);
  const top = height - barHeight;
  const middle = top + barHeight / 2;
  context.fillStyle = INK;
  context.fillRect(0, top, width, barHeight);
  context.fillStyle = ACCENT;
  context.fillRect(26 * scale, middle - 8 * scale, 16 * scale, 16 * scale);
  context.textBaseline = 'middle';
  context.fillStyle = '#ffffff';
  context.font = `700 ${Math.round(30 * scale)}px "Tektur", "Arial Black", sans-serif`;
  context.fillText(title.toUpperCase(), 60 * scale, middle, width * 0.62);
  const clock = time.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
  context.font = `${Math.round(22 * scale)}px "IBM Plex Mono", Consolas, monospace`;
  context.textAlign = 'right';
  context.fillStyle = completed ? ACCENT : '#bdbdbd';
  context.fillText(`${completed ? 'GESCHAFFT' : 'FOTO'} · photobooTH · ${clock}`, width - 26 * scale, middle);
  context.textAlign = 'left';
  context.textBaseline = 'alphabetic';
}

async function loadShareUrl() {
  const share = await fetch('/api/share').then(response => response.json()).catch(() => ({}));
  publicShareUrl = share.url || null;
}

// ----------------------------------------------------------------- overlays
function renderPeopleProgress(state) {
  peopleProgress.replaceChildren();
  peopleProgress.hidden = !state || state.required < 2;
  if (peopleProgress.hidden) return;
  for (let index = 0; index < state.required; index += 1) {
    const person = document.createElement('span');
    const met = index < state.met;
    person.className = met ? 'person met' : 'person';
    person.textContent = index + 1;
    peopleProgress.append(person);
  }
}

function drawBoxes(data, snapshot = null) {
  if (!data.frameWidth || !data.frameHeight) return;
  boxesCanvas.width = data.frameWidth;
  boxesCanvas.height = data.frameHeight;
  boxesContext.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
  if (snapshot) boxesContext.drawImage(snapshot, 0, 0, boxesCanvas.width, boxesCanvas.height);
  for (const box of data.boxes || []) drawBox(box);
}

// Overlay sizes are defined for a 640 px wide frame and scale with the actual frame.
function overlayUnit() {
  return boxesCanvas.width / 640;
}

// Thin corner brackets with a label and an optional score bar (1.0 = threshold).
function drawBox(box) {
  const unit = overlayUnit();
  const color = BOX_COLORS[box.color] || '#ffffff';
  const corner = Math.max(10 * unit, Math.min(box.width, box.height) * 0.22);
  const { x, y, width: w, height: h } = box;
  boxesContext.save();
  boxesContext.strokeStyle = box.passed ? color : '#ffffff';
  boxesContext.lineWidth = (box.passed ? 4 : 2) * unit;
  boxesContext.lineCap = 'square';
  boxesContext.beginPath();
  for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
    boxesContext.moveTo(cx + dx * corner, cy);
    boxesContext.lineTo(cx, cy);
    boxesContext.lineTo(cx, cy + dy * corner);
  }
  boxesContext.stroke();
  boxesContext.restore();

  const hasMeter = box.meter !== undefined;
  const labelHeight = 28 * unit;
  boxesContext.font = `700 ${Math.round(16 * unit)}px "IBM Plex Mono", Consolas, monospace`;
  const text = box.label.toUpperCase();
  const labelWidth = Math.max(boxesContext.measureText(text).width + 20 * unit, hasMeter ? 120 * unit : 0);
  const labelTop = Math.max(0, y - labelHeight - (hasMeter ? 10 * unit : 4 * unit));
  boxesContext.fillStyle = box.passed ? color : '#f4f4f4';
  boxesContext.fillRect(x, labelTop, labelWidth, labelHeight);
  boxesContext.fillStyle = box.passed && LIGHT_LABEL_TEXT.has(box.color) ? '#ffffff' : INK;
  boxesContext.fillText(text, x + 10 * unit, labelTop + 19 * unit);
  if (hasMeter) drawMeter(x, labelTop + labelHeight, labelWidth, box.meter, box.passed);
}

// Score bar under a label; the dark tick marks the threshold.
function drawMeter(x, y, width, meter, passed) {
  const maxMeter = 1.6;
  const unit = overlayUnit();
  const height = 6 * unit;
  boxesContext.fillStyle = '#d4d4d4';
  boxesContext.fillRect(x, y, width, height);
  boxesContext.fillStyle = passed ? ACCENT : '#6d6d6d';
  boxesContext.fillRect(x, y, width * Math.min(meter, maxMeter) / maxMeter, height);
  boxesContext.fillStyle = INK;
  boxesContext.fillRect(x + width / maxMeter - unit, y - 2 * unit, 2 * unit, height + 4 * unit);
}

// Writes the result of the analysed frame into the debug image itself.
function drawFrameCaption(data, snapshot) {
  if (!snapshot || !data.debug) return;
  const text = `Dieses Bild: ${(data.debug.hands || []).length} Hand/Hände · ${data.debug.totalFingers} Finger`;
  const unit = overlayUnit();
  boxesContext.font = `700 ${Math.round(22 * unit)}px ui-sans-serif, system-ui`;
  const width = boxesContext.measureText(text).width + 24 * unit;
  boxesContext.fillStyle = 'rgba(0, 0, 0, .65)';
  const left = (boxesCanvas.width - width) / 2;
  boxesContext.fillRect(left, boxesCanvas.height - 50 * unit, width, 38 * unit);
  boxesContext.fillStyle = '#ffffff';
  boxesContext.fillText(text, left + 12 * unit, boxesCanvas.height - 24 * unit);
}

// Draws the hand skeleton (bones between landmarks) on the overlay canvas.
function drawLines(data) {
  boxesContext.lineWidth = 5 * overlayUnit();
  boxesContext.lineCap = 'round';
  for (const line of data.lines || []) {
    boxesContext.strokeStyle = BOX_COLORS[line.color] || '#ffffff';
    boxesContext.beginPath();
    boxesContext.moveTo(line.x1, line.y1);
    boxesContext.lineTo(line.x2, line.y2);
    boxesContext.stroke();
  }
}

// Draws landmark and fingertip dots on top of the boxes (same canvas).
function drawCircles(data) {
  for (const circle of data.circles || []) {
    const color = BOX_COLORS[circle.color] || '#ffffff';
    boxesContext.beginPath();
    boxesContext.arc(circle.x, circle.y, circle.radius, 0, 2 * Math.PI);
    if (circle.filled) {
      boxesContext.fillStyle = color;
      boxesContext.fill();
    } else {
      boxesContext.strokeStyle = color;
      boxesContext.lineWidth = 2;
      boxesContext.stroke();
    }
  }
}

function updateStatus(chip, label, active = true) {
  stateChip.classList.remove('warn');
  stateChip.textContent = chip;
  detectionLabel.textContent = label;
  dot.classList.toggle('detected', Boolean(active && challenge && chip === challenge.active));
}

function showProblem(chip, label) {
  updateStatus(chip, label);
  stateChip.classList.add('warn');
}

// ------------------------------------------------------------------- debug
// Rolling statistics over the last frames, like the command-line self-check.
const DEBUG_WINDOW = 10;
let debugHistory = [];

function recordDebugFrame(debug) {
  debugHistory.push({ hasHand: (debug.hands || []).length > 0, analysisMs: debug.analysisMs || 0, at: performance.now() });
  if (debugHistory.length > DEBUG_WINDOW) debugHistory.shift();
}

function debugStatistics() {
  const frames = debugHistory.length;
  const withHand = debugHistory.filter(frame => frame.hasHand).length;
  const averageMs = debugHistory.reduce((sum, frame) => sum + frame.analysisMs, 0) / Math.max(frames, 1);
  const seconds = frames > 1 ? (debugHistory[frames - 1].at - debugHistory[0].at) / 1000 : 0;
  const fps = seconds > 0 ? (frames - 1) / seconds : 0;
  return `Letzte ${frames} Bilder: Hand in ${withHand}/${frames} · Ø ${averageMs.toFixed(0)} ms Analyse · ${fps.toFixed(1)} Bilder/s`;
}

// Shows per-hand numbers and detection statistics while a debug challenge is active.
function updateDebugPanel(debug) {
  debugPanel.hidden = !debug;
  if (!debug) { debugHistory = []; return; }
  recordDebugFrame(debug);
  const perHand = (debug.hands || []).map((hand, index) => {
    const fingerNames = hand.fingerNames || [];
    const names = fingerNames.length ? fingerNames.join(', ') : 'Faust';
    return `Hand ${index + 1}: ${hand.fingers} Finger – ${names} (Sicherheit ${Math.round(hand.confidence * 100)} %)`;
  });
  const handText = perHand.length ? perHand.join('\n') : 'Keine Hand erkannt.';
  debugDetails.textContent = `${handText}\n\n${debugStatistics()}`;
}

// ------------------------------------------------------------------ controls
function toggleSound() {
  $('#sound-button').classList.toggle('off', !sounds.toggle());
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
}

// Keyboard shortcuts for whoever runs the booth: N = next challenge, Space = photo / continue,
// F = full screen, M = sound on/off, Esc = close panels.
document.addEventListener('keydown', event => {
  if (event.target.closest('input, select, textarea') || event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === 'escape') { settingsPanel.close(); if (photoViewer.isOpen) photoViewer.close(); return; }
  if (photoViewer.isOpen) {
    if (key === ' ' || key === 'enter') { event.preventDefault(); photoViewer.close(); }
    return;
  }
  if (key === 'n' || key === 'arrowright') skipChallenge();
  else if (key === ' ' || key === 'enter') { event.preventDefault(); manualCapture(); }
  else if (key === 'f') toggleFullscreen();
  else if (key === 'm') toggleSound();
});

$('#sound-button').classList.toggle('off', !sounds.enabled);
$('#sound-button').addEventListener('click', toggleSound);
$('#fullscreen-button').addEventListener('click', toggleFullscreen);
$('#skip-challenge').addEventListener('click', skipChallenge);
$('#manual-capture').addEventListener('click', manualCapture);
$('#settings-button').addEventListener('click', () => settingsPanel.open());
airClearButton.addEventListener('click', () => airDrawing.clear());
start();
