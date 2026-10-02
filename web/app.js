// Photo booth front end: camera, analysis loop, round flow and overlays.

const EXPECTED_SERVER_VERSION = 'party-6';
const DEFAULT_ANALYSIS_INTERVAL_MS = 180;
const ANALYSIS_WIDTH = 640;   // frame width sent for analysis
const PHOTO_WIDTH = 1280;     // width of the saved photo
const RANDOM_KIND = 'random';
const ROUND = {
  holdMs: 3000,     // challenge must stay met this long (the 3-2-1 countdown); set from the settings
  graceMs: 700,     // short detection gaps during the countdown are forgiven
};
const BOX_COLORS = {
  lavender: '#b9a1e9', rose: '#ed9eb2', sage: '#75c58b', peach: '#f3a97d', blue: '#6da9ec', red: '#ed7f83',
  green: '#75c58b', yellow: '#e8c95b', white: '#ffffff', black: '#393234', skeleton: '#9b30ff', joint: '#ffffff',
  tip: '#00e676', success: '#39d98a', muted: '#d8ccd0',
};

const video = document.querySelector('#video');
const canvas = document.querySelector('#canvas');
const context = canvas.getContext('2d');
const boxesCanvas = document.querySelector('#bounding-boxes');
const boxesContext = boxesCanvas.getContext('2d');
const cameraCard = document.querySelector('.camera-card');
const progress = document.querySelector('#progress');
const countdownText = document.querySelector('#countdown');
const stateChip = document.querySelector('#state-chip');
const dot = document.querySelector('#detection-dot');
const detectionLabel = document.querySelector('#detection-label');
const challengeSelect = document.querySelector('#challenge-select');
const shareCard = document.querySelector('#share-card');
const debugPanel = document.querySelector('#debug-panel');
const debugDetails = document.querySelector('#debug-details');
const airClearButton = document.querySelector('#air-clear');
const peopleProgress = document.querySelector('#people-progress');
const airDrawing = new AirDrawing(document.querySelector('#air-canvas'));
const drawGuide = document.querySelector('#draw-guide');
const penStatus = document.querySelector('#pen-status');
airDrawing.onPenState = owned => {
  penStatus.textContent = owned ? 'Stift: belegt – die malende Hand hat ihn' : 'Stift: frei – Zeigefinger zeigen zum Übernehmen';
  penStatus.classList.toggle('owned', owned);
};
const countdown = new Countdown(document.querySelector('#countdown-overlay'), document.querySelector('#countdown-number'));
const photoViewer = new PhotoViewer(document.querySelector('#photo-viewer'));
const settingsPanel = new SettingsPanel(document.querySelector('#settings-panel'), applyTiming);

let publicShareUrl = null;
let challenge = null;
let selectedKind = RANDOM_KIND;
let analyzing = false;
// Round flow: searching -> countdown -> capturing -> viewing (full-screen photo) -> next challenge
const round = { phase: 'searching', countdownStartedAt: null, lastMetAt: 0 };

// ------------------------------------------------------------------ startup
// Warns when the page talks to an older server process than this app.js expects.
async function checkServerVersion() {
  const health = await fetch('/api/health').then(response => response.ok ? response.json() : {}).catch(() => ({}));
  if (health.version === EXPECTED_SERVER_VERSION) return true;
  updateStatus('Server veraltet', 'Alter Server läuft noch – alle Python-Fenster schließen und neu starten', false);
  return false;
}

async function startCamera() {
  try {
    if (!await checkServerVersion()) return;
    const catalogue = (await (await fetch('/api/challenges')).json()).challenges;
    challengeSelect.add(new Option('Zufall – alle Challenges', RANDOM_KIND));
    for (const item of catalogue) challengeSelect.add(new Option(item.title, item.id));
    challengeSelect.disabled = false;
    await settingsPanel.load();
    await loadNextChallenge();
    await loadShareCard();
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    await new Promise(resolve => video.onloadedmetadata = resolve);
    updateStatus('Bereit', challenge.ready);
    scheduleNextAnalysis();
    requestAnimationFrame(tickRound);
  } catch (error) {
    updateStatus('Kamera fehlt', 'Bitte erlaube den Kamerazugriff', false);
    console.error(error);
  }
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
}

function applyChallenge() {
  document.querySelector('#challenge-title').textContent = challenge.title;
  document.querySelector('#challenge-description').textContent = challenge.description;
  document.querySelector('#booth-title').textContent = challenge.boothName;
  document.title = challenge.boothName;
  const autoCapture = isAutoCapture();
  // Debug and drawing modes show the whole camera frame (no cropping): nothing is hidden at the edges.
  cameraCard.classList.toggle('full-frame', Boolean(challenge.showAnalysedFrame || challenge.airDraw));
  cameraCard.classList.toggle('air-draw', Boolean(challenge.airDraw));
  // Free modes never take a photo on their own, so the countdown would only confuse.
  document.querySelector('.progress-info').hidden = !autoCapture;
  document.querySelector('.progress-track').hidden = !autoCapture;
  document.querySelector('#skip-challenge').hidden = !autoCapture;
  // Challenge photos are taken automatically (they are the proof for the bar); free modes and the
  // stencil challenge (drawing has no natural end) are started by the guests.
  document.querySelector('#manual-capture').hidden = !challenge.manualCapture;
  document.querySelector('#manual-capture-label').textContent = challenge.stencilMode ? 'Masken aufsetzen & Foto' : 'Foto jetzt aufnehmen';
  airClearButton.hidden = !challenge.airDraw;
  renderPeopleProgress(autoCapture ? { met: 0, required: challenge.required } : null);
  drawGuide.hidden = !challenge.airDraw;
  document.querySelector('#guide-erase').textContent = challenge.stencilMode
    ? 'Hand offen 1 s halten: löschen – über einer Schablone nur diese'
    : 'Hand offen 1 s halten: alles löschen';
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

function isAutoCapture() {
  return challenge && challenge.autoCapture !== false;
}

challengeSelect.addEventListener('change', async () => {
  selectedKind = challengeSelect.value;
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
    const body = await frameBlob(.68, false, challenge.analysisWidth || ANALYSIS_WIDTH);
    // Debug views show the analysed frame itself, so image and overlay always match.
    const snapshot = challenge.showAnalysedFrame ? await createImageBitmap(canvas) : null;
    const response = await fetch(`/api/analyze?challenge=${encodeURIComponent(challenge.id)}`, { method: 'POST', body, headers: { 'Content-Type': 'image/jpeg' } });
    const data = await response.json().catch(() => ({}));
    if (analysedChallenge !== challenge) return;  // the challenge changed while waiting
    if (!response.ok) {
      // The server answered, but the analysis failed: show its reason.
      updateStatus('Fehler', data.error || `Analyse fehlgeschlagen (HTTP ${response.status})`, false);
      console.error('Analyse fehlgeschlagen:', data.error);
      return;
    }
    try {
      updateDetection(data, snapshot, sampleTime);
    } catch (error) {
      // Rendering failed although the server answered: say so instead of blaming the connection.
      updateStatus('Fehler', `Anzeigefehler: ${error.message}`, false);
      console.error(error);
    }
  } catch (error) {
    updateStatus('Verbindung fehlt', 'Lokaler Erkennungsdienst nicht erreichbar', false);
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
  const chip = round.phase === 'countdown' ? challenge.active : 'Bereit';
  const label = round.phase === 'countdown' ? 'Bleibt so – gleich wird fotografiert!' : (data.statusText || challenge.waiting);
  updateStatus(chip, label);
}

// Stencil challenge: progress of the drawing phase (the photo is started with the button).
function showStencilStatus() {
  if (round.phase !== 'searching') return;
  const required = challenge.required;
  const drawn = airDrawing.drawnStencilCount();
  const faces = airDrawing.visibleFaceCount();
  let label = `${drawn}/${required} Schablonen bemalt`;
  if (drawn >= required && faces >= required) label = 'Fertig? „Masken aufsetzen & Foto“ drücken!';
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
  airDrawing.showMasks(false);
  if (challenge) drawGuide.hidden = !challenge.airDraw;
  round.countdownStartedAt = null;
  round.lastMetAt = 0;
  countdown.hide();
  cameraCard.classList.remove('challenge-met');
  progress.style.width = '0%';
  countdownText.textContent = `${(ROUND.holdMs / 1000).toFixed(1)} s`;
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
  showPhoto(photo, completed, loadNextChallenge);
}

// Full-screen photo; afterwards the photo is deleted and the booth continues.
function showPhoto(photo, completed, afterClose) {
  round.phase = 'viewing';
  updateStatus(completed ? 'Geschafft!' : 'Foto', 'Foto wird gleich wieder gelöscht', false);
  photoViewer.open(photo, { completed, shareUrl: publicShareUrl }, afterClose);
}

async function manualCapture() {
  if (round.phase === 'capturing' || round.phase === 'viewing') return;
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
  updateStatus('Masken auf!', 'Bleibt so – gleich wird fotografiert!', false);
}

// Settings changed (or loaded): the countdown length comes from the server settings.
function applyTiming(values) {
  ROUND.holdMs = values.countdown_seconds * 1000;
  if (round.phase === 'searching') countdownText.textContent = `${(ROUND.holdMs / 1000).toFixed(1)} s`;
}

document.querySelector('#skip-challenge').addEventListener('click', () => loadNextChallenge());

// ------------------------------------------------------------------ photos
async function capturePhoto(completed = false) {
  if (!video.videoWidth) return null;
  try {
    const stamp = { title: completed ? `✓ ${challenge.title}` : challenge.title, time: new Date() };
    const body = await frameBlob(.92, true, PHOTO_WIDTH, stamp);
    const response = await fetch('/api/captures', { method: 'POST', body, headers: { 'Content-Type': 'image/jpeg' } });
    if (!response.ok) throw new Error('Speichern fehlgeschlagen');
    const photo = await response.json();
    photo.title = stamp.title;
    document.querySelector('#flash').classList.add('show');
    setTimeout(() => document.querySelector('#flash').classList.remove('show'), 500);
    return photo;
  } catch (error) {
    updateStatus('Fehler', 'Das Foto konnte nicht gespeichert werden.', false);
    return null;
  }
}

// Banner burnt into the photo: which challenge was solved and when – the proof shown at the bar.
function drawStamp(context, width, height, { title, time }) {
  const scale = width / 1280;
  const barHeight = Math.round(86 * scale);
  const gradient = context.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, 'rgba(20, 8, 36, .88)');
  gradient.addColorStop(1, 'rgba(60, 12, 70, .88)');
  context.fillStyle = gradient;
  context.fillRect(0, height - barHeight, width, barHeight);
  context.fillStyle = '#ffffff';
  context.font = `700 ${Math.round(34 * scale)}px Georgia, serif`;
  context.fillText(title, 28 * scale, height - barHeight / 2 + 12 * scale, width * 0.72);
  const clock = time.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
  context.font = `600 ${Math.round(24 * scale)}px ui-sans-serif, system-ui`;
  context.textAlign = 'right';
  context.fillStyle = '#ffd23f';
  context.fillText(`photobooTH · ${clock}`, width - 28 * scale, height - barHeight / 2 + 9 * scale);
  context.textAlign = 'left';
}

// ------------------------------------------------------------------ sharing
async function loadShareCard() {
  const response = await fetch('/api/share');
  const share = await response.json();
  if (!share.url) return;
  publicShareUrl = share.url;
  document.querySelector('#share-eyebrow').textContent = 'FOTO-SHARING BEREIT';
  document.querySelector('#share-title').textContent = 'Nimm ein Foto auf.';
  document.querySelector('#share-description').textContent = 'Nach jeder Aufnahme erscheint beim Foto ein QR-Code zum Download.';
  document.querySelector('#share-url').hidden = true;
  shareCard.hidden = false;
}

// ----------------------------------------------------------------- drawing
function renderPeopleProgress(state) {
  peopleProgress.replaceChildren();
  peopleProgress.hidden = !state || state.required < 2;
  if (peopleProgress.hidden) return;
  for (let index = 0; index < state.required; index += 1) {
    const person = document.createElement('span');
    person.className = index < state.met ? 'person met' : 'person';
    peopleProgress.append(person);
  }
}

// HUD-style corner brackets with a label and an optional score meter (1.0 = threshold).
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

function drawBox(box) {
  const unit = overlayUnit();
  const color = BOX_COLORS[box.color] || '#ffffff';
  const corner = Math.max(8 * unit, Math.min(box.width, box.height) * 0.25);
  boxesContext.save();
  boxesContext.strokeStyle = color;
  boxesContext.lineWidth = 4 * unit;
  boxesContext.lineCap = 'round';
  if (box.passed) { boxesContext.shadowColor = color; boxesContext.shadowBlur = 12; }
  const { x, y, width: w, height: h } = box;
  boxesContext.beginPath();
  for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
    boxesContext.moveTo(cx + dx * corner, cy);
    boxesContext.lineTo(cx, cy);
    boxesContext.lineTo(cx, cy + dy * corner);
  }
  boxesContext.stroke();
  boxesContext.restore();

  const hasMeter = box.meter !== undefined;
  const labelHeight = 30 * unit;
  boxesContext.font = `700 ${Math.round(20 * unit)}px ui-sans-serif, system-ui`;
  const labelWidth = Math.max(boxesContext.measureText(box.label).width + 18 * unit, hasMeter ? 120 * unit : 0);
  const labelTop = Math.max(0, y - labelHeight - (hasMeter ? 10 * unit : 0));
  boxesContext.fillStyle = color;
  boxesContext.fillRect(x, labelTop, labelWidth, labelHeight);
  boxesContext.fillStyle = box.color === 'black' ? '#ffffff' : '#2d2426';
  boxesContext.fillText(box.label, x + 9 * unit, labelTop + 21 * unit);
  if (hasMeter) drawMeter(x, labelTop + labelHeight + 2 * unit, labelWidth, box.meter, box.passed);
}

// Score bar under a label; the white tick marks the threshold.
function drawMeter(x, y, width, meter, passed) {
  const maxMeter = 1.5;
  const unit = overlayUnit();
  boxesContext.fillStyle = 'rgba(20, 16, 24, .75)';
  boxesContext.fillRect(x, y, width, 8 * unit);
  boxesContext.fillStyle = passed ? '#b6ffd9' : '#ffb27a';
  boxesContext.fillRect(x, y, width * Math.min(meter, maxMeter) / maxMeter, 8 * unit);
  boxesContext.fillStyle = '#ffffff';
  boxesContext.fillRect(x + width / maxMeter - unit, y - 2 * unit, 3 * unit, 12 * unit);
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

function updateStatus(chip, label, active = true) {
  stateChip.textContent = chip;
  detectionLabel.textContent = label;
  dot.classList.toggle('detected', Boolean(active && challenge && chip === challenge.active));
}

document.querySelector('#manual-capture').addEventListener('click', manualCapture);
document.querySelector('#settings-button').addEventListener('click', () => settingsPanel.open());
airClearButton.addEventListener('click', () => airDrawing.clear());
startCamera();
