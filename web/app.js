const HOLD_MS = 3000;
const video = document.querySelector('#video');
const canvas = document.querySelector('#canvas');
const context = canvas.getContext('2d');
const boxesCanvas = document.querySelector('#bounding-boxes');
const boxesContext = boxesCanvas.getContext('2d');
const progress = document.querySelector('#progress');
const countdown = document.querySelector('#countdown');
const stateChip = document.querySelector('#state-chip');
const dot = document.querySelector('#detection-dot');
const detectionLabel = document.querySelector('#detection-label');
const challengeSelect = document.querySelector('#challenge-select');
const shareCard = document.querySelector('#share-card');
let publicShareUrl = null;
let challenge = null;
let challenges = [];
let challengeSince = null;
let armed = true;
let analyzing = false;

async function startCamera() {
  try {
    const [selectedResponse, choicesResponse] = await Promise.all([fetch('/api/challenge'), fetch('/api/challenges')]);
    challenge = await selectedResponse.json();
    challenges = (await choicesResponse.json()).challenges;
    for (const item of challenges) challengeSelect.add(new Option(item.title, item.id));
    challengeSelect.value = challenge.id;
    challengeSelect.disabled = false;
    applyChallenge();
    await loadShareCard();
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 960 } }, audio: false });
    await new Promise(resolve => video.onloadedmetadata = resolve);
    updateStatus('Bereit', challenge.ready);
    setInterval(analyzeFrame, 180);
  } catch (error) {
    updateStatus('Kamera fehlt', 'Bitte erlaube den Kamerazugriff', false);
    console.error(error);
  }
}

async function loadShareCard() {
  const response = await fetch('/api/share');
  const share = await response.json();
  if (!share.url) return;
  publicShareUrl = share.url;
  document.querySelector('#share-eyebrow').textContent = 'FOTO-SHARING BEREIT';
  document.querySelector('#share-title').textContent = 'Nimm ein Foto auf.';
  document.querySelector('#share-description').textContent = 'Danach erscheint hier ein QR-Code für den separaten Foto-Download.';
  document.querySelector('#share-url').hidden = true;
  shareCard.hidden = false;
}

function showPhotoShare(photo) {
  if (!publicShareUrl) return;
  const photoUrl = `${publicShareUrl}/photo/${encodeURIComponent(photo.filename)}`;
  const qr = document.querySelector('#share-qr');
  qr.src = `/api/share/qr?photo=${encodeURIComponent(photo.filename)}`;
  qr.hidden = false;
  document.querySelector('#share-eyebrow').textContent = 'FOTO AUFS HANDY';
  document.querySelector('#share-title').textContent = 'Scan & save.';
  document.querySelector('#share-description').textContent = 'Der QR-Code öffnet dieses Foto über einen temporären Link.';
  const link = document.querySelector('#share-url');
  link.href = photoUrl;
  link.textContent = photoUrl;
  link.hidden = false;
  shareCard.hidden = false;
}

function applyChallenge() {
  document.querySelector('#challenge-title').textContent = challenge.title;
  document.querySelector('#challenge-description').textContent = challenge.description;
  document.querySelector('#booth-title').textContent = challenge.boothName;
  document.title = challenge.boothName;
}

challengeSelect.addEventListener('change', () => {
  challenge = challenges.find(item => item.id === challengeSelect.value);
  challengeSince = null; armed = true;
  progress.style.width = '0%'; countdown.textContent = '3.0 s';
  applyChallenge();
  updateStatus('Bereit', challenge.ready);
});

function frameBlob(quality = .82) {
  const maxWidth = 640;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  context.save(); context.scale(-1, 1); context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height); context.restore();
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
}

async function analyzeFrame() {
  if (analyzing || !video.videoWidth) return;
  analyzing = true;
  try {
    const response = await fetch(`/api/analyze?challenge=${encodeURIComponent(challenge.id)}`, { method: 'POST', body: await frameBlob(.68), headers: { 'Content-Type': 'image/jpeg' } });
    if (!response.ok) throw new Error('Analyse fehlgeschlagen');
    const data = await response.json();
    updateDetection(data);
  } catch (error) {
    updateStatus('Verbindung fehlt', 'Lokaler Erkennungsdienst nicht erreichbar', false);
  } finally { analyzing = false; }
}

function updateDetection(data) {
  drawBoxes(data);
  if (!data.complete) {
    challengeSince = null; armed = true; progress.style.width = '0%'; countdown.textContent = '3.0 s';
    const noFace = data.faceCount === 0;
    const waiting = challenge.id !== 'group' && noFace ? 'Warte auf ein Gesicht' : challenge.waiting;
    updateStatus('Bereit', waiting);
    return;
  }
  dot.classList.add('detected');
  if (!armed) return updateStatus('Gespeichert', 'Wunderbar! Verlasse kurz das Bild für ein weiteres Foto.');
  if (!challengeSince) challengeSince = performance.now();
  const remaining = Math.max(0, HOLD_MS - (performance.now() - challengeSince));
  progress.style.width = `${100 - remaining / HOLD_MS * 100}%`;
  countdown.textContent = `${(remaining / 1000).toFixed(1)} s`;
  updateStatus(challenge.active, 'Bleib noch einen kleinen Moment so.');
  if (!remaining) { armed = false; challengeSince = null; capturePhoto(); }
}

function drawBoxes(data) {
  if (!data.frameWidth || !data.frameHeight) return;
  boxesCanvas.width = data.frameWidth;
  boxesCanvas.height = data.frameHeight;
  boxesContext.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
  const colors = { lavender: '#b9a1e9', rose: '#ed9eb2', sage: '#75c58b', peach: '#f3a97d' };
  boxesContext.font = '600 15px ui-sans-serif, system-ui';
  for (const box of data.boxes || []) {
    const color = colors[box.color] || '#ffffff';
    boxesContext.strokeStyle = color;
    boxesContext.lineWidth = 3;
    boxesContext.strokeRect(box.x, box.y, box.width, box.height);
    const textWidth = boxesContext.measureText(box.label).width;
    boxesContext.fillStyle = color;
    boxesContext.fillRect(box.x, Math.max(0, box.y - 23), textWidth + 14, 23);
    boxesContext.fillStyle = '#433738';
    boxesContext.fillText(box.label, box.x + 7, Math.max(16, box.y - 7));
  }
}

function updateStatus(chip, label, active = true) { stateChip.textContent = chip; detectionLabel.textContent = label; dot.classList.toggle('detected', active && challenge && chip === challenge.active); }

async function capturePhoto() {
  if (!video.videoWidth) return;
  try {
    const response = await fetch('/api/captures', { method: 'POST', body: await frameBlob(.92), headers: { 'Content-Type': 'image/jpeg' } });
    if (!response.ok) throw new Error('Speichern fehlgeschlagen');
    const photo = await response.json();
    addToGallery(photo.url, photo.filename);
    showPhotoShare(photo);
    document.querySelector('#flash').classList.add('show');
    setTimeout(() => document.querySelector('#flash').classList.remove('show'), 500);
    updateStatus('Gespeichert', 'Dein Foto wurde lokal gespeichert.');
  } catch (error) { updateStatus('Fehler', 'Das Foto konnte nicht gespeichert werden.', false); }
}

function addToGallery(url, filename) {
  document.querySelector('.empty-gallery')?.remove();
  const element = document.querySelector('#photo-template').content.firstElementChild.cloneNode(true);
  element.href = url; element.querySelector('img').src = url; element.querySelector('img').alt = filename;
  document.querySelector('#gallery').prepend(element);
}

document.querySelector('#manual-capture').addEventListener('click', capturePhoto);
startCamera();
