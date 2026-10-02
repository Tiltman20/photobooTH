// Futuristic air drawing: neon strokes follow the index fingertip.
//
// Strokes are drawn as smooth Catmull-Rom curves *through the measured
// fingertip positions*, so fast movements become curves instead of straight
// lines. A long pause or a big jump between two samples starts a new stroke
// rather than bridging the gap with a straight line.
//
// With a StencilBoard attached (stencil challenge), strokes drawn on a face
// template are stored in face coordinates and worn by the matching real face.

const AIR_DRAW = {
  renderScale: 2,          // canvas pixels per frame pixel, for crisp lines
  cursorEasing: 0.45,      // smoothing of the HUD cursor (not of the line)
  votesToStart: 2,         // equal readings in a row to put the pen down
  votesToStop: 3,          // … and to lift it (tolerates a flickering frame)
  maxGapMs: 350,           // longer without a sample: new stroke instead of a connecting line
  maxJump: 0.35,           // bigger jump (share of frame width) between samples: new stroke
  clearHoldMs: 800,        // open hand must be held this long to erase
  staleAfterMs: 600,       // no update for this long = hand lost
  coreWidth: 3.5,
  glowWidth: 12,
  maxSparks: 120,
};

class AirDrawing {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.layer = document.createElement('canvas');      // free strokes, drawn once
    this.layerContext = this.layer.getContext('2d');
    this.artwork = document.createElement('canvas');    // free strokes + drawings on faces = what goes into the photo
    this.artworkContext = this.artwork.getContext('2d');
    this.board = null;
    this.masksOnFaces = false;   // stencil mode: drawing phase shows stencils only, photo phase shows the masks
    this.active = false;
    this.frameWidth = 0;
    this.onPenState = null;   // callback(penOwned) for the on-screen guide
    this.reset();
  }

  reset() {
    this.cursor = null;          // eased cursor position (canvas pixels)
    this.target = null;          // latest measured fingertip (canvas pixels)
    this.stroke = [];            // samples of the current stroke: {x, y, t}
    this.strokeHues = [];        // hue of the segment ending at each sample
    this.freeStrokes = [];       // finished strokes in canvas pixels: {points, hues}
    this.anchoredStrokes = [];   // finished stencil drawings in face coordinates: {key, points, hues}
    this.gesture = 'none';
    this.candidate = { gesture: 'none', votes: 0 };
    this.clearSince = null;
    this.lastUpdateAt = 0;
    this.huePhase = 0;
    this.hue = 190;
    this.sparks = [];
    this.flash = 0;
    if (this.board) this.board.reset();
  }

  // Sharper lines on big screens: canvas resolution follows the displayed size (2x-4x the analysis frame).
  fitResolution(displayWidth, frameWidth) {
    AIR_DRAW.renderScale = Math.min(4, Math.max(2, Math.round(displayWidth / frameWidth)));
    this.frameWidth = 0;   // forces new canvas sizes on the next frame
  }

  // Stencil mode: switches between drawing on the stencils and wearing the masks on the faces.
  showMasks(on) {
    this.masksOnFaces = on;
  }

  // Stencil mode: count templates, or 0 for free drawing.
  useStencils(count) {
    this.board = count ? new StencilBoard(AIR_DRAW.renderScale, count) : null;
    this.masksOnFaces = false;
    this.frameWidth = 0;   // forces a new layout on the next frame
  }

  start() {
    if (this.active) return;
    this.active = true;
    this.canvas.hidden = false;
    this.lastTick = performance.now();
    requestAnimationFrame(time => this.tick(time));
  }

  stop() {
    this.active = false;
    this.canvas.hidden = true;
    this.clear();
    this.reset();
  }

  clear() {
    this.layerContext.clearRect(0, 0, this.layer.width, this.layer.height);
    this.stroke = [];
    this.strokeHues = [];
    this.freeStrokes = [];
    this.anchoredStrokes = [];
    this.flash = 1;
  }

  // Open hand held: over a stencil only that stencil is erased, anywhere else everything.
  eraseAt(point) {
    const key = this.board && point ? this.board.keyAt(point) : null;
    if (key === null) {
      this.clear();
      return;
    }
    this.anchoredStrokes = this.anchoredStrokes.filter(stroke => stroke.key !== key);
    this.flash = 0.6;
  }

  // Number of people wearing a finished stencil drawing.
  completedCount() {
    return this.board ? this.board.completedCount(this.anchoredStrokes) : 0;
  }

  drawnStencilCount() {
    return this.board ? this.board.drawnStencils(this.anchoredStrokes).length : 0;
  }

  visibleFaceCount() {
    return this.board ? this.board.visibleFaceCount() : 0;
  }

  // One analysis result. pointer = {x, y, gesture, penOwned} in frame pixels or null;
  // faces = tracked face poses (stencil challenge); sampleTime = capture time of the analysed frame.
  update(pointer, frameWidth, frameHeight, sampleTime = performance.now(), faces = null) {
    this.resize(frameWidth, frameHeight);
    this.lastUpdateAt = performance.now();
    if (this.board) this.board.updateFaces(faces || []);
    if (this.onPenState) this.onPenState(Boolean(pointer && pointer.penOwned));
    const point = pointer && { x: pointer.x * AIR_DRAW.renderScale, y: pointer.y * AIR_DRAW.renderScale, t: sampleTime };
    if (point) {
      this.target = point;
      if (!this.cursor) this.cursor = { ...point };
    }
    this.voteGesture(pointer ? pointer.gesture : 'none');
    if (this.gesture === 'draw' && point && pointer.gesture === 'draw') this.addSample(point);
  }

  // Keeps flickering single readings from interrupting a line.
  voteGesture(gesture) {
    if (gesture === this.candidate.gesture) this.candidate.votes += 1;
    else this.candidate = { gesture, votes: 1 };
    if (gesture === this.gesture) return;
    const needed = this.gesture === 'draw' ? AIR_DRAW.votesToStop : AIR_DRAW.votesToStart;
    if (this.candidate.votes >= needed) this.setGesture(gesture);
  }

  setGesture(gesture) {
    if (this.gesture === 'draw') this.endStroke();
    this.gesture = gesture;
    this.clearSince = gesture === 'clear' ? performance.now() : null;
  }

  addSample(point) {
    const last = this.stroke[this.stroke.length - 1];
    if (last) {
      const gap = point.t - last.t;
      const jump = Math.hypot(point.x - last.x, point.y - last.y) / this.canvas.width;
      if (gap > AIR_DRAW.maxGapMs || jump > AIR_DRAW.maxJump) this.endStroke();
      else if (Math.hypot(point.x - last.x, point.y - last.y) < 2) return;  // standing still
    }
    this.stroke.push(point);
    this.strokeHues.push(this.nextHue(last, point));
    this.drawReadySegment();
    this.emitSparks(point);
  }

  nextHue(last, point) {
    if (last) this.huePhase += Math.hypot(point.x - last.x, point.y - last.y) * 0.004;
    this.hue = 250 + 70 * Math.sin(this.huePhase);   // sweeps cyan → violet → magenta along the line
    return this.hue;
  }

  // Once the sample after p2 exists, the curve p1 -> p2 is final and goes onto the layer.
  drawReadySegment() {
    const count = this.stroke.length;
    if (count < 3) return;
    const index = count - 2;   // segment ending at stroke[index]
    drawNeonCurve(this.layerContext, catmullRomWindow(this.stroke, index), this.strokeHues[index]);
  }

  // Finishes the last piece of the stroke and files it as free stroke or stencil drawing.
  endStroke() {
    const points = this.stroke;
    const hues = this.strokeHues;
    this.stroke = [];
    this.strokeHues = [];
    if (!points.length) return;
    drawNeonCurve(this.layerContext, catmullRomWindow(points, points.length - 1), hues[hues.length - 1]);
    if (!this.board) {
      this.freeStrokes.push({ points, hues });
      return;
    }
    for (const piece of this.board.splitStroke(points, hues)) {
      if (piece.key === null) this.freeStrokes.push({ points: piece.points, hues: piece.hues });
      else this.anchoredStrokes.push(piece);
    }
    this.redrawLayer();   // stencil pieces are drawn from their stencil every frame instead
  }

  redrawLayer() {
    this.layerContext.clearRect(0, 0, this.layer.width, this.layer.height);
    for (const stroke of this.freeStrokes) drawStroke(this.layerContext, stroke.points, stroke.hues);
  }

  resize(frameWidth, frameHeight) {
    if (frameWidth === this.frameWidth) return;
    this.frameWidth = frameWidth;
    for (const canvas of [this.canvas, this.layer, this.artwork]) {
      canvas.width = frameWidth * AIR_DRAW.renderScale;
      canvas.height = frameHeight * AIR_DRAW.renderScale;
    }
    if (this.board) this.board.resize(this.canvas.width, this.canvas.height);
    this.redrawLayer();
  }

  // Draws the finished artwork (incl. masks worn by faces) on top of a photo of the given size.
  compositeInto(context, width, height) {
    if (!this.active || !this.artwork.width) return;
    this.renderArtwork(true);
    context.drawImage(this.artwork, 0, 0, width, height);
  }

  renderArtwork(withMasks) {
    const context = this.artworkContext;
    context.clearRect(0, 0, this.artwork.width, this.artwork.height);
    context.drawImage(this.layer, 0, 0);
    if (!this.board || !withMasks) return;
    for (const stroke of this.anchoredStrokes) {
      const points = this.board.facePoints(stroke);
      if (points) drawStroke(context, points, stroke.hues, true);
    }
  }

  tick(time) {
    if (!this.active) return;
    const elapsed = Math.min(time - this.lastTick, 100);
    this.lastTick = time;
    if (time - this.lastUpdateAt > AIR_DRAW.staleAfterMs && this.gesture !== 'none') this.setGesture('none');
    this.moveCursor(elapsed);
    this.handleClearHold(time);
    this.render(elapsed);
    requestAnimationFrame(next => this.tick(next));
  }

  moveCursor(elapsed) {
    if (!this.cursor || !this.target) return;
    const share = 1 - Math.pow(1 - AIR_DRAW.cursorEasing, elapsed / 16.7);
    this.cursor.x += (this.target.x - this.cursor.x) * share;
    this.cursor.y += (this.target.y - this.cursor.y) * share;
  }

  emitSparks(point) {
    for (let i = 0; i < 3 && this.sparks.length < AIR_DRAW.maxSparks; i += 1) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.05 + Math.random() * 0.18;
      this.sparks.push({ x: point.x, y: point.y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 1, hue: this.hue });
    }
  }

  handleClearHold(time) {
    if (this.gesture !== 'clear' || this.clearSince === null) return;
    if (time - this.clearSince >= AIR_DRAW.clearHoldMs) {
      this.eraseAt(this.cursor);
      this.clearSince = null;   // one erase per open-hand gesture
    }
  }

  clearProgress() {
    if (this.gesture !== 'clear' || this.clearSince === null) return 0;
    return Math.min((performance.now() - this.clearSince) / AIR_DRAW.clearHoldMs, 1);
  }

  render(elapsed) {
    const context = this.context;
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.renderArtwork(this.masksOnFaces);
    context.drawImage(this.artwork, 0, 0);
    if (this.board && !this.masksOnFaces) this.board.renderOverlay(context, this.anchoredStrokes, drawStroke);
    this.renderOpenEnd();
    this.renderSparks(elapsed);
    if (this.cursor && this.gesture !== 'none') this.renderCursor();
    this.renderFlash(elapsed);
  }

  // The newest piece of the line (not final yet) is drawn live, so there is no visible lag.
  renderOpenEnd() {
    if (this.gesture !== 'draw' || this.stroke.length < 2) return;
    const index = this.stroke.length - 1;
    drawNeonCurve(this.context, catmullRomWindow(this.stroke, index), this.strokeHues[index]);
  }

  renderSparks(elapsed) {
    const context = this.context;
    context.save();
    context.globalCompositeOperation = 'lighter';
    for (const spark of this.sparks) {
      spark.x += spark.vx * elapsed * AIR_DRAW.renderScale;
      spark.y += spark.vy * elapsed * AIR_DRAW.renderScale;
      spark.life -= elapsed / 500;
      context.fillStyle = `hsla(${spark.hue}, 100%, 75%, ${Math.max(spark.life, 0)})`;
      context.beginPath();
      context.arc(spark.x, spark.y, 2.5 * AIR_DRAW.renderScale * Math.max(spark.life, 0), 0, Math.PI * 2);
      context.fill();
    }
    context.restore();
    this.sparks = this.sparks.filter(spark => spark.life > 0);
  }

  // HUD reticle: rotating dashed ring, crosshair ticks and an erase progress arc.
  renderCursor() {
    const context = this.context;
    const { x, y } = this.cursor;
    const scale = AIR_DRAW.renderScale;
    const color = { draw: `hsl(${this.hue}, 100%, 70%)`, hover: 'rgba(255, 255, 255, .85)', clear: '#ff3df2' }[this.gesture];
    const radius = (this.gesture === 'draw' ? 11 : 16) * scale;
    const rotation = performance.now() / 600;
    context.save();
    context.strokeStyle = color;
    context.shadowColor = 'rgba(0, 0, 0, .6)';
    context.shadowBlur = 6;
    context.lineWidth = 1.8 * scale;
    context.setLineDash([6 * scale, 5 * scale]);
    context.beginPath();
    context.arc(x, y, radius, rotation, rotation + Math.PI * 2);
    context.stroke();
    context.setLineDash([]);
    for (let i = 0; i < 4; i += 1) {
      const angle = rotation * -0.5 + i * Math.PI / 2;
      context.beginPath();
      context.moveTo(x + Math.cos(angle) * (radius + 3 * scale), y + Math.sin(angle) * (radius + 3 * scale));
      context.lineTo(x + Math.cos(angle) * (radius + 9 * scale), y + Math.sin(angle) * (radius + 9 * scale));
      context.stroke();
    }
    const progress = this.clearProgress();
    if (progress > 0) {
      context.lineWidth = 3.5 * scale;
      context.beginPath();
      context.arc(x, y, radius + 14 * scale, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
      context.stroke();
    }
    context.fillStyle = color;
    context.beginPath();
    context.arc(x, y, 2.5 * scale, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  // Short magenta flash across the image after erasing.
  renderFlash(elapsed) {
    if (this.flash <= 0) return;
    this.context.fillStyle = `rgba(255, 61, 242, ${this.flash * 0.25})`;
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.flash = Math.max(0, this.flash - elapsed / 350);
  }
}

// The four points around the segment that ends at points[index] (ends are repeated).
function catmullRomWindow(points, index) {
  const at = i => points[Math.min(Math.max(i, 0), points.length - 1)];
  return [at(index - 2), at(index - 1), at(index), at(index + 1)];
}

// A whole finished stroke; 'light' skips the expensive blur (stencil drawings are redrawn every frame).
function drawStroke(context, points, hues, light = false) {
  if (points.length === 1) drawNeonCurve(context, [points[0], points[0], points[0], points[0]], hues[0], light);
  for (let index = 1; index < points.length; index += 1) {
    drawNeonCurve(context, catmullRomWindow(points, index), hues[index], light);
  }
}

// Neon line through p1 -> p2 of a Catmull-Rom segment [p0, p1, p2, p3], converted to a Bézier curve.
function drawNeonCurve(context, [p0, p1, p2, p3], hue, light = false) {
  const control1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
  const control2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
  const passes = [
    { width: AIR_DRAW.glowWidth, color: `hsla(${hue}, 100%, 60%, .28)`, blur: light ? 0 : 28 },
    { width: AIR_DRAW.coreWidth * 2, color: `hsla(${hue}, 100%, 65%, .7)`, blur: light ? 0 : 12 },
    { width: AIR_DRAW.coreWidth, color: `hsla(${hue}, 100%, 92%, 1)`, blur: 0 },
  ];
  context.save();
  context.globalCompositeOperation = 'lighter';
  context.lineCap = 'round';
  context.lineJoin = 'round';
  for (const pass of passes) {
    context.strokeStyle = pass.color;
    context.lineWidth = pass.width * AIR_DRAW.renderScale / 2;
    context.shadowColor = `hsl(${hue}, 100%, 60%)`;
    context.shadowBlur = pass.blur;
    context.beginPath();
    context.moveTo(p1.x, p1.y);
    context.bezierCurveTo(control1.x, control1.y, control2.x, control2.y, p2.x, p2.y);
    context.stroke();
  }
  context.restore();
}
