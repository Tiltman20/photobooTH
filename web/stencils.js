// Stencil challenge: guests draw on face templates, the drawings are worn by real faces.
//
// Each stencil is a sketched face in *face coordinates* (1 unit = eye
// distance, origin between the eyes, v < 0 above the eyes). A stroke drawn on
// stencil i is stored in these coordinates and rendered twice: on the
// stencil, and on the i-th real face from the left, following its pose. A hat
// drawn above the stencil's head therefore sits above the real head, a
// moustache under the stencil's nose sits under the real nose.

const STENCILS = {
  area: { u0: -1.8, u1: 1.8, v0: -3.4, v1: 2.4 },  // drawable area around each stencil face
  minInk: 0.8,             // line length (eye distances) that counts as "drawn" (a moustache is short)
  keepPoseMs: 700,         // a face missing for a moment keeps wearing its drawing
  topMargin: 0.03,         // share of the canvas height above the stencils
  maxHeightShare: 0.94,    // stencils may use almost the full height (they are see-through)
};

class StencilBoard {
  constructor(renderScale, count) {
    this.renderScale = renderScale;
    this.count = count;
    this.poses = new Map();   // face id -> {x, y, ax, ay, seenAt} in canvas pixels
    this.layout = [];         // per stencil: {x, y, scale} (origin between the eyes, pixels per unit)
  }

  reset() {
    this.poses.clear();
  }

  // Places the stencils side by side at the top of the canvas.
  resize(width, height) {
    const area = STENCILS.area;
    const areaWidth = area.u1 - area.u0, areaHeight = area.v1 - area.v0;
    const scale = Math.min(width / (this.count * areaWidth * 1.02), height * STENCILS.maxHeightShare / areaHeight);
    const y = height * STENCILS.topMargin - area.v0 * scale;
    this.layout = Array.from({ length: this.count }, (_, index) => ({ x: width * (index + 0.5) / this.count, y, scale }));
  }

  updateFaces(faces = [], now = performance.now()) {
    const scale = this.renderScale;
    for (const face of faces) {
      this.poses.set(face.id, { x: face.x * scale, y: face.y * scale, ax: face.ax * scale, ay: face.ay * scale, seenAt: now });
    }
  }

  // Visible faces sorted from left to right: face k wears stencil k.
  wearers(now = performance.now()) {
    return [...this.poses.values()].filter(pose => now - pose.seenAt <= STENCILS.keepPoseMs).sort((a, b) => a.x - b.x);
  }

  visibleFaceCount() {
    return this.wearers().length;
  }

  // Stencil whose drawing area contains the point, or null.
  keyAt(point) {
    const index = this.layout.findIndex(stencil => this.inArea(this.stencilLocal(stencil, point)));
    return index >= 0 ? index : null;
  }

  // Splits a finished stroke into pieces per stencil (in face coordinates); the rest stays free.
  splitStroke(points, hues) {
    const pieces = [];
    points.forEach((point, index) => {
      const key = this.keyAt(point);
      const last = pieces[pieces.length - 1];
      if (last && last.key === key) {
        last.points.push(point);
        last.hues.push(hues[index]);
      } else {
        pieces.push({ key, points: [point], hues: [hues[index]] });
      }
    });
    return pieces.map(piece => (piece.key === null ? piece : {
      ...piece, points: piece.points.map(point => this.stencilLocal(this.layout[piece.key], point)),
    }));
  }

  // Screen points of a stencil drawing on its real face, or null if that face is not there.
  facePoints(stroke) {
    const pose = this.wearers()[stroke.key];
    return pose ? stroke.points.map(point => poseToScreen(pose, point)) : null;
  }

  stencilPoints(stroke) {
    const stencil = this.layout[stroke.key];
    return stroke.points.map(({ u, v }) => ({ x: stencil.x + u * stencil.scale, y: stencil.y + v * stencil.scale }));
  }

  // Stencils with enough drawing on them.
  drawnStencils(strokes) {
    const ink = new Array(this.count).fill(0);
    for (const stroke of strokes) ink[stroke.key] += localLength(stroke.points);
    return ink.map((length, key) => length >= STENCILS.minInk ? key : null).filter(key => key !== null);
  }

  // Number of people who wear a finished drawing (stencil drawn and a face to wear it).
  completedCount(strokes) {
    const faces = this.visibleFaceCount();
    return this.drawnStencils(strokes).filter(key => key < faces).length;
  }

  // Stencil sketches with their drawings (light rendering) and numbers above the wearers.
  renderOverlay(context, strokes, drawStrokeFn) {
    const drawn = new Set(this.drawnStencils(strokes));
    this.layout.forEach((stencil, key) => this.renderStencil(context, stencil, key, drawn.has(key)));
    for (const stroke of strokes) drawStrokeFn(context, this.stencilPoints(stroke), stroke.hues, true);
    this.wearers().forEach((pose, key) => {
      if (key < this.count) this.renderBadge(context, poseToScreen(pose, { u: -1.7, v: 0.3 }), key, drawn.has(key));
    });
  }

  renderStencil(context, stencil, key, drawn) {
    const { x, y, scale } = stencil;
    const at = (u, v) => [x + u * scale, y + v * scale];
    const area = STENCILS.area;
    context.save();
    context.fillStyle = 'rgba(10, 8, 20, .16)';
    context.strokeStyle = drawn ? 'rgba(255, 106, 19, .95)' : 'rgba(255, 255, 255, .7)';
    context.lineWidth = 2.2 * this.renderScale;
    context.setLineDash(drawn ? [] : [7 * this.renderScale, 5 * this.renderScale]);
    const [left, top] = at(area.u0, area.v0);
    context.fillRect(left, top, (area.u1 - area.u0) * scale, (area.v1 - area.v0) * scale);
    context.strokeRect(left, top, (area.u1 - area.u0) * scale, (area.v1 - area.v0) * scale);
    context.setLineDash([]);
    // Face sketch: head, eyes, nose, mouth - the reference for what goes where.
    context.strokeStyle = 'rgba(255, 255, 255, .45)';
    context.beginPath();
    context.ellipse(...at(0, 0.15), 1.15 * scale, 1.85 * scale, 0, 0, Math.PI * 2);
    context.moveTo(...at(0.68, 0)); context.arc(...at(0.5, 0), 0.18 * scale, 0, Math.PI * 2);
    context.moveTo(...at(-0.32, 0)); context.arc(...at(-0.5, 0), 0.18 * scale, 0, Math.PI * 2);
    context.moveTo(...at(0, 0.2)); context.lineTo(...at(0.12, 0.75)); context.lineTo(...at(-0.05, 0.8));
    context.moveTo(...at(-0.45, 1.2)); context.quadraticCurveTo(...at(0, 1.45), ...at(0.45, 1.2));
    context.stroke();
    context.font = `800 ${22 * this.renderScale}px ui-sans-serif, system-ui`;
    context.fillStyle = drawn ? '#ff6a13' : '#ffffff';
    context.fillText(`${key + 1}`, left + 10 * this.renderScale, top + 27 * this.renderScale);
    context.restore();
  }

  renderBadge(context, point, key, drawn) {
    const radius = 19 * this.renderScale;
    context.save();
    context.fillStyle = drawn ? 'rgba(255, 106, 19, .95)' : 'rgba(20, 16, 24, .75)';
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = drawn ? '#141414' : '#ffffff';
    context.font = `800 ${22 * this.renderScale}px ui-sans-serif, system-ui`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(String(key + 1), point.x, point.y + this.renderScale);
    context.restore();
  }

  stencilLocal(stencil, point) {
    return { u: (point.x - stencil.x) / stencil.scale, v: (point.y - stencil.y) / stencil.scale };
  }

  inArea({ u, v }) {
    const area = STENCILS.area;
    return u >= area.u0 && u <= area.u1 && v >= area.v0 && v <= area.v1;
  }
}

// Face coordinates -> screen pixels for a face pose (eye midpoint + x axis scaled to the eye distance).
function poseToScreen(pose, { u, v }) {
  return { x: pose.x + pose.ax * u - pose.ay * v, y: pose.y + pose.ay * u + pose.ax * v };
}

function localLength(points) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += Math.hypot(points[index].u - points[index - 1].u, points[index].v - points[index - 1].v);
  }
  return length;
}
