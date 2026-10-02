// Visual rewards: the big 3-2-1 countdown and a confetti burst after a photo.

const CONFETTI = {
  particles: 180,
  durationMs: 2600,
  gravity: 0.0009,   // px per ms², relative to canvas height
  colors: ['#00e5ff', '#9b30ff', '#ff3df2', '#39d98a', '#ffd23f', '#ffffff'],
};

class Countdown {
  constructor(overlay, number) {
    this.overlay = overlay;
    this.number = number;
    this.shown = null;
  }

  // Shows the whole seconds left; the pop animation restarts on every new number.
  show(remainingMs) {
    const seconds = Math.max(1, Math.ceil(remainingMs / 1000));
    this.overlay.hidden = false;
    if (seconds === this.shown) return;
    this.shown = seconds;
    this.number.textContent = seconds;
    this.number.classList.remove('pop');
    void this.number.offsetWidth;  // reflow, so the animation plays again
    this.number.classList.add('pop');
  }

  hide() {
    this.overlay.hidden = true;
    this.shown = null;
  }
}

class ConfettiBurst {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.pieces = [];
    this.running = false;
  }

  fire() {
    const { width, height } = this.canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    this.pieces = Array.from({ length: CONFETTI.particles }, () => this.createPiece());
    this.startedAt = performance.now();
    this.lastTick = this.startedAt;
    if (!this.running) {
      this.running = true;
      requestAnimationFrame(time => this.tick(time));
    }
  }

  createPiece() {
    const { width, height } = this.canvas;
    const angle = -Math.PI / 2 + (Math.random() - 0.5) * 1.6;   // mostly upwards
    const speed = (0.6 + Math.random() * 0.9) * height / 1000;
    return {
      x: width / 2 + (Math.random() - 0.5) * width * 0.3,
      y: height * 0.75,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size: (6 + Math.random() * 8) * (window.devicePixelRatio || 1),
      rotation: Math.random() * Math.PI,
      spin: (Math.random() - 0.5) * 0.02,
      color: CONFETTI.colors[Math.floor(Math.random() * CONFETTI.colors.length)],
    };
  }

  tick(time) {
    const elapsed = Math.min(time - this.lastTick, 50);
    this.lastTick = time;
    const age = time - this.startedAt;
    const { context, canvas } = this;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.globalAlpha = Math.max(0, 1 - age / CONFETTI.durationMs);
    for (const piece of this.pieces) {
      piece.vy += CONFETTI.gravity * canvas.height / 1000 * elapsed;
      piece.x += piece.vx * elapsed;
      piece.y += piece.vy * elapsed;
      piece.rotation += piece.spin * elapsed;
      context.save();
      context.translate(piece.x, piece.y);
      context.rotate(piece.rotation);
      context.fillStyle = piece.color;
      context.fillRect(-piece.size / 2, -piece.size / 4, piece.size, piece.size / 2);
      context.restore();
    }
    context.globalAlpha = 1;
    if (age < CONFETTI.durationMs) {
      requestAnimationFrame(next => this.tick(next));
    } else {
      context.clearRect(0, 0, canvas.width, canvas.height);
      this.running = false;
    }
  }
}
