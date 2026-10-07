// Party sounds, synthesised with WebAudio (no audio files needed).
// Browsers only allow sound after a user interaction; the first click/keypress unlocks it.

class SoundBoard {
  constructor() {
    this.context = null;
    this.enabled = readPreference('photobooth.sound', 'on') !== 'off';
    const unlock = () => this.audio();
    window.addEventListener('pointerdown', unlock, { once: true });
    window.addEventListener('keydown', unlock, { once: true });
  }

  audio() {
    if (!this.context) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return null;
      this.context = new AudioContext();
    }
    if (this.context.state === 'suspended') this.context.resume();
    return this.context;
  }

  toggle() {
    this.enabled = !this.enabled;
    writePreference('photobooth.sound', this.enabled ? 'on' : 'off');
    if (this.enabled) this.tick(2);
    return this.enabled;
  }

  // One short beep per countdown second; the last one is higher.
  tick(secondsLeft) {
    this.tone(secondsLeft <= 1 ? 1046 : 784, 0, 0.12, 'square', 0.08);
  }

  // Camera shutter: a short burst of filtered noise.
  shutter() {
    const context = this.ready();
    if (!context) return;
    const length = Math.floor(context.sampleRate * 0.12);
    const buffer = context.createBuffer(1, length, context.sampleRate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < length; i += 1) samples[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 3);
    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const gain = context.createGain();
    source.buffer = buffer;
    filter.type = 'highpass';
    filter.frequency.value = 1800;
    gain.gain.value = 0.5;
    source.connect(filter).connect(gain).connect(context.destination);
    source.start();
  }

  // Rising arpeggio for a solved challenge.
  fanfare() {
    [523, 659, 784, 1046].forEach((frequency, index) => this.tone(frequency, index * 0.09, 0.22, 'triangle', 0.14));
  }

  // Quick "whoosh" for a new challenge.
  whoosh() {
    const context = this.ready();
    if (!context) return;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const now = context.currentTime;
    oscillator.type = 'sawtooth';
    oscillator.frequency.setValueAtTime(220, now);
    oscillator.frequency.exponentialRampToValueAtTime(880, now + 0.25);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.06, now + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.32);
  }

  tone(frequency, delay, duration, type, volume) {
    const context = this.ready();
    if (!context) return;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const start = context.currentTime + delay;
    oscillator.type = type;
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(volume, start);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(start);
    oscillator.stop(start + duration + 0.02);
  }

  ready() {
    return this.enabled ? this.audio() : null;
  }
}

// localStorage can be unavailable (private mode, blocked storage): fall back silently.
function readPreference(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}

function writePreference(key, value) {
  try { localStorage.setItem(key, value); } catch { /* not persisted, still works for this session */ }
}
