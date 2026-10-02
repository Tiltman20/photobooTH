// Full-screen view of the photo just taken. The photo only exists on the
// server for a few seconds; when the timer ends (or "Fertig" is pressed) it
// is deleted and the booth returns to the camera page.

class PhotoViewer {
  constructor(root) {
    this.root = root;
    this.image = root.querySelector('#photo-viewer-image');
    this.title = root.querySelector('#photo-viewer-title');
    this.subtitle = root.querySelector('#photo-viewer-subtitle');
    this.note = root.querySelector('#photo-viewer-note');
    this.seconds = root.querySelector('#photo-viewer-seconds');
    this.timer = root.querySelector('#photo-viewer-timer');
    this.qr = root.querySelector('#photo-viewer-qr');
    this.confetti = new ConfettiBurst(root.querySelector('#photo-viewer-fx'));
    this.photo = null;
    this.onClose = null;
    this.interval = null;
    root.querySelector('#photo-viewer-close').addEventListener('click', () => this.close());
  }

  get isOpen() {
    return this.photo !== null;
  }

  // photo: capture response; options: {completed, shareUrl}; onClose runs after deletion.
  open(photo, { completed = false, shareUrl = null } = {}, onClose = null) {
    this.photo = photo;
    this.onClose = onClose;
    this.image.src = photo.url;
    this.title.textContent = completed ? 'Geschafft!' : 'Foto!';
    this.subtitle.textContent = photo.title || '';
    this.note.textContent = shareUrl
      ? 'QR-Code scannen, Foto aufs Handy laden und an der Theke vorzeigen.'
      : 'Zeig das Foto an der Theke vor – es wird danach automatisch gelöscht.';
    this.qr.hidden = !shareUrl;
    if (shareUrl) this.qr.src = `/api/share/qr?photo=${encodeURIComponent(photo.token)}`;
    this.root.hidden = false;
    this.endsAt = performance.now() + photo.expiresInSeconds * 1000;
    this.totalMs = photo.expiresInSeconds * 1000;
    this.updateTimer();
    this.interval = setInterval(() => this.updateTimer(), 200);
    if (completed) this.confetti.fire();
  }

  updateTimer() {
    const remaining = Math.max(0, this.endsAt - performance.now());
    this.seconds.textContent = Math.ceil(remaining / 1000);
    this.timer.style.setProperty('--remaining', remaining / this.totalMs);
    if (remaining <= 0) this.close();
  }

  async close() {
    if (!this.photo) return;
    const { token } = this.photo;
    clearInterval(this.interval);
    this.photo = null;
    this.root.hidden = true;
    this.image.removeAttribute('src');
    this.qr.removeAttribute('src');
    // The server deletes it on its own after the lifetime; this also covers closing early.
    await fetch(`/api/photos/${encodeURIComponent(token)}`, { method: 'DELETE' }).catch(() => {});
    if (this.onClose) await this.onClose();
  }
}
