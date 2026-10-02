// Settings drawer: sliders for detection thresholds and round timing.
// Changes are applied (and saved on the server) right away, so the effect is
// visible in the live score meters while the camera keeps running.

const SETTINGS_SAVE_DELAY_MS = 300;
const SETTINGS_GROUPS = { thresholds: 'Erkennung (Schwellwerte)', timing: 'Ablauf' };

class SettingsPanel {
  constructor(panel, onChange) {
    this.panel = panel;
    this.form = panel.querySelector('#settings-form');
    this.status = panel.querySelector('#settings-status');
    this.onChange = onChange;
    this.saveTimer = null;
    panel.querySelector('#settings-close').addEventListener('click', () => this.close());
    panel.querySelector('#settings-reset').addEventListener('click', () => this.reset());
  }

  async load() {
    const settings = await (await fetch('/api/settings')).json();
    this.onChange(settings.values);
    return settings;
  }

  async open() {
    this.render(await this.load());
    this.panel.hidden = false;
  }

  close() {
    this.panel.hidden = true;
  }

  render(settings) {
    this.form.replaceChildren();
    for (const [group, title] of Object.entries(SETTINGS_GROUPS)) {
      const heading = document.createElement('h3');
      heading.textContent = title;
      this.form.append(heading);
      for (const spec of settings.specs.filter(item => item.group === group)) {
        this.form.append(this.createSlider(spec, settings.values[spec.key], settings.defaults[spec.key]));
      }
    }
  }

  createSlider(spec, value, defaultValue) {
    const row = document.createElement('label');
    row.className = 'setting-row';
    const name = document.createElement('span');
    name.className = 'setting-name';
    name.textContent = spec.label;
    const output = document.createElement('output');
    output.textContent = formatSetting(value, spec.step);
    const input = document.createElement('input');
    Object.assign(input, { type: 'range', min: spec.minimum, max: spec.maximum, step: spec.step, value, name: spec.key });
    input.addEventListener('input', () => {
      output.textContent = formatSetting(Number(input.value), spec.step);
      this.scheduleSave();
    });
    const hint = document.createElement('small');
    hint.textContent = `${spec.hint} · Standard ${formatSetting(defaultValue, spec.step)}`;
    row.append(name, output, input, hint);
    return row;
  }

  scheduleSave() {
    clearTimeout(this.saveTimer);
    this.status.textContent = 'Wird übernommen …';
    this.saveTimer = setTimeout(() => this.save(), SETTINGS_SAVE_DELAY_MS);
  }

  async save() {
    const values = Object.fromEntries([...this.form.querySelectorAll('input')].map(input => [input.name, Number(input.value)]));
    const settings = await this.post({ values });
    if (settings) this.status.textContent = 'Gespeichert – gilt ab sofort.';
  }

  async reset() {
    const settings = await this.post({ reset: true });
    if (!settings) return;
    this.render(settings);
    this.status.textContent = 'Standardwerte wiederhergestellt.';
  }

  async post(payload) {
    try {
      const response = await fetch('/api/settings', { method: 'POST', body: JSON.stringify(payload), headers: { 'Content-Type': 'application/json' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const settings = await response.json();
      this.onChange(settings.values);
      return settings;
    } catch (error) {
      this.status.textContent = `Speichern fehlgeschlagen: ${error.message}`;
      return null;
    }
  }
}

function formatSetting(value, step) {
  const decimals = step >= 1 ? 0 : Math.min(3, String(step).split('.')[1]?.length || 2);
  return Number(value).toFixed(decimals);
}
