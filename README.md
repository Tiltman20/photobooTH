# photobooTH

Ein lokaler Party-Fotoautomat im Browser. Die Gäste bekommen zufällige
Challenges („2 Leute mit Brille“, „Team Schwarz: 3 Leute“, „Kussmund-Crew“ …),
neuronale Netze prüfen live, ob sie erfüllt sind, und nach einem
3-2-1-Countdown entsteht automatisch das Foto. Das Foto mit eingebranntem
Challenge-Banner ist der Nachweis, den die Gäste an der Theke vorzeigen.

Es werden **keine Fotos gespeichert**: Bilder liegen nur kurz im Arbeitsspeicher
(`src/photo_store.py`) und sind nach der Anzeigezeit weg.

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

MediaPipe bringt `opencv-contrib-python` mit; ein vorher installiertes
`opencv-python` bitte zuerst entfernen (`python -m pip uninstall -y opencv-python`).

**Vor der Party einmal mit Internet** alle Modelle laden (ca. 320 MB), danach
läuft alles offline:

```powershell
python src/models.py
```

## Starten

```powershell
python main.py
```

Der Browser öffnet sich automatisch mit `http://localhost:8000`; dort den
Kamerazugriff erlauben. Beenden mit `Strg+C`.

| Taste | Wirkung |
|---|---|
| `N` / `→` | andere Challenge |
| `Leertaste` | Foto (freie Modi) bzw. Foto-Ansicht schließen |
| `F` | Vollbild |
| `M` | Ton an/aus |
| `Esc` | Einstellungen / Foto schließen |

Tests: `python -m pytest tests`

## Ablauf einer Runde

1. Eine Challenge wird zufällig gezogen (Varianten mit mehreren Personen
   kommen häufiger – Leute kennenlernen ist der Sinn der Sache).
2. Ist sie erfüllt, startet der Countdown. Kurze Erkennungsaussetzer (< 0,8 s)
   brechen ihn nicht ab.
3. Das Foto erscheint im Vollbild und wird nach der Anzeigezeit gelöscht.
   Oben rechts zählt die Seite die gelösten Challenges der Party.

## Challenges und Erkennung

| Gruppe | Challenges | Erkennung |
|---|---|---|
| Look & Style | Brille, Brille + ohne Brille, Bart, Hut/Cap, Haarfarbe (rot, blond, bunt), Shirt-Farbe, Regenbogen-Crew | **CLIP** (Zero-Shot, `src/attributes.py`) |
| Grimassen | Grinsen, Schrei (Mund auf), Kussmund, Gefühlschaos (3 Leute, 3 Ausdrücke) | **MediaPipe Face Landmarker**, Blendshapes (`src/expressions.py`) |
| Gruppe | Gruppenfoto mit 3–6 Leuten | YuNet-Gesichtserkennung |
| Hände | Daumen hoch, Peace, offene Hände, Finger-Mathe | **MediaPipe Hand Landmarker** (`src/hand_detection.py`) |
| Kreativ | Masken-Werkstatt, Luftmalerei | Hand Landmarker + Face-Tracking |

**Wie die Look-Erkennung funktioniert:** Jede Person wird anhand der
YuNet-Landmarks aufrecht ausgeschnitten (Kopf bzw. Oberkörper). CLIP vergleicht
den Ausschnitt mit mehreren Textbeschreibungen pro Antwort, z. B. „a photo of a
person wearing glasses“ gegen „… without glasses“; ein Softmax liefert die
Wahrscheinlichkeit. Werte werden pro Person über mehrere Bilder geglättet, damit
ein einzelnes unsicheres Bild den Countdown nicht abbricht.

Neue Merkmale brauchen nur neue Prompts in `QUESTIONS` (`src/attributes.py`).
Danach einmal `python src/attributes.py` ausführen: das berechnet die
Text-Embeddings neu und speichert sie in `res/clip_prompt_cache.json`, damit zur
Laufzeit nur der Bild-Encoder nötig ist.

**Einstellungen (Regler-Symbol oben rechts):** Schwellen für jedes Merkmal,
Countdown und Anzeigedauer. Änderungen gelten sofort; im Kamerabild zeigt ein
Balken unter jedem Label den aktuellen Wert, der Strich ist die Schwelle. Die
Werte landen in `settings.json` (nicht im Git), die Standardwerte stehen in
`src/settings.py`.

## Projektstruktur

```
src/
  web_server.py      HTTP-Server, API, Start
  challenges.py      Katalog, Codes, Texte
  face_challenges.py Auswertung der Gesichts-Challenges (zählen, glätten)
  attributes.py      CLIP: Brille, Bart, Hut, Haare, Shirt
  expressions.py     Face Landmarker: Grinsen, Schrei, Kussmund
  hand_challenges.py Gesten, Luftmalerei, Hand-Debug
  hand_detection.py  Hand Landmarker + Fingerlogik
  gestures.py        Gesten und Stift-Tracking
  face_detection.py  YuNet-Gesichtserkennung
  face_geometry.py   Gesichtskoordinaten und Ausschnitte
  face_tracking.py   stabile IDs über mehrere Bilder
  models.py          Modell-Downloads
  settings.py        einstellbare Schwellen und Zeiten
  photo_store.py     Fotos nur im Arbeitsspeicher
  share.py           QR-Download über ngrok
web/                 Oberfläche (HTML/CSS/JS, keine Build-Tools)
```

### Luftmalerei und Masken-Werkstatt

Zeigefinger ausgestreckt = malen, zwei Finger oder Faust = Stift absetzen,
offene Hand halten = löschen. Der Stift bleibt bei der Hand, die angefangen hat.
In der Masken-Werkstatt wird auf Gesichts-Schablonen gemalt; die Zeichnungen
werden in Gesichtskoordinaten gespeichert und von den echten Gesichtern
getragen (Schablone 1 = Person ganz links).

### Per QR-Code teilen (ngrok)

> Derzeit deaktiviert (`NGROK_ENABLED = False` in `src/web_server.py`).

Mit [ngrok](https://ngrok.com/download) und hinterlegtem Authtoken startet
`python src/web_server.py --ngrok` einen separaten Download-Server mit
HTTPS-Tunnel. Nach jeder Aufnahme erscheint ein QR-Code, der genau dieses Foto
lädt, solange es noch existiert.
