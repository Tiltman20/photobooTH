# photobooTH

Ein lokaler OpenCV-Fotoautomat: Sobald eine erkannte Person ihre Brille fuer
drei Sekunden durchgehend im Kamerabild hat, wird automatisch ein Foto
aufgenommen.

Desktop-App starten:

```powershell
python src/main.py
```

Die Aufnahmen liegen anschliessend im Ordner `captures/`.

- `S`: Foto sofort speichern
- `R`: Aufnahme wieder aktivieren
- `Q` oder `ESC`: Fotoautomat beenden

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Wer vorher schon `opencv-python` installiert hatte, entfernt es zuerst, weil
MediaPipe die Variante `opencv-contrib-python` mitbringt:

```powershell
python -m pip uninstall -y opencv-python
python -m pip install -r requirements.txt
```

Tests ausführen:

```powershell
python -m pytest tests
```

## Lokale Website

```powershell
python src/web_server.py
```

Danach `http://localhost:8000` öffnen und den Kamerazugriff erlauben.
Mit `Strg+C` wird der Server beendet.

### Ablauf einer Runde

1. Eine Challenge wird zufällig gezogen (Mehrpersonen-Varianten bevorzugt).
2. Sobald sie erfüllt ist, läuft ein großer 3-2-1-Countdown. Kurze
   Erkennungsaussetzer (< 0,7 s) brechen ihn nicht ab.
3. Das Foto erscheint im Vollbild (mit Konfetti) und wird nach 15 s gelöscht –
   oder sofort mit `Fertig`. Ins Foto ist ein Banner mit der gelösten
   Challenge und der Uhrzeit eingebrannt: das ist der Nachweis, den die Gäste
   an der Theke vorzeigen. Challenge-Fotos entstehen nur automatisch (kein
   manueller Auslöser); mit ngrok erscheint ein QR-Code zum Download aufs Handy. Danach geht es mit der
   nächsten Challenge auf der Aufnahmeseite weiter. `Andere Challenge`
   überspringt eine unlösbare Aufgabe.

**Keine gespeicherten Bilder:** Die Website speichert Fotos nur im
Arbeitsspeicher (`src/photo_store.py`) unter einem zufälligen Token. Nach
Ablauf der Anzeigezeit sind sie weg; es entsteht kein `captures/`-Ordner mehr.
(Nur die alte Desktop-App `src/main.py` speichert noch Dateien.)

Im Dropdown lässt sich eine Art fest wählen (neue Varianten kommen trotzdem
zufällig) oder `Zufall – alle Challenges`.

| Challenge | Varianten | Prüfung (klassisch) |
|---|---|---|
| Brille | 1–3 Personen | Kanten des Brillenstegs zwischen den Augen |
| Schnurrbart | 1–2 Personen | Oberlippe dunkler als die eigenen Wangen, Kinn hell |
| Rote Haare | 1 Person | rote, texturierte Pixel über der Stirn, die nicht hautfarben sind |
| Farb-Team | 1–3 Personen gleiche Farbe | Farbanteil am Oberkörper nach Weißabgleich |
| Regenbogen-Crew | 3–4 verschiedene Farben | dominante Oberteil-Farbe pro Person |
| Gegensätze | 1× mit + 1× ohne Brille | Brillen-Prüfung |
| Gruppenfoto | 3–5 Personen | Gesichtsanzahl |
| Daumen hoch | 2–4 Hände | nur Daumen gestreckt, zeigt nach oben |
| Peace-Zeichen | 2–4 Hände | Zeige- und Mittelfinger gestreckt |
| Alle Hände hoch | 4 oder 6 offene Hände | alle Finger gestreckt |
| Finger-Summe | genau 7–15 Finger zusammen | Summe aller gestreckten Finger |
| Masken-Werkstatt (Malen) | 2–3 Schablonen: Hüte, Hörner, Schnurrbärte oder frei | Foto per Knopf; ✓ wenn jede Schablone bemalt ist |

Die Hand-Challenges nutzen die MediaPipe-Handerkennung (siehe unten).

**Zuverlässigkeit:** Alle Prüfbereiche werden aus den fünf YuNet-Landmarks
(Augen, Nase, Mundwinkel) berechnet und aufrecht in fester Auflösung
ausgeschnitten (`src/face_geometry.py`) – Kopfneigung und Abstand verschieben
sie nicht mehr. Dunkelheit wird relativ zur eigenen Hautfarbe gemessen, Farben
nach Grauwelt-Weißabgleich.

**Einstellungen (⚙ oben rechts):** Schieberegler für alle Schwellwerte, die
Countdown-Länge und die Foto-Anzeigedauer. Änderungen gelten sofort – jede
Person hat im Bild einen Messbalken, der weiße Strich ist die Schwelle. Die
Werte werden in `settings.json` gespeichert (nicht im Git); die Standardwerte
stehen in `TraitThresholds` (`src/face_checks.py`) und `RoundTiming`
(`src/settings.py`).

### Hand- und Fingererkennung (Debug)

Im Dropdown `Debug: Hand- und Fingererkennung` wählen. Angezeigt werden pro
Hand ein Rahmen mit Fingerzahl, das Hand-Skelett (lila) und die Spitzen der
ausgestreckten Finger (grün). Rechts steht, welche Finger erkannt wurden.
In dieser Ansicht wird kein Foto automatisch aufgenommen.

Die Erkennung (`src/hand_detection.py`) nutzt den **MediaPipe Hand Landmarker**
(neuronales Netz, 21 Punkte pro Hand). Ob ein Finger ausgestreckt ist, wird
anschließend geometrisch entschieden:

- Zeige-, Mittel-, Ring-, kleiner Finger: Fingerspitze deutlich weiter vom
  Handgelenk entfernt als das mittlere Fingergelenk.
- Daumen: nahezu gerade und vom Zeigefinger-Ansatz weg gerichtet.

Das Modell (`res/models/hand_landmarker.task`, ca. 8 MB) wird beim ersten Start
automatisch heruntergeladen. Klappt das nicht (Firewall/Proxy), manuell von
<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>
laden und unter diesem Pfad speichern.

Bekannte Grenzen: Finger, die direkt in die Kamera zeigen, werden schlecht
erkannt; Hände sollten gut beleuchtet und nicht zu klein im Bild sein.

### Luftmalerei

Im Dropdown `Luftmalerei` wählen und mit dem Zeigefinger in die Luft malen –
die Linie erscheint als Neon-Spur über dem unveränderten Kamerabild.

- Der Stift bleibt bei der Hand, die angefangen hat (`PenTracker` in
  `src/gestures.py`); eine andere Hand übernimmt erst, wenn diese ~0,7 s weg ist.
- Die Linie läuft als Kurve (Catmull-Rom) durch die gemessenen Fingerpunkte;
  in diesem Modus wird so schnell wie möglich mit kleinerem Bild analysiert.
- Nach einer Lücke > 0,35 s oder einem großen Sprung beginnt eine neue Linie,
  statt eine gerade Verbindung zu ziehen.

| Geste | Wirkung |
|---|---|
| Nur Zeigefinger ausgestreckt (Daumen egal) | malen |
| Zwei Finger, Faust, … | Stift absetzen, Cursor folgt weiter |
| Offene Hand ca. 1 s halten | alles löschen |

`Foto jetzt aufnehmen` speichert das Foto samt Zeichnung. Die Gesten-Logik
steht in `src/gestures.py`, die Darstellung in `web/air_draw.js`.

### Masken-Werkstatt (Schablonen)

Oben im Bild erscheinen 2–3 Gesichts-Schablonen (Kopf, Augen, Nase, Mund).
Die Gäste malen darauf mit dem Zeigefinger – je nach Runde Hüte, Hörner,
Schnurrbärte oder frei. Jede Zeichnung wird in Gesichtskoordinaten gespeichert
(`web/stencils.js`) und von einem echten Gesicht „getragen“: Schablone 1 von
der Person ganz links, Schablone 2 von der nächsten usw. – ein Hut über dem
Schablonen-Kopf sitzt also auf dem echten Kopf, ein Schnurrbart unter der
Schablonen-Nase unter der echten Nase, und beides wandert mit.
Offene Hand über einer Schablone löscht nur diese. Während des Malens sind nur
die Schablonen zu sehen. Mit `Masken aufsetzen & Foto` verschwinden sie, die
Masken sitzen auf den echten Gesichtern, und nach dem 3-2-1-Countdown entsteht
das Foto (Banner mit ✓, wenn alle Schablonen bemalt waren). Eine Anleitung zu Malen,
Absetzen, Löschen und Stift holen/abgeben steht in allen Malmodi im Bild.

### Per QR-Code teilen (ngrok)

> **Derzeit deaktiviert** (`NGROK_ENABLED = False` in `src/web_server.py`).

Installiere den [ngrok-Agenten](https://ngrok.com/download), melde dich dort an
und hinterlege einmalig deinen Token mit `ngrok config add-authtoken <TOKEN>`.
Anschliessend startet diese Variante einen temporaeren HTTPS-Tunnel zu einem
separaten Download-Server. Die Fotoautomaten-Website selbst bleibt auf
`localhost`; nach jeder Aufnahme zeigt sie einen QR-Code an, der direkt zu
diesem Foto fuehrt:

```powershell
python -m pip install -r requirements.txt
python src/web_server.py --ngrok
```

Jede Person mit dem QR-Code kann die oeffentliche Website waehrend der Laufzeit
des Tunnels aufrufen. Beende den Server mit `Strg+C`, um den Tunnel zu schliessen.
