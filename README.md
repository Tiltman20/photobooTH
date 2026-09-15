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

## Lokale Website

Die Website waehlt bei jedem Neuaufruf zufaellig eine Challenge: Brille zeigen,
rote Haare zeigen, Schnurrbart zeigen oder mindestens drei Personen im Bild.
Die Kamera im Browser und das vorhandene lokale OpenCV-Modell erkennen die
Aufgabe. Alle Fotos verbleiben im Ordner
`captures/` auf diesem Rechner.

```powershell
python src/web_server.py
```

Danach `http://localhost:8000` im Browser oeffnen und den Kamerazugriff
erlauben. Die zufaellig vorausgewaehlte Challenge kann im Dropdown geaendert
werden. Mit `Strg+C` wird der Server beendet.

### Per QR-Code teilen (ngrok)

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
