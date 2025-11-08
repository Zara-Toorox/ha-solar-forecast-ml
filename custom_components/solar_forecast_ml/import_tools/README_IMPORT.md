# Historical Data Import - Anleitung

## Übersicht

Hallo! Mit diesem Skript kannst du historische Sensordaten aus Home Assistant in die Solar Forecast ML Integration importieren. So lernt das ML-Modell von deinen vergangenen Daten und macht bessere Vorhersagen.

Version: 8.2.6 (aktualisiert am 08.11.2025)

---

## Was macht das Skript genau?

Das Skript übernimmt ein paar nützliche Aufgaben:
- Es importiert CSV-Exporte von deinen Home Assistant Sensoren.
- Es berechnet stündliche kWh-Werte aus den Leistungsdaten, indem es die Riemann-Integration anwendet.
- Es leitet Wetterdaten ab, wie Cloud Cover und Condition, basierend auf Irradiance, Lux, UV oder Regen.
- Es merged die neuen Daten mit deinen bestehenden, ohne Duplikate zu erzeugen.
- Am Ende speichert es alles direkt in die passende Datei, nämlich hourly_samples.json.

---

## Verzeichnis-Struktur

Deine Dateien sollten so organisiert sein:

Im Ordner /config/ findest du:
- solar_forecast_ml/
  - imports/  (Hier legst du deine CSV-Dateien ab!)
    - power_production.csv  (Pflicht)
    - temperature.csv  (Pflicht)
    - humidity.csv  (Pflicht)
    - wind_speed.csv  (optional)
    - rain.csv  (optional)
    - uv_index.csv  (optional)
    - lux.csv  (optional)
    - irradiance.csv  (optional)
  - ml/
    - hourly_samples.json  (Hier schreibt das Skript die Ergebnisse hin)

---

## Schritt-für-Schritt Anleitung

### Schritt 1: Verzeichnis erstellen

Falls das Import-Verzeichnis noch nicht da ist, erstelle es einfach mit diesem Befehl:

mkdir -p /config/solar_forecast_ml/imports

---

### Schritt 2: Sensordaten aus Home Assistant exportieren

Geh in Home Assistant zu Entwicklerwerkzeuge und dann zu Statistiken. Exportiere dort die folgenden Sensoren als CSV-Dateien.

Pflicht-Sensoren:
- Solar-Leistung: Speichere als power_production.csv (das ist die aktuelle Leistung in Watt).
- Temperatur: Als temperature.csv (Außentemperatur in Grad Celsius).
- Luftfeuchtigkeit: Als humidity.csv (in Prozent).

Optionale Sensoren, die ich empfehle, weil sie die ML-Ergebnisse verbessern:
- Windgeschwindigkeit: Als wind_speed.csv (in km/h).
- Regen: Als rain.csv (Niederschlag in mm).
- UV-Index: Als uv_index.csv (Werte von 0 bis 11 oder höher).
- Helligkeit: Als lux.csv (Lichtstärke in Lux).
- Irradiance: Als irradiance.csv (Sonneneinstrahlung in W/m²).

Tipp: Exportiere am besten 30 bis 90 Tage Daten, damit das ML-Modell richtig trainieren kann.

---

### Schritt 3: CSV-Dateien ins Import-Verzeichnis kopieren

Kopiere alle deine exportierten CSV-Dateien in den Ordner /config/solar_forecast_ml/imports/.

Zum Beispiel so:
cp ~/Downloads/sensor.solar_power*.csv /config/solar_forecast_ml/imports/power_production.csv
cp ~/Downloads/sensor.temperature*.csv /config/solar_forecast_ml/imports/temperature.csv
cp ~/Downloads/sensor.humidity*.csv /config/solar_forecast_ml/imports/humidity.csv

---

### Schritt 4: Skript ausführen

Du hast zwei Optionen:

Option A: Direkt aus der Home Assistant Shell:
python3 /config/custom_components/solar_forecast_ml/import_tools/import_historical_data.py

Option B: Über SSH oder Terminal (bei Docker):
docker exec homeassistant python3 /config/custom_components/solar_forecast_ml/import_tools/import_historical_data.py

---

### Schritt 5: Output prüfen

Das Skript gibt dir detaillierte Logs aus, die so aussehen könnten:

Solar Forecast ML - Historical Data Import

1. Checking required files...
  Found power_production.csv
  Found temperature.csv
  Found humidity.csv

2. Parsing required sensor data...
Parsed power_production.csv: 2160 hours, skipped 143 invalid entries
Parsed temperature.csv: 2160 hours, skipped 23 invalid entries
Parsed humidity.csv: 2160 hours, skipped 15 invalid entries

3. Parsing optional sensor data...
  Parsed lux.csv
  Parsed uv_index.csv
  Skipping wind_speed.csv (not found)

4. Merging hourly data...
Processing 2160 hours of data...
Created 1843 valid samples with production > 0

5. Calculating daily totals and percentages...
Calculated daily totals for 90 days

6. Validating samples...
Validated 1843 samples

7. Loading existing samples...
Loaded 288 existing samples

8. Merging with existing data...
Added 1555 new samples, pruned to 1843 total (keeping last 90 days)

9. Writing output file...
Successfully wrote 1843 samples to /config/solar_forecast_ml/ml/hourly_samples.json

Import completed successfully!
  Total samples: 1843
  Output file: /config/solar_forecast_ml/ml/hourly_samples.json

---

## Erfolgsprüfung

Nach dem Import:
1. Schau dir die Anzahl der Samples an:
   cat /config/solar_forecast_ml/ml/hourly_samples.json | grep '"count"'

2. Starte Home Assistant neu, damit die Integration die neuen Daten lädt.

3. Trigger das ML-Training: Geh zu Entwicklerwerkzeuge > Services und rufe den Service solar_forecast_ml.force_retrain auf.

4. Prüfe die Logs, die so aussehen könnten:
   Scheduled model training completed successfully.
   Training samples: 1843
   Model accuracy: 0.87

---

## Konfiguration

### Retention (Aufbewahrungszeit)

Standardmäßig behält es 90 Tage Daten bei (das steht in Zeile 42 des Skripts).

Wenn du das ändern möchtest, bearbeite diese Zeile:
MAX_RETENTION_DAYS = 90  # Empfohlen: 60-365

Mehr Tage bedeuten mehr Trainingsdaten, aber auch größere Dateien. Weniger Tage machen alles schneller und sparen Speicher.

---

## Troubleshooting

### Error: "Import directory not found"

Lösung: Erstelle das Verzeichnis mit:
mkdir -p /config/solar_forecast_ml/imports

### Error: "REQUIRED file missing"

Lösung: Achte darauf, dass die drei Pflichtdateien da sind:
- power_production.csv
- temperature.csv
- humidity.csv

### Error: "No valid power production data found!"

Ursache: Die CSV hat nur "unavailable" oder "unknown" Werte.

Lösung: Exportiere einen anderen Zeitraum oder prüfe, ob dein Sensor richtig arbeitet.

### Warning: "Skipped X invalid entries"

Das ist normal. Das Skript filtert automatisch ungültige Dinge wie unavailable, unknown, None oder Stunden mit null Produktion (z. B. nachts).

---

## CSV-Format

Das Skript erwartet die Standard-CSV-Exporte aus Home Assistant, die so aussehen:

entity_id,state,last_changed,last_updated
sensor.solar_power,1250.5,2025-10-01T10:00:00+00:00,2025-10-01T10:00:00+00:00
sensor.solar_power,1305.2,2025-10-01T10:05:00+00:00,2025-10-01T10:05:00+00:00
...

Wichtig:
- Timestamps sollten in UTC oder mit Zeitzone sein.
- Der State muss eine Zahl sein.
- Der CSV-Header muss dabei sein.

---

## Tipps für beste Ergebnisse

1. Exportiere immer ganze Tage, von 00:00 bis 23:59.
2. Nimm mindestens 30 Tage für das Training.
3. Verschiede Wetterbedingungen einbeziehen: sonnig, wolkig, regnerisch.
4. Verwende Irradiance (W/m²), wenn möglich – das ist besser als Lux.
5. Prüfe vor dem Export, ob deine Sensoren zuverlässig sind und wenige unavailable haben.

---

## Hinweise

- Duplikate werden automatisch vermieden, also kannst du das Skript mehrmals laufen lassen.
- Alte Daten werden nach der Retention-Zeit gelöscht.
- Es wird kein Backup erstellt – die Integration macht das selbst.
- Das Skript ist sicher und zerstört nichts; bestehende Daten bleiben erhalten.

---

## Support

Falls was schiefgeht:
1. Schau in die Logs des Skripts für detaillierte Fehlermeldungen.
2. Prüfe die Home Assistant Logs in /config/home-assistant.log.
3. Frag im GitHub: https://github.com/Zara-Toorox/ha-solar-forecast-ml/issues

Erstellt von: Zara-Toorox
Letzte Aktualisierung: 08.11.2025
Version: 8.2.6 "Sarpeidon"