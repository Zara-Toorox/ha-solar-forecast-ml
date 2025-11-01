# Historical Data Import für Solar Forecast ML Integration

## 📖 Übersicht

Dieses Skript ermöglicht den **einmaligen Import** historischer Sensordaten (bis zu 30 Tage) in die ML-Integration, um die Trainingsphase drastisch zu verkürzen.

**Vorteile:**
- ✅ **Sofortiger ML-Start** - Keine 30-tägige Wartezeit mehr
- ✅ **Isoliert** - Läuft unabhängig von der Integration (keine Absturzgefahr)
- ✅ **Flexibel** - Arbeitet mit nativen Home Assistant CSV-Exporten
- ✅ **Sicher** - Validierung & automatisches Backup

---

## 🚀 Schnellstart

### 1. CSV-Dateien exportieren

Exportiere folgende Sensoren aus Home Assistant (letzte 30 Tage):

#### **Pflicht-Sensoren:**
- `power_production.csv` - Solar-Leistungssensor (in Watt)
- `temperature.csv` - Temperatursensor
- `humidity.csv` - Luftfeuchtigkeitssensor

#### **Optional (falls vorhanden):**
- `wind_speed.csv` - Windgeschwindigkeitssensor
- `rain.csv` - Regensensor
- `uv_index.csv` - UV-Index-Sensor
- `lux.csv` - Helligkeitssensor (W/m² oder Lux)

---

### 2. CSV-Export in Home Assistant

**Schritte:**
1. Öffne **Developer Tools** → **Statistiken**
2. Wähle den entsprechenden Sensor aus
3. Klicke auf das **Download-Icon** (CSV-Export)
4. Wähle **Zeitraum**: Letzte 30 Tage
5. Speichere die Datei mit dem **exakten Namen** aus obiger Liste

**Wichtig:** Die Dateinamen müssen **genau** wie oben angegeben sein!

---

### 3. Verzeichnisstruktur erstellen

Navigiere zum Datenverzeichnis deiner Integration:

```bash
cd /config/custom_components/solar_forecast_ml/data/<ENTRY_ID>/
```

*(Ersetze `<ENTRY_ID>` mit deiner tatsächlichen Entry-ID, z.B. `01234567890abcdef`)*

Erstelle den Import-Ordner:

```bash
mkdir -p import
```

---

### 4. Dateien platzieren

Kopiere **alle exportierten CSV-Dateien** in den `import/` Ordner:

```bash
# Beispiel (passe Pfade an):
cp ~/downloads/power_production.csv import/
cp ~/downloads/temperature.csv import/
cp ~/downloads/humidity.csv import/
cp ~/downloads/wind_speed.csv import/  # optional
```

Kopiere das Import-Skript:

```bash
cp /pfad/zu/import_historical_data.py import/
```

---

### 5. Import ausführen

```bash
cd import
python3 import_historical_data.py
```

**Erwartete Ausgabe:**
```
============================================================
Solar Forecast ML - Historical Data Import
============================================================

1. Checking required files...
  ✓ Found power_production.csv
  ✓ Found temperature.csv
  ✓ Found humidity.csv

2. Parsing required sensor data...
Parsed power_production.csv: 720 hours, skipped 45 invalid entries
Parsed temperature.csv: 720 hours, skipped 12 invalid entries
...

✓ Import completed successfully!
  Total samples: 685
  Output file: /config/custom_components/.../hourly_samples.json
============================================================
```

---

### 6. Integration neu starten

Nach erfolgreichem Import:

1. **Home Assistant neu laden** (oder nur die Integration neu laden)
2. **Prüfe Log-Datei** auf Fehler
3. **Warte 2-5 Minuten** bis erstes ML-Training läuft

**Hinweis:** Die Integration erkennt automatisch die neuen Daten beim nächsten Start!

---

## 📋 CSV-Format-Referenz

Home Assistant exportiert CSVs im folgenden Format:

```csv
entity_id,state,last_changed
sensor.solar_power,1250.5,2025-09-30T12:34:56.789Z
sensor.solar_power,1248.2,2025-09-30T12:35:12.456Z
sensor.solar_power,unavailable,2025-09-30T12:36:00.123Z
sensor.solar_power,1255.0,2025-09-30T12:36:45.678Z
```

**Wichtig:**
- `state`: Sensor-Wert (numerisch oder "unavailable")
- `last_changed`: Zeitstempel in UTC (ISO 8601 Format)
- Das Skript **ignoriert** automatisch `unavailable`, `unknown` und `None`

---

## ⚙️ Technische Details

### Datenverarbeitung

1. **Zeitliche Aggregation:**
   - Alle Messwerte werden auf **volle Stunden** aggregiert
   - Zeitzone: UTC (intern), wird bei Speicherung konvertiert

2. **Leistung → Energie:**
   - Power-Sensor (Watt) wird via **Riemann-Integration** zu kWh
   - Formel: `∫ P(t) dt` über Stundenintervall

3. **Sensor-Durchschnitte:**
   - Temperatur, Feuchtigkeit, Wind, etc.: **Stundenmittelwert**
   - Bei fehlenden Werten: Defaults (0.0)

4. **Tagesberechnungen:**
   - `daily_total`: Kumulative Summe bis zur jeweiligen Stunde
   - `percentage_of_day`: Anteil der Stunde am Tagertrag

5. **Duplikat-Schutz:**
   - Vorhandene Samples werden **nicht** überschrieben
   - Nur neue Zeitstempel werden hinzugefügt

---

## 🔧 Fehlerbehandlung

### Häufige Probleme

**Problem:** `REQUIRED file missing: power_production.csv`
- **Lösung:** Stelle sicher, dass die Datei exakt `power_production.csv` heißt (Groß-/Kleinschreibung beachten!)

**Problem:** `No valid power production data found!`
- **Lösung:** 
  - Prüfe, ob die CSV gültige numerische Werte enthält
  - Vergewissere dich, dass der Zeitraum Produktionsstunden enthält (nicht nur Nacht)

**Problem:** `No valid samples after validation!`
- **Lösung:**
  - Mindestens eine der Dateien enthält keine gültigen Daten
  - Prüfe CSV-Format (Header vorhanden? Zeitstempel korrekt?)

---

## 🛡️ Sicherheit

- ✅ **Atomare Schreibvorgänge** - Keine Datenkorruption bei Abbruch
- ✅ **Validierung** - Ungültige Samples werden gefiltert
- ✅ **Isolation** - Import-Fehler betreffen NICHT die laufende Integration
- ✅ **Backup** - Bestehende Daten werden nie gelöscht, nur ergänzt

---

## 📊 Erwartete Ergebnisse

**Nach erfolgreichem Import:**
- **~240-720 Samples** (je nach Datenqualität und Produktionsstunden)
- **ML-Training startet** innerhalb von 2-5 Minuten
- **Prognose-Qualität** verbessert sich ab ~100 Samples deutlich
- **Optimale Genauigkeit** ab ~500 Samples (ca. 20-30 Tage)

---

## 🐛 Debug-Modus

Für detaillierte Fehlersuche:

```python
# In import_historical_data.py Zeile 40 ändern:
logging.basicConfig(
    level=logging.DEBUG,  # <- von INFO auf DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

---

## 📝 Beispiel-Output (Gekürzt)

```
2025-10-30 14:23:15 - INFO - Solar Forecast ML - Historical Data Import
2025-10-30 14:23:15 - INFO - Import directory: /config/.../import
2025-10-30 14:23:15 - INFO - Output file: /config/.../hourly_samples.json

2025-10-30 14:23:16 - INFO - Parsed power_production.csv: 720 hours, skipped 45
2025-10-30 14:23:16 - INFO - Parsed temperature.csv: 720 hours, skipped 12
2025-10-30 14:23:16 - INFO - Parsed humidity.csv: 720 hours, skipped 8

2025-10-30 14:23:17 - INFO - Created 685 valid samples with production > 0
2025-10-30 14:23:17 - INFO - Calculated daily totals for 28 days
2025-10-30 14:23:17 - INFO - Validated 685 samples

2025-10-30 14:23:18 - INFO - Loaded 0 existing samples
2025-10-30 14:23:18 - INFO - Added 685 new samples, pruned to 685 total
2025-10-30 14:23:18 - INFO - Successfully wrote 685 samples

✓ Import completed successfully!
  Total samples: 685
```

---

## ❓ FAQ

**F: Kann ich das Skript mehrfach ausführen?**  
A: Ja! Duplikate werden automatisch erkannt und übersprungen.

**F: Werden meine bestehenden Daten überschrieben?**  
A: Nein. Das Skript **fügt nur neue Samples hinzu**.

**F: Was passiert bei einem Fehler während des Imports?**  
A: Die Integration läuft normal weiter. Die Datei wird nur geschrieben, wenn der Import vollständig erfolgreich war.

**F: Wie viele Tage sollte ich importieren?**  
A: Empfohlen: **30 Tage**. Minimum: **7 Tage** (für erste ML-Funktionalität).

**F: Muss ich den Import wiederholen?**  
A: Nein. Nach einem erfolgreichen Import sammelt die Integration selbst weiter Daten.

---

## 🔗 Support

Bei Problemen:
1. Prüfe das **Log-Output** des Skripts
2. Aktiviere **DEBUG-Modus** (siehe oben)
3. Erstelle ein **GitHub Issue** mit:
   - Log-Ausgabe
   - Anzahl Zeilen in deinen CSV-Dateien
   - Home Assistant Version

---

## 📄 Lizenz

GNU Affero General Public License v3.0 (AGPL-3.0)  
Copyright (C) 2025 Zara-Toorox
