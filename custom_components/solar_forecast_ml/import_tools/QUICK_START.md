GER / ENG below

# 🚀 Schnellstart: Solar Forecast ML Datenimport

Mit dieser Anleitung kannst du deine Solardaten der letzten 30 Tage importieren, um das Training des Modells sofort zu starten, ohne wochenlang warten zu müssen.

### ❗ Wichtiger Hinweis

Damit der Import funktioniert, müssen die Sensoren, deren Daten du hier herunterlädst (z.B. `power_production.csv`, `temperature.csv` usw.), **exakt dieselben Sensoren sein, die du auch in der Konfiguration der "Solar Forecast ML" Integration in Home Assistant ausgewählt hast**.

-----

### Schritt 1: 🖥️ Daten herunterladen (im Browser)

Das machst du in der normalen Home Assistant Weboberfläche.

1.  Öffne **Entwicklerwerkzeuge** (Hammer-Symbol) → **Statistiken**.

2.  Suche den Sensor, klicke auf das Download-Symbol (Pfeil nach unten).

3.  Wähle den Zeitraum: **"Letzte 30 Tage"**.

4.  Speichere die Dateien mit **exakt** diesen Namen ab:

/config/custom_components/solar_forecast_ml/
├── import_historical_data.py
├── import/                          ← CSV-Dateien hier ablegen
│   ├── power_production.csv        ✅ PFLICHT
│   ├── temperature.csv             ✅ PFLICHT
│   ├── humidity.csv                ✅ PFLICHT
│   ├── wind_speed.csv              ⭕ Optional
│   ├── rain.csv                    ⭕ Optional
│   ├── uv_index.csv                ⭕ Optional
│   ├── irradiance.csv              ⭕ Optional (W/m²)
│   └── lux.csv                     ⭕ Optional (LUX)
└── hourly_samples.json             ← Output

> **Wichtig:** Lade nur die CSV-Dateien für die Sensoren herunter, die du auch wirklich in der Integration konfiguriert hast. Die ersten drei (`power`, `temperature`, `humidity`) sind aber fast immer erforderlich.

-----

### Schritt 2: 📂 CSV-Dateien hochladen (mit File Explorer)

Jetzt benutzt wir das **File Explorer** Add-on.

1.  Öffne den **File Explorer**.
2.  Klicke dich links durch die Ordner: `config` → `solar_forecast_ml` → `import`.
3.  Der Pfad oben in der Leiste sollte jetzt `/config/solar_forecast_ml/import` anzeigen.
4.  Klicke oben auf das **"Upload"**-Symbol (Pfeil nach oben).
5.  Lade **alle deine CSV-Dateien** (die du in Schritt 1 heruntergeladen hast) in diesen Ordner hoch.

-----

### Schritt 3: ✅ Daten prüfen (mit SSH Terminal)

Jetzt wechseln wir zum **SSH Terminal** Add-on.

1.  Öffne das **SSH Terminal**.
2.  Tippe den folgenden Befehl ein, um in den richtigen Ordner zu wechseln:
    ```bash
    cd /config/solar_forecast_ml/import
    ```
3.  Drücke **Enter**.
4.  Tippe nun den Befehl ein, um die **Prüfung** zu starten:
    ```bash
    python3 validate_csv_files.py
    ```
5.  Drücke **Enter**.
6.  **WICHTIG:** Schaue dir die Ausgabe an. Wenn alles gut ist, siehst du eine Erfolgsmeldung.

-----

### Schritt 4: 🏃 Import starten (NUR bei Erfolg\!)

> **STOPP\!** Führe diesen Schritt **nur** aus, wenn Schritt 3 **KEINE FEHLER** angezeigt hat.
> Bei Fehlern: Starte den Import **NICHT**. Du musst die CSV-Dateien korrigieren und sie in Schritt 2 neu hochladen.

1.  Stelle sicher, dass du noch im richtigen Ordner bist (`/config/solar_forecast_ml/import`).
2.  Tippe den Befehl ein, um den **Import** zu starten:
    ```bash
    python3 import_historical_data.py
    ```
3.  Drücke **Enter**.
4.  Warte auf die Erfolgsmeldung, z.B. `✓ Import completed successfully!`.

-----

### Schritt 5: 🔄 Neustart

Wenn der Import (Schritt 4) erfolgreich war:

1.  Gehe zu **Einstellungen** → **System**.
2.  Klicke oben rechts auf das Power-Symbol.
3.  Wähle **"Home Assistant neu starten"**.

Fertig\! Nach dem Neustart (2-5 Minuten warten) beginnt die ML-Integration sofort mit den importierten Daten zu trainieren.

_____________________
_______________
### English Version:
---------------
--------------------

# 🚀 Quick Start: Solar Forecast ML Data Import

With this guide, you can import your solar data from the last 30 days to start the model's training immediately, without having to wait for weeks.

### ❗ Important Note

For the import to work, the sensors you are downloading data for (e.g., `power_production.csv`, `temperature.csv`, etc.) **must be the exact same sensors that you have selected in the "Solar Forecast ML" integration's configuration in Home Assistant**.

-----

### Step 1: 🖥️ Download Data (in your Browser)

You do this in the normal Home Assistant web interface.

1.  Open **Developer Tools** (hammer icon) → **Statistics**.

2.  Find the sensor, click the Download icon (arrow pointing down).

3.  Select the time range: **"Last 30 days"**.

4.  Save the files with these **exact** names:

/config/custom_components/solar_forecast_ml/
├── import_historical_data.py
├── import/                          ← CSV-Dateien hier ablegen
│   ├── power_production.csv        ✅ needed
│   ├── temperature.csv             ✅ needed
│   ├── humidity.csv                ✅ needed
│   ├── wind_speed.csv              ⭕ Optional
│   ├── rain.csv                    ⭕ Optional
│   ├── uv_index.csv                ⭕ Optional
│   ├── irradiance.csv              ⭕ Optional (W/m²)
│   └── lux.csv                     ⭕ Optional (LUX)
└── hourly_samples.json             ← Output

> **Important:** Only download the CSV files for the sensors that you have actually configured in the integration. The first three (`power`, `temperature`, `humidity`) are almost always required.

-----

### Step 2: 📂 Upload CSV Files (with File Explorer)

Now, we'll use the **File Explorer** Add-on.

1.  Open the **File Explorer**.
2.  Click through the folders on the left: `config` → `solar_forecast_ml` → `import`.
3.  The path at the top should now show `/config/solar_forecast_ml/import`.
4.  Click the **"Upload"** icon at the top (arrow pointing up).
5.  Upload **all your CSV files** (that you downloaded in Step 1) into this folder.

-----

### Step 3: ✅ Validate Data (with SSH Terminal)

Now, let's switch to the **SSH Terminal** Add-on.

1.  Open the **SSH Terminal**.
2.  Type the following command to change to the correct folder:
    ```bash
    cd /config/solar_forecast_ml/import
    ```
3.  Press **Enter**.
4.  Now, type the command to start the **validation**:
    ```bash
    python3 validate_csv_files.py
    ```
5.  Press **Enter**.
6.  **IMPORTANT:** Look at the output. If everything is okay, you will see a success message.

-----

### Step 4: 🏃 Run Import (ONLY on Success\!)

> **STOP\!** Only perform this step if Step 3 showed **NO ERRORS**.
> If there were errors: **DO NOT** run the import. You must fix the CSV files and re-upload them in Step 2.

1.  Make sure you are still in the correct folder (`/config/solar_forecast_ml/import`).
2.  Type the command to start the **import**:
    ```bash
    python3 import_historical_data.py
    ```
3.  Press **Enter**.
4.  Wait for the success message, e.g., `✓ Import completed successfully!`.

-----

### Step 5: 🔄 Restart

If the import (Step 4) was successful:

1.  Go to **Settings** → **System**.
2.  Click the Power icon in the top-right corner.
3.  Select **"Restart Home Assistant"**.

Done\! After the restart (wait 2-5 minutes), the ML integration will immediately start training with the imported data.