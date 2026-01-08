# SFML Stats - Sensor Handbuch

## Allgemeine Hinweise

Alle Sensoren sind optional. Die Integration funktioniert auch mit unvollstaendiger Konfiguration, zeigt dann aber nur die verfuegbaren Daten an.

Leistungssensoren koennen in Watt (W) oder Kilowatt (kW) angegeben werden. Die Integration erkennt die Einheit automatisch.

Energiesensoren sollten in Kilowattstunden (kWh) vorliegen.

---

## Schritt 1: Grundeinstellungen

Hier werden keine Sensoren benoetigt. Es handelt sich um reine Konfigurationsoptionen fuer die automatische Generierung von Charts und das Farbschema.

---

## Schritt 2: Energiefluss-Sensoren

### sensor_solar_power
Aktuelle Gesamtleistung der Photovoltaikanlage.
- Einheit: W 
- Quelle: Wechselrichter-Integration
- Beispiel: sensor.wechselrichter_pv_leistung

### sensor_solar_to_house
Anteil der Solarleistung der direkt im Haus verbraucht wird.
- Einheit: W
- Quelle: Wechselrichter oder Energiemanagement-System
- Hinweis: Nicht alle Wechselrichter liefern diesen Wert. Falls nicht vorhanden, kann er oft aus anderen Werten berechnet werden.

### sensor_solar_to_battery
Solarleistung die in die Batterie fliesst.
- Einheit: W
- Quelle: Batterie-Management-System oder Hybrid-Wechselrichter
- Hinweis: Nur relevant bei Speichersystemen.

### sensor_grid_to_house
Leistung die aktuell aus dem Stromnetz bezogen wird.
- Einheit: W
- Quelle: Smartmeter oder Wechselrichter
- Hinweis: Manche Systeme zeigen nur den Gesamtnetzbezug ohne Aufteilung.

### sensor_grid_to_battery
Netzstrom der zum Laden der Batterie verwendet wird.
- Einheit: W
- Quelle: Batterie-Management-System
- Hinweis: Relevant bei Systemen die Netzladung erlauben.

### sensor_house_to_grid
Einspeiseleistung ins Stromnetz.
- Einheit: W
- Quelle: Smartmeter oder Wechselrichter
- Hinweis: Zeigt den aktuellen Ueberschuss der eingespeist wird.

### sensor_smartmeter_import
Gesamter Strombezug vom Smartmeter.
- Einheit: kWh
- Quelle: Smartmeter-Integration
- Hinweis: Kann ein Zaehlerstand (stetig steigend) oder ein Tageszaehler sein.

### sensor_smartmeter_export
Gesamte Einspeisung laut Smartmeter.
- Einheit: kWh
- Quelle: Smartmeter-Integration
- Hinweis: Entspricht dem Einspeisezaehler.

---

## Schritt 3: Batterie-Sensoren

### sensor_battery_soc
Aktueller Ladestand der Batterie in Prozent.
- Einheit: Prozent (0-100)
- Quelle: Batterie-Management-System
- Beispiel: sensor.batterie_ladestand

### sensor_battery_power
Aktuelle Lade- oder Entladeleistung der Batterie.
- Einheit: W
- Quelle: Batterie-Management-System
- Hinweis: Positive Werte bedeuten Laden, negative Werte Entladen. Manche Systeme verwenden die umgekehrte Logik.

### sensor_battery_to_house
Entladeleistung die ins Haus fliesst.
- Einheit: W
- Quelle: Batterie-Management-System oder Hybrid-Wechselrichter

### sensor_battery_to_grid
Entladeleistung die ins Netz eingespeist wird.
- Einheit: W
- Quelle: Batterie-Management-System
- Hinweis: Nur relevant wenn das System Batteriestrom einspeisen kann.

### sensor_home_consumption
Gesamter aktueller Stromverbrauch des Haushalts.
- Einheit: W
- Quelle: Energiemanagement-System oder eigene Berechnung
- Hinweis: Summe aus allen Quellen (Solar, Batterie, Netz).

---

## Schritt 4: Statistik-Sensoren

### sensor_solar_yield_daily
Tagesertrag der Photovoltaikanlage.
- Einheit: kWh
- Quelle: Wechselrichter oder Home Assistant Utility Meter
- Hinweis: Sollte um Mitternacht auf Null zurueckgesetzt werden.

### sensor_grid_import_daily
Tagesbezug aus dem Stromnetz.
- Einheit: kWh
- Quelle: Smartmeter oder Home Assistant Utility Meter
- Hinweis: Sollte um Mitternacht auf Null zurueckgesetzt werden.

### sensor_grid_import_yearly
Jahresbezug aus dem Stromnetz.
- Einheit: kWh
- Quelle: Smartmeter oder Home Assistant Utility Meter
- Hinweis: Wird fuer Jahresstatistiken verwendet.

### sensor_battery_charge_solar_daily
Tagesladung der Batterie aus Solarstrom.
- Einheit: kWh
- Quelle: Batterie-Management-System oder eigene Integration

### sensor_battery_charge_grid_daily
Tagesladung der Batterie aus Netzstrom.
- Einheit: kWh
- Quelle: Batterie-Management-System

### sensor_price_total
Gesamtkosten fuer den Strombezug.
- Einheit: Euro oder Cent
- Quelle: Eigene Berechnung oder Energiekosten-Integration (GPM)

### weather_entity
Wetter-Entitaet fuer Wetteranzeige in den Charts.
- Typ: Weather-Entity (nicht Sensor)
- Quelle: Beliebige Home Assistant Wetter-Integration
- Beispiel: weather.zuhause

---

## Schritt 5: Panel-Sensoren

Die Integration unterstuetzt bis zu vier separate Strings oder Panelgruppen. Fuer jede Gruppe gibt es drei Felder.

### panel_name (1-4)
Frei waehlbarer Name fuer den String.
- Typ: Text
- Beispiele: Sued, Ost, West, Garage, Carport
- Voreinstellung: String 1, String 2, String 3, String 4

### sensor_panel_power (1-4)
Aktuelle Leistung dieses Strings.
- Einheit: W
- Quelle: Wechselrichter mit MPPT-Tracking
- Hinweis: Nicht alle Wechselrichter geben einzelne Stringwerte aus.

### sensor_panel_max_today (1-4)
Hoechste erreichte Leistung des Strings am aktuellen Tag.
- Einheit: W
- Quelle: Eigene Berechnung oder Wechselrichter
- Hinweis: Kann mit einem Home Assistant Max-Sensor erstellt werden.

---

## Schritt 6: Abrechnung

Hier werden keine Sensoren benoetigt. Es werden nur Parameter fuer die Kostenberechnung festgelegt.

Der Abrechnungszeitraum legt fest ab welchem Tag und Monat die jaehrliche Energiebilanz berechnet wird. Das entspricht typischerweise dem Beginn des Stromliefervertrags.

Der Preismodus kann dynamisch (ueber eine separate Integration) oder als fester Wert in Cent pro Kilowattstunde angegeben werden.

---

## Praktische Tipps

### Fehlende Sensoren erstellen

Falls ein benoetiger Sensor nicht direkt verfuegbar ist, kann er oft in Home Assistant erstellt werden:

Fuer Tageszaehler den Utility Meter Helper verwenden mit taeglichem Reset.

Fuer Maximalwerte den Statistics Helper verwenden mit der Funktion Maximum.

Fuer berechnete Werte einen Template Sensor anlegen.

### Typische Sensorquellen nach Hersteller

Fronius: Die Integration liefert die meisten Werte direkt.

SMA: Modbus oder SMA Energy Meter Integration verwenden.

Huawei: FusionSolar Integration oder Modbus.

Kostal: Piko oder Plenticore Integration.

Growatt: Growatt Server Integration.

Sungrow: Modbus Integration.

Enphase: Envoy Integration liefert Panel-Daten.

### Smartmeter anbinden

Fuer Smartmeter gibt es verschiedene Wege:

ISKRA oder andere mit IR-Lesekopf und ESPHome.

P1 Port bei niederlaendischen und belgischen Zaehlern.

Shelly EM oder 3EM als Zwischenloesung.

Direkte Modbus-Verbindung bei modernen Zaehlern.

### Utility Meter einrichten

Fuer Tageszaehler in der configuration.yaml:

```yaml
utility_meter:
  solar_daily:
    source: sensor.wechselrichter_energie_gesamt
    cycle: daily
  grid_import_daily:
    source: sensor.smartmeter_bezug_gesamt
    cycle: daily
```

### Template Sensor fuer Hausverbrauch

Falls kein direkter Sensor vorhanden:

```yaml
template:
  - sensor:
      - name: "Hausverbrauch"
        unit_of_measurement: "W"
        state: >
          {{ states('sensor.solar_power')|float(0)
             + states('sensor.grid_import')|float(0)
             - states('sensor.grid_export')|float(0) }}
```
