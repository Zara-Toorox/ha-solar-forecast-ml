# Implementierungsplan: Monatliche Tarif-Verwaltung für EEG/Energy Sharing

## Übersicht

Dieses Feature ermöglicht die korrekte Kostenberechnung für Nutzer mit:
- Energiegemeinschaften (EEG) in Österreich
- Energy Sharing / Mieterstrom in Deutschland (ab Juli 2026)
- Variablen Tarifen mit unterschiedlichen Bezugs- und Einspeisepreisen
- Netzgebühren, die vom Jahresverbrauch abhängen

## Architektur

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SFML Stats Integration                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Hourly Aggregator│───▶│ Monthly Tariff   │◀───│ Dashboard UI  │  │
│  │                  │    │ Manager          │    │ (JavaScript)  │  │
│  │ - Speichert      │    │                  │    │               │  │
│  │   stündliche     │    │ - Berechnet Ø    │    │ - Tabelle     │  │
│  │   Preise + kWh   │    │ - Verwaltet      │    │ - Edit/Save   │  │
│  └──────────────────┘    │   Overrides      │    │ - Export      │  │
│                          │ - Neuberechnung  │    └───────────────┘  │
│                          └──────────────────┘                       │
│                                   │                                  │
│                                   ▼                                  │
│                     ┌──────────────────────────┐                    │
│                     │ monthly_tariffs.json     │                    │
│                     │ (Persistente Speicherung)│                    │
│                     └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Dateien und Änderungen

### Neue Dateien

#### 1. `services/monthly_tariff_manager.py`
```python
class MonthlyTariffManager:
    """Verwaltet monatliche Tarife mit Smart Defaults und Overrides."""

    async def get_monthly_data(self, year: int, month: int) -> dict
    async def set_monthly_override(self, year: int, month: int, data: dict) -> bool
    async def finalize_month(self, year: int, month: int) -> bool
    async def calculate_weighted_average_price(self, year: int, month: int) -> float
    async def recalculate_historical_data(self, year: int, month: int) -> bool
    async def get_all_months(self) -> list[dict]
    async def export_csv(self, start_date: date, end_date: date) -> str
```

#### 2. `data/monthly_tariffs.json` (Datenstruktur)
```json
{
  "defaults": {
    "reference_price_ct": 26.0,
    "feed_in_tariff_ct": 8.1,
    "eeg_import_price_ct": 18.0,
    "eeg_feed_in_ct": 12.0,
    "grid_fee_base_ct": 13.0,
    "grid_fee_scaling_enabled": true
  },
  "months": {
    "2025-01": {
      "auto_calculated": {
        "import_kwh": 234.5,
        "export_kwh": 45.2,
        "self_consumption_kwh": 312.8,
        "weighted_avg_price_ct": 32.47,
        "eeg_share_percent": 43.5
      },
      "overrides": {
        "import_price_ct": 31.80,
        "export_price_ct": 7.50,
        "grid_fee_ct": 18.20
      },
      "is_finalized": true,
      "finalized_at": "2025-02-15T10:30:00"
    }
  }
}
```

### Änderungen an bestehenden Dateien

#### 3. `services/hourly_aggregator.py`
- Neues Feld: `price_ct_kwh` pro Stunde speichern (bereits vorhanden, erweitern)
- Neue Methode: `get_monthly_price_data(year, month)` - liefert alle Stundenpreise

#### 4. `services/billing_calculator.py`
- Integration mit `MonthlyTariffManager`
- Neue Berechnungslogik: Monatsgenaue Preise statt globaler avg_price
- Unterscheidung: finalized vs. estimated

#### 5. `api/views.py`
Neue API-Endpunkte:
```
GET  /api/sfml_stats/monthly_tariffs
GET  /api/sfml_stats/monthly_tariffs/{year}/{month}
POST /api/sfml_stats/monthly_tariffs/{year}/{month}
POST /api/sfml_stats/monthly_tariffs/{year}/{month}/finalize
GET  /api/sfml_stats/monthly_tariffs/export?start=2025-01&end=2025-12
```

#### 6. `www/dashboard.html` + `www/js/dashboard.js`
- Neuer Tab: "Tarife & Abrechnung"
- Interaktive Tabelle mit Edit-Funktion
- Icons für Auto/Override/Finalized Status
- CSV Export Button

#### 7. `config_flow.py`
Neuer Konfigurationsschritt: "Tarif-Defaults"
- Referenzpreis (ct/kWh) - "Was würde Strom ohne PV kosten?"
- Standard-Einspeisevergütung (ct/kWh)
- Optional: EEG-Tarife (Bezug/Einspeisung)
- Optional: Netzgebühren-Skalierung aktivieren

#### 8. `const.py`
Neue Konstanten:
```python
# Monthly Tariff Feature
MONTHLY_TARIFFS_FILE = "monthly_tariffs.json"
CONF_REFERENCE_PRICE = "reference_price"
CONF_EEG_IMPORT_PRICE = "eeg_import_price"
CONF_EEG_FEED_IN_PRICE = "eeg_feed_in_price"
CONF_GRID_FEE_BASE = "grid_fee_base"
CONF_GRID_FEE_SCALING = "grid_fee_scaling"
DEFAULT_REFERENCE_PRICE = 26.0
DEFAULT_GRID_FEE_BASE = 13.0
```

#### 9. `translations/de.json` + `translations/en.json`
Neue Übersetzungen für:
- Config Flow Schritt
- Dashboard Texte
- Tooltips und Erklärungen

## Berechnungslogik

### 1. Gewichteter Durchschnittspreis
```python
def calculate_weighted_average_price(hourly_data: list) -> float:
    """
    Berechnet den verbrauchsgewichteten Durchschnittspreis.

    Beispiel:
    - Stunde 1: 0.3 kWh @ 25 ct = 7.5 ct
    - Stunde 2: 0.0 kWh @ 45 ct = 0.0 ct (PV deckt alles)
    - Stunde 3: 1.2 kWh @ 38 ct = 45.6 ct

    Gesamt: 1.5 kWh, 53.1 ct → Ø 35.4 ct/kWh
    """
    total_cost_ct = sum(h["import_kwh"] * h["price_ct"] for h in hourly_data)
    total_kwh = sum(h["import_kwh"] for h in hourly_data)
    return total_cost_ct / total_kwh if total_kwh > 0 else 0
```

### 2. EEG-Anteil Schätzung
```python
def estimate_eeg_share(
    weighted_price: float,
    standard_price: float,
    eeg_price: float
) -> float:
    """
    Schätzt den EEG-Anteil aus der Preisdifferenz.

    Formel: EEG% = (Standard - Gewichtet) / (Standard - EEG) * 100

    Beispiel:
    - Gewichteter Preis: 22.5 ct
    - Standard-Tarif: 26.0 ct
    - EEG-Tarif: 18.0 ct
    → EEG-Anteil ≈ (26-22.5)/(26-18) = 43.75%
    """
    if standard_price <= eeg_price:
        return 0.0
    return max(0, min(100,
        (standard_price - weighted_price) / (standard_price - eeg_price) * 100
    ))
```

### 3. Netzgebühren-Skalierung
```python
def calculate_grid_fee(
    annual_import_kwh: float,
    base_fee_ct: float = 13.0
) -> float:
    """
    Berechnet skalierte Netzgebühren basierend auf Jahresverbrauch.

    Typische Staffelung (vereinfacht):
    - > 5000 kWh: base_fee (z.B. 13 ct)
    - 2500-5000 kWh: base_fee * 1.3
    - 1000-2500 kWh: base_fee * 1.6
    - < 1000 kWh: base_fee * 2.0

    Dies ist eine Schätzung - echte Werte variieren je Netzbetreiber.
    """
    if annual_import_kwh > 5000:
        return base_fee_ct
    elif annual_import_kwh > 2500:
        return base_fee_ct * 1.3
    elif annual_import_kwh > 1000:
        return base_fee_ct * 1.6
    else:
        return base_fee_ct * 2.0
```

### 4. Korrigierte Einsparungsberechnung
```python
def calculate_savings(
    self_consumption_kwh: float,
    reference_price_ct: float,  # Was Strom ohne PV kosten würde
    actual_import_price_ct: float  # Was tatsächlich bezahlt wird
) -> dict:
    """
    Berechnet die echte Einsparung.

    Einsparung = Eigenverbrauch × Referenzpreis
    NICHT: Eigenverbrauch × aktueller Bezugspreis

    Denn: Der Referenzpreis ist das, was der User OHNE PV
    für den gesamten Verbrauch zahlen würde.
    """
    return {
        "savings_eur": (self_consumption_kwh * reference_price_ct) / 100,
        "reference_price_used_ct": reference_price_ct
    }
```

## UI-Design (Dashboard)

### Tab: "Tarife & Abrechnung"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Monatliche Tarife & Abrechnung                      [⚙️ Defaults] [📥 CSV] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 2025                                                        [◀ 2024]   ││
│  ├────────┬────────┬────────────┬────────┬──────────┬────────┬──────────┬─┤│
│  │ Monat  │ Bezug  │ Bezugspreis│Einspei.│ Vergütung│Referenz│Netzgeb.  │S││
│  │        │ (kWh)  │ (ct/kWh)   │ (kWh)  │ (ct/kWh) │(ct/kWh)│(ct/kWh)  │ ││
│  ├────────┼────────┼────────────┼────────┼──────────┼────────┼──────────┼─┤│
│  │ Jan    │ 234.5  │ 32.47 🤖   │ 45.2   │ 8.10 ⚙️  │ 26.0 ⚙️│ 18.2 ✏️  │✅││
│  │ Feb    │ 198.3  │ 31.80 ✏️   │ 67.8   │ 7.50 ✏️  │ 26.0 ⚙️│ 18.0 ✏️  │✅││
│  │ Mär    │ 156.2  │ 29.15 🤖   │ 123.4  │ 8.10 ⚙️  │ 26.0 ⚙️│ ~17.5 📊 │⏳││
│  │ Apr    │ —      │ —          │ —      │ —        │ —      │ —        │—││
│  └────────┴────────┴────────────┴────────┴──────────┴────────┴──────────┴─┘│
│                                                                              │
│  Legende: 🤖 Auto-berechnet  ⚙️ Default  ✏️ Manuell  📊 Geschätzt  ✅ Final │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Zusammenfassung 2025 (bis März)                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stromkosten:        189.12 €   (Bezug × Bezugspreis)                  │ │
│  │  Einspeise-Erlös:     18.74 €   (Einspeisung × Vergütung)              │ │
│  │  Einsparung:         285.67 €   (Eigenverbrauch × Referenzpreis)       │ │
│  │  ────────────────────────────────────────────────────                  │ │
│  │  Netto-Vorteil PV:   115.29 €                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edit-Modal (bei Klick auf Zeile)

```
┌─────────────────────────────────────────────────────┐
│  Tarife bearbeiten: März 2025                    ✕  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Bezugspreis (ct/kWh)                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ 29.15                              🤖 Auto  │   │
│  └─────────────────────────────────────────────┘   │
│  ☐ Manuell überschreiben: [_______]                │
│                                                     │
│  Einspeisevergütung (ct/kWh)                       │
│  ┌─────────────────────────────────────────────┐   │
│  │ 8.10                               ⚙️ Default│   │
│  └─────────────────────────────────────────────┘   │
│  ☐ Manuell überschreiben: [_______]                │
│                                                     │
│  Netzgebühren (ct/kWh)                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ ~17.5                              📊 Gesch.│   │
│  └─────────────────────────────────────────────┘   │
│  ☐ Manuell überschreiben: [_______]                │
│                                                     │
│  ─────────────────────────────────────────────     │
│  ☐ Monat als abgerechnet markieren (finalisieren)  │
│    → Historische Daten werden neu berechnet        │
│                                                     │
│            [Abbrechen]  [Speichern]                │
└─────────────────────────────────────────────────────┘
```

## Implementierungsreihenfolge

### Phase 1: Grundgerüst (Priorität: Hoch)
1. [ ] `MonthlyTariffManager` Service erstellen
2. [ ] Datenstruktur und JSON-Speicherung
3. [ ] Gewichtete Durchschnittsberechnung
4. [ ] API-Endpunkte

### Phase 2: Config & Defaults (Priorität: Hoch)
5. [ ] Config Flow erweitern (neue Defaults)
6. [ ] Bestehende `billing_calculator.py` anpassen
7. [ ] Translations (DE/EN)

### Phase 3: Dashboard UI (Priorität: Hoch)
8. [ ] Neuer Tab im Dashboard
9. [ ] Tabellen-Komponente mit Edit-Funktion
10. [ ] Status-Icons und Legende

### Phase 4: Erweiterte Features (Priorität: Mittel)
11. [ ] Rückwirkende Neuberechnung bei Finalisierung
12. [ ] CSV Export
13. [ ] Netzgebühren-Schätzung
14. [ ] EEG-Anteil Rückrechnung

### Phase 5: Polish (Priorität: Niedrig)
15. [ ] Tooltips und Hilfe-Texte
16. [ ] Mobile-optimierte Darstellung
17. [ ] Validierung und Fehlermeldungen
18. [ ] Dokumentation

## Testfälle

1. **Automatische Berechnung**: Monat ohne Overrides zeigt gewichteten Durchschnitt
2. **Override**: Manuell eingegebener Preis ersetzt Auto-Wert
3. **Finalisierung**: Historische Tage werden mit neuen Preisen neu berechnet
4. **Default-Änderung**: Neue Defaults werden nur für nicht-finalisierte Monate angewendet
5. **CSV Export**: Alle Monate korrekt exportiert
6. **Leerer Monat**: Monate ohne Daten werden korrekt als "—" angezeigt

## Offene Fragen

1. Sollen finalisierte Monate noch editierbar sein? (Vorschlag: Ja, mit Warnung)
2. Wie weit zurück sollen historische Daten neu berechnet werden? (Vorschlag: Nur aktuelle Billing-Periode)
3. Soll die Netzgebühren-Staffelung konfigurierbar sein? (Vorschlag: Später, erstmal feste Schätzung)
