# Solar Forecast ML - Benutzerhandbuch Version 10.0.0 "Lyra"

**Professionelle Dokumentation für die Solar Forecast ML Integration**

Version 10.0.0 "Lyra" | Erstellt: November 2025 | Sprache: Deutsch

---

## Inhaltsverzeichnis

1. [Einführung](#einführung)
2. [Was ist Solar Forecast ML?](#was-ist-solar-forecast-ml)
3. [Wie funktioniert die Integration?](#wie-funktioniert-die-integration)
4. [Systemvoraussetzungen](#systemvoraussetzungen)
5. [Benötigte Sensoren und Einheiten](#benötigte-sensoren-und-einheiten)
6. [Installation und Einrichtung](#installation-und-einrichtung)
7. [Funktionsübersicht](#funktionsübersicht)
8. [Dateistruktur und Datenspeicherung](#dateistruktur-und-datenspeicherung)
9. [Maschinelles Lernen - Technische Grundlagen](#maschinelles-lernen---technische-grundlagen)
10. [Sensoren und ihre Bedeutung](#sensoren-und-ihre-bedeutung)
11. [Services und Funktionen](#services-und-funktionen)
12. [Konfigurationsoptionen](#konfigurationsoptionen)
13. [Tägliches Solar-Briefing](#tägliches-solar-briefing)
14. [Fehlersuche und Problemlösung](#fehlersuche-und-problemlösung)
15. [Häufig gestellte Fragen](#häufig-gestellte-fragen)

---

## Einführung

Willkommen beim Solar Forecast ML Benutzerhandbuch. Dieses Dokument erklärt die Funktionsweise, Einrichtung und Nutzung der Integration im Detail. Es richtet sich an technisch interessierte Anwender, die verstehen möchten, wie die Integration arbeitet und wie sie optimal konfiguriert wird.

### Über dieses Handbuch

Dieses Handbuch verzichtet bewusst auf Code-Beispiele und technische Symbole. Stattdessen erklärt es die Konzepte, Funktionen und technischen Hintergründe in verständlicher Sprache. Ziel ist es, Ihnen ein fundiertes Verständnis der Integration zu vermitteln, damit Sie diese optimal nutzen können.

---

## Was ist Solar Forecast ML?

Solar Forecast ML ist eine Home Assistant Integration, die mithilfe von maschinellem Lernen präzise Vorhersagen über die zu erwartende Solarstromerzeugung Ihrer Photovoltaik-Anlage erstellt. Im Gegensatz zu einfachen Wettervorhersage-Diensten lernt die Integration kontinuierlich aus den tatsächlichen Produktionsdaten Ihrer spezifischen Anlage.

### Grundprinzip

Die Integration kombiniert drei Informationsquellen:

**Wetterdaten:** Temperatur, Bewölkung, Niederschlag und weitere meteorologische Parameter werden von Ihrem konfigurierten Wetterdienst bezogen.

**Astronomische Daten:** Sonnenstand, Sonnenauf- und -untergangszeiten, Sonnenhöhe und weitere astronomische Berechnungen werden präzise für Ihren Standort ermittelt.

**Historische Produktionsdaten:** Die tatsächliche Stromerzeugung Ihrer Anlage wird kontinuierlich erfasst und mit den Vorhersagen verglichen.

Aus diesen drei Datenquellen erstellt ein maschinelles Lernmodell personalisierte Vorhersagen, die sich täglich verbessern.

### Warum maschinelles Lernen?

Jede Solaranlage ist einzigartig. Dachausrichtung, Neigungswinkel, lokale Verschattungen, Modultyp und Wechselrichter-Effizienz beeinflussen die Stromerzeugung unterschiedlich. Ein maschinelles Lernmodell kann diese individuellen Faktoren berücksichtigen und lernt mit der Zeit, wie Ihre spezifische Anlage auf verschiedene Wetterbedingungen reagiert.

---

## Wie funktioniert die Integration?

Die Integration arbeitet in einem mehrstufigen Prozess, der sich täglich wiederholt:

### Datenerfassung

Jede Stunde erfasst die Integration verschiedene Datenpunkte:

**Aktuelle Wetterdaten** werden vom konfigurierten Wetterdienst abgerufen. Dies umfasst Parameter wie Temperatur, Luftfeuchtigkeit, Bewölkungsgrad, Windgeschwindigkeit und weitere meteorologische Werte.

**Astronomische Berechnungen** werden präzise für Ihren Standort durchgeführt. Die Integration berechnet den aktuellen Sonnenstand, die Sonnenhöhe über dem Horizont, den Azimutwinkel und die theoretisch maximal mögliche Solarstrahlung unter wolkenlosem Himmel.

**Produktionsdaten** Ihrer Anlage werden vom konfigurierten Leistungssensor ausgelesen. Die Integration erfasst die aktuelle Leistung in Watt sowie den Tagesertrag in Kilowattstunden.

### Datenverarbeitung

Die gesammelten Daten werden in ein strukturiertes Format gebracht und gespeichert. Dabei werden verschiedene abgeleitete Werte berechnet, beispielsweise Interaktionen zwischen Temperatur und Sonnenhöhe oder jahreszeitliche Muster.

### Vorhersageberechnung

Die Integration verwendet mehrere Vorhersagemethoden parallel:

**Regelbasierte Vorhersage:** Eine physikalisch fundierte Berechnung basierend auf Sonnenstand, Wetterdaten und der Anlagenkapazität. Diese Methode funktioniert ab dem ersten Tag.

**Maschinelles Lernmodell:** Ein selbstlernendes Modell, das aus historischen Daten trainiert wird. Es erkennt Muster und kann komplexe Zusammenhänge zwischen Wetterbedingungen und Stromerzeugung erfassen.

**Adaptive Gewichtung:** Die Integration kombiniert beide Methoden intelligent. Zunächst dominiert die regelbasierte Vorhersage. Mit zunehmender Trainingsdatenmenge erhält das maschinelle Lernmodell mehr Gewicht, sofern es sich als zuverlässig erweist.

### Kontinuierliche Verbesserung

Nach jedem Tag vergleicht die Integration die Vorhersage mit der tatsächlichen Produktion. Diese Informationen fließen in das maschinelle Lernmodell ein, wodurch zukünftige Vorhersagen präziser werden.

---

## Systemvoraussetzungen

### Home Assistant Version

Die Integration erfordert Home Assistant Core in Version 2023.1 oder neuer. Empfohlen wird die jeweils aktuelle stabile Version.

### Recorder Integration

Die Recorder-Integration muss aktiviert sein, da historische Daten für das maschinelle Lernen benötigt werden. Die Standard-Einstellungen sind ausreichend.

### Python-Pakete

Folgende Python-Bibliotheken werden automatisch installiert:

**aiofiles:** Ermöglicht asynchrone Dateizugriffe für optimale Performance ohne Blockierung des Home Assistant Systems.

**numpy:** Stellt mathematische Funktionen für die Datenverarbeitung und das maschinelle Lernen bereit.

Diese Pakete werden bei der Installation automatisch heruntergeladen. Eine manuelle Installation ist nicht erforderlich.

### Speicherplatz

Die Integration benötigt etwa 5 bis 20 Megabyte Speicherplatz im Home Assistant Konfigurationsverzeichnis. Der genaue Bedarf hängt von der Datenmenge ab, wächst aber nur langsam mit der Zeit.

### Rechenleistung

Die Integration ist ressourcenschonend konzipiert. Auf einem Raspberry Pi 4 verursacht sie eine durchschnittliche CPU-Last von unter einem Prozent. Das maschinelle Lernen findet in optimierten Intervallen statt und blockiert das System nicht.

---

## Benötigte Sensoren und Einheiten

Für die korrekte Funktion der Integration müssen bestimmte Sensoren in Home Assistant vorhanden sein. Diese Abschnitt erklärt jeden erforderlichen und optionalen Sensor im Detail.

### Pflicht-Sensoren

Diese Sensoren müssen zwingend konfiguriert werden:

#### Wetter-Entity

**Beschreibung:** Eine Home Assistant Wetter-Integration, die aktuelle Wetterdaten und Vorhersagen bereitstellt.

**Beispiele:** Deutscher Wetterdienst, Met.no, OpenWeatherMap, AccuWeather oder andere Wetterdienste.

**Erforderliche Daten:** Die Wetter-Entity sollte mindestens Temperatur, Bewölkungsgrad und eine Wettervorhersage für die kommenden Tage liefern.

**Wichtig:** Die Wettervorhersage muss zuverlässig sein, da sie direkt die Genauigkeit der Solarvorhersage beeinflusst.

#### Leistungssensor

**Beschreibung:** Ein Sensor, der die aktuelle Leistung Ihrer Solaranlage in Echtzeit misst.

**Einheit:** Watt

**Aktualisierungsrate:** Mindestens alle 60 Sekunden, idealerweise häufiger.

**Quelle:** Dieser Sensor stammt typischerweise direkt von Ihrem Wechselrichter über eine Integration wie Modbus, SolarEdge, SMA, Fronius oder ähnliche.

**Wichtig:** Der Sensor muss die tatsächliche Erzeugungsleistung anzeigen, nicht die Netzeinspeisung oder den Hausverbrauch.

#### Tagesertragssensor

**Beschreibung:** Ein Sensor, der den gesamten erzeugten Solarstrom des aktuellen Tages anzeigt.

**Einheit:** Kilowattstunden

**Aktualisierung:** Kontinuierlich während des Tages steigend.

**Mitternachts-Reset:** Der Sensor muss sich jeden Tag um Mitternacht automatisch auf Null zurücksetzen. Dies ist kritisch für die korrekte Funktion.

**Quelle:** Üblicherweise vom Wechselrichter bereitgestellt.

**Wichtig:** Falls Ihr Wechselrichter keinen Tagesertragssensor mit automatischem Mitternachts-Reset bereitstellt, können Sie einen mit dem Utility Meter Helper in Home Assistant erstellen.

### Optionale Sensoren

Diese Sensoren verbessern die Vorhersagegenauigkeit, sind aber nicht zwingend erforderlich:

#### Anlagenkapazität

**Beschreibung:** Die installierte Spitzenleistung Ihrer Solaranlage.

**Einheit:** Kilowatt Peak

**Beispiel:** Eine Anlage mit 10 Modulen à 400 Watt hat eine Kapazität von 4 Kilowatt Peak.

**Verwendung:** Ermöglicht die Berechnung theoretischer Maximalwerte und relativer Effizienz.

#### Temperatur-Sensor

**Beschreibung:** Ein externer Temperatursensor, unabhängig von der Wetter-Entity.

**Einheit:** Grad Celsius

**Vorteil:** Kann lokale Temperaturunterschiede erfassen, die von der Wetterstation abweichen.

**Quelle:** Eigener Außentemperatursensor oder Smart Home Wetterstation.

#### Luftfeuchtigkeits-Sensor

**Beschreibung:** Misst die relative Luftfeuchtigkeit.

**Einheit:** Prozent

**Vorteil:** Luftfeuchtigkeit korreliert mit Wolkenbildung und Dunst, was die Solarstrahlung beeinflusst.

#### Wind-Sensor

**Beschreibung:** Misst die Windgeschwindigkeit.

**Einheit:** Meter pro Sekunde oder Kilometer pro Stunde

**Vorteil:** Starker Wind kann Wolken bewegen und die Kühlung der Solarmodule beeinflussen, was die Effizienz erhöht.

#### Regen-Sensor

**Beschreibung:** Erkennt Niederschlag.

**Typ:** Sensor oder Niederschlagsmenge

**Vorteil:** Regen korreliert stark mit reduzierter Solarproduktion.

#### UV-Index Sensor

**Beschreibung:** Misst die UV-Strahlung.

**Einheit:** UV-Index

**Vorteil:** UV-Strahlung ist ein direkter Indikator für Sonneneinstrahlung.

#### Helligkeits-Sensor

**Beschreibung:** Misst die Umgebungshelligkeit.

**Einheit:** Lux

**Vorteil:** Direktes Maß für verfügbares Sonnenlicht.

#### Verbrauchs- und Netz-Sensoren

Falls konfiguriert, können zusätzliche Sensoren die Autarkie-Berechnung ermöglichen:

**Gesamtverbrauch heute:** Kilowattstunden Hausverbrauch
**Netzbezug heute:** Kilowattstunden vom Stromnetz bezogen
**Netzeinspeisung heute:** Kilowattstunden ins Stromnetz eingespeist

---

## Installation und Einrichtung

### Installation über HACS

HACS ist die empfohlene Installationsmethode:

Öffnen Sie HACS in Home Assistant und navigieren Sie zum Bereich Integrationen.

Klicken Sie auf die drei Punkte oben rechts und wählen Sie "Benutzerdefinierte Repositories".

Fügen Sie die URL des GitHub-Repositories hinzu und wählen Sie als Kategorie "Integration".

Suchen Sie nach "Solar Forecast ML" und klicken Sie auf "Herunterladen".

Nach dem Download starten Sie Home Assistant neu.

### Manuelle Installation

Alternativ können Sie die Integration manuell installieren:

Laden Sie die neueste Version von GitHub herunter.

Entpacken Sie das Archiv und kopieren Sie den Ordner "solar_forecast_ml" in das Verzeichnis "custom_components" Ihrer Home Assistant Installation.

Starten Sie Home Assistant neu.

### Einrichtung der Integration

Nach der Installation und dem Neustart:

Navigieren Sie zu Einstellungen, dann Geräte und Dienste.

Klicken Sie auf "Integration hinzufügen".

Suchen Sie nach "Solar Forecast ML" und wählen Sie diese aus.

Der Einrichtungsassistent führt Sie durch folgende Schritte:

**Schritt 1 - Wetter-Entity:** Wählen Sie Ihre Wetter-Integration aus der Liste.

**Schritt 2 - Leistungssensor:** Wählen Sie den Sensor, der die aktuelle Solarleistung in Watt anzeigt.

**Schritt 3 - Tagesertrag:** Wählen Sie den Sensor für den täglichen Solarertrag in Kilowattstunden. Dieser muss sich um Mitternacht automatisch zurücksetzen.

**Schritt 4 - Anlagenkapazität:** Geben Sie die Spitzenleistung Ihrer Anlage in Kilowatt Peak ein.

**Schritt 5 - Optionale Sensoren:** Wählen Sie zusätzliche Sensoren aus, falls vorhanden. Diese können übersprungen werden.

Nach Abschluss der Einrichtung beginnt die Integration sofort mit der Arbeit.

### Erste Schritte nach der Installation

In den ersten Tagen arbeitet die Integration ausschließlich mit regelbasierten Vorhersagen. Das maschinelle Lernmodell beginnt mit dem Training, sobald ausreichend Daten gesammelt wurden. Dies dauert typischerweise 4 bis 14 Tage.

Sie können die Datensammlung in den Diagnose-Sensoren verfolgen. Der Sensor "ML Model Status" zeigt den aktuellen Trainingszustand an.

---

## Funktionsübersicht

Die Integration bietet eine Vielzahl von Funktionen, die in diesem Abschnitt detailliert erklärt werden.

### Solarvorhersage

Die Kernfunktion ist die Vorhersage der Solarstromerzeugung für drei Tage:

**Heute:** Aktualisiert kontinuierlich während des Tages. Am Morgen wird die Vorhersage "gesperrt", um Konsistenz zu gewährleisten.

**Morgen:** Vorhersage für den kommenden Tag, aktualisiert mehrmals täglich basierend auf aktuellen Wetterdaten.

**Übermorgen:** Vorausschau für den übernächsten Tag, hilfreich für längerfristige Planung.

Die Vorhersagen werden als Kilowattstunden für den gesamten Tag angegeben.

### Stündliche Vorhersagen

Optional können Sie stündliche Vorhersagen aktivieren. Diese zeigen für jede Stunde des Tages die erwartete Stromerzeugung. Dies ist besonders nützlich für:

Zeitgesteuerte Automatisierungen, die energieintensive Geräte zu Zeiten hoher Solarproduktion aktivieren.

Planung von Ladezeitpunkten für Elektrofahrzeuge oder Batteriespeicher.

Optimierung des Eigenverbrauchs.

### Spitzenlastzeit-Erkennung

Die Integration identifiziert automatisch die Stunde mit der höchsten erwarteten Solarproduktion. Dies ist typischerweise um die solare Mittagszeit herum, kann aber je nach Wetterlage variieren.

Diese Information kann für Automatisierungen genutzt werden, um beispielsweise Waschmaschine, Trockner oder Poolpumpe genau dann zu betreiben, wenn die Solarproduktion am höchsten ist.

### Produktionszeit-Berechnung

Die Integration berechnet täglich, wann die Solaranlage voraussichtlich Strom erzeugen wird. Dies beginnt typischerweise kurz nach Sonnenaufgang und endet kurz vor Sonnenuntergang. Die exakten Zeiten hängen von Wetterbedingungen und Jahreszeit ab.

### Genauigkeits-Tracking

Nach jedem Tag wird die Vorhersage mit der tatsächlichen Produktion verglichen. Die Abweichung wird als Prozentsatz berechnet und gespeichert. So können Sie die Vorhersagequalität überwachen und sehen, wie sich die Genauigkeit mit der Zeit verbessert.

### Autarkie-Berechnung

Falls Sie die entsprechenden Sensoren konfiguriert haben, berechnet die Integration Ihren Autarkiegrad. Dies ist der Prozentsatz Ihres Stromverbrauchs, der durch Ihre Solaranlage gedeckt wird.

Die Formel lautet: Solarertrag minus Netzeinspeisung, geteilt durch Gesamtverbrauch, multipliziert mit 100.

Ein Autarkiegrad von 70 Prozent bedeutet beispielsweise, dass Sie 70 Prozent Ihres Strombedarfs selbst erzeugen.

### Durchschnitts-Berechnung

Die Integration berechnet kontinuierlich den durchschnittlichen Tagesertrag über verschiedene Zeiträume. Dies hilft Ihnen, saisonale Schwankungen und langfristige Trends zu erkennen.

### Tägliches Solar-Briefing

Eine neue Funktion in Version 10.0.0 ist das tägliche Solar-Briefing. Jeden Morgen erhalten Sie eine formatierte Benachrichtigung mit:

Der Vorhersage für den aktuellen Tag
Einem Vergleich zur gestrigen Produktion
Einer Wetter-Interpretation basierend auf der Vorhersage
Astronomischen Daten wie Sonnenaufgang, Sonnenuntergang und Tageslichtdauer
Der besten Produktionszeit

Das Briefing kann in Deutsch oder Englisch konfiguriert werden.

---

## Dateistruktur und Datenspeicherung

Die Integration speichert alle Daten als JSON-Dateien in einem eigenen Verzeichnis innerhalb Ihrer Home Assistant Konfiguration. Dieser Abschnitt erklärt jede Datei und ihren Zweck.

### Verzeichnisstruktur

Alle Daten befinden sich im Ordner:
config/solar_forecast_ml/

Innerhalb dieses Ordners gibt es mehrere Unterverzeichnisse:

**stats:** Enthält alle produktionsbezogenen Daten und Vorhersagen
**ml:** Enthält maschinelles Lernmodell und Trainingsdaten
**production:** Temporäre Dateien für laufende Prozesse


### Wichtige Dateien im stats-Verzeichnis

#### hourly_predictions.json

**Zweck:** Das Herzstück der Datensammlung. Diese Datei enthält für jede Stunde des Tages eine Vorhersage und später den tatsächlichen Ertrag.

**Inhalt:** Für jede Stunde werden gespeichert:
- Datum und Uhrzeit
- Wetterdaten zum Vorhersagezeitpunkt
- Astronomische Daten
- Vorhergesagte Stromerzeugung
- Tatsächliche Stromerzeugung nach Ablauf der Stunde
- Sensorwerte externer Sensoren

**Verwendung:** Diese Daten bilden die Grundlage für das maschinelle Lernen. Das Modell lernt aus dem Vergleich zwischen Vorhersage und tatsächlichem Ertrag.

**Wachstum:** Die Datei wächst kontinuierlich, wird aber automatisch bereinigt, um nur relevante historische Daten zu behalten.

#### daily_forecasts.json

**Zweck:** Speichert die Tagesvorhersagen und deren Ergebnisse.

**Inhalt:**
- Vorhersage für heute, morgen und übermorgen
- Historische Vorhersagen vergangener Tage
- Vergleich zwischen Vorhersage und tatsächlichem Ertrag
- Genauigkeitsmetriken
- Statistiken über die Vorhersagequalität

**Verwendung:** Diese Datei ermöglicht die Nachvollziehbarkeit und Analyse der Vorhersagegenauigkeit über längere Zeiträume.

#### astronomy_cache.json

**Zweck:** Speichert vorberechnete astronomische Daten.

**Inhalt:** Für jeden Tag und jede Stunde:
- Sonnenaufgangs- und -untergangszeit
- Sonnenhöhe über dem Horizont
- Sonnen-Azimutwinkel
- Theoretische maximale Solarstrahlung unter wolkenlosem Himmel
- Solare Mittagszeit
- Tageslichtdauer

**Verwendung:** Diese Berechnungen sind rechenintensiv und werden daher vorberechnet und zwischengespeichert. Das spart Rechenleistung und beschleunigt Vorhersagen.

**Aktualisierung:** Der Cache wird automatisch für zukünftige Tage erweitert.

#### daily_summaries.json

**Zweck:** Enthält tägliche Zusammenfassungen der Solarproduktion.

**Inhalt:**
- Gesamtertrag des Tages
- Produktionsstunden
- Spitzenlast-Stunde
- Durchschnittswerte
- Vergleichsdaten zu Vorhersagen

**Verwendung:** Ermöglicht schnelle Übersichten und Langzeitanalysen ohne Durchsuchen der stündlichen Daten.

#### prediction_history.json

**Zweck:** Langzeitarchiv vergangener Vorhersagen.

**Inhalt:** Historische Vorhersagen mit Vergleichswerten zur tatsächlichen Produktion.

**Verwendung:** Dient der Qualitätssicherung und Analyse langfristiger Trends in der Vorhersagegenauigkeit.

### Wichtige Dateien im ml-Verzeichnis

#### hourly_samples.json

**Zweck:** Unabhängige Sammlung von Trainingsdaten.

**Inhalt:** Stündliche Datenpunkte mit allen relevanten Merkmalen für das maschinelle Lernen.

**Besonderheit:** Diese Sammlung läuft unabhängig vom Hauptsystem. Selbst wenn andere Komponenten Fehler aufweisen, werden hier kontinuierlich Daten gesammelt. Dies stellt sicher, dass keine Trainingsdaten verloren gehen.

#### learned_weights.json

**Zweck:** Speichert das trainierte maschinelle Lernmodell.

**Inhalt:**
- Modell-Gewichte für alle 44 Eingabedaten-Merkmale
- Skalierungsparameter
- Modellgenauigkeit auf Trainingsdaten
- Anzahl verwendeter Trainingsbeispiele
- Trainingsdatum

**Verwendung:** Das gespeicherte Modell wird für Vorhersagen verwendet. Nach jedem Training wird diese Datei aktualisiert.

**Wichtig:** Diese Datei enthält Ihr personalisiertes Modell, das auf Ihre spezifische Anlage trainiert wurde.

#### model_state.json

**Zweck:** Verfolgt den Zustand des maschinellen Lernmodells.

**Inhalt:**
- Aktueller Trainingsstatus
- Anzahl verfügbarer Trainingsbeispiele
- Letztes Trainingsdatum
- Modellversion
- Fehlerprotokolle

**Verwendung:** Ermöglicht der Integration, den Trainingszustand zu verfolgen und Trainings bei Bedarf auszulösen.

#### hourly_profile.json

**Zweck:** Speichert typische Produktionsmuster für jede Stunde.

**Inhalt:** Durchschnittliche Produktion und Standardabweichung für jede Tagesstunde, basierend auf historischen Daten.

**Verwendung:** Hilft bei der Identifikation ungewöhnlicher Produktionsmuster und unterstützt Vorhersagen.

### Systemdatei

#### versinfo.json

**Zweck:** Versionsinformationen und Migrationsstatus.

**Inhalt:**
- Installierte Version der Integration
- Datum der Installation
- Migrationsstatus
- Systemkonfiguration

**Verwendung:** Stellt sicher, dass bei Updates Datenmigrationen korrekt durchgeführt werden.

### Warum JSON-Dateien?

Die Integration verwendet JSON-Dateien statt einer Datenbank aus mehreren Gründen:

**Portabilität:** JSON-Dateien können einfach gesichert, übertragen und wiederhergestellt werden.

**Transparenz:** Sie können die Dateien mit einem Texteditor öffnen und inspizieren, um die gespeicherten Daten zu verstehen.

**Unabhängigkeit:** Keine Abhängigkeit von zusätzlichen Datenbank-Systemen.

**Performance:** Für die Datenmenge sind JSON-Dateien effizient und schnell.

**Fehlertoleranz:** Beschädigte Dateien betreffen nicht das gesamte System, sondern nur einzelne Komponenten.

### Datensicherung

Alle JSON-Dateien sollten in Ihrer regulären Home Assistant Sicherung enthalten sein. Zusätzlich erstellt die Integration automatische Sicherungen vor kritischen Operationen wie dem Morgen-Update.

---

## Maschinelles Lernen - Technische Grundlagen

Dieser Abschnitt erklärt, wie das maschinelle Lernen in der Integration funktioniert, ohne tief in mathematische Details zu gehen.

### Was ist maschinelles Lernen?

Maschinelles Lernen ist eine Methode, bei der ein Computer-Algorithmus aus Beispielen lernt, Muster zu erkennen und Vorhersagen zu treffen. Im Fall von Solar Forecast ML bedeutet dies:

Der Algorithmus erhält historische Daten: Wetterbedingungen, astronomische Parameter und die tatsächliche Stromproduktion.

Er erkennt Zusammenhänge: Beispielsweise, dass bei hoher Sonneneinstrahlung und geringer Bewölkung die Produktion hoch ist, oder dass Ihre Anlage im Sommer bei gleicher Bewölkung mehr produziert als im Winter.

Er erstellt ein Modell: Dieses Modell kann für neue Situationen Vorhersagen treffen, auch wenn diese exakte Kombination von Bedingungen noch nie vorkam.

### Die 44 Eingangsmerkmale

Das maschinelle Lernmodell analysiert 44 verschiedene Merkmale für jede Vorhersage:

**Zeitliche Merkmale:** Tag im Jahr, Monat, Wochentag, Jahreszeit, Stunde der Vorhersage. Diese erfassen saisonale und tägliche Muster.

**Wettervorhersage:** Temperatur, Luftfeuchtigkeit, Windgeschwindigkeit, Wolkenbedeckung, Taupunkt, Regenwahrscheinlichkeit, Windrichtung, Luftdruck, Wetterzustand. Diese Parameter beschreiben die erwarteten Wetterbedingungen.

**Aktuelle Sensordaten:** Wenn verfügbar, werden aktuelle Messwerte von Temperatur, Feuchtigkeit, Wind, Regen, UV-Index und Helligkeit verwendet.

**Astronomische Grunddaten:** Sonnenaufgangszeit, Sonnenuntergangszeit, solare Mittagszeit, Tageslichtdauer. Diese beschreiben den täglichen Sonnenverlauf.

**Erweiterte Astronomie:** Sonnenhöhe, Sonnen-Azimut, theoretische Himmelsstrahlung, theoretische Maximalleistung, Zeit seit solarer Mittagszeit, Tagesfortschritt. Diese Parameter erfassen die exakte Position der Sonne.

**Historische Vergleiche:** Produktion von gestern, Produktion zur gleichen Stunde gestern. Diese Werte erfassen kurzfristige Trends.

**Kontext:** Zeitlicher Abstand der Vorhersage zum vorhergesagten Zeitpunkt. Vorhersagen für den aktuellen Tag sind typischerweise genauer als für übermorgen.

**Abgeleitete Merkmale:** Temperatur-Sonnenhöhen-Interaktion, Strahlungseffizienz, saisonale Muster und weitere berechnete Werte, die komplexe Zusammenhänge erfassen.

### Wie lernt das Modell?

Der Lernprozess läuft in Phasen ab:

**Datensammlung:** In den ersten Tagen sammelt die Integration Daten ohne maschinelles Lernen. Vorhersagen basieren auf regelbasierten Berechnungen.

**Erstes Training:** Sobald ausreichend Daten vorliegen, typischerweise nach 4 bis 14 Tagen, wird das erste Modell trainiert.

**Kontinuierliches Lernen:** Täglich werden neue Daten hinzugefügt. Periodisch wird das Modell neu trainiert, um diese neuen Erkenntnisse einzuarbeiten.

**Qualitätssicherung:** Die Integration überwacht die Vorhersagequalität. Falls das maschinelle Lernmodell schlechtere Ergebnisse liefert als die regelbasierte Methode, wird seine Gewichtung reduziert.

### Ridge-Regression

Die Integration verwendet ein Verfahren namens Ridge-Regression. Dies ist eine mathematische Methode, die:

Linear ist: Jedes Eingangsmerkmal hat einen bestimmten Einfluss auf die Vorhersage, proportional zu seinem Wert.

Regularisiert ist: Verhindert, dass das Modell zu stark auf Trainings-daten spezialisiert wird und dadurch bei neuen Daten versagt.

Robust ist: Weniger anfällig für Ausreißer in den Daten.

Schnell ist: Benötigt wenig Rechenzeit, ideal für ein System wie Home Assistant.

### Adaptive Gewichtung

Die Integration kombiniert intelligenterweise regelbasierte und ML-basierte Vorhersagen:

**Anfangsphase:** Regelbasierte Vorhersage hat 100 Prozent Gewicht, da noch kein trainiertes Modell existiert.

**Aufbauphase:** Mit zunehmenden Trainingsdaten erhält das ML-Modell graduell mehr Gewicht.

**Stabile Phase:** Bei ausreichender Datenmenge und guter Modellqualität kann das ML-Modell bis zu 90 Prozent Gewicht erreichen.

**Qualitätskontrolle:** Falls das ML-Modell unplausible Vorhersagen liefert, wird automatisch mehr Gewicht auf die regelbasierte Methode gelegt.

Diese adaptive Strategie stellt sicher, dass Sie immer sinnvolle Vorhersagen erhalten, unabhängig vom Trainingszustand.

---

## Sensoren und ihre Bedeutung

Die Integration erstellt zahlreiche Sensoren in Home Assistant. Dieser Abschnitt erklärt jeden Sensor und seine Verwendung.

### Hauptvorhersage-Sensoren

**Solar Forecast Heute:** Zeigt die vorhergesagte Stromerzeugung für den aktuellen Tag in Kilowattstunden. Dieser Wert wird am Morgen "gesperrt", um den ganzen Tag über konsistent zu bleiben.

**Solar Forecast Morgen:** Vorhersage für den kommenden Tag. Aktualisiert sich mehrmals täglich basierend auf aktuellen Wetterdaten.

**Solar Forecast Übermorgen:** Vorhersage für den übernächsten Tag. Nützlich für vorausschauende Planung.

**Spitzenlast-Stunde:** Gibt an, in welcher Stunde die höchste Solarproduktion erwartet wird. Nützlich für zeitgesteuerte Automatisierungen.

**Produktionszeit:** Zeigt an, wann die Anlage voraussichtlich Strom erzeugen wird, typischerweise von kurz nach Sonnenaufgang bis kurz vor Sonnenuntergang.

### Analyse-Sensoren

**Durchschnittlicher Ertrag:** Berechnet kontinuierlich den durchschnittlichen Tagesertrag über verschiedene Zeiträume.

**Autarkie:** Zeigt den Prozentsatz Ihres Stromverbrauchs, der durch Solar gedeckt wird. Erfordert Konfiguration von Verbrauchs- und Netz-Sensoren.

**Solar-Genauigkeit:** Misst, wie genau die Vorhersagen im Durchschnitt sind. Ein Wert von 95 Prozent bedeutet, dass die Vorhersagen im Mittel nur 5 Prozent vom tatsächlichen Wert abweichen.

**Gestrige Abweichung:** Zeigt die prozentuale Abweichung der gestrigen Vorhersage vom tatsächlichen Ertrag.

### ML-Status-Sensoren

**ML-Modellstatus:** Zeigt den aktuellen Zustand des maschinellen Lernmodells an. Mögliche Werte sind: Keine Daten, Sammelt Daten, Training, Aktiv, Fehler.

**Letztes ML-Training:** Zeitstempel des letzten Modell-Trainings.

**Trainingsbeispiele:** Anzahl der verfügbaren Datenpunkte für das Training.

**Modellgenauigkeit:** Technische Metrik der Modellqualität auf Trainingsdaten.

### System-Status-Sensoren

**Diagnosestatus:** Gesamtstatus der Integration. Zeigt an, ob alle Komponenten ordnungsgemäß funktionieren.

**Letztes Coordinator-Update:** Zeitpunkt der letzten Datenaktualisierung.

**Update-Alter:** Zeit seit dem letzten Update. Hilft zu erkennen, ob das System noch aktiv aktualisiert.

**Nächstes geplantes Update:** Zeitpunkt des nächsten automatischen Updates.



### Stündliche Vorhersage-Sensoren

Falls aktiviert, erstellt die Integration für jede Stunde des Tages einen eigenen Sensor. Diese zeigen die erwartete Stromerzeugung für die jeweilige Stunde in Kilowattstunden.

Beispiel: "Solar Forecast 12 Uhr" zeigt die erwartete Produktion zwischen 12 und 13 Uhr.

### Diagnose-Sensoren

Im erweiterten Diagnosemodus werden zusätzliche technische Sensoren erstellt:

**Feature-Vektoren:** Zeigen die aktuellen Werte aller 44 Eingangsmerkmale.

**Modellgewichte:** Die gelernten Gewichtungen des ML-Modells.

**Skalierungsparameter:** Technische Parameter der Datenvorverarbeitung.

**Fehlerprotokolle:** Detaillierte Informationen über Fehler und Warnungen.

Diese Sensoren sind primär für Entwickler und technisch versierte Nutzer interessant.


---

## Konfigurationsoptionen

Nach der Ersteinrichtung können Sie die Integration über Einstellungen, Geräte und Dienste konfigurieren. Wählen Sie die Solar Forecast ML Integration und klicken Sie auf Konfigurieren.

### Allgemeine Optionen

**Update-Intervall:** Bestimmt, wie oft die Integration Vorhersagen aktualisiert. Der Standardwert von 3600 Sekunden (60 Minuten) ist für die meisten Wetterdienste optimal. Kürzere Intervalle bringen selten Vorteile, da sich Wettervorhersagen nicht minütlich ändern, können aber API-Limits belasten.

**Diagnosemodus:** Aktiviert zusätzliche technische Sensoren für Entwickler und Experten. Im Normalbetrieb nicht erforderlich.

**Stündliche Vorhersagen:** Erstellt Sensoren für jede Stunde des Tages. Aktivieren Sie dies nur, wenn Sie diese Informationen für Automatisierungen benötigen, da es die Anzahl der Entities erhöht.

### Benachrichtigungsoptionen

**Startup-Benachrichtigungen:** Zeigt eine Benachrichtigung beim Start der Integration mit Versions- und Statusinformationen.

**Vorhersage-Benachrichtigungen:** Benachrichtigt Sie bei jeder Vorhersage-Aktualisierung. Kann häufig sein, daher standardmäßig deaktiviert.

**Lern-Benachrichtigungen:** Informiert Sie, wenn das maschinelle Lernmodell gerade trainiert. Nützlich, um den Lernfortschritt zu verfolgen.

**Erfolgreiche-Training-Benachrichtigungen:** Benachrichtigt nach erfolgreichem Abschluss eines Trainings mit Informationen zur Modellgenauigkeit.

### Batterie-Management (BETA)

Falls Sie einen Batteriespeicher besitzen, können Sie das optionale Batterie-Management aktivieren:

**Batterie-Management aktivieren:** Hauptschalter für Batterie-Funktionen.

**Batteriekapazität:** Nutzbare Kapazität Ihrer Batterie in Kilowattstunden.

**Batterie SOC Entity:** Sensor für den Ladezustand in Prozent.

**Batterie-Leistungs-Entity:** Sensor für aktuelle Lade- oder Entladeleistung in Watt.

**Batterie-Ladung heute:** Sensor für die heute geladene Energie in Kilowattstunden.

**Batterie-Entladung heute:** Sensor für die heute entnommene Energie in Kilowattstunden.

**Batterietemperatur:** Optional. Temperatursensor der Batterie.

Mit diesen Einstellungen erstellt die Integration zusätzliche Sensoren für Batterieanalyse, erwartete Ladung aus Solar, verbleibende Laufzeit und erweiterte Autarkie-Berechnungen.

### Strompreis-Integration

Falls Sie variable Strompreise haben, können Sie die aWATTar-Integration aktivieren:

**Strompreise aktivieren:** Hauptschalter für Strompreis-Funktionen.

**Land:** Deutschland oder Österreich. Bestimmt die Preisregion.

Die aWATTar-API ist kostenlos und erfordert keinen API-Schlüssel. Die Integration ruft täglich um 13 Uhr die aktuellen Börsenpreise ab und erstellt Sensoren für aktuelle Preise, Durchschnittswerte, günstigste Stunden und Ladeempfehlungen.

---

## Tägliches Solar-Briefing

Das tägliche Solar-Briefing ist eine neue Funktion in Version 10.0.0, die Ihnen jeden Morgen eine übersichtliche Zusammenfassung der erwarteten Solarproduktion liefert.

### Was enthält das Briefing?

Das Briefing ist eine formatierte Benachrichtigung mit folgenden Informationen:

**Tagesvorhersage:** Die erwartete Gesamtproduktion für den Tag in Kilowattstunden.

**Wetter-Interpretation:** Eine textuelle Beschreibung basierend auf der Vorhersage, beispielsweise "Guter Solar-Tag erwartet" oder "Schwacher Solar-Tag erwartet". Diese wird durch passende Emojis visualisiert.

**Vergleich zu gestern:** Falls verfügbar, wird die heutige Vorhersage mit der gestrigen tatsächlichen Produktion verglichen. Beispiel: "Sechsmal besser als gestern" oder "Ähnlich wie gestern".

**Beste Produktionszeit:** Die Uhrzeit der solaren Mittagszeit, wenn die Sonne am höchsten steht und die Produktion typischerweise maximal ist.

**Astronomische Daten:** Sonnenaufgang, Sonnenuntergang und die Tageslichtdauer in Stunden und Minuten.

**Abschlussbotschaft:** Ein motivierender Satz, der zur Vorhersage passt.

### Benachrichtigungsmodi

Das Briefing unterstützt zwei Modi:

**Nur Home Assistant UI:** Die vollständige Benachrichtigung wird in der Home Assistant Benutzeroberfläche angezeigt und ist über das Benachrichtigungssymbol abrufbar.

**Dual-Modus:** Zusätzlich zur UI-Benachrichtigung erhalten Sie eine kurze Push-Benachrichtigung auf Ihrem mobilen Gerät mit dem wichtigsten Wert. Die Details sind dann in der Home Assistant App verfügbar.

### Einrichtung in Automationen

Um das tägliche Briefing automatisch zu erhalten, erstellen Sie eine Automation:

Trigger: Zeitpunkt, beispielsweise 7 Uhr morgens.
Aktion: Service "solar_forecast_ml.send_daily_briefing" aufrufen.
Parameter: Sprache auf "de" oder "en" setzen, optional einen Benachrichtigungsdienst angeben.

Die Automation kann auch Bedingungen enthalten, beispielsweise nur an Wochentagen oder nur bei guter Vorhersage.

### Sprachunterstützung

Das Briefing ist vollständig in Deutsch und Englisch verfügbar. Alle Texte, Wochentage und Interpretationen werden korrekt übersetzt.

### Wetter-Klassifikation

Die Vorhersage wird automatisch kategorisiert:

Über 15 Kilowattstunden: Sehr guter Solar-Tag, sonnig und ideal.
10 bis 15 Kilowattstunden: Guter Solar-Tag, sonnig bis teilweise bewölkt.
5 bis 10 Kilowattstunden: Ordentlicher Solar-Tag, teilweise bewölkt.
2 bis 5 Kilowattstunden: Mäßiger Solar-Tag, bewölkt mit sonnigen Phasen.
0,5 bis 2 Kilowattstunden: Schwacher Solar-Tag, stark bewölkt.
Unter 0,5 Kilowattstunden: Kaum Solar-Produktion, bedeckt oder Regen.

Diese Klassifikation hilft Ihnen, auf einen Blick einzuschätzen, was Sie erwarten können.

---

## Fehlersuche und Problemlösung

Dieser Abschnitt hilft Ihnen bei häufigen Problemen.

### Integration startet nicht

**Symptom:** Nach der Installation erscheint die Integration nicht in der Liste oder zeigt Fehler beim Start.

**Lösungen:**

Überprüfen Sie, dass alle Pflicht-Sensoren existieren und Daten liefern.

Stellen Sie sicher, dass der Tagesertragssensor sich um Mitternacht zurücksetzt.

Prüfen Sie die Home Assistant Logs unter Einstellungen, System, Protokolle auf Fehlermeldungen.

Verifizieren Sie, dass die Wetter-Entity funktioniert und Vorhersagen liefert.

### Ungenaue Vorhersagen

**Symptom:** Die Vorhersagen weichen stark von der tatsächlichen Produktion ab.

**Lösungen:**

Geben Sie dem System Zeit. In den ersten Wochen lernt das Modell noch. Die Genauigkeit verbessert sich kontinuierlich.

Überprüfen Sie die Qualität Ihrer Wetterdaten. Ein unzuverlässiger Wetterdienst führt zu ungenauen Vorhersagen.

Stellen Sie sicher, dass der Leistungs- und Ertragssensor korrekte Werte liefern.

Prüfen Sie, ob die Anlagenkapazität korrekt konfiguriert ist.

Verwenden Sie den "Retrain Model" Service, um das Modell mit allen verfügbaren Daten neu zu trainieren.

### Fehlende Sensoren

**Symptom:** Erwartete Sensoren werden nicht angezeigt.

**Lösungen:**

Aktivieren Sie den Diagnosemodus in den Konfigurationsoptionen, falls Sie technische Sensoren vermissen.

Aktivieren Sie stündliche Vorhersagen, falls Sie diese Sensoren benötigen.

Starten Sie Home Assistant nach Änderungen an der Konfiguration neu.

Prüfen Sie in der Entity-Registry, ob Sensoren möglicherweise deaktiviert wurden.

### Maschinelles Lernen trainiert nicht

**Symptom:** Der ML-Modellstatus zeigt dauerhaft "Sammelt Daten" oder "Keine Daten".

**Lösungen:**

Das System benötigt eine Mindestanzahl an Datenpunkten. Warten Sie mindestens 7 bis 14 Tage.

Überprüfen Sie die Datei hourly_predictions.json im stats-Verzeichnis. Sie sollte kontinuierlich wachsen.

Prüfen Sie, ob der Tagesertragssensor korrekt funktioniert und tatsächliche Produktionsdaten liefert.

Schauen Sie in die Logs nach Fehlermeldungen während der Datensammlung.

### Briefing-Benachrichtigungen funktionieren nicht

**Symptom:** Das tägliche Briefing wird nicht gesendet oder enthält Fehler.

**Lösungen:**


Stellen Sie sicher, dass der Benachrichtigungsdienst korrekt konfiguriert ist. Bei mobilen Apps muss der Name exakt dem Entity-Namen entsprechen, ohne das Präfix "notify".

Prüfen Sie, ob Vorhersagedaten verfügbar sind. Der Sensor "Solar Forecast Heute" sollte einen Wert zeigen.

Verifizieren Sie die Spracheinstellung. Verwenden Sie "de" für Deutsch oder "en" für Englisch.

### Hohe CPU-Last

**Symptom:** Die Integration verursacht hohe Prozessor-Auslastung.

**Lösungen:**

Dies ist ungewöhnlich. Prüfen Sie die Logs auf Fehler, die eine Endlosschleife verursachen könnten.

Reduzieren Sie das Update-Intervall auf 3600 Sekunden (Standard).

Deaktivieren Sie den Diagnosemodus, falls aktiviert.

Deaktivieren Sie stündliche Vorhersagen, falls diese nicht benötigt werden.

Kontaktieren Sie den Entwickler über GitHub Issues mit Details zu Ihrem System.

---

## Häufig gestellte Fragen

### Wie lange dauert es, bis das maschinelle Lernen funktioniert?

Das System beginnt sofort mit regelbasierten Vorhersagen. Das maschinelle Lernmodell benötigt etwa 4 bis 14 Tage Datensammlung für das erste Training. Die volle Genauigkeit wird typischerweise nach 4 bis 6 Wochen erreicht, da das Modell dann verschiedene Wetterbedingungen kennengelernt hat.

### Muss ich etwas tun, damit das System lernt?

Nein. Das Lernen geschieht vollautomatisch im Hintergrund. Sie müssen lediglich sicherstellen, dass die Sensoren korrekt konfiguriert sind und kontinuierlich Daten liefern.

### Kann ich die historischen Daten sichern?

Ja. Alle Daten befinden sich im Verzeichnis cconfig/solar_forecast_ml. Sichern Sie diesen Ordner regelmäßig. Die Dateien sind im JSON-Format und können einfach auf ein neues System übertragen werden.

### Was passiert nach einem Home Assistant Neustart?

Die Integration lädt automatisch alle gespeicherten Daten und setzt die Arbeit nahtlos fort. Trainierte Modelle bleiben erhalten. Es gehen keine Daten verloren.

### Funktioniert die Integration offline?

Teilweise. Für Vorhersagen werden Wetterdaten benötigt, die von Online-Diensten stammen. Die Datensammlung und das Training funktionieren jedoch auch offline, sofern Ihre Wetter-Integration gecachte Daten liefert.

### Wie genau sind die Vorhersagen?

Die Genauigkeit hängt von vielen Faktoren ab. Typischerweise erreichen gut trainierte Modelle eine Genauigkeit von 85 bis 95 Prozent. Dies bedeutet, dass die durchschnittliche Abweichung zwischen Vorhersage und tatsächlicher Produktion 5 bis 15 Prozent beträgt.

### Kann ich mehrere Solaranlagen überwachen?

Derzeit ist die Integration auf eine Anlage pro Home Assistant Instanz ausgelegt. Für mehrere Anlagen müssten Sie separate Instanzen oder eine angepasste Konfiguration verwenden.

### Unterstützt die Integration Batteriespeicher?

Ja. Die optionale Batterie-Management-Erweiterung bietet umfassende Überwachung und Analyse von Batteriespeichern, einschließlich erwarteter Ladung aus Solar, Laufzeitberechnung und erweiterter Autarkie-Metriken.

### Werden meine Daten irgendwohin gesendet?

Nein. Alle Daten bleiben lokal auf Ihrem Home Assistant System. Es erfolgt keine Kommunikation mit externen Servern, außer dem Abruf von Wetterdaten von Ihrem konfigurierten Wetterdienst. Das maschinelle Lernen findet komplett lokal statt.

### Wie kann ich die Vorhersagen in Automatisierungen nutzen?

Verwenden Sie die Vorhersage-Sensoren als Trigger oder Bedingungen. Beispielsweise können Sie energieintensive Geräte nur dann automatisch starten, wenn die Vorhersage über einem bestimmten Schwellwert liegt.

### Was ist der Unterschied zwischen regelbasiert und maschinellem Lernen?

Regelbasierte Vorhersagen verwenden physikalische Formeln und Wetterdaten. Sie funktionieren sofort, sind aber generisch. Maschinelles Lernen erkennt spezifische Muster Ihrer Anlage und lernt aus historischen Daten, wodurch die Vorhersagen personalisiert und genauer werden.

### Kann ich das Modell manuell zurücksetzen?

Ja. Löschen Sie die Datei learned_weights.json im ml-Verzeichnis. Beim nächsten Training wird ein neues Modell von Grund auf erstellt. Dies kann sinnvoll sein nach größeren Änderungen an Ihrer Anlage.

### Funktioniert die Integration mit allen Wetterdiensten?

Die Integration funktioniert mit allen Standard-Home-Assistant-Wetter-Integrationen. Getestet wurde mit DWD, Met.no, OpenWeatherMap und weiteren. Wichtig ist, dass die Wetter-Entity Temperatur, Bewölkung und eine Mehrtages-Vorhersage (numerisch) liefert.

### Wie oft sollte ich das Modell neu trainieren?

Normalerweise ist dies nicht nötig. Die Integration trainiert automatisch periodisch. Ein manuelles Retraining kann sinnvoll sein nach längeren Ausfallzeiten, nach Änderungen an der Anlage oder wenn Sie vermuten, dass das Modell nicht optimal arbeitet.

### Was bedeutet Autarkie genau?

Autarkie ist der Prozentsatz Ihres Stromverbrauchs, der durch selbst erzeugten Solarstrom gedeckt wird. 100 Prozent Autarkie bedeutet vollständige Eigenversorgung. Die Berechnung berücksichtigt Solarertrag, Verbrauch, Netzbezug und Einspeisung.

---

## Schlusswort

Dieses Handbuch hat Ihnen einen umfassenden Überblick über die Solar Forecast ML Integration gegeben. Sie verstehen nun, wie die Integration funktioniert, welche Sensoren benötigt werden, wie die Daten gespeichert werden und wie das maschinelle Lernen arbeitet.

Für weitere Unterstützung steht die Community auf GitHub zur Verfügung. Bei Fragen, Problemen oder Verbesserungsvorschlägen können Sie Issues erstellen oder an Diskussionen teilnehmen.

Viel Erfolg mit Ihrer Solaranlage und präzisen Vorhersagen!

---

**Solar Forecast ML Version 10.0.0 "Lyra"**

Entwickelt von Zara-Toorox

Open Source unter AGPL-3.0 Lizenz

November 2025
