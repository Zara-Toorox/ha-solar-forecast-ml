# Protected Code Notice / Hinweis zu geschütztem Code

## Deutsch

Einige Dateien in dieser Integration sind mit **PyArmor** obfuskiert (verschlüsselt).

### Warum ist der Code geschützt?

1. **Schutz vor AI-Training**: Ich möchte verhindern, dass mein Quellcode von AI-Modellen wie ChatGPT, Claude, Gemini oder anderen Large Language Models (LLMs) ohne Genehmigung zum Training verwendet wird.

2. **Schutz geistigen Eigentums**: Die Algorithmen für Solarprognosen, AI-Learning und Wetter-Analyse wurden mit erheblichem Aufwand entwickelt und stellen mein geistiges Eigentum dar.

3. **Open Source mit Grenzen**: Diese Integration ist kostenlos für den persönlichen Gebrauch, aber der Quellcode ist proprietär und unterliegt einer Non-Commercial License.

4. **Leider notwendig**: Da in der Vergangenheit bereits Code ohne meine Zustimmung kopiert, in kommerzielle Anwendungen übernommen und mittels KI versucht wurde, ihn auszulesen und zu verändern, sehe ich mich leider gezwungen, den Quellcode zu schützen.

5. **Transparenz**: Bei berechtigtem Interesse gebe ich selbstverständlich gerne Auskunft über den Code oder lege ihn offen. Kontaktiere mich einfach über GitHub Issues oder Discussions.

### Welche Dateien sind geschützt?

Die folgenden Kern-Module sind obfuskiert:

| Modul | Beschreibung |
|-------|--------------|
| `ai/ai_predictor.py` | Haupt-AI-Vorhersage-Engine |
| `ai/ai_tiny_lstm.py` | LSTM Neural Network Modell |
| `ai/ai_feature_engineering.py` | Feature-Extraktion für ML |
| `ai/ai_grid_search.py` | Hyperparameter-Optimierung |
| `physics/physics_calibrator.py` | Physik-basierte Kalibrierung |
| `data/data_weather_expert_blender.py` | Multi-Source Wetter-Blending |
| `data/data_weather_corrector.py` | Wetter-Korrektur-Algorithmen |
| `data/data_forecast_handler.py` | Forecast-Logik |
| `data/data_weather_precision.py` | Präzisions-Algorithmen |
| `data/data_weather_pipeline_manager.py` | Pipeline-Steuerung |
| `data/data_hourly_predictions.py` | Stündliche Vorhersage-Logik |
| `data/data_shadow_detection.py` | Schatten-Erkennung |
| `production/production_scheduled_tasks.py` | Scheduling & Produktions-Algorithmen |
| `forecast/forecast_rule_based_strategy.py` | Regelbasierte Strategien |

### Funktionalität

Die Obfuskierung hat **keinen Einfluss auf die Funktionalität**. Die Integration funktioniert identisch zur nicht-obfuskierten Version. Der Runtime-Overhead ist minimal.

---

## English

Some files in this integration are obfuscated (encrypted) with **PyArmor**.

### Why is the code protected?

1. **Protection against AI Training**: I want to prevent my source code from being used to train AI models like ChatGPT, Claude, Gemini, or other Large Language Models (LLMs) without permission.

2. **Intellectual Property Protection**: The algorithms for solar forecasting, AI-learning, and weather analysis were developed with considerable effort and represent my intellectual property.

3. **Open Source with Limits**: This integration is free for personal use, but the source code is proprietary and subject to a Non-Commercial License.

4. **Unfortunately necessary**: Since code has been copied without my consent, incorporated into commercial applications, and attempts have been made to read and modify it using AI in the past, I unfortunately feel compelled to protect the source code.

5. **Transparency**: If you have a legitimate interest, I'm happy to provide information about the code or disclose it. Just contact me via GitHub Issues or Discussions.

### Which files are protected?

The following core modules are obfuscated:

| Module | Description |
|--------|-------------|
| `ai/ai_predictor.py` | Main AI prediction engine |
| `ai/ai_tiny_lstm.py` | LSTM neural network model |
| `ai/ai_feature_engineering.py` | Feature extraction for ML |
| `ai/ai_grid_search.py` | Hyperparameter optimization |
| `physics/physics_calibrator.py` | Physics-based calibration |
| `data/data_weather_expert_blender.py` | Multi-source weather blending |
| `data/data_weather_corrector.py` | Weather correction algorithms |
| `data/data_forecast_handler.py` | Forecast logic |
| `data/data_weather_precision.py` | Precision algorithms |
| `data/data_weather_pipeline_manager.py` | Pipeline management |
| `data/data_hourly_predictions.py` | Hourly prediction logic |
| `data/data_shadow_detection.py` | Shadow detection |
| `production/production_scheduled_tasks.py` | Scheduling & production algorithms |
| `forecast/forecast_rule_based_strategy.py` | Rule-based strategies |

### Functionality

The obfuscation has **no impact on functionality**. The integration works identically to the non-obfuscated version. Runtime overhead is minimal.

---

*Solar Forecast ML - Copyright (C) 2025 Zara-Toorox*
*Protected with PyArmor 9.2.3*
