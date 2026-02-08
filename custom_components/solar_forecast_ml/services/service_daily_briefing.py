# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# *****************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# Refactored: JSON replaced with DatabaseManager @zara
# *****************************************************************************

"""
Daily briefing service for Solar Forecast ML.
Generates and sends daily solar briefing notifications.
Uses DatabaseManager for all data operations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, date as date_type
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..core.core_helpers import SafeDateTimeUtil as dt_util
from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class DailyBriefingService:
    """Service for generating and sending daily solar briefing notifications. @zara"""

    def __init__(self, hass: HomeAssistant, coordinator) -> None:
        """Initialize the daily briefing service. @zara"""
        self.hass = hass
        self.coordinator = coordinator

    @property
    def db_manager(self) -> Optional[DatabaseManager]:
        """Get database manager from coordinator. @zara V16.1 fix"""
        # V16.1: Correct path is coordinator.data_manager._db_manager @zara
        data_manager = getattr(self.coordinator, "data_manager", None)
        if data_manager:
            return getattr(data_manager, "_db_manager", None)
        return None

    async def send_daily_briefing(
        self,
        notify_service: str = "persistent_notification",
        language: str = "de",
    ) -> Dict[str, Any]:
        """Generate and send daily solar briefing notification. @zara

        Args:
            notify_service: Name of the notify service (e.g., "mobile_app_iphone")
            language: Language code ("de" or "en")

        Returns:
            Dictionary with result status and message preview
        """
        try:
            # Validate notify service if specified
            if (
                notify_service
                and notify_service != "persistent_notification"
                and "mobile_app" in notify_service
            ):
                service_name = notify_service.replace("notify.", "")
                if not self.hass.services.has_service("notify", service_name):
                    error_msg = f"Notify service not found: notify.{service_name}"
                    _LOGGER.error(error_msg)
                    return {"success": False, "error": error_msg}

            # Get forecast data from DB
            forecast_data = await self._get_today_forecast_data()
            if not forecast_data:
                _LOGGER.error("Failed to retrieve today's forecast data for briefing")
                return {"success": False, "error": "No forecast data available"}

            # Get yesterday's actual data from DB
            yesterday_data = await self._get_yesterday_actual_data()

            # Get astronomy data from cache
            astro_data = await self._get_astronomy_data()

            # Get weather data from coordinator
            weather_data = await self._get_today_weather_data()

            # Generate briefing message
            message_data = await self._generate_briefing_message(
                forecast_data, yesterday_data, astro_data, weather_data, language
            )

            # Send to persistent notification
            persistent_notification = {
                "title": message_data["title"],
                "message": message_data["message"],
                "data": {
                    "notification_id": "solar_briefing_daily",
                    "tag": "solar_briefing",
                },
            }

            await self.hass.services.async_call(
                "notify",
                "persistent_notification",
                persistent_notification,
                blocking=True,
            )

            _LOGGER.info("Full briefing sent to persistent_notification (HA UI)")

            # Send to mobile if configured
            if (
                notify_service
                and notify_service != "persistent_notification"
                and "mobile_app" in notify_service
            ):
                prediction_kwh = forecast_data["prediction_kwh"]
                clouds = weather_data.get("clouds") if weather_data else None
                weather_emoji, _ = self._interpret_weather(prediction_kwh, clouds, language)
                weather_desc = self._get_weather_description(clouds, language)

                temp_str = ""
                if weather_data and weather_data.get("temperature") is not None:
                    temp = weather_data["temperature"]
                    temp_str = f", {temp:.0f}C"

                mobile_message = f"{weather_emoji} {prediction_kwh:.2f} kWh | {weather_desc}{temp_str}"

                mobile_notification = {
                    "title": message_data["title"],
                    "message": mobile_message,
                    "data": {
                        "push": {"interruption-level": "time-sensitive"},
                        "presentation_options": ["alert", "sound"],
                    },
                }

                await self.hass.services.async_call(
                    "notify",
                    notify_service.replace("notify.", ""),
                    mobile_notification,
                    blocking=True,
                )

                _LOGGER.info(f"Additional mobile push notification sent to {notify_service}")

            _LOGGER.info(f"Daily solar briefing sent via {notify_service} (language: {language})")

            return {
                "success": True,
                "title": message_data["title"],
                "message_preview": message_data["message"][:100] + "...",
            }

        except Exception as err:
            _LOGGER.error(f"Failed to send daily briefing: {err}", exc_info=True)
            return {"success": False, "error": str(err)}

    async def _get_today_forecast_data(self) -> Optional[Dict[str, Any]]:
        """Get today's forecast data from database. @zara"""
        try:
            db = self.db_manager
            if not db:
                _LOGGER.warning("Database manager not available for forecast data")
                return None

            today = dt_util.now().date().isoformat()

            # Get today's forecast from daily_forecasts table
            row = await db.fetchone(
                """SELECT prediction_kwh, source, locked
                   FROM daily_forecasts
                   WHERE forecast_date = ? AND forecast_type = 'today'""",
                (today,),
            )

            if row:
                return {
                    "date": today,
                    "prediction_kwh": row[0] or 0.0,
                    "source": row[1] or "unknown",
                    "locked": bool(row[2]) if row[2] is not None else False,
                }

            # Fallback: calculate from hourly predictions
            sum_row = await db.fetchone(
                """SELECT SUM(prediction_kwh)
                   FROM hourly_predictions
                   WHERE target_date = ?""",
                (today,),
            )

            return {
                "date": today,
                "prediction_kwh": sum_row[0] if sum_row and sum_row[0] else 0.0,
                "source": "hourly_sum",
                "locked": False,
            }

        except Exception as err:
            _LOGGER.error(f"Error loading today forecast from DB: {err}")
            return None

    async def _get_yesterday_actual_data(self) -> Optional[Dict[str, Any]]:
        """Get yesterday's actual production from database. @zara"""
        try:
            db = self.db_manager
            if not db:
                return None

            yesterday = (dt_util.now().date() - timedelta(days=1)).isoformat()

            # Get from daily_summaries
            row = await db.fetchone(
                """SELECT actual_total_kwh, predicted_total_kwh, accuracy_percent
                   FROM daily_summaries
                   WHERE date = ?""",
                (yesterday,),
            )

            if row and row[0]:
                return {
                    "date": yesterday,
                    "actual_kwh": row[0] or 0.0,
                    "forecast_kwh": row[1] or 0.0,
                    "accuracy": row[2] or 0.0,
                }

            return None

        except Exception as err:
            _LOGGER.error(f"Error loading yesterday data from DB: {err}")
            return None

    async def _get_astronomy_data(self) -> Optional[Dict[str, Any]]:
        """Get today's astronomy data from cache. @zara"""
        try:
            from ..astronomy.astronomy_cache_manager import get_cache_manager

            astronomy_manager = get_cache_manager()

            today = dt_util.now().date()

            astro_data = astronomy_manager.get_day_data(today)
            if astro_data:
                return {
                    "sunrise": astro_data.get("sunrise_time"),
                    "sunset": astro_data.get("sunset_time"),
                    "solar_noon": astro_data.get("solar_noon"),
                    "daylight_hours": astro_data.get("day_length_hours", 0.0),
                }
            return None

        except Exception as err:
            _LOGGER.error(f"Error loading astronomy data: {err}")
            return None

    async def _get_today_weather_data(self) -> Optional[Dict[str, Any]]:
        """Get today's weather data from coordinator or DB fallback. @zara"""
        try:
            # Try to get from coordinator's weather cache
            if self.coordinator:
                weather_pipeline = getattr(self.coordinator, "weather_pipeline_manager", None)
                if weather_pipeline:
                    weather_corrector = getattr(weather_pipeline, "weather_corrector", None)
                    if weather_corrector:
                        current_weather = getattr(weather_corrector, "_current_weather_cache", None)
                        if current_weather:
                            return {
                                "temperature": current_weather.get("temperature"),
                                "clouds": current_weather.get("clouds"),
                                "wind": current_weather.get("wind"),
                                "humidity": current_weather.get("humidity"),
                            }

            # Fallback: get average weather from prediction_weather DB
            db = self.db_manager
            if db:
                today = dt_util.now().date().isoformat()
                row = await db.fetchone(
                    """SELECT AVG(pw.clouds), AVG(pw.temperature)
                       FROM prediction_weather pw
                       JOIN hourly_predictions hp ON pw.prediction_id = hp.prediction_id
                       WHERE hp.target_date = ? AND hp.is_production_hour = 1""",
                    (today,),
                )
                if row and row[0] is not None:
                    return {
                        "temperature": round(row[1], 1) if row[1] is not None else None,
                        "clouds": round(row[0], 0),
                        "wind": None,
                        "humidity": None,
                    }

            return None

        except Exception as err:
            _LOGGER.warning(f"Error loading weather data: {err}")
            return None

    async def _get_peak_hour(self) -> Optional[Dict[str, Any]]:
        """Get the hour with highest predicted production today. @zara"""
        try:
            db = self.db_manager
            if not db:
                return None
            today = dt_util.now().date().isoformat()
            row = await db.fetchone(
                """SELECT target_hour, prediction_kwh, prediction_method
                   FROM hourly_predictions
                   WHERE target_date = ? AND prediction_kwh > 0
                   ORDER BY prediction_kwh DESC LIMIT 1""",
                (today,),
            )
            if row:
                return {"hour": row[0], "kwh": row[1], "method": row[2]}
            return None
        except Exception:
            return None

    async def _get_system_status(self, language: str) -> List[str]:
        """Get system status info for briefing. @zara"""
        lines = []
        try:
            db = self.db_manager
            if not db:
                return lines

            yesterday = (dt_util.now().date() - timedelta(days=1)).isoformat()

            # Check if EOD ran yesterday (daily_summaries entry exists)
            eod_row = await db.fetchone(
                "SELECT actual_total_kwh, accuracy_percent FROM daily_summaries WHERE date = ?",
                (yesterday,),
            )
            if eod_row and eod_row[0] is not None:
                acc = eod_row[1]
                if acc is not None:
                    if language == "de":
                        lines.append(f"   Gestern: Forecast war {acc:.0f}% genau")
                    else:
                        lines.append(f"   Yesterday: Forecast was {acc:.0f}% accurate")

            # Check method_performance_learning
            learn_row = await db.fetchone(
                """SELECT COUNT(*), MAX(sample_count)
                   FROM method_performance_learning
                   WHERE last_updated >= ?""",
                (yesterday,),
            )
            if learn_row and learn_row[0] and learn_row[0] > 0:
                buckets = learn_row[0]
                samples = learn_row[1] or 0
                if language == "de":
                    lines.append(f"   KI-Lernstatus: {buckets} Wetter-Buckets aktiv ({samples} Samples)")
                else:
                    lines.append(f"   AI learning: {buckets} weather buckets active ({samples} samples)")

            # Check AI model info
            ai_predictor = getattr(self.coordinator, "ai_predictor", None)
            if ai_predictor:
                model_info = getattr(ai_predictor, "_model_info", None)
                if model_info:
                    r2 = model_info.get("r2_score")
                    if r2 is not None:
                        if language == "de":
                            lines.append(f"   KI-Modell: R²={r2:.3f}")
                        else:
                            lines.append(f"   AI model: R²={r2:.3f}")

            # Check exclude_from_learning count yesterday
            excl_row = await db.fetchone(
                """SELECT COUNT(*) FROM hourly_predictions
                   WHERE target_date = ? AND exclude_from_learning = TRUE""",
                (yesterday,),
            )
            if excl_row and excl_row[0] and excl_row[0] > 0:
                if language == "de":
                    lines.append(f"   Gestern: {excl_row[0]}h vom Lernen ausgeschlossen (Schnee/Frost)")
                else:
                    lines.append(f"   Yesterday: {excl_row[0]}h excluded from learning (snow/frost)")

        except Exception as e:
            _LOGGER.debug(f"System status info error: {e}")
        return lines

    def _get_daily_quote(self, clouds: Optional[float], prediction_kwh: float, day_of_year: int, language: str) -> str:
        """Get a rotating weather-appropriate daily quote. @zara"""
        if language == "de":
            sunny_quotes = [
                "Die Sonne meint es gut mit uns heute!",
                "Perfekter Tag fuer die Panels!",
                "Heute ernten wir Sonnenstrahlen.",
                "Die Solarzellen freuen sich schon!",
                "Ein guter Tag fuer saubere Energie.",
                "Sonne satt - die Panels laufen heiss!",
                "Heute tanken die Module ordentlich auf.",
            ]
            cloudy_quotes = [
                "Auch hinter Wolken scheint die Sonne.",
                "Jedes bisschen Licht zaehlt!",
                "Die Panels machen auch bei Wolken nicht frei.",
                "Grau ist auch eine Farbe - die Panels arbeiten trotzdem.",
                "Diffuses Licht ist besser als kein Licht!",
                "Die Module trotzen den Wolken tapfer.",
                "Nicht jeder Tag kann ein Sonnentag sein.",
            ]
            overcast_quotes = [
                "Heute ist ein Ruhetag fuer die Panels.",
                "Selbst an trueben Tagen - das System lernt weiter!",
                "Die KI nutzt auch wolkige Tage zum Lernen.",
                "Morgen scheint die Sonne bestimmt wieder!",
                "Geduld - bessere Tage kommen.",
                "Auch ein kleiner Ertrag ist ein Ertrag.",
                "Die Technik ruht sich aus fuer den naechsten Sonnentag.",
            ]
        else:
            sunny_quotes = [
                "The sun is smiling on us today!",
                "Perfect day for the panels!",
                "Today we harvest sunbeams.",
                "The solar cells are ready and waiting!",
                "A great day for clean energy.",
                "Sunshine galore - panels running hot!",
                "The modules are charging up nicely today.",
            ]
            cloudy_quotes = [
                "The sun shines behind the clouds too.",
                "Every bit of light counts!",
                "Panels don't take days off for clouds.",
                "Diffuse light is better than no light!",
                "The modules bravely face the clouds.",
                "Not every day can be a sunny day.",
                "Even cloudy days have their moments.",
            ]
            overcast_quotes = [
                "A rest day for the panels.",
                "Even on grey days, the system keeps learning!",
                "The AI uses cloudy days to learn too.",
                "Tomorrow the sun will shine again!",
                "Patience - better days are coming.",
                "A small yield is still a yield.",
                "The tech rests up for the next sunny day.",
            ]

        # Select pool based on conditions
        if clouds is not None:
            if clouds < 40:
                pool = sunny_quotes
            elif clouds < 75:
                pool = cloudy_quotes
            else:
                pool = overcast_quotes
        elif prediction_kwh > 5:
            pool = sunny_quotes
        elif prediction_kwh > 1:
            pool = cloudy_quotes
        else:
            pool = overcast_quotes

        # Rotate based on day of year
        return pool[day_of_year % len(pool)]

    async def _generate_briefing_message(
        self,
        forecast_data: Dict[str, Any],
        yesterday_data: Optional[Dict[str, Any]],
        astro_data: Optional[Dict[str, Any]],
        weather_data: Optional[Dict[str, Any]],
        language: str,
    ) -> Dict[str, str]:
        """Generate formatted briefing message. @zara

        Args:
            forecast_data: Today's forecast data
            yesterday_data: Yesterday's actual data (optional)
            astro_data: Today's astronomy data (optional)
            weather_data: Today's weather data (optional)
            language: Language code ("de" or "en")

        Returns:
            Dictionary with "title" and "message" keys
        """
        try:
            date_obj = datetime.strptime(forecast_data["date"], "%Y-%m-%d")
        except (ValueError, TypeError, KeyError):
            _LOGGER.error("Invalid date format in forecast_data")
            date_obj = dt_util.now()

        day_of_year = date_obj.timetuple().tm_yday

        # Build title with emoji
        if language == "de":
            weekday = date_obj.strftime("%A")
            weekday_de = {
                "Monday": "Montag",
                "Tuesday": "Dienstag",
                "Wednesday": "Mittwoch",
                "Thursday": "Donnerstag",
                "Friday": "Freitag",
                "Saturday": "Samstag",
                "Sunday": "Sonntag",
            }.get(weekday, weekday)
            title = f"\u2600\ufe0f Solar Forecast - {weekday_de}, {date_obj.strftime('%d. %b')}"
        else:
            title = f"\u2600\ufe0f Solar Forecast - {date_obj.strftime('%A, %b %d')}"

        message_parts = []

        # Daily quote
        prediction_kwh = forecast_data["prediction_kwh"]
        clouds = weather_data.get("clouds") if weather_data else None
        quote = self._get_daily_quote(clouds, prediction_kwh, day_of_year, language)
        message_parts.append(f"\U0001f4ac {quote}")
        message_parts.append("")

        # --- FORECAST SECTION ---
        weather_emoji, weather_text = self._interpret_weather(prediction_kwh, clouds, language)
        message_parts.append(f"{weather_emoji} {weather_text}")
        message_parts.append("")

        # Forecast line with bolt emoji
        message_parts.append(f"\u26a1 Forecast: {prediction_kwh:.2f} kWh")

        # Yesterday comparison
        if yesterday_data:
            yesterday_actual = yesterday_data["actual_kwh"]
            if yesterday_actual > 0 and prediction_kwh > 0:
                ratio = prediction_kwh / yesterday_actual
                if ratio > 1.5:
                    if language == "de":
                        message_parts.append(f"   \u2197\ufe0f {ratio:.1f}x besser als gestern ({yesterday_actual:.2f} kWh)")
                    else:
                        message_parts.append(f"   \u2197\ufe0f {ratio:.1f}x better than yesterday ({yesterday_actual:.2f} kWh)")
                elif ratio < 0.67:
                    if language == "de":
                        message_parts.append(f"   \u2198\ufe0f {(1 / ratio):.1f}x weniger als gestern ({yesterday_actual:.2f} kWh)")
                    else:
                        message_parts.append(f"   \u2198\ufe0f {(1 / ratio):.1f}x less than yesterday ({yesterday_actual:.2f} kWh)")
                else:
                    if language == "de":
                        message_parts.append(f"   \u2194\ufe0f Aehnlich wie gestern ({yesterday_actual:.2f} kWh)")
                    else:
                        message_parts.append(f"   \u2194\ufe0f Similar to yesterday ({yesterday_actual:.2f} kWh)")

        message_parts.append("")

        # --- BEST HOUR ---
        peak_hour = await self._get_peak_hour()
        if peak_hour:
            h = peak_hour["hour"]
            if language == "de":
                message_parts.append(f"\U0001f31f Beste Stunde: {h:02d}:00-{h+1:02d}:00 Uhr ({peak_hour['kwh']:.3f} kWh)")
            else:
                message_parts.append(f"\U0001f31f Best hour: {h:02d}:00-{h+1:02d}:00 ({peak_hour['kwh']:.3f} kWh)")

        # --- WEATHER & DAYLIGHT ---
        weather_desc = self._get_weather_description(clouds, language)
        temp_str = ""
        if weather_data and weather_data.get("temperature") is not None:
            temp_str = f" ({weather_data['temperature']:.0f}\u00b0C)"
        if language == "de":
            message_parts.append(f"\U0001f324\ufe0f Wetter: {weather_desc}{temp_str}")
        else:
            message_parts.append(f"\U0001f324\ufe0f Weather: {weather_desc}{temp_str}")

        if astro_data:
            daylight_hours = astro_data.get("daylight_hours", 0.0)
            hours = int(daylight_hours)
            minutes = int((daylight_hours - hours) * 60)

            sunrise = astro_data.get("sunrise")
            sunset = astro_data.get("sunset")
            sun_times = ""
            if sunrise and sunset:
                try:
                    sr_str = str(sunrise)
                    sr = sr_str.split("T")[1][:5] if "T" in sr_str else sr_str.split(" ")[1][:5] if " " in sr_str else sr_str[:5]
                    ss_str = str(sunset)
                    ss = ss_str.split("T")[1][:5] if "T" in ss_str else ss_str.split(" ")[1][:5] if " " in ss_str else ss_str[:5]
                    sun_times = f" ({sr} - {ss})"
                except Exception:
                    pass

            if language == "de":
                message_parts.append(f"\U0001f305 Tageslicht: {hours}h {minutes}min{sun_times}")
            else:
                message_parts.append(f"\U0001f305 Daylight: {hours}h {minutes}min{sun_times}")

        message_parts.append("")

        # --- SYSTEM STATUS ---
        status_lines = await self._get_system_status(language)
        if status_lines:
            if language == "de":
                message_parts.append("\U0001f916 System-Status:")
            else:
                message_parts.append("\U0001f916 System Status:")
            message_parts.extend(status_lines)
            message_parts.append("")

        # Shadow summary from DB
        shadow_summary = await self._get_yesterday_shadow_summary(language)
        if shadow_summary:
            message_parts.append(shadow_summary)
            message_parts.append("")

        # Closing quote
        closing = self._get_closing_message(prediction_kwh, clouds, language)
        message_parts.append(closing)

        message = "\n".join(message_parts)

        return {"title": title, "message": message}

    def _interpret_weather(
        self, prediction_kwh: float, clouds: Optional[float], language: str
    ) -> tuple[str, str]:
        """Interpret weather from cloud cover and prediction value. @zara"""
        if clouds is not None:
            if clouds < 20:
                emoji = "\u2600\ufe0f"
                text = (
                    "Voraussichtlich sonnig - gute Bedingungen!"
                    if language == "de"
                    else "Likely sunny - good conditions!"
                )
            elif clouds < 40:
                emoji = "\U0001f324\ufe0f"
                text = (
                    "Ueberwiegend sonnig erwartet"
                    if language == "de"
                    else "Mostly sunny expected"
                )
            elif clouds < 60:
                emoji = "\u26c5"
                text = (
                    "Wechselhaft - Wolken moeglich"
                    if language == "de"
                    else "Variable - clouds possible"
                )
            elif clouds < 80:
                emoji = "\U0001f325\ufe0f"
                text = (
                    "Eher bewoelkt erwartet" if language == "de" else "Rather cloudy expected"
                )
            else:
                emoji = "\u2601\ufe0f"
                text = (
                    "Ueberwiegend bewoelkt erwartet"
                    if language == "de"
                    else "Mostly cloudy expected"
                )
        else:
            # Fallback based on prediction
            if prediction_kwh > 15:
                emoji = "\u2600\ufe0f"
                text = "Gute Bedingungen moeglich" if language == "de" else "Good conditions possible"
            elif prediction_kwh > 10:
                emoji = "\U0001f324\ufe0f"
                text = "Ordentliche Produktion moeglich" if language == "de" else "Decent production possible"
            elif prediction_kwh > 5:
                emoji = "\u26c5"
                text = "Moderate Produktion erwartet" if language == "de" else "Moderate production expected"
            elif prediction_kwh > 2:
                emoji = "\U0001f325\ufe0f"
                text = "Eingeschraenkte Produktion wahrscheinlich" if language == "de" else "Limited production likely"
            elif prediction_kwh > 0.5:
                emoji = "\u2601\ufe0f"
                text = "Geringe Produktion erwartet" if language == "de" else "Low production expected"
            else:
                emoji = "\U0001f327\ufe0f"
                text = "Kaum Produktion erwartet" if language == "de" else "Minimal production expected"

        return (emoji, text)

    def _get_weather_description(self, clouds: Optional[float], language: str) -> str:
        """Get detailed weather description from cloud cover. @zara"""
        if clouds is None:
            return "Keine Wetterdaten" if language == "de" else "No weather data"

        if clouds < 10:
            return "Voraussichtlich klar" if language == "de" else "Expected clear"
        elif clouds < 25:
            return "Voraussichtlich sonnig" if language == "de" else "Expected sunny"
        elif clouds < 50:
            return "Wolken moeglich" if language == "de" else "Clouds possible"
        elif clouds < 75:
            return "Eher bewoelkt" if language == "de" else "Rather cloudy"
        elif clouds < 90:
            return "Stark bewoelkt erwartet" if language == "de" else "Heavy clouds expected"
        else:
            return "Ueberwiegend bedeckt erwartet" if language == "de" else "Overcast expected"

    def _get_closing_message(
        self, prediction_kwh: float, clouds: Optional[float], language: str
    ) -> str:
        """Get closing message based on prediction and cloud cover. @zara"""
        if clouds is not None:
            if clouds < 20:
                return "\u2600\ufe0f Gute Chancen auf Sonne!" if language == "de" else "\u2600\ufe0f Good chance of sun!"
            elif clouds < 40:
                return "\U0001f324\ufe0f Sonnenschein wahrscheinlich" if language == "de" else "\U0001f324\ufe0f Sunshine likely"
            elif clouds < 60:
                return "\u26c5 Sonne moeglich" if language == "de" else "\u26c5 Sun possible"
            elif clouds < 80:
                return "\U0001f325\ufe0f Wenig Sonne erwartet" if language == "de" else "\U0001f325\ufe0f Little sun expected"
            else:
                return "\u2601\ufe0f Bewoelkung wahrscheinlich" if language == "de" else "\u2601\ufe0f Clouds likely"

        if prediction_kwh > 10:
            return "\u2600\ufe0f Gute Chancen auf Sonne!" if language == "de" else "\u2600\ufe0f Good chance of sun!"
        elif prediction_kwh > 5:
            return "\U0001f324\ufe0f Ordentliche Produktion moeglich" if language == "de" else "\U0001f324\ufe0f Decent production possible"
        elif prediction_kwh > 2:
            return "\u26c5 Etwas Sonne moeglich" if language == "de" else "\u26c5 Some sun possible"
        else:
            return "\u2601\ufe0f Wenig Sonne erwartet" if language == "de" else "\u2601\ufe0f Little sun expected"

    async def _get_yesterday_shadow_summary(self, language: str) -> Optional[str]:
        """Get yesterday's shadow detection summary from DB. @zara"""
        try:
            db = self.db_manager
            if not db:
                return None

            yesterday = (dt_util.now().date() - timedelta(days=1)).isoformat()

            # Get shadow data from shadow_learning_history + hourly_predictions
            rows = await db.fetchall(
                """SELECT slh.hour, slh.shadow_detected, slh.root_cause,
                          hp.actual_kwh, hp.prediction_kwh
                   FROM shadow_learning_history slh
                   LEFT JOIN hourly_predictions hp
                       ON hp.target_date = slh.date AND hp.target_hour = slh.hour
                   WHERE slh.date = ? AND slh.shadow_detected = 1""",
                (yesterday,),
            )

            if not rows:
                return None

            shadow_hours = len(rows)
            if shadow_hours == 0:
                return None

            # Calculate loss
            total_loss = 0.0
            dominant_cause = "unknown"
            cause_counts = {}

            for row in rows:
                actual = row[3] or 0
                predicted = row[4] or 0
                if predicted > actual:
                    total_loss += predicted - actual

                cause = row[2] or "unknown"
                cause_counts[cause] = cause_counts.get(cause, 0) + 1

            if cause_counts:
                dominant_cause = max(cause_counts.items(), key=lambda x: x[1])[0]

            # Build summary
            if language == "de":
                header = "Schatten-Analyse (Gestern):"
                hours_text = f"   {shadow_hours}h Verschattung erkannt"
                loss_text = f"   Verlust: {total_loss:.2f} kWh"

                cause_map = {
                    "weather_clouds": "Wolken",
                    "building_tree_obstruction": "Gebaeude/Baum",
                    "normal_variation": "Normale Variation",
                    "unknown": "Unbekannt",
                }
                cause_text = cause_map.get(dominant_cause, dominant_cause)
                cause_line = f"   Ursache: {cause_text}"
            else:
                header = "Shadow Analysis (Yesterday):"
                hours_text = f"   {shadow_hours}h shadowing detected"
                loss_text = f"   Loss: {total_loss:.2f} kWh"

                cause_map = {
                    "weather_clouds": "Clouds",
                    "building_tree_obstruction": "Building/Tree",
                    "normal_variation": "Normal variation",
                    "unknown": "Unknown",
                }
                cause_text = cause_map.get(dominant_cause, dominant_cause)
                cause_line = f"   Cause: {cause_text}"

            return "\n".join([header, hours_text, loss_text, cause_line])

        except Exception as e:
            _LOGGER.warning(f"Failed to get shadow summary from DB: {e}")
            return None
