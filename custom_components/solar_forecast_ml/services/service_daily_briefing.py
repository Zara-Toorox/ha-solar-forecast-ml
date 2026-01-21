# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
import homeassistant.util.dt as dt_util

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)

class DailyBriefingService:
    """Service for generating and sending daily solar briefing notifications."""

    def __init__(self, hass: HomeAssistant, coordinator) -> None:
        """Initialize the daily briefing service. @zara"""
        self.hass = hass
        self.coordinator = coordinator

    async def send_daily_briefing(
        self,
        notify_service: str = "persistent_notification",
        language: str = "de",
    ) -> dict[str, Any]:
        """Generate and send daily solar briefing notification.

        Args:
            notify_service: Name of the notify service (e.g., "mobile_app_iphone")
            language: Language code ("de" or "en")

        Returns:
            Dictionary with result status and message preview
        """
        try:

            if notify_service and notify_service != "persistent_notification" and "mobile_app" in notify_service:
                service_name = notify_service.replace("notify.", "")
                if not self.hass.services.has_service("notify", service_name):
                    error_msg = f"Notify service not found: notify.{service_name}"
                    _LOGGER.error(error_msg)
                    return {"success": False, "error": error_msg}

            forecast_data = await self._get_today_forecast_data()
            if not forecast_data:
                _LOGGER.error("Failed to retrieve today's forecast data for briefing")
                return {"success": False, "error": "No forecast data available"}

            yesterday_data = await self._get_yesterday_actual_data()

            astro_data = await self._get_astronomy_data()

            weather_data = await self._get_today_weather_data()

            message_data = await self._generate_briefing_message(
                forecast_data, yesterday_data, astro_data, weather_data, language
            )

            persistent_notification = {
                "title": message_data["title"],
                "message": message_data["message"],
                "data": {
                    "notification_id": "solar_briefing_daily",
                    "tag": "solar_briefing",
                }
            }

            await self.hass.services.async_call(
                "notify",
                "persistent_notification",
                persistent_notification,
                blocking=True,
            )

            _LOGGER.info("Full briefing sent to persistent_notification (HA UI)")

            if notify_service and notify_service != "persistent_notification" and "mobile_app" in notify_service:

                prediction_kwh = forecast_data['prediction_kwh']
                clouds = weather_data.get("clouds") if weather_data else None
                weather_emoji, _ = self._interpret_weather(prediction_kwh, clouds, language)
                weather_desc = self._get_weather_description(clouds, language)

                temp_str = ""
                if weather_data and weather_data.get("temperature") is not None:
                    temp = weather_data["temperature"]
                    temp_str = f", {temp:.0f}°C"

                mobile_message = f"{weather_emoji} {prediction_kwh:.2f} kWh | {weather_desc}{temp_str}"

                mobile_notification = {
                    "title": message_data["title"],
                    "message": mobile_message,
                    "data": {
                        "push": {
                            "interruption-level": "time-sensitive"
                        },
                        "presentation_options": ["alert", "sound"],
                    }
                }

                await self.hass.services.async_call(
                    "notify",
                    notify_service.replace("notify.", ""),
                    mobile_notification,
                    blocking=True,
                )

                _LOGGER.info(f"Additional mobile push notification sent to {notify_service}")

            _LOGGER.info(
                f"Daily solar briefing sent via {notify_service} (language: {language})"
            )

            return {
                "success": True,
                "title": message_data["title"],
                "message_preview": message_data["message"][:100] + "...",
            }

        except Exception as err:
            _LOGGER.error(f"Failed to send daily briefing: {err}", exc_info=True)
            return {"success": False, "error": str(err)}

    async def _get_today_forecast_data(self) -> dict[str, Any] | None:
        """Get today's forecast data from daily_forecasts.json. @zara"""
        try:
            data_manager = self.coordinator.data_manager
            data = await data_manager.load_daily_forecasts()

            today = data.get("today", {})
            forecast_day = today.get("forecast_day", {})

            return {
                "date": today.get("date"),
                "prediction_kwh": forecast_day.get("prediction_kwh", 0.0),
                "source": forecast_day.get("source", "unknown"),
                "locked": forecast_day.get("locked", False),
            }
        except Exception as err:
            _LOGGER.error(f"Error loading today forecast: {err}")
            return None

    async def _get_yesterday_actual_data(self) -> dict[str, Any] | None:
        """Get yesterday's actual production from history. @zara"""
        try:
            data_manager = self.coordinator.data_manager
            data = await data_manager.load_daily_forecasts()

            history = data.get("history", [])
            if history:

                yesterday = history[0]
                return {
                    "date": yesterday.get("date"),
                    "actual_kwh": yesterday.get("actual_kwh", 0.0),
                    "forecast_kwh": yesterday.get("forecast_kwh", 0.0),
                    "accuracy": yesterday.get("accuracy", 0.0),
                }
            return None
        except Exception as err:
            _LOGGER.error(f"Error loading yesterday data: {err}")
            return None

    async def _get_astronomy_data(self) -> dict[str, Any] | None:
        """Get today's astronomy data from astronomy_cache.json. @zara"""
        try:

            from ..astronomy.astronomy_cache_manager import get_cache_manager

            astronomy_manager = get_cache_manager()

            today = dt_util.now().date()
            date_str = today.strftime("%Y-%m-%d")

            astro_data = astronomy_manager.get_day_data(date_str)
            if astro_data:
                return {
                    "sunrise": astro_data.get("sunrise_local"),
                    "sunset": astro_data.get("sunset_local"),
                    "solar_noon": astro_data.get("solar_noon_local"),
                    "daylight_hours": astro_data.get("daylight_hours", 0.0),
                }
            return None
        except Exception as err:
            _LOGGER.error(f"Error loading astronomy data: {err}")
            return None

    async def _get_today_weather_data(self) -> dict[str, Any] | None:
        """Get today's weather data from weather_forecast_corrected.json @zara"""
        try:

            if not self.coordinator.weather_pipeline_manager or not self.coordinator.weather_pipeline_manager.weather_corrector:
                _LOGGER.warning("Weather pipeline manager or corrector not available")
                return None

            weather_corrector = self.coordinator.weather_pipeline_manager.weather_corrector
            weather_corrected = await weather_corrector._read_json_file(
                weather_corrector.corrected_file, None
            )

            if not weather_corrected:
                _LOGGER.warning("weather_forecast_corrected.json is empty or missing")
                return None

            today = dt_util.now().date()
            date_str = today.strftime("%Y-%m-%d")
            current_hour = dt_util.now().hour

            forecast = weather_corrected.get("forecast", {})
            today_forecast = forecast.get(date_str, {})

            for hour_offset in range(0, 6):
                check_hour = str((current_hour + hour_offset) % 24)
                hour_data = today_forecast.get(check_hour)
                if hour_data:
                    return {
                        "temperature": hour_data.get("temperature"),
                        "clouds": hour_data.get("clouds"),
                        "wind": hour_data.get("wind"),
                        "humidity": hour_data.get("humidity"),
                    }

            return None
        except Exception as err:
            _LOGGER.warning(f"Error loading weather data: {err}")
            return None

    async def _generate_briefing_message(
        self,
        forecast_data: dict[str, Any],
        yesterday_data: dict[str, Any] | None,
        astro_data: dict[str, Any] | None,
        weather_data: dict[str, Any] | None,
        language: str,
    ) -> dict[str, str]:
        """Generate formatted briefing message.

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
        except (ValueError, TypeError, KeyError) as err:
            _LOGGER.error(f"Invalid date format in forecast_data: {err}")

            date_obj = dt_util.now()

        if language == "de":
            weekday = date_obj.strftime("%A")
            weekday_de = {
                "Monday": "Montag", "Tuesday": "Dienstag", "Wednesday": "Mittwoch",
                "Thursday": "Donnerstag", "Friday": "Freitag", "Saturday": "Samstag",
                "Sunday": "Sonntag"
            }.get(weekday, weekday)
            title = f"☀️ Solar Forecast - {weekday_de}, {date_obj.strftime('%d. %b')}"
        else:
            title = f"☀️ Solar Forecast - {date_obj.strftime('%A, %b %d')}"

        message_parts = []

        prediction_kwh = forecast_data["prediction_kwh"]
        clouds = weather_data.get("clouds") if weather_data else None
        weather_emoji, weather_text = self._interpret_weather(prediction_kwh, clouds, language)
        message_parts.append(f"{weather_emoji} {weather_text}")
        message_parts.append("")

        if language == "de":
            message_parts.append(f"📊 Forecast: {prediction_kwh:.2f} kWh")
        else:
            message_parts.append(f"📊 Forecast: {prediction_kwh:.2f} kWh")

        if yesterday_data:
            yesterday_actual = yesterday_data["actual_kwh"]
            if yesterday_actual > 0 and prediction_kwh > 0:
                ratio = prediction_kwh / yesterday_actual
                if ratio > 1.5:
                    comparison_emoji = "📈"
                    if language == "de":
                        comparison_text = f"→ {ratio:.1f}x besser als gestern ({yesterday_actual:.2f} kWh)"
                    else:
                        comparison_text = f"→ {ratio:.1f}x better than yesterday ({yesterday_actual:.2f} kWh)"
                elif ratio < 0.67:
                    comparison_emoji = "📉"
                    if language == "de":
                        comparison_text = f"→ {(1/ratio):.1f}x schlechter als gestern ({yesterday_actual:.2f} kWh)"
                    else:
                        comparison_text = f"→ {(1/ratio):.1f}x worse than yesterday ({yesterday_actual:.2f} kWh)"
                else:
                    comparison_emoji = "➡️"
                    if language == "de":
                        comparison_text = f"→ Ähnlich wie gestern ({yesterday_actual:.2f} kWh)"
                    else:
                        comparison_text = f"→ Similar to yesterday ({yesterday_actual:.2f} kWh)"
            elif prediction_kwh == 0:

                comparison_emoji = "⚠️"
                if language == "de":
                    comparison_text = f"→ Keine Produktion erwartet (gestern: {yesterday_actual:.2f} kWh)"
                else:
                    comparison_text = f"→ No production expected (yesterday: {yesterday_actual:.2f} kWh)"

                message_parts.append(f"   {comparison_emoji} {comparison_text}")

        message_parts.append("")

        # V12.9.1: Removed temperature from weather line - was often inaccurate or confusing
        weather_desc = self._get_weather_description(clouds, language)
        if language == "de":
            message_parts.append(f"🌤️ Wetter: {weather_desc}")
        else:
            message_parts.append(f"🌤️ Weather: {weather_desc}")

        if astro_data and astro_data.get("solar_noon"):
            solar_noon = astro_data["solar_noon"]

            try:
                solar_noon_time = solar_noon.split("T")[1][:5]
                if language == "de":
                    message_parts.append(f"   Beste Zeit: {solar_noon_time} Uhr (Solar Noon)")
                else:
                    message_parts.append(f"   Best Time: {solar_noon_time} (Solar Noon)")
            except Exception:
                pass

        message_parts.append("")

        if astro_data:
            daylight_hours = astro_data.get("daylight_hours", 0.0)
            hours = int(daylight_hours)
            minutes = int((daylight_hours - hours) * 60)

            if language == "de":
                message_parts.append(f"⏰ Tageslicht: {hours}h {minutes}min")
            else:
                message_parts.append(f"⏰ Daylight: {hours}h {minutes}min")

            sunrise = astro_data.get("sunrise")
            sunset = astro_data.get("sunset")
            if sunrise and sunset:
                try:
                    sunrise_time = sunrise.split("T")[1][:5]
                    sunset_time = sunset.split("T")[1][:5]
                    message_parts.append(f"   🌅 Aufgang: {sunrise_time} Uhr" if language == "de" else f"   🌅 Sunrise: {sunrise_time}")
                    message_parts.append(f"   🌇 Untergang: {sunset_time} Uhr" if language == "de" else f"   🌇 Sunset: {sunset_time}")
                except Exception:
                    pass

        message_parts.append("")

        shadow_summary = await self._get_yesterday_shadow_summary(language)
        if shadow_summary:
            message_parts.append(shadow_summary)
            message_parts.append("")

        # V12.9.1: Pass cloud cover to closing message for consistency
        closing = self._get_closing_message(prediction_kwh, clouds, language)
        message_parts.append(closing)

        message = "\n".join(message_parts)

        return {"title": title, "message": message}

    def _interpret_weather(self, prediction_kwh: float, clouds: float | None, language: str) -> tuple[str, str]:
        """Interpret weather from cloud cover and prediction value @zara

        Args:
            prediction_kwh: Today's forecast in kWh
            clouds: Cloud cover percentage (0-100) from weather data
            language: Language code ("de" or "en")
        """
        # V12.9.1: Use cautious/tentative wording - forecasts are never 100% certain
        # Use cloud cover if available, otherwise fall back to prediction-based estimation
        if clouds is not None:
            if clouds < 20:
                emoji = "🌞"
                text = "Voraussichtlich sonnig - gute Bedingungen möglich" if language == "de" else "Likely sunny - good conditions possible"
            elif clouds < 40:
                emoji = "☀️"
                text = "Überwiegend sonnig erwartet" if language == "de" else "Mostly sunny expected"
            elif clouds < 60:
                emoji = "⛅"
                text = "Wechselhaft - Wolken möglich" if language == "de" else "Variable - clouds possible"
            elif clouds < 80:
                emoji = "🌥️"
                text = "Eher bewölkt erwartet" if language == "de" else "Rather cloudy expected"
            else:
                emoji = "☁️"
                text = "Überwiegend bewölkt erwartet" if language == "de" else "Mostly cloudy expected"
        else:
            # Fallback: use prediction_kwh if no cloud data available
            if prediction_kwh > 15:
                emoji, text = "🌞", "Gute Bedingungen möglich" if language == "de" else "Good conditions possible"
            elif prediction_kwh > 10:
                emoji, text = "☀️", "Ordentliche Produktion möglich" if language == "de" else "Decent production possible"
            elif prediction_kwh > 5:
                emoji, text = "⛅", "Moderate Produktion erwartet" if language == "de" else "Moderate production expected"
            elif prediction_kwh > 2:
                emoji, text = "🌥️", "Eingeschränkte Produktion wahrscheinlich" if language == "de" else "Limited production likely"
            elif prediction_kwh > 0.5:
                emoji, text = "☁️", "Geringe Produktion erwartet" if language == "de" else "Low production expected"
            else:
                emoji, text = "🌧️", "Kaum Produktion erwartet" if language == "de" else "Minimal production expected"

        return (emoji, text)

    def _get_weather_description(self, clouds: float | None, language: str) -> str:
        """Get detailed weather description from cloud cover @zara

        V12.9.1: Use tentative wording - these are forecasts, not guarantees

        Args:
            clouds: Cloud cover percentage (0-100) from weather data
            language: Language code ("de" or "en")
        """
        if clouds is None:
            return "Keine Wetterdaten" if language == "de" else "No weather data"

        # Use tentative wording with "voraussichtlich" / "expected"
        if clouds < 10:
            return "Voraussichtlich klar" if language == "de" else "Expected clear"
        elif clouds < 25:
            return "Voraussichtlich sonnig" if language == "de" else "Expected sunny"
        elif clouds < 50:
            return "Wolken möglich" if language == "de" else "Clouds possible"
        elif clouds < 75:
            return "Eher bewölkt" if language == "de" else "Rather cloudy"
        elif clouds < 90:
            return "Stark bewölkt erwartet" if language == "de" else "Heavy clouds expected"
        else:
            return "Überwiegend bedeckt erwartet" if language == "de" else "Overcast expected"

    def _get_closing_message(self, prediction_kwh: float, clouds: float | None, language: str) -> str:
        """Get closing message based on prediction value AND cloud cover for consistency. @zara

        V12.9.1: Now considers cloud cover and uses tentative/cautious wording.
        Forecasts are estimates, not guarantees.
        """
        # Priority 1: Use cloud cover if available (more accurate for closing message)
        if clouds is not None:
            if clouds < 20:
                return "Gute Chancen auf Sonne! ☀️" if language == "de" else "Good chance of sun! ☀️"
            elif clouds < 40:
                return "Sonnenschein wahrscheinlich ⚡" if language == "de" else "Sunshine likely ⚡"
            elif clouds < 60:
                return "Sonne möglich 🌤️" if language == "de" else "Sun possible 🌤️"
            elif clouds < 80:
                return "Wenig Sonne erwartet 🌥️" if language == "de" else "Little sun expected 🌥️"
            else:
                return "Bewölkung wahrscheinlich ☁️" if language == "de" else "Clouds likely ☁️"

        # Fallback: Use prediction-based estimation if no cloud data
        if prediction_kwh > 10:
            return "Gute Chancen auf Sonne! ☀️" if language == "de" else "Good chance of sun! ☀️"
        elif prediction_kwh > 5:
            return "Ordentliche Produktion möglich ⚡" if language == "de" else "Decent production possible ⚡"
        elif prediction_kwh > 2:
            return "Etwas Sonne möglich 🌤️" if language == "de" else "Some sun possible 🌤️"
        else:
            return "Wenig Sonne erwartet ☁️" if language == "de" else "Little sun expected ☁️"

    async def _get_yesterday_shadow_summary(self, language: str) -> str | None:
        """Get yesterday's shadow detection summary for briefing @zara"""
        try:

            from ..data.data_shadow_detection import get_performance_analyzer

            yesterday = (dt_util.now().date() - dt_util.dt.timedelta(days=1)).isoformat()

            hourly_predictions = self.coordinator.data.get("hourly_predictions_handler")
            if not hourly_predictions:
                return None

            data = hourly_predictions._read_json()
            yesterday_predictions = [
                p for p in data.get("predictions", [])
                if p.get("target_date") == yesterday
                and p.get("shadow_detection") is not None
            ]

            if not yesterday_predictions:
                return None

            analyzer = get_performance_analyzer()
            shadow_analysis = await analyzer.analyze_daily_shadow(yesterday, yesterday_predictions)

            shadow_hours = shadow_analysis.get("shadow_hours_count", 0)
            if shadow_hours == 0:

                return None

            daily_loss_percent = shadow_analysis.get("daily_loss_percent", 0)
            cumulative_loss = shadow_analysis.get("cumulative_loss_kwh", 0)
            dominant_cause = shadow_analysis.get("dominant_cause", "unknown")

            if language == "de":
                header = "🌑 Schatten-Analyse (Gestern):"
                hours_text = f"   ⚠️ {shadow_hours}h Verschattung erkannt"
                loss_text = f"   📉 Verlust: {cumulative_loss:.2f} kWh (-{daily_loss_percent:.0f}%)"

                cause_map = {
                    "weather_clouds": "Wolken",
                    "building_tree_obstruction": "Gebäude/Baum",
                    "normal_variation": "Normale Variation",
                    "unknown": "Unbekannt"
                }
                cause_text = cause_map.get(dominant_cause, dominant_cause)
                cause_line = f"   🏢 Ursache: {cause_text}"

            else:
                header = "🌑 Shadow Analysis (Yesterday):"
                hours_text = f"   ⚠️ {shadow_hours}h shadowing detected"
                loss_text = f"   📉 Loss: {cumulative_loss:.2f} kWh (-{daily_loss_percent:.0f}%)"

                cause_map = {
                    "weather_clouds": "Clouds",
                    "building_tree_obstruction": "Building/Tree",
                    "normal_variation": "Normal variation",
                    "unknown": "Unknown"
                }
                cause_text = cause_map.get(dominant_cause, dominant_cause)
                cause_line = f"   🏢 Cause: {cause_text}"

            summary_lines = [header, hours_text, loss_text, cause_line]

            return "\n".join(summary_lines)

        except Exception as e:
            _LOGGER.warning(f"Failed to get shadow summary: {e}", exc_info=False)
            return None
