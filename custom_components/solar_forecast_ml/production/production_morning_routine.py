# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

class MorningRoutineHandler:
    """Bulletproof morning routine with backup/restore and retry"""

    def __init__(self, data_manager, coordinator):
        self.data_manager = data_manager
        self.coordinator = coordinator
        self.backup_file = None

    async def execute_morning_routine_with_retry(
        self, date: str, hourly_forecast: List, weather_hourly: List, astronomy_data: Dict, sensor_config: Dict
    ) -> bool:
        """
        Execute morning routine with bulletproof retry logic

        Returns:
            True if successful, False if all retries failed
        """
        max_retries = 3
        backup_created = False

        try:

            backup_created = await self._create_backup()

            if not backup_created:
                _LOGGER.error(
                    "✗ CRITICAL: Backup creation failed - ABORTING morning routine\n"
                    "   → Safety mechanism triggered to prevent data loss\n"
                    "   → Manual intervention required"
                )
                return False

            for attempt in range(1, max_retries + 1):
                _LOGGER.info(f"=== Morning routine attempt {attempt}/{max_retries} for {date} ===")

                try:

                    if not await self._validate_prerequisites(date, hourly_forecast, weather_hourly, astronomy_data):
                        raise ValueError("Prerequisites validation failed")

                    # Get inverter max power for forecast capping
                    inverter_max_power = getattr(self.coordinator, 'inverter_max_power', 0.0)

                    success = await self.data_manager.hourly_predictions.create_daily_predictions(
                        date=date,
                        hourly_forecast=hourly_forecast,
                        weather_forecast=weather_hourly,
                        astronomy_data=astronomy_data,
                        sensor_config=sensor_config,
                        inverter_max_power=inverter_max_power,
                    )

                    if not success:
                        raise RuntimeError("Prediction creation returned False")

                    if not await self._verify_integrity(date):
                        raise RuntimeError("Integrity verification failed")

                    await self._delete_backup()

                    _LOGGER.info(
                        f"✓ Morning routine SUCCESSFUL (attempt {attempt}/{max_retries})\n"
                        f"   → Predictions created and validated\n"
                        f"   → Backup removed\n"
                        f"   → System ready for day {date}"
                    )

                    return True

                except Exception as e:
                    _LOGGER.error(
                        f"✗ Morning routine attempt {attempt}/{max_retries} FAILED: {e}", exc_info=True
                    )

                    if attempt < max_retries:

                        restore_success = await self._restore_backup()

                        if not restore_success:
                            _LOGGER.error(
                                "✗ CRITICAL: Backup restore FAILED!\n"
                                "   → Cannot retry safely\n"
                                "   → Aborting morning routine"
                            )
                            return False

                        wait_time = 60 * attempt
                        _LOGGER.info(f"→ Waiting {wait_time}s before retry {attempt + 1}...")
                        await asyncio.sleep(wait_time)

                    else:

                        _LOGGER.error(
                            f"✗ CRITICAL: Morning routine FAILED after {max_retries} attempts!\n"
                            f"   → Restoring backup\n"
                            f"   → Manual intervention required"
                        )

                        await self._restore_backup()
                        await self._send_critical_alert("Morning routine failure", str(e))

                        return False

        except Exception as e:
            _LOGGER.error(f"✗ CRITICAL: Morning routine handler crashed: {e}", exc_info=True)

            if backup_created:
                await self._restore_backup()

            return False

    async def _create_backup(self) -> bool:
        """Create atomic backup of hourly_predictions.json @zara"""
        try:
            source_file = self.data_manager.data_dir / "stats" / "hourly_predictions.json"

            if not source_file.exists():
                _LOGGER.warning("No existing hourly_predictions.json to backup - creating new")
                return True

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_file = source_file.with_suffix(f".json.backup_{timestamp}")

            def _copy_sync():
                shutil.copy2(source_file, self.backup_file)

            await asyncio.get_event_loop().run_in_executor(None, _copy_sync)

            _LOGGER.info(f"✓ Backup created: {self.backup_file.name}")
            return True

        except Exception as e:
            _LOGGER.error(f"✗ Backup creation failed: {e}", exc_info=True)
            return False

    async def _restore_backup(self) -> bool:
        """Restore from backup @zara"""
        try:
            if not self.backup_file or not self.backup_file.exists():
                _LOGGER.error("✗ No backup file to restore from!")
                return False

            source_file = self.data_manager.data_dir / "stats" / "hourly_predictions.json"

            def _restore_sync():
                shutil.copy2(self.backup_file, source_file)

            await asyncio.get_event_loop().run_in_executor(None, _restore_sync)

            _LOGGER.info(f"✓ Restored from backup: {self.backup_file.name}")
            return True

        except Exception as e:
            _LOGGER.error(f"✗ Backup restore failed: {e}", exc_info=True)
            return False

    async def _delete_backup(self) -> bool:
        """Delete backup after successful operation @zara"""
        try:
            if self.backup_file and self.backup_file.exists():
                self.backup_file.unlink()
                _LOGGER.debug(f"✓ Backup deleted: {self.backup_file.name}")

            return True

        except Exception as e:
            _LOGGER.warning(f"Could not delete backup: {e}")
            return False

    async def _validate_prerequisites(
        self, date: str, hourly_forecast: List, weather_hourly: List, astronomy_data: Dict
    ) -> bool:
        """Validate all prerequisites before creating predictions"""
        try:

            datetime.fromisoformat(date)

            if not hourly_forecast or not isinstance(hourly_forecast, list):
                _LOGGER.error(f"✗ Invalid hourly_forecast: {type(hourly_forecast)}")
                return False

            if len(hourly_forecast) < 8:
                _LOGGER.warning(
                    f"⚠️  Only {len(hourly_forecast)} hours in forecast (expected 8+ for winter, 12+ for summer)"
                )

            if not weather_hourly or not isinstance(weather_hourly, list):
                _LOGGER.error(f"✗ Invalid weather_hourly: {type(weather_hourly)}")
                return False

            if not astronomy_data or not isinstance(astronomy_data, dict):
                _LOGGER.error(f"✗ Invalid astronomy_data: {type(astronomy_data)}")
                return False

            _LOGGER.debug("✓ All prerequisites validated")
            return True

        except Exception as e:
            _LOGGER.error(f"✗ Prerequisite validation failed: {e}", exc_info=True)
            return False

    async def _verify_integrity(self, date: str) -> bool:
        """Verify data integrity after creation @zara"""
        try:

            data = await self.data_manager.hourly_predictions._read_json_async()

            today_predictions = [p for p in data.get("predictions", []) if p.get("target_date") == date]

            if not today_predictions:
                _LOGGER.error(f"✗ No predictions found for {date} after creation!")
                return False

            ids = [p["id"] for p in data["predictions"]]
            duplicates = [id for id in ids if ids.count(id) > 1]

            if duplicates:
                _LOGGER.error(f"✗ Duplicate IDs found: {set(duplicates)}")
                return False

            _LOGGER.debug(f"✓ Integrity verified: {len(today_predictions)} predictions for {date}, no duplicates")
            return True

        except Exception as e:
            _LOGGER.error(f"✗ Integrity verification failed: {e}", exc_info=True)
            return False

    async def _send_critical_alert(self, title: str, details: str) -> None:
        """Send critical alert about failure @zara"""
        try:

            error_log = self.data_manager.data_dir / "stats" / "morning_routine_errors.log"

            def _write_error_log():
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log, "a") as f:
                    import traceback
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Title: {title}\n")
                    f.write(f"Details: {details}\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}\n")

            await self.coordinator.hass.async_add_executor_job(_write_error_log)
            _LOGGER.info(f"✓ Critical alert logged to morning_routine_errors.log")

            if hasattr(self.coordinator, "hass"):
                await self.coordinator.hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": f"⚠️ Solar Forecast ML: {title}",
                        "message": f"{details}\n\nCheck logs for details.",
                        "notification_id": "solar_forecast_ml_critical",
                    },
                )

        except Exception as e:
            _LOGGER.error(f"Could not send critical alert: {e}")
