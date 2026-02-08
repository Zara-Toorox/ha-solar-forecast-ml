# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Startup Initializer for Solar Forecast ML V16.2.0.
Synchronous initializer - guarantees database and directories exist before async startup.
Initializes database schema and default data.

@zara
"""

import logging
from datetime import datetime, timedelta, time, date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class StartupInitializer:
    """Synchronous initializer - guarantees database exists before async startup. @zara

    Creates:
    1. Directory structure (backups/auto/)
    2. Database file with schema
    3. Default astronomy cache data
    4. Default weather forecast data

    This runs BEFORE any async operations to prevent race conditions.
    """

    def __init__(self, data_dir: Path, config: dict[str, Any]):
        """Initialize startup initializer. @zara

        Args:
            data_dir: Base data directory path
            config: Configuration dictionary with location settings
        """
        self.data_dir = Path(data_dir)
        self.config = config

        self.latitude = config.get("latitude", 52.52)
        self.longitude = config.get("longitude", 13.40)
        self.solar_capacity_kwp = config.get("solar_capacity", 2.0)
        self.timezone_str = config.get("timezone", "Europe/Berlin")

        try:
            self.timezone = ZoneInfo(self.timezone_str)
        except Exception:
            _LOGGER.warning(
                "Invalid timezone '%s', using Europe/Berlin", self.timezone_str
            )
            self.timezone = ZoneInfo("Europe/Berlin")
            self.timezone_str = "Europe/Berlin"

        # Database path @zara
        self.db_path = self.data_dir / "solar_forecast.db"

    def initialize_all(self) -> bool:
        """Initialize critical components synchronously. @zara

        MUST run BEFORE any async operations to prevent race conditions.

        Returns:
            True if all critical components are ready
        """
        _LOGGER.info("=" * 60)
        _LOGGER.info("STARTUP INITIALIZER - Creating critical pre-async components")
        _LOGGER.info("=" * 60)

        success = True

        # Step 1: Create directory structure @zara
        if not self._ensure_directories():
            _LOGGER.error("Failed to create directory structure")
            success = False
        else:
            _LOGGER.info("Directory structure ready")

        # Step 2: Create database with schema @zara
        if not self._ensure_database():
            _LOGGER.error("Failed to create database")
            success = False
        else:
            _LOGGER.info("Database ready")

        _LOGGER.info("=" * 60)
        if success:
            _LOGGER.info("STARTUP INITIALIZER complete - database ready")
        else:
            _LOGGER.error("STARTUP INITIALIZER failed")
        _LOGGER.info("=" * 60)

        return success

    def _ensure_directories(self) -> bool:
        """Create all required directories. @zara"""
        directories = [
            self.data_dir,
            self.data_dir / "backups",
            self.data_dir / "backups" / "auto",
            self.data_dir / "backups" / "manual",
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            _LOGGER.debug("Created/verified %d directories", len(directories))
            return True
        except Exception as e:
            _LOGGER.error("Failed to create directories: %s", e)
            return False

    def _ensure_database(self) -> bool:
        """Ensure database file exists. @zara"""
        try:
            import sqlite3

            conn = sqlite3.connect(str(self.db_path))

            schema_path = Path(__file__).parent / "schema.sql"
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                conn.executescript(schema_sql)
                _LOGGER.debug("Database schema initialized")

            self._init_default_data(conn)

            conn.close()
            return True

        except Exception as e:
            _LOGGER.error("Failed to create database: %s", e)
            return False

    def _init_default_data(self, conn) -> None:
        """Initialize default data in database. @zara"""
        cursor = conn.cursor()
        today = date.today()

        # Initialize astronomy system info if not exists @zara
        cursor.execute(
            "SELECT id FROM astronomy_system_info WHERE id = 1"
        )
        if cursor.fetchone() is None:
            cursor.execute(
                """INSERT INTO astronomy_system_info
                   (id, latitude, longitude, timezone, installed_capacity_kwp)
                   VALUES (1, ?, ?, ?, ?)""",
                (self.latitude, self.longitude, self.timezone_str, self.solar_capacity_kwp)
            )
            _LOGGER.debug("Initialized astronomy system info")

        cursor.execute("SELECT COUNT(*) FROM astronomy_hourly_peaks")
        if cursor.fetchone()[0] == 0:
            hourly_peaks_data = [(hour, 0) for hour in range(24)]
            cursor.executemany(
                "INSERT INTO astronomy_hourly_peaks (hour, kwh) VALUES (?, ?)",
                hourly_peaks_data
            )
            _LOGGER.debug("Initialized hourly peaks")

        # Initialize astronomy cache for next 7 days @zara
        cursor.execute(
            "SELECT COUNT(*) FROM astronomy_cache WHERE cache_date >= ?",
            (today.isoformat(),)
        )
        if cursor.fetchone()[0] == 0:
            self._init_astronomy_cache(cursor, today)
            _LOGGER.debug("Initialized astronomy cache")

        # Initialize weather forecast for next 3 days @zara
        cursor.execute(
            "SELECT COUNT(*) FROM weather_forecast WHERE forecast_date >= ?",
            (today.isoformat(),)
        )
        if cursor.fetchone()[0] == 0:
            self._init_weather_forecast(cursor, today)
            _LOGGER.debug("Initialized weather forecast")

        conn.commit()

    def _init_astronomy_cache(self, cursor, today: date) -> None:
        """Initialize astronomy cache with baseline data. @zara"""
        astronomy_data = []
        for i in range(7):
            target_date = today + timedelta(days=i)
            date_str = target_date.isoformat()

            sunrise = datetime.combine(target_date, time(6, 0), tzinfo=self.timezone)
            sunset = datetime.combine(target_date, time(18, 0), tzinfo=self.timezone)
            solar_noon = datetime.combine(target_date, time(12, 0), tzinfo=self.timezone)

            for hour in range(24):
                if 6 <= hour <= 18:
                    elevation = max(0, 60 * (1 - abs(hour - 12) / 6))
                    azimuth = 90 + (hour - 6) * 15
                    clear_sky_rad = max(0, 1000 * (elevation / 60) ** 1.5)
                    theoretical_max = (clear_sky_rad / 1000) * self.solar_capacity_kwp
                else:
                    elevation = azimuth = clear_sky_rad = theoretical_max = 0

                astronomy_data.append((
                    date_str, hour, round(elevation, 1), round(azimuth, 1),
                    round(clear_sky_rad, 0), round(theoretical_max, 3),
                    sunrise.isoformat(), sunset.isoformat(),
                    solar_noon.isoformat(), 12.0
                ))

        cursor.executemany(
            """INSERT OR IGNORE INTO astronomy_cache
               (cache_date, hour, sun_elevation_deg, sun_azimuth_deg,
                clear_sky_radiation_wm2, theoretical_max_kwh,
                sunrise, sunset, solar_noon, daylight_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            astronomy_data
        )

    def _init_weather_forecast(self, cursor, today: date) -> None:
        """Initialize weather forecast with baseline data. @zara"""
        weather_data = []
        for i in range(3):
            target_date = today + timedelta(days=i)
            date_str = target_date.isoformat()

            for hour in range(24):
                if 6 <= hour <= 18:
                    elevation = max(0, 60 * (1 - abs(hour - 12) / 6))
                    clear_sky_ghi = max(0, 1000 * (elevation / 60) ** 1.5)
                    direct_rad = int(clear_sky_ghi * 0.35)
                    diffuse_rad = int(clear_sky_ghi * 0.24)
                    solar_rad = int(clear_sky_ghi * 0.5)
                else:
                    direct_rad = diffuse_rad = solar_rad = 0

                weather_data.append((
                    date_str, hour, 10.0, 70, 50, 0.0, 3.0, 1013.0,
                    direct_rad, diffuse_rad, solar_rad
                ))

        cursor.executemany(
            """INSERT OR IGNORE INTO weather_forecast
               (forecast_date, hour, temperature, humidity, clouds,
                rain, wind, pressure, direct_radiation, diffuse_radiation,
                solar_radiation_wm2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            weather_data
        )

    def get_db_path(self) -> str:
        """Get the database file path. @zara"""
        return str(self.db_path)

    def get_config(self) -> dict[str, Any]:
        """Get the configuration. @zara"""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "solar_capacity_kwp": self.solar_capacity_kwp,
            "timezone": self.timezone_str,
            "data_dir": str(self.data_dir),
            "db_path": str(self.db_path),
        }
