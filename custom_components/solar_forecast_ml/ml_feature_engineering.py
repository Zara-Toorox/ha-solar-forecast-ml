"""
Feature Engineering for the Solar Forecast ML Predictor.
Extracts and creates features from weather and sensor data.

This program is free software: you can redistribute it and/or modify
it under a terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
import math
import logging # Import logging
from datetime import datetime
from typing import Dict, Any, List, Optional # Added Optional

# Use SafeDateTimeUtil for consistent timezone handling
from .helpers import SafeDateTimeUtil as dt_util

_LOGGER = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles the creation and extraction of features used by the ML model.
    Includes base weather features, lag features, polynomial features,
    and interaction features.
    """

    def __init__(self):
        """Initializes the FeatureEngineer and defines the feature sets."""
        # Define the core input features expected
        self.base_features = [
            "temperature",          # Current temperature
            "humidity",             # Current humidity
            "cloudiness",           # Current cloud cover percentage
            "wind_speed",           # Current wind speed
            "hour_of_day",          # Hour of the day (0-23)
            "seasonal_factor",      # Cosine-based factor representing time of year
            "weather_trend",        # Combined score based on cloudiness and wind
            "production_yesterday", # Total production from the previous day (lag feature)
            "production_last_hour", # Production from the previous hour (lag feature)
        ]

        # Define derived polynomial features (e.g., squared terms)
        self.polynomial_features = [
            "temperature_sq",       # temperature^2
            "cloudiness_sq",        # cloudiness^2
            "hour_of_day_sq",       # hour_of_day^2
            "seasonal_factor_sq",   # seasonal_factor^2
        ]

        # Define derived interaction features (products of two base features)
        self.interaction_features = [
            "cloudiness_x_hour",    # cloudiness * hour_of_day
            "temperature_x_seasonal", # temperature * seasonal_factor
            "humidity_x_cloudiness",# humidity * cloudiness
            "wind_x_hour",          # wind_speed * hour_of_day
            "weather_trend_x_seasonal", # weather_trend * seasonal_factor
        ]

        # Combine all feature names into a single list, defining the order
        # This list is crucial for aligning features with model weights.
        self.feature_names = (
            self.base_features +
            self.polynomial_features +
            self.interaction_features
        )
        _LOGGER.debug(f"FeatureEngineer initialized with {len(self.feature_names)} features: {self.feature_names}")


    async def extract_features(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        prediction_hour: Optional[int] = None # Hour to extract features for
    ) -> Dict[str, float]:
        """
        Asynchronously extracts and calculates all defined features for a given time context.

        Args:
            weather_data: Dictionary containing current weather information.
            sensor_data: Dictionary containing sensor readings (including lag features).
            prediction_hour: The hour (0-23) for which to calculate features.
                               Defaults to the current UTC hour if None.

        Returns:
            A dictionary mapping feature names to their calculated float values.
            Returns default features if extraction fails.
        """
        try:
            # Determine timestamp and hour for feature calculation
            now_utc = dt_util.utcnow()
            target_hour = prediction_hour if prediction_hour is not None else now_utc.hour

            # --- Extract Base Weather Features ---
            # Use _safe_extract to handle missing keys and type conversions gracefully
            temp = self._safe_extract(weather_data, 'temperature', default=15.0)
            humidity = self._safe_extract(weather_data, 'humidity', default=60.0)
            # Allow 'cloud_cover' or 'cloudiness'
            cloudiness = self._safe_extract(weather_data, 'cloudiness',
                                            weather_data.get('cloud_cover', 50.0))
            wind_speed = self._safe_extract(weather_data, 'wind_speed', default=5.0)
            # pressure = self._safe_extract(weather_data, 'pressure', default=1013.0) # Not currently used as base feature

            # --- Extract Lag Features (Expected to be in sensor_data) ---
            prod_yesterday = self._safe_extract(sensor_data, 'production_yesterday', default=0.0)
            prod_last_hour = self._safe_extract(sensor_data, 'production_last_hour', default=0.0)

            # --- Calculate Time-Based Features ---
            # Use current date for seasonal factor calculation
            day_of_year = now_utc.timetuple().tm_yday # Day number (1-366)
            # Cosine function peaks in summer (around day 172) and troughs in winter
            seasonal_factor = 0.5 + 0.5 * math.cos((day_of_year - 172) * 2 * math.pi / 365.25)

            # --- Calculate Derived Features ---
            weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)

            # --- Assemble Base Feature Dictionary ---
            # Use float for hour_of_day for consistency in model input
            hour_float = float(target_hour)
            base_features_dict = {
                "temperature": temp,
                "humidity": humidity,
                "cloudiness": cloudiness,
                "wind_speed": wind_speed,
                "hour_of_day": hour_float,
                "seasonal_factor": seasonal_factor,
                "weather_trend": weather_trend,
                "production_yesterday": prod_yesterday,
                "production_last_hour": prod_last_hour,
            }

            # --- Calculate Polynomial and Interaction Features ---
            # Use values from base_features_dict to ensure consistency
            poly_interaction_features = {
                # Polynomial
                "temperature_sq": base_features_dict["temperature"] ** 2,
                "cloudiness_sq": base_features_dict["cloudiness"] ** 2,
                "hour_of_day_sq": base_features_dict["hour_of_day"] ** 2,
                "seasonal_factor_sq": base_features_dict["seasonal_factor"] ** 2,
                # Interaction
                "cloudiness_x_hour": base_features_dict["cloudiness"] * base_features_dict["hour_of_day"],
                "temperature_x_seasonal": base_features_dict["temperature"] * base_features_dict["seasonal_factor"],
                "humidity_x_cloudiness": base_features_dict["humidity"] * base_features_dict["cloudiness"],
                "wind_x_hour": base_features_dict["wind_speed"] * base_features_dict["hour_of_day"],
                "weather_trend_x_seasonal": base_features_dict["weather_trend"] * base_features_dict["seasonal_factor"]
            }

            # --- Combine all features ---
            all_features = {**base_features_dict, **poly_interaction_features}

            # --- Final Check: Ensure all expected features are present ---
            # (Optional but good practice)
            # missing_keys = [name for name in self.feature_names if name not in all_features]
            # if missing_keys:
            #     _LOGGER.warning(f"Feature extraction resulted in missing keys: {missing_keys}. Check logic.")
            #     # Handle missing keys, e.g., fill with 0 or raise error? Fill with 0 for now.
            #     for key in missing_keys: all_features[key] = 0.0

            _LOGGER.debug(f"Features extracted for hour {target_hour}: { {k: round(v, 2) for k, v in all_features.items()} }") # Log rounded values
            return all_features

        except Exception as e:
            # Catch any unexpected error during feature extraction
            _LOGGER.error(f"Feature extraction failed: {e}", exc_info=True)
            # Return a default feature set to allow prediction fallback
            return self.get_default_features(prediction_hour or dt_util.utcnow().hour)


    def extract_features_sync(
        self,
        weather_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        record: Dict[str, Any] # Expecting a record containing 'timestamp'
    ) -> Dict[str, float]:
        """
        Synchronously extracts and calculates all defined features, typically used during training.
        Derives time context from the provided record's timestamp.

        Args:
            weather_data: Dictionary containing weather information relevant to the record's time.
            sensor_data: Dictionary containing sensor readings (including lag features) relevant to the record's time.
            record: The historical record dictionary, must contain a parseable 'timestamp'.

        Returns:
            A dictionary mapping feature names to their calculated float values.
            Returns default features if extraction fails.
        """
        try:
            # --- Determine Time Context ---
            timestamp_str = record.get('timestamp')
            if not timestamp_str:
                 raise ValueError("Record is missing 'timestamp' key.")

            timestamp = dt_util.parse_datetime(timestamp_str)
            if not timestamp:
                 raise ValueError(f"Could not parse timestamp: '{timestamp_str}'")

            target_hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday

            # --- Extract/Calculate Features (Similar logic as async version) ---
            temp = self._safe_extract(weather_data, 'temperature', default=15.0)
            humidity = self._safe_extract(weather_data, 'humidity', default=60.0)
            cloudiness = self._safe_extract(weather_data, 'cloudiness',
                                            weather_data.get('cloud_cover', 50.0))
            wind_speed = self._safe_extract(weather_data, 'wind_speed', default=5.0)

            # Lag features expected in sensor_data (added by caller, e.g., train_model)
            prod_yesterday = self._safe_extract(sensor_data, 'production_yesterday', default=0.0)
            prod_last_hour = self._safe_extract(sensor_data, 'production_last_hour', default=0.0) # Should be in record's sensor_data

            seasonal_factor = 0.5 + 0.5 * math.cos((day_of_year - 172) * 2 * math.pi / 365.25)
            weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)

            hour_float = float(target_hour)
            base_features_dict = {
                "temperature": temp, "humidity": humidity, "cloudiness": cloudiness,
                "wind_speed": wind_speed, "hour_of_day": hour_float, "seasonal_factor": seasonal_factor,
                "weather_trend": weather_trend, "production_yesterday": prod_yesterday,
                "production_last_hour": prod_last_hour,
            }

            poly_interaction_features = {
                "temperature_sq": temp ** 2, "cloudiness_sq": cloudiness ** 2,
                "hour_of_day_sq": hour_float ** 2, "seasonal_factor_sq": seasonal_factor ** 2,
                "cloudiness_x_hour": cloudiness * hour_float,
                "temperature_x_seasonal": temp * seasonal_factor,
                "humidity_x_cloudiness": humidity * cloudiness,
                "wind_x_hour": wind_speed * hour_float,
                "weather_trend_x_seasonal": weather_trend * seasonal_factor
            }

            all_features = {**base_features_dict, **poly_interaction_features}

            # Optional: Add validation step to ensure all keys in self.feature_names exist
            # for key in self.feature_names:
            #     if key not in all_features: all_features[key] = 0.0 # Add missing keys with 0

            return all_features

        except Exception as e:
            # Catch errors during synchronous extraction
            _LOGGER.warning(f"Sync feature extraction failed for record {record.get('timestamp')}: {e}")
            # Return default features based on the record's hour (if possible)
            fallback_hour = dt_util.utcnow().hour # Default hour if record timestamp fails
            try: fallback_hour = dt_util.parse_datetime(record.get('timestamp')).hour
            except: pass
            return self.get_default_features(fallback_hour)


    def get_default_features(self, hour: int) -> Dict[str, float]:
        """
        Provides a default set of feature values, used as a fallback if
        real data extraction fails. Uses average/typical values.

        Args:
            hour: The hour (0-23) for which to generate default features.

        Returns:
            A dictionary mapping all feature names to default float values.
        """
        _LOGGER.debug(f"Generating default features for hour {hour}.")
        # Use typical mid-range values for weather
        temp = 15.0
        humidity = 60.0
        cloudiness = 50.0
        wind_speed = 5.0
        # Calculate seasonal factor for an average day (e.g., mid-year)
        avg_day_of_year = 182
        seasonal_factor = 0.5 + 0.5 * math.cos((avg_day_of_year - 172) * 2 * math.pi / 365.25)
        weather_trend = self._calculate_weather_trend(cloudiness, wind_speed)
        hour_float = float(hour)

        # Base defaults
        defaults = {
            "temperature": temp,
            "humidity": humidity,
            "cloudiness": cloudiness,
            "wind_speed": wind_speed,
            "hour_of_day": hour_float,
            "seasonal_factor": seasonal_factor,
            "weather_trend": weather_trend,
            "production_yesterday": 0.0, # Default lag features to 0
            "production_last_hour": 0.0,
        }

        # Calculate derived defaults based on base values
        defaults.update({
            "temperature_sq": defaults["temperature"] ** 2,
            "cloudiness_sq": defaults["cloudiness"] ** 2,
            "hour_of_day_sq": defaults["hour_of_day"] ** 2,
            "seasonal_factor_sq": defaults["seasonal_factor"] ** 2,
            "cloudiness_x_hour": defaults["cloudiness"] * defaults["hour_of_day"],
            "temperature_x_seasonal": defaults["temperature"] * defaults["seasonal_factor"],
            "humidity_x_cloudiness": defaults["humidity"] * defaults["cloudiness"],
            "wind_x_hour": defaults["wind_speed"] * defaults["hour_of_day"],
            "weather_trend_x_seasonal": defaults["weather_trend"] * defaults["seasonal_factor"]
        })

        # Ensure all expected features are present (safety net)
        # for name in self.feature_names:
        #      defaults.setdefault(name, 0.0) # Set to 0 if somehow missed

        return defaults


    def _safe_extract(
        self,
        data: Optional[Dict[str, Any]],
        key: str,
        default: float = 0.0
    ) -> float:
        """
        Safely extracts a value from a dictionary, attempts to convert it to float,
        and returns a default value if the key is missing, value is None, or conversion fails.

        Args:
            data: The dictionary to extract from. Can be None.
            key: The key to look for.
            default: The default float value to return on failure.

        Returns:
            The extracted float value or the default.
        """
        if data is None:
            return default

        value = data.get(key)
        if value is None:
            return default # Return default if key missing or value is None

        try:
            # Attempt conversion to float
            return float(value)
        except (ValueError, TypeError):
            # Log warning and return default if conversion fails
            _LOGGER.debug(f"Could not convert value for key '{key}' ('{value}') to float. Using default {default}.")
            return default


    def _calculate_weather_trend(self, cloudiness: float, wind_speed: float) -> float:
        """
        Calculates a simple weather trend score (0-1), where higher is better (sunnier, calmer).
        Primarily based on cloud cover, with a minor adjustment for wind.

        Args:
            cloudiness: Cloud cover percentage (0-100).
            wind_speed: Wind speed (any unit, magnitude matters).

        Returns:
            A trend score between 0.0 and 1.0.
        """
        try:
            # Ensure inputs are floats and within reasonable ranges
            cloud = max(0.0, min(100.0, float(cloudiness)))
            wind = max(0.0, float(wind_speed))

            # Score based on lack of clouds (1.0 = clear, 0.0 = overcast)
            cloud_score = (100.0 - cloud) / 100.0

            # Wind factor (slightly reduces score for very high wind, max reduction ~15%)
            # Assume high wind (> 30 m/s or ~100 km/h) is detrimental or indicates storm
            wind_penalty = min(wind / 50.0, 0.15) # Max 15% penalty for wind > 50
            wind_factor = 1.0 - wind_penalty

            # Combine scores, giving cloud cover much higher weight
            trend_score = (cloud_score * 0.9 + wind_factor * 0.1)

            # Ensure final score is clamped between 0.0 and 1.0
            return max(0.0, min(1.0, trend_score))

        except (ValueError, TypeError) as e:
            _LOGGER.warning(f"Weather trend calculation failed (clouds='{cloudiness}', wind='{wind_speed}'): {e}. Using default 0.5.")
            return 0.5 # Return neutral default on error