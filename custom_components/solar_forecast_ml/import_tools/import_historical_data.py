#!/usr/bin/env python3
"""
Standalone Historical Data Import Script for Solar Forecast ML Integration.

This script imports historical sensor data from Home Assistant CSV exports
into the hourly_samples.json file required by the ML integration.

CRITICAL: This script is ISOLATED from the main integration to prevent
any import errors from affecting the running integration.

File Requirements (place in same directory as this script):
- power_production.csv (REQUIRED) - Solar power sensor in Watts
- temperature.csv (REQUIRED) - Temperature sensor
- humidity.csv (REQUIRED) - Humidity sensor  
- wind_speed.csv (optional) - Wind speed sensor
- rain.csv (optional) - Rain rate sensor
- uv_index.csv (optional) - UV index sensor
- lux.csv (optional) - Solar radiation sensor

CSV Format (Home Assistant native export):
entity_id,state,last_changed
sensor.name,value,2025-09-30T12:34:56.789Z

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
_LOGGER = logging.getLogger(__name__)

# Constants
ML_MODEL_VERSION = "1.0.0"
DATA_VERSION = "1.0.0"


class HistoricalDataImporter:
    """Imports historical sensor data into hourly_samples.json format."""
    
    def __init__(self, import_dir: Path, output_file: Path):
        """
        Initialize the importer.
        
        Args:
            import_dir: Directory containing CSV files
            output_file: Path to hourly_samples.json
        """
        self.import_dir = import_dir
        self.output_file = output_file
        
        # Required files
        self.required_files = {
            'power': 'power_production.csv',
            'temperature': 'temperature.csv', 
            'humidity': 'humidity.csv'
        }
        
        # Optional files
        self.optional_files = {
            'wind_speed': 'wind_speed.csv',
            'rain': 'rain.csv',
            'uv_index': 'uv_index.csv',
            'lux': 'lux.csv'
        }
        
    def parse_csv_file(self, filepath: Path) -> Dict[str, List[tuple]]:
        """
        Parse a Home Assistant CSV export file.
        
        Returns:
            Dict mapping hour_key (YYYY-MM-DD-HH) to list of (timestamp, value) tuples
        """
        hourly_data = defaultdict(list)
        skipped = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Skip invalid states
                        state = row['state'].strip()
                        if state in ['unavailable', 'unknown', 'None', '']:
                            skipped += 1
                            continue
                            
                        # Parse timestamp (UTC)
                        timestamp_str = row['last_changed'].strip()
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        # Convert to float
                        value = float(state)
                        
                        # Create hour key (YYYY-MM-DD-HH in UTC)
                        hour_key = timestamp.strftime('%Y-%m-%d-%H')
                        
                        hourly_data[hour_key].append((timestamp, value))
                        
                    except (ValueError, KeyError) as e:
                        skipped += 1
                        continue
                        
            _LOGGER.info(f"Parsed {filepath.name}: {len(hourly_data)} hours, skipped {skipped} invalid entries")
            return dict(hourly_data)
            
        except FileNotFoundError:
            _LOGGER.warning(f"File not found: {filepath}")
            return {}
        except Exception as e:
            _LOGGER.error(f"Error parsing {filepath}: {e}")
            return {}
    
    def calculate_hourly_average(self, values: List[tuple]) -> float:
        """Calculate average from list of (timestamp, value) tuples."""
        if not values:
            return 0.0
        return sum(v[1] for v in values) / len(values)
    
    def calculate_hourly_kwh_riemann(self, values: List[tuple]) -> float:
        """
        Calculate kWh using Riemann sum integration from power values (Watts).
        
        Args:
            values: List of (timestamp, watts) tuples sorted by timestamp
            
        Returns:
            Energy in kWh
        """
        if not values:
            return 0.0
            
        # Sort by timestamp
        sorted_values = sorted(values, key=lambda x: x[0])
        
        total_wh = 0.0
        
        for i in range(len(sorted_values) - 1):
            t1, w1 = sorted_values[i]
            t2, w2 = sorted_values[i + 1]
            
            # Time delta in hours
            delta_hours = (t2 - t1).total_seconds() / 3600.0
            
            # Average power in this interval
            avg_power = (w1 + w2) / 2.0
            
            # Energy = Power * Time
            total_wh += avg_power * delta_hours
        
        # Convert Wh to kWh
        return total_wh / 1000.0
    
    def merge_hourly_data(
        self,
        power_data: Dict[str, List[tuple]],
        temp_data: Dict[str, List[tuple]],
        humidity_data: Dict[str, List[tuple]],
        optional_data: Dict[str, Dict[str, List[tuple]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge all sensor data into hourly samples.
        
        Returns:
            List of hourly sample dictionaries
        """
        samples = []
        
        # Get all unique hour keys from power data (required)
        hour_keys = sorted(power_data.keys())
        
        _LOGGER.info(f"Processing {len(hour_keys)} hours of data...")
        
        for hour_key in hour_keys:
            try:
                # Parse hour_key to get timestamp for this hour
                year, month, day, hour = hour_key.split('-')
                hour_start_utc = datetime(
                    int(year), int(month), int(day), int(hour),
                    tzinfo=timezone.utc
                )
                
                # Convert to local time for storage (OPTION A from integration)
                # For import, we keep UTC and let integration handle conversion
                hour_start_local = hour_start_utc
                
                # Calculate actual kWh for this hour using Riemann integration
                actual_kwh = self.calculate_hourly_kwh_riemann(power_data[hour_key])
                
                # Skip hours with zero production (unless it's nighttime data)
                if actual_kwh <= 0.0001:
                    continue
                
                # Get weather data (averages)
                temperature = self.calculate_hourly_average(temp_data.get(hour_key, []))
                humidity_val = self.calculate_hourly_average(humidity_data.get(hour_key, []))
                
                # Get optional sensor data
                wind_speed = self.calculate_hourly_average(
                    optional_data.get('wind_speed', {}).get(hour_key, [])
                )
                rain = self.calculate_hourly_average(
                    optional_data.get('rain', {}).get(hour_key, [])
                )
                uv_index = self.calculate_hourly_average(
                    optional_data.get('uv_index', {}).get(hour_key, [])
                )
                lux = self.calculate_hourly_average(
                    optional_data.get('lux', {}).get(hour_key, [])
                )
                
                # Build weather_data dict (matching integration format)
                weather_data = {
                    'temperature': round(temperature, 2),
                    'humidity': round(max(0.0, min(100.0, humidity_val)), 2),
                    'cloudiness': 50.0,  # Default - not available from sensors
                    'wind_speed': round(max(0.0, wind_speed), 2),
                    'pressure': 1013.0  # Default - not available from sensors
                }
                
                # Build sensor_data dict (matching integration format)
                sensor_data = {
                    'temperature': round(temperature, 2),
                    'wind_speed': round(max(0.0, wind_speed), 2),
                    'rain': round(max(0.0, rain), 2),
                    'uv_index': round(max(0.0, uv_index), 2),
                    'lux': round(max(0.0, lux), 2),
                    'humidity': round(max(0.0, min(100.0, humidity_val)), 2)
                }
                
                # Create sample (without daily_total and percentage yet)
                sample = {
                    'timestamp': hour_start_local.isoformat(),
                    'actual_kwh': round(actual_kwh, 4),
                    'weather_data': weather_data,
                    'sensor_data': sensor_data,
                    'model_version': ML_MODEL_VERSION
                }
                
                samples.append(sample)
                
            except Exception as e:
                _LOGGER.warning(f"Error processing hour {hour_key}: {e}")
                continue
        
        _LOGGER.info(f"Created {len(samples)} valid samples with production > 0")
        return samples
    
    def calculate_daily_totals(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate daily_total and percentage_of_day for each sample.
        
        Args:
            samples: List of samples sorted by timestamp
            
        Returns:
            Updated samples with daily_total and percentage_of_day
        """
        # Group samples by date
        daily_groups = defaultdict(list)
        
        for sample in samples:
            timestamp = datetime.fromisoformat(sample['timestamp'])
            date_key = timestamp.date()
            daily_groups[date_key].append(sample)
        
        # Calculate daily totals
        updated_samples = []
        
        for date_key, day_samples in daily_groups.items():
            # Sort samples by hour
            day_samples.sort(key=lambda x: x['timestamp'])
            
            # Calculate cumulative totals for each hour of the day
            cumulative = 0.0
            
            for sample in day_samples:
                cumulative += sample['actual_kwh']
                
                # Store cumulative total up to and including this hour
                sample['daily_total'] = round(cumulative, 4)
                
                # Calculate percentage
                if cumulative > 0.01:
                    percentage = sample['actual_kwh'] / cumulative
                else:
                    percentage = 0.0
                    
                sample['percentage_of_day'] = round(percentage, 4)
                
                updated_samples.append(sample)
        
        # Sort all samples by timestamp
        updated_samples.sort(key=lambda x: x['timestamp'])
        
        _LOGGER.info(f"Calculated daily totals for {len(daily_groups)} days")
        return updated_samples
    
    def validate_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter samples."""
        valid_samples = []
        
        for sample in samples:
            try:
                # Check required fields
                required_fields = [
                    'timestamp', 'actual_kwh', 'daily_total',
                    'percentage_of_day', 'weather_data', 'sensor_data'
                ]
                
                if not all(field in sample for field in required_fields):
                    continue
                
                # Validate values
                if sample['actual_kwh'] <= 0:
                    continue
                    
                if sample['daily_total'] <= 0:
                    continue
                
                valid_samples.append(sample)
                
            except Exception as e:
                _LOGGER.debug(f"Invalid sample: {e}")
                continue
        
        _LOGGER.info(f"Validated {len(valid_samples)} samples")
        return valid_samples
    
    def load_existing_samples(self) -> Dict[str, Any]:
        """Load existing hourly_samples.json if it exists."""
        try:
            if self.output_file.exists():
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _LOGGER.info(f"Loaded {data.get('count', 0)} existing samples")
                    return data
        except Exception as e:
            _LOGGER.warning(f"Could not load existing samples: {e}")
        
        # Return default structure
        return {
            'version': DATA_VERSION,
            'samples': [],
            'count': 0,
            'last_updated': None
        }
    
    def merge_with_existing(
        self,
        new_samples: List[Dict[str, Any]],
        existing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge new samples with existing, avoiding duplicates.
        
        Args:
            new_samples: New samples to add
            existing_data: Existing hourly_samples.json structure
            
        Returns:
            Merged data structure
        """
        existing_samples = existing_data.get('samples', [])
        
        # Create set of existing timestamps
        existing_timestamps = {s['timestamp'] for s in existing_samples}
        
        # Add only new samples (not duplicates)
        added = 0
        for sample in new_samples:
            if sample['timestamp'] not in existing_timestamps:
                existing_samples.append(sample)
                added += 1
        
        # Sort by timestamp
        existing_samples.sort(key=lambda x: x['timestamp'])
        
        # Prune to last 60 days
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=60)
        
        pruned_samples = [
            s for s in existing_samples
            if datetime.fromisoformat(s['timestamp']) >= cutoff
        ]
        
        _LOGGER.info(f"Added {added} new samples, pruned to {len(pruned_samples)} total")
        
        # Update structure
        return {
            'version': DATA_VERSION,
            'samples': pruned_samples,
            'count': len(pruned_samples),
            'last_updated': now.isoformat()
        }
    
    def write_output(self, data: Dict[str, Any]) -> bool:
        """Write data to hourly_samples.json atomically."""
        try:
            # Write to temp file first
            temp_file = self.output_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            temp_file.replace(self.output_file)
            
            _LOGGER.info(f"Successfully wrote {data['count']} samples to {self.output_file}")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to write output: {e}")
            return False
    
    def run(self) -> bool:
        """Execute the import process."""
        _LOGGER.info("=" * 60)
        _LOGGER.info("Solar Forecast ML - Historical Data Import")
        _LOGGER.info("=" * 60)
        
        # Check required files exist
        _LOGGER.info("\n1. Checking required files...")
        for key, filename in self.required_files.items():
            filepath = self.import_dir / filename
            if not filepath.exists():
                _LOGGER.error(f"REQUIRED file missing: {filename}")
                return False
            _LOGGER.info(f"  ✓ Found {filename}")
        
        # Parse required files
        _LOGGER.info("\n2. Parsing required sensor data...")
        power_data = self.parse_csv_file(self.import_dir / self.required_files['power'])
        temp_data = self.parse_csv_file(self.import_dir / self.required_files['temperature'])
        humidity_data = self.parse_csv_file(self.import_dir / self.required_files['humidity'])
        
        if not power_data:
            _LOGGER.error("No valid power production data found!")
            return False
        
        # Parse optional files
        _LOGGER.info("\n3. Parsing optional sensor data...")
        optional_data = {}
        for key, filename in self.optional_files.items():
            filepath = self.import_dir / filename
            if filepath.exists():
                optional_data[key] = self.parse_csv_file(filepath)
                _LOGGER.info(f"  ✓ Parsed {filename}")
            else:
                _LOGGER.info(f"  - Skipping {filename} (not found)")
        
        # Merge all data
        _LOGGER.info("\n4. Merging hourly data...")
        samples = self.merge_hourly_data(
            power_data, temp_data, humidity_data, optional_data
        )
        
        if not samples:
            _LOGGER.error("No valid samples created!")
            return False
        
        # Calculate daily totals
        _LOGGER.info("\n5. Calculating daily totals and percentages...")
        samples = self.calculate_daily_totals(samples)
        
        # Validate
        _LOGGER.info("\n6. Validating samples...")
        samples = self.validate_samples(samples)
        
        if not samples:
            _LOGGER.error("No valid samples after validation!")
            return False
        
        # Load existing data
        _LOGGER.info("\n7. Loading existing samples...")
        existing_data = self.load_existing_samples()
        
        # Merge with existing
        _LOGGER.info("\n8. Merging with existing data...")
        merged_data = self.merge_with_existing(samples, existing_data)
        
        # Write output
        _LOGGER.info("\n9. Writing output file...")
        success = self.write_output(merged_data)
        
        if success:
            _LOGGER.info("\n" + "=" * 60)
            _LOGGER.info("✓ Import completed successfully!")
            _LOGGER.info(f"  Total samples: {merged_data['count']}")
            _LOGGER.info(f"  Output file: {self.output_file}")
            _LOGGER.info("=" * 60)
        
        return success


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    
    # Check if running from import directory
    if script_dir.name == 'import':
        import_dir = script_dir
        output_file = script_dir.parent / 'hourly_samples.json'
    else:
        # Assume script was copied elsewhere, look for import folder
        import_dir = script_dir / 'import'
        output_file = script_dir / 'hourly_samples.json'
    
    _LOGGER.info(f"Import directory: {import_dir}")
    _LOGGER.info(f"Output file: {output_file}")
    
    # Verify import directory exists
    if not import_dir.exists():
        _LOGGER.error(f"Import directory not found: {import_dir}")
        _LOGGER.error("Please create the 'import' directory and place CSV files there.")
        return 1
    
    # Create importer and run
    importer = HistoricalDataImporter(import_dir, output_file)
    
    try:
        success = importer.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        _LOGGER.info("\nImport cancelled by user")
        return 1
    except Exception as e:
        _LOGGER.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
