"""
Chart Generator Service for Solar Forecast ML Integration

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
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

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import json
import asyncio
from functools import partial

_LOGGER = logging.getLogger(__name__)

# DO NOT import matplotlib at module level - it causes blocking I/O!
# Matplotlib will be imported lazily inside the executor thread


def _check_matplotlib_available() -> bool:
    """Check if matplotlib is available without importing it"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("matplotlib")
        return spec is not None
    except Exception:
        return False


class ChartGenerator:
    """Generates PNG charts from solar forecast data"""

    def __init__(self, data_dir: Path):
        """Initialize chart generator"""
        self.data_dir = data_dir
        # Charts dir should be inside solar_forecast_ml directory
        self.charts_dir = data_dir / "exports" / "pictures"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    async def generate_daily_forecast_chart(
        self,
        target_date: Optional[date] = None
    ) -> Optional[str]:
        """
        Generate Forecast vs Actual chart for a specific day

        Returns path to generated chart or None if failed
        """
        if not _check_matplotlib_available():
            _LOGGER.warning(
                "Cannot generate chart - matplotlib not available. "
                "Install matplotlib to enable chart generation: pip install matplotlib"
            )
            return None

        try:
            if target_date is None:
                target_date = date.today()

            # Run blocking operations in executor
            loop = asyncio.get_event_loop()
            chart_path = await loop.run_in_executor(
                None,
                partial(self._generate_daily_chart_sync, target_date)
            )

            return chart_path

        except Exception as e:
            _LOGGER.error(f"Failed to generate daily forecast chart: {e}", exc_info=True)
            return None

    def _generate_daily_chart_sync(self, target_date: date) -> Optional[str]:
        """Synchronous chart generation (runs in executor)"""
        try:
            # Import matplotlib here - inside executor thread!
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.figure import Figure

            # Load data from prediction_history.json
            history_file = self.data_dir / "stats" / "prediction_history.json"
            if not history_file.exists():
                _LOGGER.error(f"prediction_history.json not found at {history_file}")
                return None

            with open(history_file, 'r') as f:
                history_data = json.load(f)

            predictions = history_data.get('predictions', [])

            # Load daily_forecasts.json for metadata
            forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"
            day_metadata = {}
            if forecasts_file.exists():
                with open(forecasts_file, 'r') as f:
                    forecasts_data = json.load(f)
                    # Check if this is today, yesterday, or in history
                    target_str = target_date.strftime('%Y-%m-%d')
                    if forecasts_data.get('today', {}).get('date') == target_str:
                        day_metadata = forecasts_data.get('today', {})
                    else:
                        # Search in history
                        for hist in forecasts_data.get('history', []):
                            if hist.get('date') == target_str:
                                day_metadata = hist
                                break

            # Filter predictions for target date
            target_str = target_date.strftime('%Y-%m-%d')
            day_predictions = [
                p for p in predictions
                if p.get('date') == target_str
            ]

            if not day_predictions:
                _LOGGER.warning(f"No predictions found for {target_str}")
                return None

            # Sort by timestamp
            day_predictions.sort(key=lambda x: x.get('timestamp', ''))

            # Extract data
            timestamps = []
            predicted_values = []
            actual_values = []

            for pred in day_predictions:
                ts_str = pred.get('timestamp')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        timestamps.append(ts)

                        # Handle None values (future hours without actual data)
                        predicted_val = pred.get('predicted_value')
                        actual_val = pred.get('actual_value')

                        predicted_values.append(predicted_val if predicted_val is not None else 0)
                        actual_values.append(actual_val if actual_val is not None else 0)
                    except ValueError:
                        continue

            if not timestamps:
                _LOGGER.warning(f"No valid timestamps for {target_str}")
                return None

            # Filter to production hours only (dynamic range)
            # Find first and last hour with significant production (>0.01 kWh)
            production_start_idx = None
            production_end_idx = None

            for i, (pred, actual) in enumerate(zip(predicted_values, actual_values)):
                if pred > 0.01 or actual > 0.01:
                    if production_start_idx is None:
                        production_start_idx = max(0, i - 1)  # Include 1 hour before
                    production_end_idx = min(len(timestamps) - 1, i + 1)  # Include 1 hour after

            # Fallback to 06:00-20:00 if no production detected
            if production_start_idx is None or production_end_idx is None:
                production_start_idx = 0
                production_end_idx = len(timestamps) - 1
                for i, ts in enumerate(timestamps):
                    if ts.hour == 6:
                        production_start_idx = i
                    if ts.hour == 20:
                        production_end_idx = i
                        break

            # Slice data to production hours
            timestamps = timestamps[production_start_idx:production_end_idx + 1]
            predicted_values = predicted_values[production_start_idx:production_end_idx + 1]
            actual_values = actual_values[production_start_idx:production_end_idx + 1]

            # Create chart
            fig = Figure(figsize=(14, 7), dpi=100)
            ax = fig.add_subplot(111)

            # Plot data
            ax.plot(timestamps, predicted_values,
                   label='Forecast', color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
            ax.plot(timestamps, actual_values,
                   label='Actual', color='#2ca02c', linewidth=2.5, marker='s', markersize=5)

            # Add production time markers if available
            production_time = day_metadata.get('production_time', {})
            start_time_str = production_time.get('start_time')
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    ax.axvline(start_time, color='green', linestyle=':', linewidth=2, alpha=0.6, label='Production Start')
                except:
                    pass

            # Add peak hour marker if available
            peak_today = day_metadata.get('peak_today', {})
            peak_at_str = peak_today.get('at')
            peak_power = peak_today.get('power_w')
            if peak_at_str and peak_power:
                try:
                    peak_time = datetime.fromisoformat(peak_at_str.replace('Z', '+00:00'))
                    ax.axvline(peak_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak: {peak_power:.0f}W')
                except:
                    pass

            # Formatting
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_ylabel('Energy (kWh)', fontsize=12, fontweight='bold')
            ax.set_title(f'Solar Forecast vs Actual Production - {target_str}',
                        fontsize=15, fontweight='bold', pad=20)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            fig.autofmt_xdate()

            # Add statistics text box
            stats_lines = []

            # Calculate totals
            actual_total = sum(actual_values)
            predicted_total = sum(predicted_values)

            if actual_total > 0:
                accuracy = (1 - abs(predicted_total - actual_total) / actual_total) * 100
                accuracy = max(0, min(100, accuracy))
                stats_lines.append(f'Accuracy: {accuracy:.1f}%')

            stats_lines.append(f'Forecast: {predicted_total:.2f} kWh')
            stats_lines.append(f'Actual: {actual_total:.2f} kWh')

            # Add production hours if available from history
            if 'production_hours' in day_metadata:
                stats_lines.append(f'Duration: {day_metadata["production_hours"]}')

            # Add autarky if available
            autarky_val = day_metadata.get('autarky')
            if autarky_val is not None:
                # Handle both dict format (from today) and direct value (from history)
                if isinstance(autarky_val, dict):
                    autarky_val = autarky_val.get('percent')
                if autarky_val is not None:
                    stats_lines.append(f'Autarky: {autarky_val:.1f}%')

            # Add consumption if available
            consumption_val = day_metadata.get('consumption_kwh')
            if consumption_val is not None:
                stats_lines.append(f'Consumption: {consumption_val:.2f} kWh')

            stats_text = '\n'.join(stats_lines)
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8))

            # Tight layout
            fig.tight_layout()

            # Save chart
            chart_path = self.charts_dir / f"daily_forecast_{target_str}.png"
            fig.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            _LOGGER.info(f"Chart generated successfully: {chart_path}")
            return str(chart_path)

        except ImportError as e:
            _LOGGER.error(f"Matplotlib import failed: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"Error in sync chart generation: {e}", exc_info=True)
            return None

    async def generate_weekly_accuracy_chart(self) -> Optional[str]:
        """Generate weekly accuracy bar chart"""
        if not _check_matplotlib_available():
            _LOGGER.error("Cannot generate chart - matplotlib not available")
            return None

        try:
            # Run blocking operations in executor
            loop = asyncio.get_event_loop()
            chart_path = await loop.run_in_executor(
                None,
                self._generate_weekly_chart_sync
            )

            return chart_path

        except Exception as e:
            _LOGGER.error(f"Failed to generate weekly accuracy chart: {e}", exc_info=True)
            return None

    def _generate_weekly_chart_sync(self) -> Optional[str]:
        """Synchronous weekly chart generation (runs in executor)"""
        try:
            # Import matplotlib here - inside executor thread!
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure

            # Load data
            forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"

            if not forecasts_file.exists():
                _LOGGER.error(f"daily_forecasts.json not found at {forecasts_file}")
                return None

            with open(forecasts_file, 'r') as f:
                forecasts_data = json.load(f)

            history = forecasts_data.get('history', [])
            _LOGGER.debug(f"Found {len(history)} total history entries")

            if len(history) < 2:
                _LOGGER.warning(f"Not enough history data for weekly chart (found {len(history)}, need at least 2)")
                return None

            # Get last 7 days with data
            history_with_accuracy = [h for h in history if h.get('accuracy') is not None]
            _LOGGER.debug(f"Found {len(history_with_accuracy)} entries with accuracy data")

            if len(history_with_accuracy) < 2:
                _LOGGER.warning(f"Not enough entries with accuracy for weekly chart (found {len(history_with_accuracy)}, need at least 2)")
                return None

            history_sorted = sorted(
                history_with_accuracy,
                key=lambda x: x.get('date', ''),
                reverse=True
            )[:7]
            history_sorted.reverse()  # Chronological order

            dates = [h.get('date', '') for h in history_sorted]
            accuracies = [h.get('accuracy', 0) for h in history_sorted]

            _LOGGER.debug(f"Generating weekly chart with {len(dates)} days: {dates}")

            # Create chart
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)

            # Bar chart
            bars = ax.bar(range(len(dates)), accuracies, color='#2ca02c', alpha=0.7)

            # Color code based on accuracy
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                if acc >= 80:
                    bar.set_color('#2ca02c')  # Green
                elif acc >= 60:
                    bar.set_color('#ff7f0e')  # Orange
                else:
                    bar.set_color('#d62728')  # Red

            # Formatting
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title('Weekly Forecast Accuracy', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d[-5:] for d in dates], rotation=45)  # Show MM-DD
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

            # Add value labels on bars
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

            # Tight layout
            fig.tight_layout()

            # Save chart
            chart_path = self.charts_dir / "weekly_accuracy.png"
            fig.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            _LOGGER.info(f"Weekly accuracy chart generated: {chart_path}")
            return str(chart_path)

        except ImportError as e:
            _LOGGER.error(f"Matplotlib import failed: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"Error in sync weekly chart generation: {e}", exc_info=True)
            return None

    async def generate_production_weather_chart(
        self,
        target_date: Optional[date] = None
    ) -> Optional[str]:
        """Generate Production vs Weather chart for a specific day"""
        if not _check_matplotlib_available():
            _LOGGER.warning("Cannot generate chart - matplotlib not available")
            return None

        try:
            if target_date is None:
                target_date = date.today()

            # Run blocking operations in executor
            loop = asyncio.get_event_loop()
            chart_path = await loop.run_in_executor(
                None,
                partial(self._generate_production_weather_sync, target_date)
            )

            return chart_path

        except Exception as e:
            _LOGGER.error(f"Failed to generate production-weather chart: {e}", exc_info=True)
            return None

    def _generate_production_weather_sync(self, target_date: date) -> Optional[str]:
        """Synchronous production-weather chart generation (runs in executor)"""
        try:
            # Import matplotlib here - inside executor thread!
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.figure import Figure

            # Load prediction history
            history_file = self.data_dir / "stats" / "prediction_history.json"
            if not history_file.exists():
                _LOGGER.error(f"prediction_history.json not found")
                return None

            with open(history_file, 'r') as f:
                history_data = json.load(f)

            predictions = history_data.get('predictions', [])

            # Load daily_forecasts.json for metadata
            forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"
            day_metadata = {}
            if forecasts_file.exists():
                with open(forecasts_file, 'r') as f:
                    forecasts_data = json.load(f)
                    target_str = target_date.strftime('%Y-%m-%d')
                    if forecasts_data.get('today', {}).get('date') == target_str:
                        day_metadata = forecasts_data.get('today', {})
                    else:
                        for hist in forecasts_data.get('history', []):
                            if hist.get('date') == target_str:
                                day_metadata = hist
                                break

            # Filter for target date
            target_str = target_date.strftime('%Y-%m-%d')
            day_predictions = [
                p for p in predictions
                if p.get('date') == target_str
            ]

            if not day_predictions:
                _LOGGER.warning(f"No predictions found for {target_str}")
                return None

            # Sort by timestamp
            day_predictions.sort(key=lambda x: x.get('timestamp', ''))

            # Extract data
            timestamps = []
            predicted_values = []
            actual_values = []
            cloud_cover = []
            temperature = []

            for pred in day_predictions:
                ts_str = pred.get('timestamp')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        timestamps.append(ts)

                        # Production data
                        predicted_val = pred.get('predicted_value')
                        actual_val = pred.get('actual_value')
                        predicted_values.append(predicted_val if predicted_val is not None else 0)
                        actual_values.append(actual_val if actual_val is not None else 0)

                        # Weather data
                        weather = pred.get('weather_data', {})
                        cloud_cover.append(weather.get('cloud_cover', 0))
                        temperature.append(weather.get('temperature', 0))
                    except ValueError:
                        continue

            if not timestamps:
                _LOGGER.warning(f"No valid data for {target_str}")
                return None

            # Filter to relevant time range: from forecast creation to production end
            # Start: When forecast was created (e.g., 6:00 for auto_6am source)
            # End: Last production hour + 1

            # Determine start time from forecast source
            forecast_source = day_metadata.get('forecast_day', {}).get('source', '')
            forecast_start_hour = 6  # Default to 6 AM

            if 'auto_6am' in forecast_source or '6am' in forecast_source.lower():
                forecast_start_hour = 6
            elif 'midnight' in forecast_source.lower():
                forecast_start_hour = 0

            # Find start index (forecast creation time)
            production_start_idx = 0
            for i, ts in enumerate(timestamps):
                if ts.hour >= forecast_start_hour:
                    production_start_idx = i
                    break

            # Find end index (last production + 1 hour)
            production_end_idx = len(timestamps) - 1
            for i in range(len(predicted_values) - 1, -1, -1):
                if predicted_values[i] > 0.01 or actual_values[i] > 0.01:
                    production_end_idx = min(len(timestamps) - 1, i + 1)
                    break

            # Fallback: if no production detected, show 06:00-18:00
            if production_end_idx <= production_start_idx:
                production_start_idx = 0
                production_end_idx = len(timestamps) - 1
                for i, ts in enumerate(timestamps):
                    if ts.hour == 6:
                        production_start_idx = i
                    if ts.hour == 18:
                        production_end_idx = i
                        break

            # Slice data to relevant time range
            timestamps = timestamps[production_start_idx:production_end_idx + 1]
            predicted_values = predicted_values[production_start_idx:production_end_idx + 1]
            actual_values = actual_values[production_start_idx:production_end_idx + 1]
            cloud_cover = cloud_cover[production_start_idx:production_end_idx + 1]
            temperature = temperature[production_start_idx:production_end_idx + 1]

            # Create chart with dual y-axis
            fig = Figure(figsize=(14, 7), dpi=100)
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            # Production data on primary axis
            ax1.plot(timestamps, predicted_values, label='Forecast',
                    color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
            ax1.plot(timestamps, actual_values, label='Actual',
                    color='#2ca02c', linewidth=2.5, marker='s', markersize=5)

            # Cloud cover as filled area on secondary axis
            ax2.fill_between(timestamps, cloud_cover, alpha=0.3, color='gray', label='Cloud Cover')

            # Temperature as line on secondary axis
            ax2.plot(timestamps, temperature, label='Temperature',
                    color='#ff7f0e', linewidth=2, linestyle='--', marker='^', markersize=4)

            # Add production time markers if available
            production_time = day_metadata.get('production_time', {})
            start_time_str = production_time.get('start_time')
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    ax1.axvline(start_time, color='green', linestyle=':', linewidth=2, alpha=0.6)
                except:
                    pass

            # Add peak hour marker if available
            peak_today = day_metadata.get('peak_today', {})
            peak_at_str = peak_today.get('at')
            peak_power = peak_today.get('power_w')
            if peak_at_str and peak_power:
                try:
                    peak_time = datetime.fromisoformat(peak_at_str.replace('Z', '+00:00'))
                    ax1.axvline(peak_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
                except:
                    pass

            # Formatting
            ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Energy (kWh)', fontsize=12, fontweight='bold', color='#2ca02c')
            ax2.set_ylabel('Cloud Cover (%) / Temperature (°C)', fontsize=12, fontweight='bold', color='gray')

            ax1.set_title(f'Production vs Weather - {target_str}',
                         fontsize=15, fontweight='bold', pad=20)

            # Legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

            # Grid
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            fig.autofmt_xdate()

            # Color the y-axis labels
            ax1.tick_params(axis='y', labelcolor='#2ca02c')
            ax2.tick_params(axis='y', labelcolor='gray')

            # Tight layout
            fig.tight_layout()

            # Save chart
            chart_path = self.charts_dir / f"production_weather_{target_str}.png"
            fig.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            _LOGGER.info(f"Production-Weather chart generated: {chart_path}")
            return str(chart_path)

        except ImportError as e:
            _LOGGER.error(f"Matplotlib import failed: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"Error in sync production-weather chart: {e}", exc_info=True)
            return None

    async def generate_monthly_heatmap(self) -> Optional[str]:
        """Generate Monthly Production Heatmap"""
        if not _check_matplotlib_available():
            _LOGGER.warning("Cannot generate chart - matplotlib not available")
            return None

        try:
            # Run blocking operations in executor
            loop = asyncio.get_event_loop()
            chart_path = await loop.run_in_executor(
                None,
                self._generate_monthly_heatmap_sync
            )

            return chart_path

        except Exception as e:
            _LOGGER.error(f"Failed to generate monthly heatmap: {e}", exc_info=True)
            return None

    def _generate_monthly_heatmap_sync(self) -> Optional[str]:
        """Synchronous monthly heatmap generation (runs in executor)"""
        try:
            # Import matplotlib here - inside executor thread!
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import numpy as np
            from datetime import timedelta

            # Load daily forecasts history
            forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"
            if not forecasts_file.exists():
                _LOGGER.error(f"daily_forecasts.json not found")
                return None

            with open(forecasts_file, 'r') as f:
                forecasts_data = json.load(f)

            history = forecasts_data.get('history', [])

            if len(history) < 2:
                _LOGGER.warning(f"Not enough history for heatmap (found {len(history)}, need at least 2)")
                return None

            # Get last 30 days
            history_sorted = sorted(
                [h for h in history if h.get('actual_kwh') is not None],
                key=lambda x: x.get('date', ''),
                reverse=True
            )[:30]

            if len(history_sorted) < 2:
                _LOGGER.warning(f"Not enough data with actual_kwh for heatmap")
                return None

            # Create date -> production mapping
            date_production = {}
            max_production = 0
            for entry in history_sorted:
                date_str = entry.get('date')
                production = entry.get('actual_kwh', 0)
                if date_str and production is not None:
                    date_production[date_str] = production
                    max_production = max(max_production, production)

            # Find date range
            dates = sorted(date_production.keys())
            start_date = datetime.strptime(dates[0], '%Y-%m-%d').date()
            end_date = datetime.strptime(dates[-1], '%Y-%m-%d').date()

            # Extend to full weeks
            # Start from Monday of the week
            start_date -= timedelta(days=start_date.weekday())
            # End on Sunday
            end_date += timedelta(days=(6 - end_date.weekday()))

            # Build matrix (weeks x 7 days)
            current_date = start_date
            weeks = []
            current_week = []

            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                production = date_production.get(date_str, None)
                current_week.append(production)

                if len(current_week) == 7:
                    weeks.append(current_week)
                    current_week = []

                current_date += timedelta(days=1)

            if current_week:
                weeks.append(current_week)

            # Convert to numpy array
            heatmap_data = np.array(weeks, dtype=float)

            # Create chart
            fig = Figure(figsize=(12, len(weeks) * 0.8 + 2), dpi=100)
            ax = fig.add_subplot(111)

            # Create heatmap
            cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
            im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=max_production)

            # Set ticks and labels
            ax.set_xticks(np.arange(7))
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            ax.set_yticks(np.arange(len(weeks)))

            # Week labels (show date of Monday)
            week_labels = []
            current_date = start_date
            for _ in range(len(weeks)):
                week_labels.append(current_date.strftime('%m/%d'))
                current_date += timedelta(days=7)
            ax.set_yticklabels(week_labels)

            # Add values to cells
            for i in range(len(weeks)):
                for j in range(7):
                    value = heatmap_data[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < max_production * 0.5 else 'black'
                        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                               color=text_color, fontsize=9, fontweight='bold')

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=30)
            cbar.set_label('Daily Production (kWh)', fontsize=11, fontweight='bold')

            # Title
            ax.set_title(f'Monthly Production Heatmap\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                        fontsize=14, fontweight='bold', pad=15)

            ax.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
            ax.set_ylabel('Week Starting', fontsize=11, fontweight='bold')

            # Tight layout
            fig.tight_layout()

            # Save chart
            chart_path = self.charts_dir / "monthly_heatmap.png"
            fig.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            _LOGGER.info(f"Monthly heatmap generated: {chart_path}")
            return str(chart_path)

        except ImportError as e:
            _LOGGER.error(f"Matplotlib import failed: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"Error in sync monthly heatmap: {e}", exc_info=True)
            return None

    async def generate_sensor_correlation_chart(self) -> Optional[str]:
        """Generate Weather Sensor Correlation Chart"""
        if not _check_matplotlib_available():
            _LOGGER.warning("Cannot generate chart - matplotlib not available")
            return None

        try:
            # Run blocking operations in executor
            loop = asyncio.get_event_loop()
            chart_path = await loop.run_in_executor(
                None,
                self._generate_sensor_correlation_sync
            )

            return chart_path

        except Exception as e:
            _LOGGER.error(f"Failed to generate sensor correlation chart: {e}", exc_info=True)
            return None

    def _generate_sensor_correlation_sync(self) -> Optional[str]:
        """Synchronous sensor correlation chart generation (runs in executor)"""
        try:
            # Import matplotlib here - inside executor thread!
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import numpy as np

            # Load prediction history with weather data
            history_file = self.data_dir / "stats" / "prediction_history.json"
            if not history_file.exists():
                _LOGGER.error(f"prediction_history.json not found")
                return None

            with open(history_file, 'r') as f:
                history_data = json.load(f)

            predictions = history_data.get('predictions', [])

            if len(predictions) < 10:
                _LOGGER.warning(f"Not enough prediction data for correlation (found {len(predictions)}, need at least 10)")
                return None

            # Extract data for correlation analysis
            # We'll analyze: forecast error vs weather conditions
            forecast_errors = []
            cloud_cover_values = []
            temperature_values = []
            humidity_values = []
            wind_speed_values = []

            for pred in predictions:
                predicted = pred.get('predicted_value')
                actual = pred.get('actual_value')
                weather = pred.get('weather_data', {})

                # Only include completed predictions with actual values
                if predicted is not None and actual is not None and actual > 0:
                    # Calculate error as percentage: (predicted - actual) / actual * 100
                    error_percent = ((predicted - actual) / actual) * 100 if actual > 0 else 0
                    forecast_errors.append(error_percent)

                    cloud_cover_values.append(weather.get('cloud_cover', 0))
                    temperature_values.append(weather.get('temperature', 0))
                    humidity_values.append(weather.get('humidity', 0))
                    wind_speed_values.append(weather.get('wind_speed', 0))

            if len(forecast_errors) < 10:
                _LOGGER.warning(f"Not enough completed predictions for correlation (found {len(forecast_errors)}, need at least 10)")
                return None

            # Calculate correlation coefficients
            correlations = {}
            sensor_data = {
                'Cloud Cover (%)': cloud_cover_values,
                'Temperature (°C)': temperature_values,
                'Humidity (%)': humidity_values,
                'Wind Speed (m/s)': wind_speed_values
            }

            for sensor_name, values in sensor_data.items():
                if len(values) > 0:
                    # Calculate Pearson correlation coefficient
                    corr = np.corrcoef(forecast_errors, values)[0, 1]
                    if not np.isnan(corr):
                        correlations[sensor_name] = corr

            if not correlations:
                _LOGGER.warning("No valid correlations could be calculated")
                return None

            # Create figure with 2 subplots: bar chart and scatter plots
            fig = Figure(figsize=(14, 10), dpi=100)

            # Subplot 1: Correlation bar chart
            ax1 = fig.add_subplot(2, 1, 1)

            sensors = list(correlations.keys())
            corr_values = list(correlations.values())

            colors = ['#d62728' if v < 0 else '#2ca02c' for v in corr_values]
            bars = ax1.barh(sensors, corr_values, color=colors, alpha=0.7)

            ax1.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
            ax1.set_title('Weather Sensor Correlation with Forecast Error\n(Negative = Better accuracy in those conditions)',
                         fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlim(-1, 1)
            ax1.axvline(0, color='black', linestyle='-', linewidth=0.8)
            ax1.grid(True, alpha=0.3, axis='x', linestyle='--')

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, corr_values)):
                x_pos = val + (0.05 if val > 0 else -0.05)
                ha = 'left' if val > 0 else 'right'
                ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha=ha, va='center', fontsize=10, fontweight='bold')

            # Add interpretation text
            interpretation = (
                "Interpretation:\n"
                "• Positive correlation: Higher sensor values lead to larger forecast errors (over/under-prediction)\n"
                "• Negative correlation: Higher sensor values lead to smaller forecast errors (better accuracy)\n"
                "• Values close to 0: Sensor has little impact on forecast accuracy"
            )
            ax1.text(0.02, 0.98, interpretation, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8))

            # Subplot 2: Scatter plot grid (2x2) for most significant correlations
            # Sort by absolute correlation value
            sorted_sensors = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_sensors = sorted_sensors[:4]  # Top 4 most correlated sensors

            for idx, (sensor_name, corr_val) in enumerate(top_sensors, 1):
                ax = fig.add_subplot(2, 4, 4 + idx)

                sensor_values = sensor_data[sensor_name]

                # Scatter plot
                scatter = ax.scatter(sensor_values, forecast_errors,
                                    alpha=0.6, s=30, c=forecast_errors,
                                    cmap='RdYlGn_r', edgecolors='black', linewidths=0.5)

                # Add trend line
                z = np.polyfit(sensor_values, forecast_errors, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(sensor_values), max(sensor_values), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

                ax.set_xlabel(sensor_name, fontsize=9)
                ax.set_ylabel('Forecast Error (%)', fontsize=9)
                ax.set_title(f'r = {corr_val:.3f}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

            # Overall title for scatter plots
            fig.text(0.5, 0.48, 'Top Correlations - Detailed View',
                    ha='center', fontsize=13, fontweight='bold')

            # Tight layout
            fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=3.0)

            # Save chart
            chart_path = self.charts_dir / "sensor_correlation.png"
            fig.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            _LOGGER.info(f"Sensor correlation chart generated: {chart_path}")
            return str(chart_path)

        except ImportError as e:
            _LOGGER.error(f"Matplotlib import failed: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"Error in sync sensor correlation chart: {e}", exc_info=True)
            return None
