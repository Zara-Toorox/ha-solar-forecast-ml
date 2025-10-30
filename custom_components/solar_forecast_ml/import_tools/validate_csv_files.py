#!/usr/bin/env python3
"""
CSV Validation Tool for Solar Forecast ML Historical Import.

Quick pre-flight check to validate CSV files before running the import.
Helps users identify issues early and provides actionable feedback.

Usage:
    python3 validate_csv_files.py

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# ANSI color codes for pretty output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def check_file_exists(filepath: Path) -> Tuple[bool, str]:
    """Check if a file exists and is readable."""
    if not filepath.exists():
        return False, f"File not found: {filepath.name}"
    
    if filepath.stat().st_size == 0:
        return False, f"File is empty: {filepath.name}"
    
    return True, f"✓ Found {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)"


def validate_csv_structure(filepath: Path) -> Tuple[bool, str, int, int]:
    """
    Validate CSV structure and count valid/invalid entries.
    
    Returns:
        (is_valid, message, valid_count, invalid_count)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check header
            if 'entity_id' not in reader.fieldnames or \
               'state' not in reader.fieldnames or \
               'last_changed' not in reader.fieldnames:
                return False, "Invalid CSV header (missing required columns)", 0, 0
            
            valid = 0
            invalid = 0
            
            for row in reader:
                try:
                    # Try to parse state
                    state = row['state'].strip()
                    if state in ['unavailable', 'unknown', 'None', '']:
                        invalid += 1
                        continue
                    
                    # Try to parse value
                    float(state)
                    
                    # Try to parse timestamp
                    timestamp_str = row['last_changed'].strip()
                    datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    valid += 1
                    
                except (ValueError, KeyError):
                    invalid += 1
                    continue
            
            if valid == 0:
                return False, "No valid data entries found", 0, invalid
            
            return True, f"✓ {valid} valid entries, {invalid} skipped", valid, invalid
            
    except Exception as e:
        return False, f"Error reading file: {str(e)}", 0, 0


def analyze_date_range(filepath: Path) -> Tuple[str, str, int]:
    """
    Analyze date range in CSV file.
    
    Returns:
        (first_date, last_date, days_covered)
    """
    try:
        timestamps = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    timestamp_str = row['last_changed'].strip()
                    ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
        
        if not timestamps:
            return "N/A", "N/A", 0
        
        timestamps.sort()
        first = timestamps[0]
        last = timestamps[-1]
        days = (last - first).days + 1
        
        return first.strftime('%Y-%m-%d'), last.strftime('%Y-%m-%d'), days
        
    except Exception:
        return "N/A", "N/A", 0


def main():
    """Main validation routine."""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}CSV File Validation for Solar Forecast ML Import{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")
    
    # Determine import directory
    script_dir = Path(__file__).parent.resolve()
    import_dir = script_dir
    
    print(f"Checking directory: {BOLD}{import_dir}{RESET}\n")
    
    # Define required and optional files
    required_files = {
        'power_production.csv': 'Solar power sensor (Watts)',
        'temperature.csv': 'Temperature sensor',
        'humidity.csv': 'Humidity sensor'
    }
    
    optional_files = {
        'wind_speed.csv': 'Wind speed sensor',
        'rain.csv': 'Rain rate sensor',
        'uv_index.csv': 'UV index sensor',
        'lux.csv': 'Solar radiation sensor'
    }
    
    all_valid = True
    validation_results = []
    
    # Check required files
    print(f"{BOLD}1. Required Files:{RESET}\n")
    
    for filename, description in required_files.items():
        filepath = import_dir / filename
        exists, msg = check_file_exists(filepath)
        
        if exists:
            print(f"  {GREEN}{msg}{RESET}")
            print(f"     {description}")
            
            # Validate structure
            valid, validation_msg, valid_count, invalid_count = validate_csv_structure(filepath)
            
            if valid:
                print(f"     {GREEN}{validation_msg}{RESET}")
                
                # Analyze date range
                first_date, last_date, days = analyze_date_range(filepath)
                print(f"     Date range: {first_date} to {last_date} ({days} days)")
                
                validation_results.append((filename, True, valid_count, days))
            else:
                print(f"     {RED}{validation_msg}{RESET}")
                all_valid = False
                validation_results.append((filename, False, 0, 0))
        else:
            print(f"  {RED}✗ {msg}{RESET}")
            print(f"     {description}")
            all_valid = False
            validation_results.append((filename, False, 0, 0))
        
        print()
    
    # Check optional files
    print(f"\n{BOLD}2. Optional Files:{RESET}\n")
    
    optional_found = 0
    for filename, description in optional_files.items():
        filepath = import_dir / filename
        exists, msg = check_file_exists(filepath)
        
        if exists:
            print(f"  {GREEN}{msg}{RESET}")
            print(f"     {description}")
            
            # Validate structure
            valid, validation_msg, valid_count, invalid_count = validate_csv_structure(filepath)
            
            if valid:
                print(f"     {GREEN}{validation_msg}{RESET}")
                
                # Analyze date range
                first_date, last_date, days = analyze_date_range(filepath)
                print(f"     Date range: {first_date} to {last_date} ({days} days)")
                
                optional_found += 1
            else:
                print(f"     {YELLOW}Warning: {validation_msg}{RESET}")
        else:
            print(f"  {YELLOW}- {filename} not found (optional){RESET}")
            print(f"     {description}")
        
        print()
    
    # Summary
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}Validation Summary:{RESET}\n")
    
    if all_valid:
        print(f"{GREEN}✓ All required files are valid!{RESET}")
        print(f"  Found {optional_found} optional file(s)")
        
        # Estimate samples
        power_valid = next((r[2] for r in validation_results if r[0] == 'power_production.csv'), 0)
        power_days = next((r[3] for r in validation_results if r[0] == 'power_production.csv'), 0)
        
        estimated_samples = min(power_valid, power_days * 15)  # ~15 hours/day production
        
        print(f"\n{BOLD}Estimated Import Results:{RESET}")
        print(f"  • Valid power entries: ~{power_valid}")
        print(f"  • Days covered: {power_days}")
        print(f"  • Estimated samples: ~{estimated_samples}")
        
        if estimated_samples < 50:
            print(f"\n{YELLOW}⚠ Warning: Less than 50 samples expected{RESET}")
            print(f"   ML training requires at least 100 samples for good results.")
            print(f"   Consider adding more days of data.")
        elif estimated_samples < 100:
            print(f"\n{YELLOW}⚠ Note: 100+ samples recommended for optimal results{RESET}")
        else:
            print(f"\n{GREEN}✓ Sufficient data for ML training!{RESET}")
        
        print(f"\n{GREEN}{BOLD}Ready to run import!{RESET}")
        print(f"\nRun: {BOLD}python3 import_historical_data.py{RESET}")
        
        return 0
        
    else:
        print(f"{RED}✗ Validation failed!{RESET}")
        print(f"\n{BOLD}Issues found:{RESET}")
        
        for filename, valid, _, _ in validation_results:
            if not valid:
                print(f"  {RED}✗ {filename}{RESET}")
        
        print(f"\n{BOLD}Recommendations:{RESET}")
        print(f"  1. Check that all required CSV files are present")
        print(f"  2. Ensure files are named exactly as specified")
        print(f"  3. Verify CSV format matches Home Assistant export")
        print(f"  4. Check that files contain actual sensor data")
        
        print(f"\n{RED}Import cannot proceed until issues are resolved.{RESET}")
        
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Validation cancelled by user.{RESET}")
        sys.exit(1)
