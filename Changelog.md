changelog:


Fuel my late-night ideas with a coffee? I'd really appreciate it ♡
<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee ' /></a>

**Solar Forecast ML**

**Update to version 13.1.0**
- Updated to the latest Home Assistant version
- New algorithm with winter and low-sun corrections
- DNI/DHI ratio now dependent on solar elevation and season
- Fog detection via dew point / humidity
- Migration: Removal of the obsolete “Default” panel group
- Improved handling of weather events
- Frost constant with improved threshold
- Retrospective migration moves historical frost hours into new buckets
- Adjusted radiation model: Increased calculated diffuse fraction at low solar angles, which mainly stabilizes morning and evening forecasts
- New optional “Winter Mode”: From November to February, specialized algorithms are automatically used  
  → This feature can be enabled/disabled in the options as needed — or left on “Auto”
- Improved logging
- New algorithm parts in physics and weather modules
- Improved cache logic
- Extended notifications
- SFML Stats (x86)

**Update to version 6.8.0**
- New consumer feature: Heat pump, electric heating element and wallbox can now be configured and monitored
- Cost calculation with COP (Coefficient of Performance) for heat pumps
- New consumer cards in the Energy Dashboard (3D isometric + 2D classic) with detail modal
- Dashboard sections can now be reordered via drag & drop
- Touch optimization: 2-second long-press with visual progress ring on mobile devices
- Persistent storage of section order (localStorage + cookie fallback for HA Companion App)
- New reset button in header to restore default sorting
- List mode for quick overview
- Fixes to mathematical formulas
- Fixes to electricity price input
- Fixes related to feed-in tariffs
- Significantly improved descriptions during initial setup and re-configuration
- Performance improvements
- Only one graph shown when using a single panel group
- Rearranged icons in 2D & 3D views to create space for heat pump, EV, heating
- New card for external consumers

**IMPORTANT NOTICE:**  
Heat pump, EV and electric heating (HZ) functionality can only be tested to a limited extent.  
→ I am therefore especially dependent on clear, logical feedback with precise error descriptions!!!  
There may still be issues in visualization and calculations in these areas!!!

Let me know if you'd like any part toned more/less formal or adjusted for length.