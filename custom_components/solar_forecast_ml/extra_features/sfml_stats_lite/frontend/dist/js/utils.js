/* ============================================================
   SFML Stats Lite Dashboard - Utility Functions
   ============================================================ */

const SFMLUtils = {
    // Format time from decimal hours or minutes
    formatTime(value, unit = 'hours') {
        if (value == null) return '—';

        let totalMinutes;
        if (unit === 'hours') {
            totalMinutes = Math.round(value * 60);
        } else {
            totalMinutes = Math.round(value);
        }

        const hours = Math.floor(totalMinutes / 60);
        const minutes = totalMinutes % 60;

        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        return `${minutes}m`;
    },

    // Format time short (e.g., "08:15")
    formatTimeShort(isoString) {
        if (!isoString) return '—';

        try {
            // Handle both "08:15" and "2025-12-15T08:15:00+01:00" formats
            if (isoString.includes('T')) {
                const date = new Date(isoString);
                return date.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
            }
            return isoString;
        } catch (e) {
            return isoString;
        }
    },

    // Format duration in minutes to human readable
    formatDuration(minutes) {
        if (minutes == null || isNaN(minutes)) return '—';

        const hours = Math.floor(minutes / 60);
        const mins = Math.round(minutes % 60);

        if (hours > 0) {
            return `${hours}h ${mins}m`;
        }
        return `${mins}m`;
    },

    // Format number with unit
    formatNumber(value, decimals = 2, unit = '') {
        if (value == null || isNaN(value)) return '—';
        return `${parseFloat(value).toFixed(decimals)}${unit ? ' ' + unit : ''}`;
    },

    // Format power (W/kW)
    formatPower(watts) {
        if (watts == null || isNaN(watts)) return '—';

        if (Math.abs(watts) >= 1000) {
            return `${(watts / 1000).toFixed(2)} kW`;
        }
        return `${Math.round(watts)} W`;
    },

    // Format energy (Wh/kWh)
    formatEnergy(wh) {
        if (wh == null || isNaN(wh)) return '—';

        if (Math.abs(wh) >= 1000) {
            return `${(wh / 1000).toFixed(2)} kWh`;
        }
        return `${Math.round(wh)} Wh`;
    },

    // Format currency
    formatCurrency(value, currency = 'EUR') {
        if (value == null || isNaN(value)) return '—';

        return new Intl.NumberFormat('de-DE', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    },

    // Format percentage
    formatPercent(value, decimals = 1) {
        if (value == null || isNaN(value)) return '—';
        return `${parseFloat(value).toFixed(decimals)}%`;
    },

    // Get weather icon based on condition
    getWeatherIcon(condition) {
        if (!condition) return '?';

        const cond = condition.toLowerCase();

        if (cond.includes('clear') || cond.includes('sunny')) return 'sunny';
        if (cond.includes('partly') || cond.includes('few clouds')) return 'partly_cloudy_day';
        if (cond.includes('cloudy') || cond.includes('overcast')) return 'cloud';
        if (cond.includes('rain') || cond.includes('drizzle')) return 'rainy';
        if (cond.includes('thunder') || cond.includes('storm')) return 'thunderstorm';
        if (cond.includes('snow')) return 'ac_unit';
        if (cond.includes('fog') || cond.includes('mist')) return 'foggy';
        if (cond.includes('wind')) return 'air';
        if (cond.includes('night') || cond.includes('clear-night')) return 'nights_stay';

        return 'wb_sunny';
    },

    // Get weather condition text
    getWeatherCondition(condition) {
        if (!condition) return 'Unbekannt';

        const translations = {
            'sunny': 'Sonnig',
            'clear': 'Klar',
            'clear-night': 'Klare Nacht',
            'partly cloudy': 'Teilweise bewölkt',
            'partlycloudy': 'Teilweise bewölkt',
            'cloudy': 'Bewölkt',
            'overcast': 'Bedeckt',
            'rainy': 'Regnerisch',
            'rain': 'Regen',
            'drizzle': 'Nieselregen',
            'thunder': 'Gewitter',
            'thunderstorm': 'Gewitter',
            'snow': 'Schnee',
            'snowy': 'Schnee',
            'fog': 'Nebel',
            'foggy': 'Neblig',
            'mist': 'Dunst',
            'windy': 'Windig',
            'exceptional': 'Außergewöhnlich'
        };

        const cond = condition.toLowerCase().replace(/-/g, ' ').replace(/_/g, ' ');
        return translations[cond] || condition;
    },

    // Calculate deviation percentage
    calculateDeviation(actual, predicted) {
        if (!predicted || predicted === 0) return null;
        return ((actual - predicted) / predicted) * 100;
    },

    // Get deviation color class
    getDeviationColor(deviation) {
        if (deviation == null) return 'var(--text-muted)';

        const absDeviation = Math.abs(deviation);

        if (absDeviation <= 10) return 'var(--neon-green)';
        if (absDeviation <= 25) return 'var(--neon-yellow)';
        return 'var(--neon-pink)';
    },

    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle function
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func(...args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Get current date string in YYYY-MM-DD format (local timezone)
    getLocalDateString(date = new Date()) {
        return date.getFullYear() + '-' +
            String(date.getMonth() + 1).padStart(2, '0') + '-' +
            String(date.getDate()).padStart(2, '0');
    },

    // Get tomorrow's date string
    getTomorrowDateString() {
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        return this.getLocalDateString(tomorrow);
    },

    // Get day after tomorrow's date string
    getDayAfterTomorrowDateString() {
        const dayAfter = new Date();
        dayAfter.setDate(dayAfter.getDate() + 2);
        return this.getLocalDateString(dayAfter);
    },

    // Parse hour from various time formats
    parseHour(timeString) {
        if (typeof timeString === 'number') return timeString;
        if (!timeString) return null;

        if (timeString.includes('T')) {
            return parseInt(timeString.split('T')[1].split(':')[0]);
        }
        if (timeString.includes(':')) {
            return parseInt(timeString.split(':')[0]);
        }
        return parseInt(timeString);
    }
};

// Export for global access
window.SFMLUtils = SFMLUtils;
