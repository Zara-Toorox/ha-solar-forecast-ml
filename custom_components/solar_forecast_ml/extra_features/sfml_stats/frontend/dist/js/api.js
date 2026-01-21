/* ============================================================
   SFML Stats Dashboard - API Client with Caching
   ============================================================ */

// API client with caching and deduplication
const SFMLApi = {
    cache: new Map(),
    pendingRequests: new Map(),
    defaultTTL: 30000, // 30 seconds

    async fetch(endpoint, options = {}) {
        const { ttl = this.defaultTTL, forceRefresh = false } = options;
        const cacheKey = endpoint;

        // Check cache first (unless force refresh)
        if (!forceRefresh && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < ttl) {
                return cached.data;
            }
        }

        // Deduplicate concurrent requests to the same endpoint
        if (this.pendingRequests.has(cacheKey)) {
            return this.pendingRequests.get(cacheKey);
        }

        // Make the request
        const requestPromise = (async () => {
            try {
                const response = await fetch(endpoint);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                const data = await response.json();

                // Cache the result
                this.cache.set(cacheKey, {
                    data,
                    timestamp: Date.now()
                });

                return data;
            } finally {
                this.pendingRequests.delete(cacheKey);
            }
        })();

        this.pendingRequests.set(cacheKey, requestPromise);
        return requestPromise;
    },

    // Convenience methods for common endpoints
    async getSummary(forceRefresh = false) {
        return this.fetch('/api/sfml_stats/summary', { forceRefresh });
    },

    async getSolar(days = 7, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/solar?days=${days}`, { forceRefresh });
    },

    async getPrices(days = 2, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/prices?days=${days}`, { forceRefresh });
    },

    async getEnergyFlow(forceRefresh = false) {
        return this.fetch('/api/sfml_stats/energy_flow', { forceRefresh });
    },

    async getStatistics(forceRefresh = false) {
        return this.fetch('/api/sfml_stats/statistics', { forceRefresh });
    },

    async getBilling(forceRefresh = false) {
        return this.fetch('/api/sfml_stats/billing', { forceRefresh });
    },

    async getPowerSourcesHistory(hours = 24, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/power_sources_history?hours=${hours}`, { forceRefresh, ttl: 60000 });
    },

    async getSolarHistory(days = 30, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/solar_history?days=${days}`, { forceRefresh, ttl: 300000 });
    },

    async getBatteryHistory(hours = 24, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/battery_history?hours=${hours}`, { forceRefresh, ttl: 60000 });
    },

    async getHouseHistory(hours = 24, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/house_history?hours=${hours}`, { forceRefresh, ttl: 60000 });
    },

    async getGridHistory(hours = 24, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/grid_history?hours=${hours}`, { forceRefresh, ttl: 60000 });
    },

    async getWeatherHistory(days = 7, forceRefresh = false) {
        return this.fetch(`/api/sfml_stats/weather_history?days=${days}`, { forceRefresh, ttl: 300000 });
    },

    async getClothingRecommendation(forceRefresh = false) {
        return this.fetch('/api/sfml_stats/clothing_recommendation', { forceRefresh, ttl: 300000 });
    },

    async getForecastComparison(forceRefresh = false) {
        return this.fetch('/api/sfml_stats/forecast_comparison', { forceRefresh, ttl: 300000 });
    },

    // Clear cache (useful for forcing refresh)
    clearCache(endpoint = null) {
        if (endpoint) {
            this.cache.delete(endpoint);
        } else {
            this.cache.clear();
        }
    }
};

// Export for global access
window.SFMLApi = SFMLApi;
