// =================== API SERVICE ===================

/**
 * Base API configuration
 */
const API = {
    baseURL: '',
    
    /**
     * Make API request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise} Response promise
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };
        
        try {
            const response = await fetch(url, defaultOptions);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Request failed');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    /**
     * GET request
     */
    async get(endpoint) {
        // Add cache-busting timestamp to prevent stale data
        const separator = endpoint.includes('?') ? '&' : '?';
        const cacheBuster = `${separator}_t=${Date.now()}`;
        return this.request(endpoint + cacheBuster, { method: 'GET' });
    },
    
    /**
     * POST request
     */
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * Upload file
     */
    async uploadFile(endpoint, file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const url = `${this.baseURL}${endpoint}`;
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        return data;
    }
};

// =================== DATASET API ===================

const DatasetAPI = {
    /**
     * Upload dataset file
     */
    async uploadDataset(file) {
        return API.uploadFile('/api/upload-dataset', file);
    },
    
    /**
     * Load sample dataset
     */
    async loadSampleDataset(datasetName) {
        return API.post('/api/load-sample-dataset', { dataset_name: datasetName });
    },
    
    /**
     * Get dataset information
     */
    async getDatasetInfo() {
        return API.get('/api/get-dataset-info');
    }
};

// =================== PREPROCESSING API ===================

const PreprocessingAPI = {
    /**
     * Handle missing values
     */
    async handleMissingValues(strategy, columns = null) {
        return API.post('/api/preprocess/handle-missing', {
            strategy,
            columns
        });
    },
    
    /**
     * Encode categorical features
     */
    async encodeFeatures(method, columns = null) {
        return API.post('/api/preprocess/encode', {
            method,
            columns
        });
    },
    
    /**
     * Scale features
     */
    async scaleFeatures(method, columns = null) {
        return API.post('/api/preprocess/scale', {
            method,
            columns
        });
    },
    
    /**
     * Remove duplicate rows
     */
    async removeDuplicates() {
        return API.post('/api/preprocess/remove-duplicates', {});
    },
    
    /**
     * Remove columns
     */
    async removeColumns(columns) {
        return API.post('/api/preprocess/remove-columns', {
            columns
        });
    },
    
    /**
     * Get preprocessing log
     */
    async getPreprocessingLog() {
        return API.get('/api/get-preprocessing-log');
    }
};

// =================== TRAINING API ===================

const TrainingAPI = {
    /**
     * Train model
     */
    async trainModel(config) {
        return API.post('/api/train', {
            target: config.target,
            model: config.model,
            test_size: config.testSize,
            tuning_method: config.tuningMethod,
            use_cv: config.useCv
        });
    },
    
    /**
     * Get evaluation results
     */
    async getEvaluation() {
        return API.get('/api/evaluate');
    }
};

// =================== SESSION API ===================

const SessionAPI = {
    /**
     * Reset session
     */
    async resetSession() {
        return API.post('/api/reset', {});
    }
};

// Export API modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        API,
        DatasetAPI,
        PreprocessingAPI,
        TrainingAPI,
        SessionAPI
    };
}
