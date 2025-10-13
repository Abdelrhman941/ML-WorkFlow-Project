// =================== HOME PAGE LOGIC ===================

document.addEventListener('DOMContentLoaded', function() {
    initializeHomePage();
});

/**
 * Initialize home page
 */
function initializeHomePage() {
    initFileUpload();
    initSampleDatasets();
    loadInitialStatus();
}

/**
 * Initialize file upload functionality
 */
function initFileUpload() {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!fileUploadArea || !fileInput) return;
    
    // Click to upload
    fileUploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Drag and drop
    fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadArea.style.borderColor = 'var(--primary-color)';
        fileUploadArea.style.background = 'var(--bg-tertiary)';
    });
    
    fileUploadArea.addEventListener('dragleave', () => {
        fileUploadArea.style.borderColor = 'var(--border-color)';
        fileUploadArea.style.background = '';
    });
    
    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadArea.style.borderColor = 'var(--border-color)';
        fileUploadArea.style.background = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

/**
 * Handle file upload
 */
async function handleFileUpload(file) {
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }
    
    // Validate file size (800MB max)
    const maxSize = 800 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File size must be less than 800MB', 'error');
        return;
    }
    
    showLoading('Uploading dataset...', 'Reading and validating file');
    
    try {
        const result = await DatasetAPI.uploadDataset(file);
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            
            // Redirect directly to data exploration
            setTimeout(() => {
                window.location.href = '/data-exploration';
            }, 800);
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Failed to upload dataset');
    }
}

/**
 * Initialize sample dataset buttons
 */
function initSampleDatasets() {
    const sampleButtons = document.querySelectorAll('.sample-btn');
    
    sampleButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const datasetName = this.dataset.dataset;
            await loadSampleDataset(datasetName);
        });
    });
}

/**
 * Load sample dataset
 */
async function loadSampleDataset(datasetName) {
    showLoading(`Loading ${datasetName} dataset...`, 'Preparing sample data');
    
    try {
        const result = await DatasetAPI.loadSampleDataset(datasetName);
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            updateStatusCards(result.overview);
            
            // Redirect to data exploration
            setTimeout(() => {
                window.location.href = '/data-exploration';
            }, 1500);
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Failed to load sample dataset');
    }
}

/**
 * Load initial status
 */
async function loadInitialStatus() {
    try {
        const result = await DatasetAPI.getDatasetInfo();
        
        if (result.success && result.data) {
            updateStatusCards(result.data);
        }
    } catch (error) {
        // No dataset loaded - expected behavior
        console.log('No dataset loaded yet');
    }
}

/**
 * Update status cards on home page
 */
function updateStatusCards(overview) {
    if (!overview) return;
    
    // Update dataset status
    const datasetStatus = document.getElementById('datasetStatus');
    if (datasetStatus) {
        const rows = overview.shape ? overview.shape[0] : 0;
        const cols = overview.shape ? overview.shape[1] : 0;
        datasetStatus.textContent = `${formatNumber(rows)} × ${cols}`;
        datasetStatus.style.color = 'var(--success-color)';
    }
    
    // Update preprocessing status
    const preprocessingStatus = document.getElementById('preprocessingStatus');
    if (preprocessingStatus) {
        const missingCount = overview.missing_values 
            ? Object.values(overview.missing_values).reduce((sum, val) => sum + val, 0)
            : 0;
        const duplicates = overview.duplicates || 0;
        
        if (missingCount === 0 && duplicates === 0) {
            preprocessingStatus.textContent = 'Ready';
            preprocessingStatus.style.color = 'var(--success-color)';
        } else {
            preprocessingStatus.textContent = 'Required';
            preprocessingStatus.style.color = 'var(--warning-color)';
        }
    }
    
    // Animate the cards
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.transform = 'translateY(-4px)';
            setTimeout(() => {
                card.style.transform = '';
            }, 300);
        }, index * 100);
    });
}

/**
 * Create dataset overview HTML
 */
function createDatasetOverview(data) {
    const rows = data.shape[0];
    const cols = data.shape[1];
    const missing = Object.values(data.missing_values || {}).reduce((sum, val) => sum + val, 0);
    const duplicates = data.duplicates || 0;
    
    return `
        <div class="overview-grid">
            ${createInfoCard('Total Rows', formatNumber(rows), 'table', 'primary')}
            ${createInfoCard('Total Columns', cols, 'columns', 'info')}
            ${createInfoCard('Missing Values', formatNumber(missing), 'question-circle', missing > 0 ? 'warning' : 'success')}
            ${createInfoCard('Duplicates', formatNumber(duplicates), 'clone', duplicates > 0 ? 'danger' : 'success')}
        </div>
    `;
}
