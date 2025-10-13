// =================== MAIN APPLICATION LOGIC ===================

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize application
 */
function initializeApp() {
    // Apply saved theme
    const savedTheme = getCurrentTheme();
    setTheme(savedTheme);
    
    // Initialize theme toggle
    initThemeToggle();
    
    // Initialize reset button
    initResetButton();
    
    // Initialize toast close button
    initToastClose();
    
    // Check for dataset on page load
    checkDatasetStatus();
}

/**
 * Initialize theme toggle
 */
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;
    
    const currentTheme = getCurrentTheme();
    
    // Set initial checkbox state (checked = dark mode)
    themeToggle.checked = currentTheme === 'dark';
    
    themeToggle.addEventListener('change', function() {
        const newTheme = this.checked ? 'dark' : 'light';
        setTheme(newTheme);
    });
}

/**
 * Initialize reset button
 */
function initResetButton() {
    const resetBtn = document.getElementById('resetBtn');
    if (!resetBtn) return;
    
    resetBtn.addEventListener('click', async function() {
        showLoading('Resetting session...', 'Clearing all data');
        
        try {
            const result = await SessionAPI.resetSession();
            hideLoading();
            
            if (result.success) {
                showToast(result.message, 'success');
                // Redirect to home after short delay
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            }
        } catch (error) {
            hideLoading();
            showToast('Failed to reset session: ' + error.message, 'error');
        }
    });
}

/**
 * Initialize toast close button
 */
function initToastClose() {
    const toastClose = document.querySelector('.toast-close');
    if (!toastClose) return;
    
    toastClose.addEventListener('click', function() {
        const toast = document.getElementById('toast');
        toast.classList.remove('show');
    });
}

/**
 * Check dataset status
 */
async function checkDatasetStatus() {
    try {
        const result = await DatasetAPI.getDatasetInfo();
        
        if (result.success && result.data) {
            updateDatasetStatus(result.data);
        }
    } catch (error) {
        // Dataset not loaded, ignore error
        console.log('No dataset loaded');
    }
}

/**
 * Update dataset status indicators
 */
function updateDatasetStatus(data) {
    // Update status elements if they exist
    const datasetStatus = document.getElementById('datasetStatus');
    if (datasetStatus) {
        datasetStatus.textContent = `${formatNumber(data.shape[0])} rows`;
        datasetStatus.style.color = 'var(--success-color)';
    }
    
    const preprocessingStatus = document.getElementById('preprocessingStatus');
    if (preprocessingStatus) {
        const missingCount = Object.values(data.missing_values || {})
            .reduce((sum, val) => sum + val, 0);
        
        if (missingCount === 0 && data.duplicates === 0) {
            preprocessingStatus.textContent = 'Clean';
            preprocessingStatus.style.color = 'var(--success-color)';
        } else {
            preprocessingStatus.textContent = 'Needs Work';
            preprocessingStatus.style.color = 'var(--warning-color)';
        }
    }
}

/**
 * Format data for display
 */
function formatDataValue(value, decimals = 4) {
    if (value === null || value === undefined) {
        return '-';
    }
    
    if (typeof value === 'number') {
        if (Number.isInteger(value)) {
            return formatNumber(value);
        }
        return value.toFixed(decimals);
    }
    
    return value;
}

/**
 * Create info card HTML
 */
function createInfoCard(title, value, icon, color = 'primary') {
    return `
        <div class="stat-card">
            <div class="stat-icon" style="background: var(--${color}-color)">
                <i class="fas fa-${icon}"></i>
            </div>
            <div class="stat-info">
                <div class="stat-value">${value}</div>
                <div class="stat-label">${title}</div>
            </div>
        </div>
    `;
}

/**
 * Create metric card HTML
 */
function createMetricCard(label, value, icon, color = 'primary') {
    return `
        <div class="metric-card">
            <div class="metric-icon bg-${color}">
                <i class="fas fa-${icon}"></i>
            </div>
            <div class="metric-info">
                <div class="metric-label">${label}</div>
                <div class="metric-value">${value}</div>
            </div>
        </div>
    `;
}

/**
 * Populate select options
 */
function populateSelectOptions(selectElement, options, includeEmpty = false) {
    selectElement.innerHTML = '';
    
    if (includeEmpty) {
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = 'Select an option...';
        selectElement.appendChild(emptyOption);
    }
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        
        if (typeof option === 'object') {
            optionElement.value = option.value;
            optionElement.textContent = option.label;
        } else {
            optionElement.value = option;
            optionElement.textContent = option;
        }
        
        selectElement.appendChild(optionElement);
    });
}

/**
 * Get selected values from multi-select
 */
function getSelectedValues(selectElement) {
    const selectedOptions = Array.from(selectElement.selectedOptions);
    return selectedOptions.map(option => option.value);
}

/**
 * Create progress bar
 */
function createProgressBar(percentage, label = '') {
    const color = getScoreColor(percentage / 100);
    
    return `
        <div class="progress-container">
            ${label ? `<div class="progress-label">${label}</div>` : ''}
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${percentage}%; background: ${color}"></div>
            </div>
            <div class="progress-value">${percentage.toFixed(1)}%</div>
        </div>
    `;
}

/**
 * Handle API errors
 */
function handleAPIError(error, defaultMessage = 'An error occurred') {
    console.error('API Error:', error);
    const message = error.message || defaultMessage;
    showToast(message, 'error', 5000);
}

/**
 * Confirm action
 */
function confirmAction(message) {
    return confirm(message);
}

/**
 * Animate number counter
 */
function animateNumber(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        
        element.textContent = Math.round(current);
    }, 16);
}

/**
 * Format timestamp
 */
function formatTimestamp(date = new Date()) {
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Scroll to element
 */
function scrollToElement(element, offset = 80) {
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - offset;
    
    window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
    });
}

// Export functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeApp,
        updateDatasetStatus,
        formatDataValue,
        createInfoCard,
        createMetricCard,
        populateSelectOptions,
        getSelectedValues,
        createProgressBar,
        handleAPIError,
        confirmAction,
        animateNumber,
        formatTimestamp,
        scrollToElement
    };
}
