// =================== UTILITY FUNCTIONS ===================

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type of notification (success, error, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.getElementById('toast');
    const toastMessage = toast.querySelector('.toast-message');
    const toastIcon = toast.querySelector('.toast-icon i');
    
    // Set message
    toastMessage.textContent = message;
    
    // Remove previous type classes
    toast.className = 'toast';
    toast.classList.add(type);
    
    // Set icon based on type
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toastIcon.className = `fas ${icons[type] || icons.info}`;
    
    // Show toast
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Hide toast after duration
    setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

/**
 * Show loading overlay
 * @param {string} text - Loading text to display
 * @param {string} subtext - Optional subtext to display
 */
function showLoading(text = 'Processing...', subtext = 'Please wait') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = overlay.querySelector('.loading-text');
    const loadingSubtext = overlay.querySelector('.loading-subtext');
    
    loadingText.textContent = text;
    loadingSubtext.textContent = subtext;
    overlay.classList.add('active');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('active');
}

/**
 * Show confirmation modal
 * @param {string} title - Modal title
 * @param {string} message - Main message
 * @param {string} details - Additional details (HTML)
 * @param {string} type - Type (danger, warning, info)
 * @returns {Promise<boolean>} True if confirmed, false if cancelled
 */
function showConfirmModal(title, message, details = '', type = 'warning') {
    return new Promise((resolve) => {
        console.log('🟡 showConfirmModal called');
        let resolved = false;
        
        // Function to close modal
        const closeModal = (confirmed) => {
            console.log('🟡 closeModal called with:', confirmed);
            if (!resolved) {
                resolved = true;
                const modal = document.getElementById('confirmModal');
                const backdrop = document.querySelector('.modal-backdrop-custom');
                
                if (modal) {
                    modal.classList.remove('show');
                }
                
                if (backdrop) {
                    backdrop.classList.remove('show');
                }
                
                setTimeout(() => {
                    if (modal) {
                        modal.remove();
                    }
                    if (backdrop) {
                        backdrop.remove();
                    }
                    console.log('🟡 Resolving promise with:', confirmed);
                    resolve(confirmed);
                }, 300);
            }
        };
        
        // Create modal HTML with beautiful card design
        const modalHTML = `
            <div class="modal-backdrop-custom" id="confirmModalBackdrop"></div>
            <div class="confirm-modal-card" id="confirmModal">
                <div class="card-content-custom">
                    <p class="card-heading-custom">${title}</p>
                    <p class="card-description-custom">${message}</p>
                    ${details}
                </div>
                <div class="card-button-wrapper-custom">
                    <button class="card-button-custom secondary-custom" id="confirmModalCancel">
                        <i class="fas fa-times"></i> Cancel
                    </button>
                    <button class="card-button-custom primary-custom" id="confirmBtn">
                        <i class="fas fa-check"></i> Confirm
                    </button>
                </div>
                <button class="exit-button-custom" id="confirmModalClose">
                    <svg height="20px" viewBox="0 0 384 512">
                        <path d="M342.6 150.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L192 210.7 86.6 105.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L146.7 256 41.4 361.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L192 301.3 297.4 406.6c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L237.3 256 342.6 150.6z"></path>
                    </svg>
                </button>
            </div>
        `;
        
        // Add styles if not already added
        if (!document.getElementById('confirmModalStyles')) {
            const styles = document.createElement('style');
            styles.id = 'confirmModalStyles';
            styles.textContent = `
                .modal-backdrop-custom {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    z-index: 9998;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                }
                .modal-backdrop-custom.show {
                    opacity: 1;
                }
                .confirm-modal-card {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%) scale(0.7);
                    width: 400px;
                    max-width: 90%;
                    height: fit-content;
                    background: rgb(255, 255, 255);
                    border-radius: 20px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 20px;
                    padding: 40px 30px 30px 30px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    z-index: 9999;
                    opacity: 0;
                    transition: all 0.3s ease;
                }
                .confirm-modal-card.show {
                    opacity: 1;
                    transform: translate(-50%, -50%) scale(1);
                }
                .card-content-custom {
                    width: 100%;
                    height: fit-content;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .card-heading-custom {
                    font-size: 24px;
                    font-weight: 700;
                    color: rgb(27, 27, 27);
                    margin: 0;
                }
                .card-description-custom {
                    font-size: 15px;
                    font-weight: 400;
                    color: rgb(102, 102, 102);
                    margin: 0;
                    line-height: 1.5;
                }
                .card-button-wrapper-custom {
                    width: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                }
                .card-button-custom {
                    flex: 1;
                    height: 40px;
                    border-radius: 10px;
                    border: none;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 14px;
                    transition: all 0.2s ease;
                }
                .primary-custom {
                    background-color: rgb(255, 114, 109);
                    color: white;
                }
                .primary-custom:hover {
                    background-color: rgb(255, 73, 66);
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(255, 114, 109, 0.4);
                }
                .secondary-custom {
                    background-color: #ddd;
                    color: #333;
                }
                .secondary-custom:hover {
                    background-color: rgb(197, 197, 197);
                    transform: translateY(-2px);
                }
                .exit-button-custom {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: none;
                    background-color: transparent;
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    cursor: pointer;
                    padding: 5px;
                    border-radius: 5px;
                    transition: background-color 0.2s ease;
                }
                .exit-button-custom:hover {
                    background-color: rgba(0, 0, 0, 0.05);
                }
                .exit-button-custom:hover svg {
                    fill: black;
                }
                .exit-button-custom svg {
                    fill: rgb(175, 175, 175);
                    transition: fill 0.2s ease;
                }
                .alert-warning {
                    background-color: #fff3cd;
                    border: 1px solid #ffc107;
                    border-radius: 10px;
                    padding: 15px;
                    margin-top: 10px;
                }
                .alert-warning strong {
                    color: #856404;
                }
                .badge-danger {
                    background-color: rgb(255, 114, 109);
                    color: white;
                    padding: 4px 10px;
                    border-radius: 12px;
                    font-size: 13px;
                    display: inline-block;
                    margin: 3px;
                }
                .text-muted {
                    color: rgb(102, 102, 102);
                    font-size: 13px;
                    margin-top: 10px;
                }
            `;
            document.head.appendChild(styles);
        }
        
        // Remove existing modal if any
        const existingModal = document.getElementById('confirmModal');
        const existingBackdrop = document.getElementById('confirmModalBackdrop');
        if (existingModal) existingModal.remove();
        if (existingBackdrop) existingBackdrop.remove();
        
        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        console.log('🟡 Modal HTML added to DOM');
        
        const modal = document.getElementById('confirmModal');
        const backdrop = document.getElementById('confirmModalBackdrop');
        const confirmBtn = document.getElementById('confirmBtn');
        const cancelBtn = document.getElementById('confirmModalCancel');
        const closeBtn = document.getElementById('confirmModalClose');
        
        console.log('🟡 Buttons found:', {
            confirmBtn: !!confirmBtn,
            cancelBtn: !!cancelBtn,
            closeBtn: !!closeBtn
        });
        
        // Handle confirm button
        if (confirmBtn) {
            confirmBtn.onclick = function() {
                console.log('🟢 Confirm button clicked');
                closeModal(true);
            };
        }
        
        // Handle cancel button
        if (cancelBtn) {
            cancelBtn.onclick = function() {
                console.log('🔴 Cancel button clicked');
                closeModal(false);
            };
        }
        
        // Handle close button
        if (closeBtn) {
            closeBtn.onclick = function() {
                console.log('🔴 Close button clicked');
                closeModal(false);
            };
        }
        
        // Show modal with animation
        setTimeout(() => {
            console.log('🟡 Showing modal...');
            backdrop.classList.add('show');
            modal.classList.add('show');
            console.log('🟡 Modal visible');
        }, 10);
    });
}

/**
 * Format number to percentage
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted percentage
 */
function formatPercentage(value, decimals = 2) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Format large numbers with commas
 * @param {number} num - Number to format
 * @param {number} decimals - Number of decimal places (optional)
 * @returns {string} Formatted number
 */
function formatNumber(num, decimals) {
    if (typeof num !== 'number') {
        num = parseFloat(num);
    }
    
    if (isNaN(num)) {
        return '0';
    }
    
    // If decimals specified, round to that many places
    if (typeof decimals === 'number') {
        num = num.toFixed(decimals);
    }
    
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Debounce function
 * @param {function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Create HTML table from data
 * @param {Array} data - Array of objects
 * @param {Array} columns - Column names to display
 * @returns {HTMLElement} Table element
 */
function createTable(data, columns = null) {
    if (!data || data.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'empty-state';
        emptyDiv.innerHTML = '<p>No data available</p>';
        return emptyDiv;
    }
    
    // Get columns from first row if not provided
    if (!columns) {
        columns = Object.keys(data[0]);
    }
    
    const table = document.createElement('table');
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            const value = row[col];
            // Format numbers
            if (typeof value === 'number') {
                td.textContent = value.toFixed(4);
            } else {
                td.textContent = value !== null && value !== undefined ? value : '-';
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    return table;
}

/**
 * Create bar chart using Chart.js
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Array} labels - Chart labels
 * @param {Array} data - Chart data
 * @param {string} label - Dataset label
 * @returns {Chart} Chart instance
 */
function createBarChart(canvas, labels, data, label = 'Values') {
    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: 'rgba(91, 192, 190, 0.6)',
                borderColor: 'rgba(91, 192, 190, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

/**
 * Create line chart using Chart.js
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Array} labels - Chart labels
 * @param {Array} data - Chart data
 * @param {string} label - Dataset label
 * @returns {Chart} Chart instance
 */
function createLineChart(canvas, labels, data, label = 'Values') {
    return new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: 'rgba(91, 192, 190, 0.2)',
                borderColor: 'rgba(91, 192, 190, 1)',
                borderWidth: 2,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

/**
 * Validate form inputs
 * @param {HTMLFormElement} form - Form element to validate
 * @returns {boolean} True if valid
 */
function validateForm(form) {
    const inputs = form.querySelectorAll('[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value) {
            input.style.borderColor = 'var(--danger-color)';
            isValid = false;
        } else {
            input.style.borderColor = 'var(--border-color)';
        }
    });
    
    return isValid;
}

/**
 * Get score color based on value
 * @param {number} score - Score value (0-1)
 * @returns {string} Color value
 */
function getScoreColor(score) {
    if (score >= 0.9) return '#28a745'; // Excellent - Green
    if (score >= 0.8) return '#5bc0be'; // Good - Primary
    if (score >= 0.7) return '#17a2b8'; // Fair - Info
    if (score >= 0.6) return '#ffc107'; // Warning - Yellow
    return '#dc3545'; // Poor - Red
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success', 2000);
    } catch (err) {
        showToast('Failed to copy', 'error', 2000);
    }
}

/**
 * Download data as JSON file
 * @param {Object} data - Data to download
 * @param {string} filename - File name
 */
function downloadJSON(data, filename = 'data.json') {
    const dataStr = JSON.stringify(data, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Parse CSV string to array
 * @param {string} csv - CSV string
 * @returns {Array} Parsed data
 */
function parseCSV(csv) {
    const lines = csv.split('\n');
    const headers = lines[0].split(',');
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header.trim()] = values[index] ? values[index].trim() : null;
        });
        data.push(row);
    }
    
    return data;
}

/**
 * Get current theme
 * @returns {string} Theme name
 */
function getCurrentTheme() {
    return localStorage.getItem('theme') || 'dark';
}

/**
 * Set theme
 * @param {string} theme - Theme name (light/dark)
 */
function setTheme(theme) {
    if (theme === 'light') {
        document.body.classList.add('light-theme');
    } else {
        document.body.classList.remove('light-theme');
    }
    localStorage.setItem('theme', theme);
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showToast,
        showLoading,
        hideLoading,
        formatPercentage,
        formatNumber,
        debounce,
        createTable,
        createBarChart,
        createLineChart,
        validateForm,
        getScoreColor,
        copyToClipboard,
        downloadJSON,
        parseCSV,
        getCurrentTheme,
        setTheme
    };
}
