// =================== PREPROCESSING PAGE LOGIC ===================

document.addEventListener('DOMContentLoaded', function() {
    initializePreprocessing();
});

/**
 * Initialize preprocessing page
 */
function initializePreprocessing() {
    loadDatasetColumns();
    initForms();
    initRemoveDuplicatesButton();
    loadPreprocessingLog();
    updateDatasetInfo();
}

/**
 * Load dataset columns
 */
async function loadDatasetColumns() {
    try {
        const result = await DatasetAPI.getDatasetInfo();
        
        if (result.success && result.data) {
            populateColumnSelects(result.data);
            updateDuplicateInfo(result.data);
        }
    } catch (error) {
        console.error('Failed to load dataset columns:', error);
    }
}

/**
 * Populate column select elements
 */
function populateColumnSelects(data) {
    const allColumns = data.columns || [];
    const numericColumns = data.numeric_columns || [];
    const categoricalColumns = data.categorical_columns || [];
    const scalableColumns = data.scalable_columns || numericColumns; // Use scalable if available
    const missingValues = data.missing_values || {};
    
    // DEBUG: Log what we're receiving from backend
    console.log('📊 populateColumnSelects called with:');
    console.log('  All columns:', allColumns);
    console.log('  Categorical columns:', categoricalColumns);
    console.log('  Numeric columns:', numericColumns);
    console.log('  Scalable columns:', scalableColumns);
    
    // Store dataset info for column notes
    window.datasetInfo = data;
    
    // Missing values columns - show only columns with missing values
    const missingColumns = document.getElementById('missingColumns');
    if (missingColumns) {
        const columnsWithMissing = Object.keys(missingValues).filter(col => missingValues[col] > 0);
        if (columnsWithMissing.length > 0) {
            populateSelectOptions(missingColumns, columnsWithMissing);
        } else {
            missingColumns.innerHTML = '<option value="">No missing values found</option>';
        }
    }
    
    // Encoding columns
    const encodingColumns = document.getElementById('encodingColumns');
    if (encodingColumns) {
        populateSelectOptions(encodingColumns, categoricalColumns);
    }
    
    // Remove columns - show all columns
    const removeColumns = document.getElementById('removeColumns');
    if (removeColumns) {
        populateSelectOptions(removeColumns, allColumns);
    }
    
    // Scaling columns - use scalable_columns (excludes binary/encoded categorical)
    const scalingColumns = document.getElementById('scalingColumns');
    if (scalingColumns) {
        populateSelectOptions(scalingColumns, scalableColumns);
    }
    
    // Add event listeners for column preview
    addColumnPreviewListeners(missingColumns, 'missingColumns');
    addColumnPreviewListeners(encodingColumns, 'encodingColumns');
    addColumnPreviewListeners(removeColumns, 'removeColumns');
    addColumnPreviewListeners(scalingColumns, 'scalingColumns');
    
    // Add hover listeners for column notes
    addColumnHoverListeners(missingColumns);
    addColumnHoverListeners(encodingColumns);
    addColumnHoverListeners(removeColumns);
    addColumnHoverListeners(scalingColumns);
}

/**
 * Update duplicate info
 */
function updateDuplicateInfo(data) {
    const duplicateCount = document.getElementById('duplicateCount');
    if (duplicateCount) {
        duplicateCount.textContent = formatNumber(data.duplicates || 0);
        duplicateCount.style.color = data.duplicates > 0 ? 'var(--danger-color)' : 'var(--success-color)';
    }
}

/**
 * Initialize forms
 */
function initForms() {
    // Missing values form
    const missingForm = document.getElementById('missingValuesForm');
    if (missingForm) {
        missingForm.addEventListener('submit', handleMissingValues);
    }
    
    // Remove columns form
    const removeColumnsForm = document.getElementById('removeColumnsForm');
    if (removeColumnsForm) {
        removeColumnsForm.addEventListener('submit', handleRemoveColumns);
    }
    
    // Encoding form
    const encodingForm = document.getElementById('encodingForm');
    if (encodingForm) {
        encodingForm.addEventListener('submit', handleEncoding);
    }
    
    // Scaling form
    const scalingForm = document.getElementById('scalingForm');
    if (scalingForm) {
        scalingForm.addEventListener('submit', handleScaling);
    }
}

/**
 * Handle missing values form submission
 */
async function handleMissingValues(e) {
    e.preventDefault();
    
    const strategy = document.getElementById('missingStrategy').value;
    const columnsSelect = document.getElementById('missingColumns');
    const columns = getSelectedValues(columnsSelect);
    
    showLoading('Handling missing values...', `Using ${strategy} strategy`);
    
    try {
        const result = await PreprocessingAPI.handleMissingValues(
            strategy,
            columns.length > 0 ? columns : null
        );
        
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            loadPreprocessingLog();
            updateDatasetInfo();
            loadDatasetColumns(); // Refresh columns
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Failed to handle missing values');
    }
}

/**
 * Handle remove columns form submission
 */
async function handleRemoveColumns(e) {
    e.preventDefault();
    console.log('🔵 handleRemoveColumns called');
    
    const columnsSelect = document.getElementById('removeColumns');
    const columns = getSelectedValues(columnsSelect);
    
    console.log('🔵 Selected columns:', columns);
    
    if (columns.length === 0) {
        showToast('Please select at least one column to remove', 'warning');
        return;
    }
    
    console.log('🔵 About to show confirmation modal...');
    
    // Show beautiful confirmation modal
    try {
        const confirmed = await showConfirmModal(
            'Remove Columns',
            `You are about to remove <strong>${columns.length}</strong> column(s) from your dataset.`,
            `<div class="alert alert-warning mt-2">
                <i class="fas fa-exclamation-triangle"></i> 
                <strong>Columns to be removed:</strong>
                <div class="mt-2">
                    ${columns.map(col => `<span class="badge badge-danger mr-1">${col}</span>`).join('')}
                </div>
            </div>
            <p class="text-muted mt-3"><i class="fas fa-info-circle"></i> This action cannot be undone.</p>`,
            'danger'
        );
        
        console.log('🔵 Modal result - confirmed:', confirmed);
        
        if (!confirmed) {
            console.log('❌ User cancelled removal');
            return;
        }
        
        console.log('✅ User confirmed removal of:', columns);
        showLoading('Removing columns...', `Removing ${columns.length} column(s)`);
        
        console.log('🔵 Calling API...');
        const result = await PreprocessingAPI.removeColumns(columns);
        console.log('✅ API response:', result);
        
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            await loadPreprocessingLog();
            
            // Update all column lists with fresh data
            if (result.categorical_columns !== undefined) {
                // Update window.datasetInfo
                if (!window.datasetInfo) window.datasetInfo = {};
                window.datasetInfo.numeric_columns = result.numeric_columns;
                window.datasetInfo.categorical_columns = result.categorical_columns;
                window.datasetInfo.columns = result.columns;
                
                // Update all dropdowns
                const removeColumnsSelect = document.getElementById('removeColumns');
                if (removeColumnsSelect) {
                    populateSelectOptions(removeColumnsSelect, result.columns);
                    addColumnHoverListeners(removeColumnsSelect);
                }
                
                const encodingColumns = document.getElementById('encodingColumns');
                if (encodingColumns) {
                    populateSelectOptions(encodingColumns, result.categorical_columns);
                    addColumnHoverListeners(encodingColumns);
                }
                
                const scalingColumns = document.getElementById('scalingColumns');
                if (scalingColumns) {
                    populateSelectOptions(scalingColumns, result.numeric_columns);
                    addColumnHoverListeners(scalingColumns);
                }
            }
            
            await updateDatasetInfo();
        }
    } catch (error) {
        console.error('❌ Error in handleRemoveColumns:', error);
        hideLoading();
        handleAPIError(error, 'Failed to remove columns');
    }
}

/**
 * Handle encoding form submission
 */
async function handleEncoding(e) {
    e.preventDefault();
    
    const methodValue = document.getElementById('encodingMethod').value;
    const columnsSelect = document.getElementById('encodingColumns');
    const columns = getSelectedValues(columnsSelect);
    
    if (columns.length === 0) {
        showToast('Please select at least one column to encode', 'warning');
        return;
    }
    
    // Map frontend values to backend method names
    const methodMap = {
        'label': 'Label Encoding',
        'onehot': 'One-Hot Encoding'
    };
    
    const method = methodMap[methodValue] || methodValue;
    
    showLoading('Encoding features...', `Using ${method}`);
    
    try {
        const result = await PreprocessingAPI.encodeFeatures(method, columns);
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            await loadPreprocessingLog();
            
            // Use the returned column lists from the backend response
            // This is more reliable than fetching again
            if (result.categorical_columns) {
                // Update window.datasetInfo first so column notes work
                if (!window.datasetInfo) window.datasetInfo = {};
                window.datasetInfo.numeric_columns = result.numeric_columns;
                window.datasetInfo.categorical_columns = result.categorical_columns;
                window.datasetInfo.scalable_columns = result.scalable_columns;
                window.datasetInfo.columns = result.columns;
                
                // Update the dropdowns directly with fresh data
                const encodingColumns = document.getElementById('encodingColumns');
                const scalingColumns = document.getElementById('scalingColumns');
                
                if (encodingColumns) {
                    populateSelectOptions(encodingColumns, result.categorical_columns);
                    addColumnHoverListeners(encodingColumns);
                }
                
                if (scalingColumns) {
                    // Use scalable_columns (excludes binary encoded columns)
                    const columnsToScale = result.scalable_columns || result.numeric_columns;
                    populateSelectOptions(scalingColumns, columnsToScale);
                    addColumnHoverListeners(scalingColumns);
                }
            }
            
            // Update the dataset info display (row count, etc.)
            await updateDatasetInfo();
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Failed to encode features');
    }
}

/**
 * Handle scaling form submission
 */
async function handleScaling(e) {
    e.preventDefault();
    
    const methodValue = document.getElementById('scalingMethod').value;
    const columnsSelect = document.getElementById('scalingColumns');
    const columns = getSelectedValues(columnsSelect);
    
    // Map frontend values to backend scaler names
    const methodMap = {
        'standard': 'StandardScaler',
        'minmax': 'MinMaxScaler',
        'robust': 'RobustScaler'
    };
    
    const method = methodMap[methodValue] || methodValue;
    
    showLoading('Scaling features...', `Using ${method}`);
    
    try {
        const result = await PreprocessingAPI.scaleFeatures(
            method,
            columns.length > 0 ? columns : null
        );
        
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            loadPreprocessingLog();
            updateDatasetInfo();
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Failed to scale features');
    }
}

/**
 * Initialize remove duplicates button
 */
function initRemoveDuplicatesButton() {
    const btn = document.getElementById('removeDuplicatesBtn');
    if (btn) {
        btn.addEventListener('click', handleRemoveDuplicates);
    }
}

/**
 * Handle remove duplicates
 */
async function handleRemoveDuplicates() {
    showLoading('Removing duplicates...', 'Analyzing dataset');
    
    try {
        const result = await PreprocessingAPI.removeDuplicates();
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            loadPreprocessingLog();
            updateDatasetInfo();
            loadDatasetColumns(); // Refresh duplicate count
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Failed to remove duplicates');
    }
}

/**
 * Load preprocessing log
 */
async function loadPreprocessingLog() {
    try {
        const result = await PreprocessingAPI.getPreprocessingLog();
        
        if (result.success) {
            displayPreprocessingLog(result.log);
        }
    } catch (error) {
        console.error('Failed to load preprocessing log:', error);
    }
}

/**
 * Display preprocessing log
 */
function displayPreprocessingLog(log) {
    const container = document.getElementById('preprocessingLog');
    if (!container) return;
    
    if (!log || log.length === 0) {
        container.innerHTML = '<p class="text-muted">No preprocessing steps performed yet</p>';
        return;
    }
    
    const html = log.map(entry => {
        return `<div class="log-entry">${entry}</div>`;
    }).join('');
    
    container.innerHTML = html;
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

/**
 * Update dataset info
 */
async function updateDatasetInfo() {
    try {
        const result = await DatasetAPI.getDatasetInfo();
        
        if (result.success && result.data) {
            const data = result.data;
            
            // Update info displays
            const currentRows = document.getElementById('currentRows');
            if (currentRows) {
                currentRows.textContent = formatNumber(data.shape[0]);
            }
            
            const currentCols = document.getElementById('currentCols');
            if (currentCols) {
                currentCols.textContent = data.shape[1];
            }
            
            const currentMissing = document.getElementById('currentMissing');
            if (currentMissing) {
                const missing = Object.values(data.missing_values || {})
                    .reduce((sum, val) => sum + val, 0);
                currentMissing.textContent = formatNumber(missing);
                currentMissing.style.color = missing > 0 ? 'var(--danger-color)' : 'var(--success-color)';
            }
            
            const currentDuplicates = document.getElementById('currentDuplicates');
            if (currentDuplicates) {
                currentDuplicates.textContent = formatNumber(data.duplicates || 0);
                currentDuplicates.style.color = data.duplicates > 0 ? 'var(--danger-color)' : 'var(--success-color)';
            }
        }
    } catch (error) {
        console.error('Failed to update dataset info:', error);
    }
}

// Add custom styles
const style = document.createElement('style');
style.textContent = `
    .log-entry {
        padding: var(--spacing-xs);
        margin-bottom: var(--spacing-xs);
        border-left: 3px solid var(--success-color);
        background: rgba(40, 167, 69, 0.1);
        border-radius: var(--radius-sm);
        font-size: 0.875rem;
    }
    
    .duplicate-info {
        background: var(--bg-tertiary);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        margin-bottom: var(--spacing-md);
    }
    
    .form-actions {
        margin-top: var(--spacing-lg);
    }
    
    .column-preview {
        margin-top: var(--spacing-lg);
        padding: var(--spacing-md);
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
    }
    
    .preview-table {
        width: 100%;
        overflow-x: auto;
        margin-top: var(--spacing-md);
    }
    
    .preview-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .preview-table th,
    .preview-table td {
        padding: var(--spacing-sm);
        text-align: left;
        border: 1px solid var(--border-color);
    }
    
    .preview-table th {
        background: var(--bg-tertiary);
        font-weight: 600;
    }
    
    .preview-table .selected-column {
        background: rgba(91, 192, 190, 0.2);
        box-shadow: inset 0 0 0 2px var(--primary-color);
    }
`;
document.head.appendChild(style);

/**
 * Add event listeners for column preview
 */
function addColumnPreviewListeners(selectElement, selectId) {
    if (!selectElement) return;
    // Previously this triggered a visual preview table on every selection change.
    // To avoid sudden UI changes when the mouse moves, we no longer auto-show
    // the big preview on change. Keep the handler minimal so selections behave
    // normally but don't open the preview.
    selectElement.addEventListener('change', function() {
        // no-op: selection stored but we intentionally do not show the preview
        // to prevent flicker when users move the mouse between options.
    });
}

/**
 * Show column preview table
 */
async function showColumnPreview(selectedColumns) {
    try {
        const result = await DatasetAPI.getDatasetInfo();
        
        if (result.success && result.data && result.data.head) {
            displayPreviewTable(result.data.head, result.data.columns, selectedColumns);
        }
    } catch (error) {
        console.error('Failed to load preview:', error);
    }
}

/**
 * Display preview table
 */
function displayPreviewTable(headData, allColumns, selectedColumns) {
    // Find or create preview container
    let previewContainer = document.getElementById('columnPreview');
    
    if (!previewContainer) {
        // Create new preview container after preprocessing log
        const logCard = document.querySelector('.log-card');
        if (logCard) {
            previewContainer = document.createElement('div');
            previewContainer.id = 'columnPreview';
            previewContainer.className = 'column-preview';
            logCard.parentNode.insertBefore(previewContainer, logCard.nextSibling);
        } else {
            return;
        }
    }
    
    // Build table HTML
    let tableHTML = `
        <h4><i class="fas fa-table"></i> Data Preview (First 5 Rows)</h4>
        <div class="preview-table">
            <table>
                <thead>
                    <tr>
    `;
    
    // Table headers
    allColumns.forEach(col => {
        const isSelected = selectedColumns.includes(col);
        tableHTML += `<th class="${isSelected ? 'selected-column' : ''}">${col}</th>`;
    });
    
    tableHTML += `
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Table rows
    for (let i = 0; i < Math.min(5, headData.length); i++) {
        tableHTML += '<tr>';
        allColumns.forEach(col => {
            const isSelected = selectedColumns.includes(col);
            const value = headData[i][col] !== null && headData[i][col] !== undefined 
                ? headData[i][col] 
                : '<em style="opacity: 0.5;">null</em>';
            tableHTML += `<td class="${isSelected ? 'selected-column' : ''}">${value}</td>`;
        });
        tableHTML += '</tr>';
    }
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    previewContainer.innerHTML = tableHTML;
    
    // Scroll to preview
    setTimeout(() => {
        previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

/**
 * Add hover event listeners to show column notes
 */
function addColumnHoverListeners(selectElement) {
    if (!selectElement) return;

    // Show popover only on click. This avoids rapid popover changes while
    // users navigate options with the mouse. Click is also mobile-friendly.
    selectElement.addEventListener('click', function(e) {
        // Safari/Chrome may not dispatch click directly on option for some
        // select types; find the selected option if necessary.
        let opt = e.target;
        if (opt.tagName !== 'OPTION') {
            // try to get the currently selected option
            opt = selectElement.options[selectElement.selectedIndex];
        }

        if (opt && opt.value) {
            showColumnNotes(opt.value, opt);
        }
    });
}

/**
 * Show column notes with preview and statistics
 */
function showColumnNotes(columnName, targetElement) {
    if (!window.datasetInfo) return;
    
    const data = window.datasetInfo;
    const popover = document.getElementById('columnNotesPopover');
    const columnNameEl = document.getElementById('popoverColumnName');
    const content = document.getElementById('columnNotesContent');
    
    if (!popover || !columnNameEl || !content) return;
    
    // Update column name
    columnNameEl.textContent = columnName;
    
    // Get column information
    const dtype = data.dtypes ? data.dtypes[columnName] : 'unknown';
    const uniqueCount = data.unique_counts ? data.unique_counts[columnName] : 0;
    const missingCount = data.missing_values ? (data.missing_values[columnName] || 0) : 0;
    const valueCounts = data.value_counts ? data.value_counts[columnName] : null;
    const columnRange = data.column_ranges ? data.column_ranges[columnName] : null;
    
    // We intentionally do NOT include the first-5 preview in the popover to
    // reduce visual noise. The popover will only show concise statistics and
    // value distribution which is what users need when clicking a column.
    let html = '';
    
    // Build statistics (only)
    html += '<div class="column-stats">';
    html += '<h4><i class="fas fa-chart-bar"></i> Statistics</h4>';
    html += '<ul>';
    html += `<li><strong>Type:</strong> <span>${dtype}</span></li>`;
    html += `<li><strong>Unique:</strong> <span>${formatNumber(uniqueCount)}</span></li>`;
    
    if (missingCount > 0) {
        html += `<li><strong>Missing:</strong> <span style="color: var(--danger-color);">${formatNumber(missingCount)}</span></li>`;
    }
    
    // Show min/max range for numeric columns (helpful for scaling decisions)
    if (columnRange && columnRange.min !== null && columnRange.max !== null) {
        html += `<li><strong>Range:</strong> <span style="color: var(--info-color);">${formatNumber(columnRange.min, 2)} → ${formatNumber(columnRange.max, 2)}</span></li>`;
    }
    
    // Show value counts for categorical or low-cardinality columns
    if (valueCounts) {
        html += '<li><strong>Distribution:</strong></li>';
        html += '<ul class="value-counts">';
        for (const [value, count] of Object.entries(valueCounts)) {
            html += `<li>${value} (${formatNumber(count)})</li>`;
        }
        html += '</ul>';
    }
    
    html += '</ul></div>';
    
    // Set content
    content.innerHTML = html;
    
    // Position popover near the select element
    positionPopover(popover, targetElement);
    
    // Show the popover
    popover.style.display = 'block';
    
    // Add event listener to keep popover open when hovering over it
    popover.addEventListener('mouseenter', function() {
        clearTimeout(popover.hideTimeout);
    });
    
    popover.addEventListener('mouseleave', function() {
        popover.hideTimeout = setTimeout(() => {
            hideColumnNotes();
        }, 300);
    });
}

/**
 * Position popover near the target element
 */
function positionPopover(popover, targetElement) {
    if (!targetElement) return;
    
    // Get the select element (parent of option)
    const selectElement = targetElement.closest('select');
    if (!selectElement) return;
    
    const selectRect = selectElement.getBoundingClientRect();
    const popoverWidth = 400;
    const popoverMaxHeight = 400;
    
    // Calculate position
    let left = selectRect.right + 20; // 20px gap from select
    let top = selectRect.top;
    
    // Check if popover would go off-screen on the right
    if (left + popoverWidth > window.innerWidth) {
        // Position to the left of select instead
        left = selectRect.left - popoverWidth - 20;
    }
    
    // Check if still off-screen (very narrow viewport)
    if (left < 10) {
        left = 10;
    }
    
    // Adjust vertical position if needed
    if (top + popoverMaxHeight > window.innerHeight) {
        top = Math.max(10, window.innerHeight - popoverMaxHeight - 10);
    }
    
    // Apply position
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
}

/**
 * Hide column notes
 */
function hideColumnNotes() {
    const popover = document.getElementById('columnNotesPopover');
    if (popover) {
        popover.style.display = 'none';
    }
}

// Make hideColumnNotes globally accessible for the close button
window.hideColumnNotes = hideColumnNotes;
