// =================== DATA EXPLORATION PAGE LOGIC ===================

document.addEventListener('DOMContentLoaded', function() {
    initializeDataExploration();
});

/**
 * Initialize data exploration page
 */
function initializeDataExploration() {
    loadDatasetInfo();
    initRefreshButton();
}

/**
 * Initialize refresh button
 */
function initRefreshButton() {
    const refreshBtn = document.getElementById('refreshDataBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadDatasetInfo);
    }
}

/**
 * Load dataset information
 */
async function loadDatasetInfo() {
    showLoading('Loading dataset information...', 'Analyzing data structure');
    
    try {
        const result = await DatasetAPI.getDatasetInfo();
        hideLoading();
        
        console.log('Dataset info result:', result); // Debug log
        
        if (result.success && result.data) {
            console.log('Displaying data...'); // Debug log
            displayDatasetOverview(result.data);
            displayStatistics(result.data);
            displayDataSample(result.data);
            displayMissingValues(result.data);
            displayDataTypes(result.data);
        } else {
            console.error('Result not successful or no data:', result);
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading dataset info:', error);
        hideLoading();
        showEmptyState();
    }
}

/**
 * Show empty state in overview container
 */
function showEmptyState() {
    const overviewContainer = document.getElementById('datasetOverview');
    if (overviewContainer) {
        overviewContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>No dataset loaded</p>
                <a href="/" class="btn btn-primary">Go to Home</a>
            </div>
        `;
    }
}

/**
 * Display dataset overview
 */
function displayDatasetOverview(data) {
    const container = document.getElementById('datasetOverview');
    if (!container) return;
    
    const rows = data.shape[0];
    const cols = data.shape[1];
    const missing = Object.values(data.missing_values || {}).reduce((sum, val) => sum + val, 0);
    const duplicates = data.duplicates || 0;
    const numericCols = data.numeric_columns?.length || 0;
    const categoricalCols = data.categorical_columns?.length || 0;
    
    container.innerHTML = `
        <div class="info-grid">
            <div class="info-item">
                <span class="info-label"><i class="fas fa-table"></i> Rows</span>
                <span class="info-value">${formatNumber(rows)}</span>
            </div>
            <div class="info-item">
                <span class="info-label"><i class="fas fa-columns"></i> Columns</span>
                <span class="info-value">${cols}</span>
            </div>
            <div class="info-item">
                <span class="info-label"><i class="fas fa-hashtag"></i> Numeric</span>
                <span class="info-value">${numericCols}</span>
            </div>
            <div class="info-item">
                <span class="info-label"><i class="fas fa-font"></i> Categorical</span>
                <span class="info-value">${categoricalCols}</span>
            </div>
            <div class="info-item">
                <span class="info-label"><i class="fas fa-question-circle"></i> Missing</span>
                <span class="info-value" style="color: ${missing > 0 ? 'var(--warning-color)' : 'var(--success-color)'}">${formatNumber(missing)}</span>
            </div>
            <div class="info-item">
                <span class="info-label"><i class="fas fa-clone"></i> Duplicates</span>
                <span class="info-value" style="color: ${duplicates > 0 ? 'var(--danger-color)' : 'var(--success-color)'}">${duplicates}</span>
            </div>
        </div>
    `;
}

/**
 * Display statistics
 */
function displayStatistics(data) {
    const container = document.getElementById('statisticsContainer');
    if (!container) return;
    
    // Check if description exists and has data
    if (!data.description || Object.keys(data.description).length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-info-circle"></i>
                <p>No numeric columns for statistical summary</p>
            </div>
        `;
        return;
    }
    
    const stats = data.description;
    const columns = Object.keys(stats);
    
    if (columns.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-info-circle"></i>
                <p>No numeric columns for statistical summary</p>
            </div>
        `;
        return;
    }
    
    // Create statistics table - SHOW ALL COLUMNS with horizontal scroll
    let html = '<div class="table-container" style="overflow-x: auto;"><table>';
    
    // Header with tooltips
    html += '<thead><tr><th>Statistic</th>';
    columns.forEach(col => {
        const tooltip = createColumnTooltip(col, data);
        html += `<th class="column-with-tooltip" data-tooltip="${escapeHtml(tooltip)}">${col}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    html += '<tbody>';
    const metrics = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
    
    metrics.forEach(metric => {
        html += '<tr>';
        html += `<td><strong>${metric}</strong></td>`;
        
        columns.forEach(col => {
            const value = stats[col] && stats[col][metric] !== undefined ? stats[col][metric] : 'N/A';
            html += `<td>${formatDataValue(value, 2)}</td>`;
        });
        
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    
    container.innerHTML = html;
    
    // Add tooltip event listeners
    addColumnTooltipListeners();
}

/**
 * Display data sample
 */
function displayDataSample(data) {
    const container = document.getElementById('dataSampleContainer');
    if (!container) return;
    
    // Check if sample data exists
    if (!data.sample || data.sample.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-table"></i>
                <p>No sample data available</p>
            </div>
        `;
        return;
    }
    
    // Create custom table with all columns and tooltips
    const sampleData = data.sample;
    const columns = Object.keys(sampleData[0]);
    
    if (columns.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-table"></i>
                <p>Dataset has no columns</p>
            </div>
        `;
        return;
    }
    
    let html = '<div style="overflow-x: auto;"><table>';
    
    // Header with tooltips
    html += '<thead><tr>';
    columns.forEach(col => {
        const tooltip = createColumnTooltip(col, data);
        html += `<th class="column-with-tooltip" data-tooltip="${escapeHtml(tooltip)}">${col}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    html += '<tbody>';
    sampleData.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            html += `<td>${formatDataValue(value, 4)}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    
    container.innerHTML = html;
    
    // Add tooltip event listeners
    addColumnTooltipListeners();
}

/**
 * Display missing values analysis
 */
function displayMissingValues(data) {
    const container = document.getElementById('missingValuesChart');
    if (!container || !data.missing_values) return;
    
    const missingData = data.missing_values;
    const columnsWithMissing = Object.entries(missingData)
        .filter(([col, count]) => count > 0)
        .sort((a, b) => b[1] - a[1]);
    
    if (columnsWithMissing.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-check-circle" style="color: var(--success-color)"></i>
                <p>No missing values found! 🎉</p>
            </div>
        `;
        return;
    }
    
    // Create missing values visualization
    const labels = columnsWithMissing.map(([col]) => col);
    const values = columnsWithMissing.map(([, count]) => count);
    const percentages = values.map(count => ((count / data.shape[0]) * 100).toFixed(2));
    
    let html = '<div class="missing-values-list">';
    
    columnsWithMissing.forEach(([col, count], index) => {
        const percentage = percentages[index];
        const color = percentage > 50 ? 'var(--danger-color)' : 
                     percentage > 20 ? 'var(--warning-color)' : 'var(--info-color)';
        
        html += `
            <div class="missing-value-item">
                <div class="missing-info">
                    <strong>${col}</strong>
                    <span>${count} values (${percentage}%)</span>
                </div>
                <div class="missing-bar">
                    <div class="missing-fill" style="width: ${percentage}%; background: ${color}"></div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Display data types
 */
function displayDataTypes(data) {
    const container = document.getElementById('dataTypesContainer');
    if (!container || !data.dtypes) return;
    
    const dtypes = data.dtypes;
    const typeGroups = {};
    
    // Group by data type
    Object.entries(dtypes).forEach(([col, dtype]) => {
        if (!typeGroups[dtype]) {
            typeGroups[dtype] = [];
        }
        typeGroups[dtype].push(col);
    });
    
    let html = '<div class="data-types-grid">';
    
    Object.entries(typeGroups).forEach(([dtype, columns]) => {
        const icon = getDataTypeIcon(dtype);
        const color = getDataTypeColor(dtype);
        
        html += `
            <div class="data-type-card">
                <div class="data-type-header" style="background: ${color}">
                    <i class="fas fa-${icon}"></i>
                    <span>${dtype}</span>
                    <span class="badge">${columns.length}</span>
                </div>
                <div class="data-type-columns">
                    ${columns.slice(0, 5).map(col => `<div class="column-tag">${col}</div>`).join('')}
                    ${columns.length > 5 ? `<div class="column-tag">+${columns.length - 5} more</div>` : ''}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Get icon for data type
 */
function getDataTypeIcon(dtype) {
    if (dtype.includes('int') || dtype.includes('float')) return 'hashtag';
    if (dtype.includes('object') || dtype.includes('string')) return 'font';
    if (dtype.includes('bool')) return 'check-square';
    if (dtype.includes('datetime')) return 'calendar';
    return 'question';
}

/**
 * Get color for data type
 */
function getDataTypeColor(dtype) {
    if (dtype.includes('int') || dtype.includes('float')) return 'var(--info-color)';
    if (dtype.includes('object') || dtype.includes('string')) return 'var(--primary-color)';
    if (dtype.includes('bool')) return 'var(--success-color)';
    if (dtype.includes('datetime')) return 'var(--warning-color)';
    return 'var(--text-muted)';
}

// Add custom styles for data exploration
const style = document.createElement('style');
style.textContent = `
    .missing-values-list {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-md);
    }
    
    .missing-value-item {
        background: var(--bg-tertiary);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
    }
    
    .missing-info {
        display: flex;
        justify-content: space-between;
        margin-bottom: var(--spacing-xs);
    }
    
    .missing-bar {
        height: 8px;
        background: var(--bg-secondary);
        border-radius: var(--radius-sm);
        overflow: hidden;
    }
    
    .missing-fill {
        height: 100%;
        transition: width var(--transition-base);
    }
    
    .data-types-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: var(--spacing-md);
    }
    
    .data-type-card {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    .data-type-header {
        padding: var(--spacing-md);
        color: white;
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
    }
    
    .badge {
        margin-left: auto;
        background: rgba(255,255,255,0.3);
        padding: 0.2rem 0.5rem;
        border-radius: var(--radius-sm);
        font-size: 0.875rem;
    }
    
    .data-type-columns {
        padding: var(--spacing-md);
        display: flex;
        flex-wrap: wrap;
        gap: var(--spacing-xs);
    }
    
    .column-tag {
        background: var(--bg-tertiary);
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-sm);
        font-size: 0.875rem;
    }
    
    /* Column tooltip styles */
    .column-with-tooltip {
        cursor: help;
        position: relative;
    }
    
    .column-with-tooltip:hover {
        background: rgba(91, 192, 190, 0.1);
    }
    
    .column-tooltip {
        position: fixed;
        background: var(--bg-card);
        border: 2px solid var(--primary-color);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        max-width: 350px;
        font-size: 0.875rem;
        line-height: 1.6;
        white-space: pre-wrap;
        font-family: var(--font-mono);
        animation: tooltipFadeIn 0.2s ease-in-out;
    }
    
    @keyframes tooltipFadeIn {
        from {
            opacity: 0;
            transform: translateY(-5px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// =================== TOOLTIP HELPER FUNCTIONS ===================

/**
 * Create tooltip content for a column
 */
function createColumnTooltip(columnName, data) {
    const dtype = data.dtypes[columnName] || 'unknown';
    const uniqueCount = data.unique_counts ? data.unique_counts[columnName] : 'N/A';
    const isNumeric = data.numeric_columns && data.numeric_columns.includes(columnName);
    const isCategorical = data.categorical_columns && data.categorical_columns.includes(columnName);
    const type = isNumeric ? 'Numeric' : (isCategorical ? 'Categorical' : 'Other');
    
    let tooltip = `📊 ${columnName}\n`;
    tooltip += `━━━━━━━━━━━━━━━━\n`;
    tooltip += `Type: ${type}\n`;
    tooltip += `Data Type: ${dtype}\n`;
    tooltip += `Unique Values: ${uniqueCount}\n`;
    
    // Add value counts if available
    if (data.value_counts && data.value_counts[columnName]) {
        tooltip += `\nTop Values:\n`;
        const valueCounts = data.value_counts[columnName];
        const entries = Object.entries(valueCounts).slice(0, 5);
        entries.forEach(([value, count]) => {
            tooltip += `  • ${value}: ${count}\n`;
        });
        if (Object.keys(valueCounts).length > 5) {
            tooltip += `  ... and more\n`;
        }
    }
    
    return tooltip;
}

/**
 * Escape HTML for tooltip attribute
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/"/g, '&quot;');
}

/**
 * Add tooltip event listeners to columns
 */
function addColumnTooltipListeners() {
    const tooltipElements = document.querySelectorAll('.column-with-tooltip');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function(e) {
            showColumnTooltip(e.target);
        });
        
        element.addEventListener('mouseleave', function() {
            hideColumnTooltip();
        });
    });
}

/**
 * Show column tooltip
 */
function showColumnTooltip(element) {
    // Remove any existing tooltip
    hideColumnTooltip();
    
    const tooltipText = element.getAttribute('data-tooltip');
    if (!tooltipText) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'column-tooltip';
    tooltip.id = 'columnTooltip';
    
    // Format the tooltip text with proper line breaks
    const lines = tooltipText.split('\n');
    lines.forEach(line => {
        const p = document.createElement('div');
        p.textContent = line;
        if (line.includes('━')) {
            p.style.opacity = '0.5';
            p.style.fontSize = '0.75rem';
        } else if (line.includes(':')) {
            p.style.marginBottom = '4px';
        }
        tooltip.appendChild(p);
    });
    
    document.body.appendChild(tooltip);
    
    // Position tooltip
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = (rect.bottom + 5) + 'px';
    
    // Adjust if tooltip goes off screen
    const tooltipRect = tooltip.getBoundingClientRect();
    if (tooltipRect.right > window.innerWidth) {
        tooltip.style.left = (window.innerWidth - tooltipRect.width - 10) + 'px';
    }
    if (tooltipRect.bottom > window.innerHeight) {
        tooltip.style.top = (rect.top - tooltipRect.height - 5) + 'px';
    }
}

/**
 * Hide column tooltip
 */
function hideColumnTooltip() {
    const tooltip = document.getElementById('columnTooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

