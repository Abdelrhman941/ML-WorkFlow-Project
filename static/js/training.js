// =================== TRAINING PAGE LOGIC ===================

let featureImportanceChart = null;
let cvScoresChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeTraining();
});

/**
 * Initialize training page
 */
function initializeTraining() {
    loadDatasetColumns();
    initTrainingForm();
}

/**
 * Load dataset columns for target selection
 */
async function loadDatasetColumns() {
    try {
        const result = await DatasetAPI.getDatasetInfo();
        
        if (result.success && result.data) {
            const targetSelect = document.getElementById('targetColumn');
            if (targetSelect) {
                populateSelectOptions(targetSelect, result.data.columns, true);
            }
        }
    } catch (error) {
        showToast('No dataset loaded. Please upload data first.', 'error');
        setTimeout(() => {
            window.location.href = '/';
        }, 2000);
    }
}

/**
 * Initialize training form
 */
function initTrainingForm() {
    const form = document.getElementById('trainingForm');
    if (!form) return;
    
    form.addEventListener('submit', handleTraining);
}

/**
 * Handle training form submission
 */
async function handleTraining(e) {
    e.preventDefault();
    
    if (!validateForm(e.target)) {
        showToast('Please fill in all required fields', 'warning');
        return;
    }
    
    const config = {
        target: document.getElementById('targetColumn').value,
        model: document.getElementById('modelSelect').value,
        testSize: parseFloat(document.getElementById('testSize').value),
        tuningMethod: document.getElementById('tuningMethod').value,
        useCv: document.getElementById('useCrossValidation').value === 'true'
    };
    
    if (!config.target) {
        showToast('Please select a target column', 'warning');
        return;
    }
    
    // Show loading with detailed message
    const tuningText = config.tuningMethod !== 'none' ? ' with hyperparameter tuning' : '';
    const cvText = config.useCv ? ' and cross-validation' : '';
    showLoading(
        `Training ${config.model}${tuningText}${cvText}...`,
        'This may take a few moments'
    );
    
    try {
        const result = await TrainingAPI.trainModel(config);
        hideLoading();
        
        if (result.success) {
            showToast(result.message, 'success');
            displayTrainingResults(result.results);
            
            // Show evaluation button
            const evalBtn = document.getElementById('evaluationBtn');
            if (evalBtn) {
                evalBtn.style.display = 'inline-flex';
            }
        }
    } catch (error) {
        hideLoading();
        handleAPIError(error, 'Training failed');
    }
}

/**
 * Display training results
 */
function displayTrainingResults(results) {
    const resultsCard = document.getElementById('resultsCard');
    if (!resultsCard) return;
    
    resultsCard.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        scrollToElement(resultsCard);
    }, 300);
    
    // Model info
    document.getElementById('resultModelName').textContent = results.model_name;
    document.getElementById('resultTaskType').textContent = results.task_type;
    document.getElementById('resultTarget').textContent = 
        document.getElementById('targetColumn').value;
    
    // Performance metrics
    const trainScore = results.train_score;
    const testScore = results.test_score;
    const cvScores = results.cv_scores || [];
    
    displayScore('trainScore', trainScore);
    displayScore('testScore', testScore);
    
    if (cvScores.length > 0) {
        const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
        const cvStd = Math.sqrt(
            cvScores.reduce((sum, score) => sum + Math.pow(score - cvMean, 2), 0) / cvScores.length
        );
        displayScore('cvScore', cvMean, cvStd);
        displayCVScores(cvScores);
    }
    
    // Best parameters
    if (results.best_params && Object.keys(results.best_params).length > 0) {
        displayBestParameters(results.best_params);
    }
    
    // Feature importance
    if (results.feature_importance && results.feature_importance.length > 0) {
        displayFeatureImportance(results.feature_importance);
    }
}

/**
 * Display score with color coding
 */
function displayScore(elementId, score, std = null) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const percentage = (score * 100).toFixed(2);
    const color = getScoreColor(score);
    
    if (std !== null) {
        const stdPercentage = (std * 100).toFixed(2);
        element.textContent = `${percentage}% ± ${stdPercentage}%`;
    } else {
        element.textContent = `${percentage}%`;
    }
    
    element.style.color = color;
}

/**
 * Display best hyperparameters
 */
function displayBestParameters(params) {
    const section = document.getElementById('bestParamsSection');
    const container = document.getElementById('bestParams');
    
    if (!section || !container) return;
    
    section.style.display = 'block';
    
    let html = '<div class="params-grid">';
    
    Object.entries(params).forEach(([key, value]) => {
        html += `
            <div class="param-item">
                <span class="param-key">${key}</span>
                <span class="param-value">${value}</span>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Display feature importance chart
 */
function displayFeatureImportance(features) {
    const section = document.getElementById('featureImportanceSection');
    const canvas = document.getElementById('featureImportanceChart');
    
    if (!section || !canvas) return;
    
    section.style.display = 'block';
    
    // Destroy existing chart
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    const labels = features.map(f => f.feature);
    const data = features.map(f => f.importance);
    
    featureImportanceChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: data,
                backgroundColor: 'rgba(91, 192, 190, 0.6)',
                borderColor: 'rgba(91, 192, 190, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Importance: ' + context.parsed.x.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b949e'
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#8b949e'
                    }
                }
            }
        }
    });
}

/**
 * Display cross-validation scores
 */
function displayCVScores(scores) {
    const section = document.getElementById('cvScoresSection');
    const canvas = document.getElementById('cvScoresChart');
    
    if (!section || !canvas) return;
    
    section.style.display = 'block';
    
    // Destroy existing chart
    if (cvScoresChart) {
        cvScoresChart.destroy();
    }
    
    const labels = scores.map((_, i) => `Fold ${i + 1}`);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    
    cvScoresChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'CV Scores',
                    data: scores,
                    backgroundColor: 'rgba(91, 192, 190, 0.2)',
                    borderColor: 'rgba(91, 192, 190, 1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Mean Score',
                    data: Array(scores.length).fill(mean),
                    borderColor: 'rgba(255, 193, 7, 1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#8b949e'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b949e',
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b949e'
                    }
                }
            }
        }
    });
}

// Add custom styles
const style = document.createElement('style');
style.textContent = `
    .result-section {
        margin-bottom: var(--spacing-xl);
    }
    
    .result-section h3 {
        margin-bottom: var(--spacing-md);
        padding-bottom: var(--spacing-sm);
        border-bottom: 2px solid var(--primary-color);
    }
    
    .params-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--spacing-md);
    }
    
    .param-item {
        background: var(--bg-tertiary);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .param-key {
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .param-value {
        font-weight: 600;
        color: var(--primary-color);
    }
`;
document.head.appendChild(style);
