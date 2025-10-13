// =================== EVALUATION PAGE LOGIC ===================

let cvChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeEvaluation();
});

/**
 * Initialize evaluation page
 */
function initializeEvaluation() {
    loadEvaluationResults();
    initRefreshButton();
    initExportButtons();
}

/**
 * Initialize refresh button
 */
function initRefreshButton() {
    const refreshBtn = document.getElementById('refreshEvalBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadEvaluationResults);
    }
}

/**
 * Initialize export buttons
 */
function initExportButtons() {
    const exportResultsBtn = document.getElementById('exportResultsBtn');
    if (exportResultsBtn) {
        exportResultsBtn.addEventListener('click', exportResults);
    }
    
    const exportReportBtn = document.getElementById('exportReportBtn');
    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', generateReport);
    }
}

/**
 * Load evaluation results
 */
async function loadEvaluationResults() {
    showLoading('Loading evaluation results...', 'Analyzing model performance');
    
    try {
        const result = await TrainingAPI.getEvaluation();
        hideLoading();
        
        if (result.success && result.results) {
            displayEvaluationResults(result.results);
        }
    } catch (error) {
        hideLoading();
        
        // Show empty state
        const container = document.getElementById('evaluationResults');
        if (container) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-robot"></i>
                    <p>No model trained yet</p>
                    <a href="/training" class="btn btn-primary">Go to Training</a>
                </div>
            `;
        }
    }
}

/**
 * Display evaluation results
 */
function displayEvaluationResults(results) {
    // Hide empty state and show cards
    const container = document.getElementById('evaluationResults');
    const metricsCard = document.getElementById('metricsCard');
    const insightsCard = document.getElementById('insightsCard');
    const exportCard = document.getElementById('exportCard');
    
    if (container) {
        container.style.display = 'none';
    }
    
    if (metricsCard) {
        metricsCard.style.display = 'block';
    }
    
    if (insightsCard) {
        insightsCard.style.display = 'block';
    }
    
    if (exportCard) {
        exportCard.style.display = 'block';
    }
    
    // Display model information
    displayModelInfo(results);
    
    // Display performance scores
    displayPerformanceScores(results);
    
    // Display CV details if available
    if (results.cv_scores && results.cv_scores.length > 0) {
        displayCVDetails(results.cv_scores);
    }
    
    // Generate insights
    generateInsights(results);
}

/**
 * Display model information
 */
function displayModelInfo(results) {
    const modelName = document.getElementById('evalModelName');
    const taskType = document.getElementById('evalTaskType');
    
    if (modelName) {
        modelName.textContent = results.model_name;
    }
    
    if (taskType) {
        const taskBadge = `<span class="badge ${results.task_type === 'classification' ? 'bg-primary' : 'bg-info'}">${results.task_type}</span>`;
        taskType.innerHTML = taskBadge;
    }
}

/**
 * Display performance scores
 */
function displayPerformanceScores(results) {
    // Train score
    displayScoreWithBar('evalTrainScore', 'trainScoreBar', results.train_score);
    
    // Test score
    displayScoreWithBar('evalTestScore', 'testScoreBar', results.test_score);
    
    // CV score
    if (results.cv_mean !== null) {
        const cvScoreCard = document.getElementById('cvScoreCard');
        if (cvScoreCard) {
            cvScoreCard.style.display = 'block';
        }
        
        const cvScore = document.getElementById('evalCvScore');
        if (cvScore) {
            const mean = (results.cv_mean * 100).toFixed(2);
            const std = (results.cv_std * 100).toFixed(2);
            cvScore.textContent = `${mean}% ± ${std}%`;
            cvScore.style.color = getScoreColor(results.cv_mean);
        }
        
        const cvScoreBar = document.getElementById('cvScoreBar');
        if (cvScoreBar) {
            cvScoreBar.style.width = `${results.cv_mean * 100}%`;
            cvScoreBar.style.background = getScoreColor(results.cv_mean);
        }
    }
}

/**
 * Display score with progress bar
 */
function displayScoreWithBar(scoreId, barId, score) {
    const scoreElement = document.getElementById(scoreId);
    const barElement = document.getElementById(barId);
    
    if (scoreElement) {
        const percentage = (score * 100).toFixed(2);
        scoreElement.textContent = `${percentage}%`;
        scoreElement.style.color = getScoreColor(score);
    }
    
    if (barElement) {
        setTimeout(() => {
            barElement.style.width = `${score * 100}%`;
            barElement.style.background = getScoreColor(score);
        }, 100);
    }
}

/**
 * Display CV details chart
 */
function displayCVDetails(cvScores) {
    const cvDetailsGroup = document.getElementById('cvDetailsGroup');
    const canvas = document.getElementById('cvDetailsChart');
    
    if (!cvDetailsGroup || !canvas) return;
    
    cvDetailsGroup.style.display = 'block';
    
    // Destroy existing chart
    if (cvChart) {
        cvChart.destroy();
    }
    
    const labels = cvScores.map((_, i) => `Fold ${i + 1}`);
    const mean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
    const std = Math.sqrt(
        cvScores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / cvScores.length
    );
    
    cvChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'CV Score',
                data: cvScores,
                backgroundColor: cvScores.map(score => {
                    const alpha = 0.6;
                    const color = getScoreColor(score);
                    return color.replace(')', `, ${alpha})`).replace('rgb', 'rgba');
                }),
                borderColor: cvScores.map(score => getScoreColor(score)),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Score: ' + (context.parsed.y * 100).toFixed(2) + '%';
                        },
                        afterLabel: function(context) {
                            const diff = ((context.parsed.y - mean) * 100).toFixed(2);
                            return `Diff from mean: ${diff > 0 ? '+' : ''}${diff}%`;
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
                        display: false
                    },
                    ticks: {
                        color: '#8b949e'
                    }
                }
            }
        }
    });
    
    // Add mean and std info
    const info = document.createElement('div');
    info.className = 'cv-stats';
    info.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Mean Score:</span>
            <span class="stat-value" style="color: ${getScoreColor(mean)}">${(mean * 100).toFixed(2)}%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Std Deviation:</span>
            <span class="stat-value">${(std * 100).toFixed(2)}%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Min Score:</span>
            <span class="stat-value">${(Math.min(...cvScores) * 100).toFixed(2)}%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Max Score:</span>
            <span class="stat-value">${(Math.max(...cvScores) * 100).toFixed(2)}%</span>
        </div>
    `;
    
    const chartContainer = canvas.parentElement;
    const existingStats = chartContainer.querySelector('.cv-stats');
    if (existingStats) {
        existingStats.remove();
    }
    chartContainer.appendChild(info);
}

/**
 * Generate insights
 */
function generateInsights(results) {
    const container = document.getElementById('modelInsights');
    if (!container) return;
    
    const insights = [];
    
    // Overall performance
    const testScore = results.test_score;
    if (testScore >= 0.9) {
        insights.push({
            type: 'success',
            icon: 'trophy',
            title: 'Excellent Performance',
            message: `Your model achieved an outstanding test score of ${(testScore * 100).toFixed(2)}%!`
        });
    } else if (testScore >= 0.8) {
        insights.push({
            type: 'success',
            icon: 'check-circle',
            title: 'Good Performance',
            message: `Your model shows good performance with a test score of ${(testScore * 100).toFixed(2)}%.`
        });
    } else if (testScore >= 0.7) {
        insights.push({
            type: 'warning',
            icon: 'info-circle',
            title: 'Fair Performance',
            message: `Your model has fair performance (${(testScore * 100).toFixed(2)}%). Consider feature engineering or hyperparameter tuning.`
        });
    } else {
        insights.push({
            type: 'danger',
            icon: 'exclamation-triangle',
            title: 'Needs Improvement',
            message: `Test score of ${(testScore * 100).toFixed(2)}% indicates room for improvement. Try different algorithms or more data preprocessing.`
        });
    }
    
    // Overfitting check
    const trainScore = results.train_score;
    const gap = trainScore - testScore;
    if (gap > 0.1) {
        insights.push({
            type: 'warning',
            icon: 'exclamation-circle',
            title: 'Possible Overfitting',
            message: `Training score (${(trainScore * 100).toFixed(2)}%) is significantly higher than test score (${(testScore * 100).toFixed(2)}%). Consider regularization or more training data.`
        });
    } else if (gap < 0.05) {
        insights.push({
            type: 'info',
            icon: 'balance-scale',
            title: 'Well-Balanced Model',
            message: 'Training and test scores are well-balanced, indicating good generalization.'
        });
    }
    
    // CV consistency
    if (results.cv_scores && results.cv_scores.length > 0) {
        const cvStd = results.cv_std;
        if (cvStd < 0.05) {
            insights.push({
                type: 'success',
                icon: 'chart-line',
                title: 'Consistent Cross-Validation',
                message: `Low standard deviation (${(cvStd * 100).toFixed(2)}%) indicates stable and reliable model performance.`
            });
        } else if (cvStd > 0.1) {
            insights.push({
                type: 'warning',
                icon: 'random',
                title: 'Variable Performance',
                message: `High CV standard deviation (${(cvStd * 100).toFixed(2)}%) suggests inconsistent performance across folds. Consider more data or different validation strategy.`
            });
        }
    }
    
    // Render insights
    let html = '<div class="insights-list">';
    insights.forEach(insight => {
        html += `
            <div class="insight-card insight-${insight.type}">
                <div class="insight-icon">
                    <i class="fas fa-${insight.icon}"></i>
                </div>
                <div class="insight-content">
                    <h4>${insight.title}</h4>
                    <p>${insight.message}</p>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

/**
 * Export results as JSON
 */
function exportResults() {
    // Get evaluation results
    TrainingAPI.getEvaluation()
        .then(result => {
            if (result.success) {
                const timestamp = new Date().toISOString().split('T')[0];
                const filename = `ml-studio-results-${timestamp}.json`;
                downloadJSON(result.results, filename);
                showToast('Results exported successfully!', 'success');
            }
        })
        .catch(error => {
            handleAPIError(error, 'Failed to export results');
        });
}

/**
 * Generate report
 */
function generateReport() {
    showToast('Report generation coming soon!', 'info');
    // TODO: Implement PDF report generation
}

// Add custom styles
const style = document.createElement('style');
style.textContent = `
    .score-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: var(--spacing-lg);
        margin: var(--spacing-lg) 0;
    }
    
    .score-card {
        background: var(--bg-tertiary);
        padding: var(--spacing-lg);
        border-radius: var(--radius-lg);
    }
    
    .score-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-bottom: var(--spacing-xs);
    }
    
    .score-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: var(--spacing-md);
    }
    
    .score-bar {
        height: 8px;
        background: var(--bg-secondary);
        border-radius: var(--radius-sm);
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        width: 0;
        transition: width 1s ease-out;
    }
    
    .cv-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: var(--spacing-md);
        margin-top: var(--spacing-lg);
        padding: var(--spacing-md);
        background: var(--bg-tertiary);
        border-radius: var(--radius-md);
    }
    
    .stat-item {
        display: flex;
        flex-direction: column;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .insights-list {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-md);
    }
    
    .insight-card {
        display: flex;
        gap: var(--spacing-md);
        padding: var(--spacing-lg);
        border-radius: var(--radius-lg);
        border-left: 4px solid;
    }
    
    .insight-success {
        background: rgba(40, 167, 69, 0.1);
        border-color: var(--success-color);
    }
    
    .insight-warning {
        background: rgba(255, 193, 7, 0.1);
        border-color: var(--warning-color);
    }
    
    .insight-danger {
        background: rgba(220, 53, 69, 0.1);
        border-color: var(--danger-color);
    }
    
    .insight-info {
        background: rgba(23, 162, 184, 0.1);
        border-color: var(--info-color);
    }
    
    .insight-icon {
        font-size: 2rem;
        width: 50px;
        flex-shrink: 0;
    }
    
    .insight-success .insight-icon { color: var(--success-color); }
    .insight-warning .insight-icon { color: var(--warning-color); }
    .insight-danger .insight-icon { color: var(--danger-color); }
    .insight-info .insight-icon { color: var(--info-color); }
    
    .insight-content h4 {
        margin-bottom: var(--spacing-xs);
        color: var(--text-primary);
    }
    
    .insight-content p {
        margin: 0;
        color: var(--text-secondary);
    }
    
    .export-options {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--spacing-md);
    }
    
    .export-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--spacing-sm);
        padding: var(--spacing-xl);
        background: var(--bg-tertiary);
        border: 2px solid var(--border-color);
        border-radius: var(--radius-lg);
        color: var(--text-primary);
        cursor: pointer;
        transition: all var(--transition-base);
    }
    
    .export-btn:hover {
        background: var(--primary-color);
        border-color: var(--primary-color);
        color: white;
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .export-btn i {
        font-size: 2rem;
    }
`;
document.head.appendChild(style);
