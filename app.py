"""
ML Studio - Flask Web Application
Merged and optimized single-file application
"""

import os
import io
import json
import pickle
import hashlib
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from werkzeug.utils import secure_filename

# ML imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_CONFIG = {
    "title": "ML Studio",
    "version": "3.0.0",
}

MODEL_CONFIG = {
    "available_models": ["Random Forest", "XGBoost", "LightGBM"],
    "default_test_size": 0.2,
    "default_random_state": 42,
    "default_cv_folds": 5,
}

PREPROCESSING_CONFIG = {
    "missing_value_strategies": ["Mean imputation", "Median imputation", "Mode imputation", "Drop rows", "Drop columns"],
    "encoding_methods": ["Label Encoding", "One-Hot Encoding"],
    "scaling_methods": ["StandardScaler", "MinMaxScaler", "RobustScaler"],
}

SAMPLE_DATASETS = [
    "Iris (Classification)",
    "Wine (Classification)",
    "Breast Cancer (Classification)",
    "Diabetes (Regression)",
]

HYPERPARAMETER_GRIDS = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
    }
}

# ============================================================================
# PREPROCESSING CLASS
# ============================================================================

class MLPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.steps_log = []
    
    def log_step(self, step_name: str, details: str) -> None:
        """Log preprocessing steps"""
        log_entry = f"✅ {step_name}: {details}"
        self.steps_log.append(log_entry)
    
    def get_preprocessing_summary(self) -> List[str]:
        """Get all preprocessing steps"""
        return self.steps_log.copy()
    
    def handle_missing_data(self, df: pd.DataFrame, strategy: Dict, advanced_imputation: bool = False) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        df = df.copy()
        
        for col, strat in strategy.items():
            if col not in df.columns:
                continue
                
            if strat == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strat == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif strat == 'mode':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0], inplace=True)
            elif strat == 'drop':
                df.dropna(subset=[col], inplace=True)
        
        self.log_step("Missing Values", f"Applied strategy to {len(strategy)} columns")
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'StandardScaler', columns: List[str] = None) -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'StandardScaler':
            scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif method == 'RobustScaler':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers[method] = scaler
        self.log_step("Feature Scaling", f"Scaled {len(columns)} columns using {method}")
        
        return df

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_task_type(y):
    """Detect if task is classification or regression"""
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    is_numeric = pd.api.types.is_numeric_dtype(y)
    
    if not is_numeric:
        return 'classification'
    
    unique_values = y.nunique()
    
    if y.dtype in ['int64', 'int32'] and unique_values <= 20:
        return 'classification'
    elif unique_values / len(y) < 0.05:
        return 'classification'
    else:
        return 'regression'

def load_sample_dataset(dataset_name: str):
    """Load sample datasets"""
    task_type = 'classification'
    
    if dataset_name == "Iris (Classification)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "Wine (Classification)":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "Breast Cancer (Classification)":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "Diabetes (Regression)":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task_type = 'regression'
    else:
        return None, None
    
    return df, task_type

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def create_model(model_name: str, task_type: str, **params):
    """Create ML model instance"""
    if task_type == 'classification':
        if model_name == 'Random Forest':
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == 'XGBoost':
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss', **params)
        elif model_name == 'LightGBM':
            return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
    else:  # regression
        if model_name == 'Random Forest':
            return RandomForestRegressor(random_state=42, **params)
        elif model_name == 'XGBoost':
            return xgb.XGBRegressor(random_state=42, eval_metric='rmse', **params)
        elif model_name == 'LightGBM':
            return lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
    
    raise ValueError(f"Unsupported model: {model_name} for task: {task_type}")

def train_model_with_tuning(X_train, X_test, y_train, y_test, model_name, task_type, 
                           tuning_method='None', use_cv=True, **model_params):
    """Train model with optional hyperparameter tuning"""
    results = {
        'model': None,
        'best_params': {},
        'train_score': 0,
        'test_score': 0,
        'cv_scores': [],
        'model_name': model_name,
        'task_type': task_type
    }
    
    try:
        # Create and train model
        if tuning_method == 'None':
            model = create_model(model_name, task_type, **model_params)
            model.fit(X_train, y_train)
            results['model'] = model
            results['best_params'] = model_params
        else:
            # Hyperparameter tuning
            base_model = create_model(model_name, task_type)
            param_grid = HYPERPARAMETER_GRIDS.get(model_name, {})
            
            scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
            
            if tuning_method == 'Grid Search':
                search = GridSearchCV(base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
            else:  # Random Search
                search = RandomizedSearchCV(base_model, param_grid, cv=5, n_iter=20,
                                          scoring=scoring, n_jobs=-1, random_state=42)
            
            search.fit(X_train, y_train)
            results['model'] = search.best_estimator_
            results['best_params'] = search.best_params_
        
        # Calculate scores
        model = results['model']
        
        if task_type == 'classification':
            results['train_score'] = accuracy_score(y_train, model.predict(X_train))
            results['test_score'] = accuracy_score(y_test, model.predict(X_test))
            
            if use_cv:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                results['cv_scores'] = cv_scores.tolist()
        else:
            results['train_score'] = r2_score(y_train, model.predict(X_train))
            results['test_score'] = r2_score(y_test, model.predict(X_test))
            
            if use_cv:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                results['cv_scores'] = cv_scores.tolist()
        
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return results

def evaluate_model(model, X_test, y_test, task_type):
    """Evaluate trained model"""
    try:
        predictions = model.predict(X_test)
        
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'f1_score': f1_score(y_test, predictions, average='weighted'),
            }
            
            if hasattr(model, 'predict_proba'):
                try:
                    if len(np.unique(y_test)) == 2:
                        proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, proba)
                    else:
                        proba = model.predict_proba(X_test)
                        metrics['roc_auc'] = roc_auc_score(y_test, proba, multi_class='ovr')
                except:
                    metrics['roc_auc'] = None
        else:
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions)
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return {}

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.secret_key = 'ml-studio-super-secret-key-2025'  # Fixed secret key for persistent sessions
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024  # 800MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Try to use Flask-Session if available, otherwise use default sessions
try:
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = 'flask_session'
    Session(app)
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
except Exception as e:
    logger.warning(f"⚠️ Flask-Session not available, using default sessions: {e}")
    logger.warning("Install Flask-Session for better performance: pip install Flask-Session")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create data storage directory
DATA_STORAGE = 'session_data'
os.makedirs(DATA_STORAGE, exist_ok=True)

# Global preprocessor
preprocessor = MLPreprocessor()

# Helper functions
def serialize_dataframe(df):
    """Save DataFrame to file and return reference"""
    if df is None:
        return None
    
    # Create unique ID for this dataset
    dataset_id = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
    filepath = os.path.join(DATA_STORAGE, f'{dataset_id}.pkl')
    
    # Save DataFrame to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)
    
    return {
        'id': dataset_id,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }

def deserialize_dataframe(data):
    """Load DataFrame from file reference"""
    if data is None:
        logger.error("❌ deserialize_dataframe: data is None!")
        return None
    
    dataset_id = data.get('id')
    if not dataset_id:
        logger.error(f"❌ deserialize_dataframe: no id in data: {data}")
        return None
    
    filepath = os.path.join(DATA_STORAGE, f'{dataset_id}.pkl')
    
    if not os.path.exists(filepath):
        logger.error(f"❌ deserialize_dataframe: file not found: {filepath}")
        return None
    
    # Load DataFrame from pickle file
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
        return df

def cleanup_old_dataset(session_data):
    """Delete old dataset file when loading new one"""
    if session_data and 'id' in session_data:
        old_filepath = os.path.join(DATA_STORAGE, f'{session_data["id"]}.pkl')
        if os.path.exists(old_filepath):
            try:
                os.remove(old_filepath)
            except Exception as e:
                logger.warning(f"Could not remove old dataset file: {e}")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/data-exploration')
def data_exploration():
    """Data exploration page"""
    return render_template('data_exploration.html')

@app.route('/preprocessing')
def preprocessing_page():
    """Preprocessing page"""
    return render_template('preprocessing.html')

@app.route('/training')
def training():
    """Training page"""
    return render_template('training.html')

@app.route('/evaluation')
def evaluation():
    """Evaluation page"""
    return render_template('evaluation.html')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload CSV dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding='latin-1')
            except Exception:
                df = pd.read_csv(file, encoding='ISO-8859-1')
        
        # Validate dataset is not empty
        if df.empty or len(df) == 0:
            return jsonify({'success': False, 'error': 'Dataset is empty'}), 400
        
        if len(df.columns) == 0:
            return jsonify({'success': False, 'error': 'Dataset has no columns'}), 400
        
        # Reset preprocessor for new dataset
        global preprocessor
        preprocessor = MLPreprocessor()
        
        # Cleanup old dataset file if exists
        if 'dataset' in session:
            cleanup_old_dataset(session['dataset'])
        
        # Store in session
        session.clear()  # Clear any old session data
        serialized_data = serialize_dataframe(df)
        session['dataset'] = serialized_data
        session['dataset_name'] = file.filename
        session.modified = True  # Ensure session is saved
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {file.filename} successfully',
            'shape': df.shape,
            'columns': df.columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/load-sample-dataset', methods=['POST'])
def load_sample():
    """Load sample dataset"""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        
        df, task_type = load_sample_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Reset preprocessor for new dataset
        global preprocessor
        preprocessor = MLPreprocessor()
        
        # Cleanup old dataset file if exists
        if 'dataset' in session:
            cleanup_old_dataset(session['dataset'])
        
        # Store in session
        session.clear()  # Clear any old session data
        session['dataset'] = serialize_dataframe(df)
        session['dataset_name'] = dataset_name
        session['task_type'] = task_type
        session.modified = True  # Ensure session is saved
        
        return jsonify({
            'success': True,
            'message': f'Loaded {dataset_name}',
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'task_type': task_type
        })
        
    except Exception as e:
        logger.error(f"Load sample error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-dataset-info', methods=['GET'])
def get_dataset_info():
    """Get dataset information"""
    try:
        if 'dataset' not in session:
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        df = deserialize_dataframe(session['dataset'])
        
        # Get unique value counts and distributions for each column
        unique_info = {}
        value_counts = {}
        column_ranges = {}  # For numeric columns: store min/max
        
        for col in df.columns:
            unique_info[col] = int(df[col].nunique())
            
            # For numeric columns, get min/max range
            if pd.api.types.is_numeric_dtype(df[col]):
                column_ranges[col] = {
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None
                }
            
            # For categorical columns or columns with few unique values, get value counts
            if df[col].dtype == 'object' or df[col].nunique() <= 20:
                vc = df[col].value_counts().head(10).to_dict()
                # Convert numpy types to Python types for JSON serialization
                value_counts[col] = {str(k): int(v) for k, v in vc.items()}
        
        # Get numeric columns suitable for scaling (exclude binary/encoded categorical and IDs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scalable_columns = []
        
        for col in numeric_cols:
            col_lower = col.lower()
            unique_vals = df[col].nunique()
            total_rows = len(df)
            
            # Skip ID-like columns (column name contains 'id', 'number', 'index', etc.)
            if any(keyword in col_lower for keyword in ['id', 'number', 'index', 'key', 'code']):
                continue
            
            # Skip columns where unique values = total rows (likely unique identifiers)
            if unique_vals == total_rows:
                continue
            
            # Skip binary columns (0/1) which are usually from one-hot encoding
            if unique_vals == 2:
                # Check if it's binary (0/1)
                unique_set = set(df[col].dropna().unique())
                if unique_set.issubset({0, 1, 0.0, 1.0}):
                    continue  # Skip binary encoded columns
            
            # Skip columns with very few unique values (likely categorical)
            if unique_vals <= 2:
                continue
            
            # Skip columns with high cardinality ratio (>95% unique = likely IDs)
            if unique_vals / total_rows > 0.95:
                continue
            
            scalable_columns.append(col)
        
        # Get description - handle small datasets and all-categorical datasets
        try:
            description_df = df.describe(include='all')
            description = {}
            for col in description_df.columns:
                col_desc = {}
                for idx in description_df.index:
                    val = description_df.loc[idx, col]
                    # Handle NaN and infinity values for JSON serialization
                    if pd.isna(val):
                        col_desc[idx] = None
                    elif np.isinf(val):
                        col_desc[idx] = None
                    else:
                        col_desc[idx] = float(val) if isinstance(val, (int, float, np.number)) else str(val)
                description[col] = col_desc
        except Exception as e:
            # Fallback for very small or problematic datasets
            description = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            description[col] = {
                                'count': float(len(col_data)),
                                'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                                'std': float(col_data.std()) if len(col_data) > 1 else None,
                                'min': float(col_data.min()) if len(col_data) > 0 else None,
                                '25%': float(col_data.quantile(0.25)) if len(col_data) > 0 else None,
                                '50%': float(col_data.quantile(0.50)) if len(col_data) > 0 else None,
                                '75%': float(col_data.quantile(0.75)) if len(col_data) > 0 else None,
                                'max': float(col_data.max()) if len(col_data) > 0 else None,
                            }
                    except Exception:
                        pass
        
        # Get sample data and handle NaN/inf values
        sample_df = df.head(min(5, len(df))).replace([np.inf, -np.inf], np.nan)
        sample_records = sample_df.to_dict('records')
        
        # Convert NaN to None for JSON serialization
        for record in sample_records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'scalable_columns': scalable_columns,
            'description': description,
            'sample': sample_records,
            'head': sample_records,
            'duplicates': int(df.duplicated().sum()),
            'unique_counts': unique_info,
            'value_counts': value_counts,
            'column_ranges': column_ranges
        }
        
        return jsonify({'success': True, 'data': info})
        
    except Exception as e:
        logger.error(f"Get info error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess/handle-missing', methods=['POST'])
def handle_missing_values():
    """Handle missing values"""
    try:
        if 'dataset' not in session:
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        data = request.get_json()
        strategy = data.get('strategy', 'mean')
        columns = data.get('columns', [])
        
        df = deserialize_dataframe(session['dataset'])
        
        # Handle drop rows/columns specially
        if strategy == 'drop_rows':
            df = df.dropna()
            preprocessor.log_step("Drop Rows", "Dropped rows with missing values")
        elif strategy == 'drop_columns':
            cols_to_drop = [col for col in columns if col in df.columns]
            df = df.drop(columns=cols_to_drop)
            preprocessor.log_step("Drop Columns", f"Dropped {len(cols_to_drop)} columns")
        else:
            # Build strategy dict
            strategy_dict = {col: strategy for col in columns if col in df.columns}
            df = preprocessor.handle_missing_data(df, strategy=strategy_dict)
        
        session['dataset'] = serialize_dataframe(df)
        
        return jsonify({
            'success': True,
            'message': 'Missing values handled successfully',
            'missing_values': df.isnull().sum().to_dict()
        })
        
    except Exception as e:
        logger.error(f"Handle missing error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess/remove-columns', methods=['POST'])
def remove_columns():
    """Remove selected columns from dataset"""
    try:
        if 'dataset' not in session:
            logger.error("❌ No dataset in session")
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        data = request.get_json()
        
        columns_to_remove = data.get('columns', [])
        
        if not columns_to_remove:
            return jsonify({'success': False, 'error': 'No columns selected'}), 400
        
        df = deserialize_dataframe(session['dataset'])
        
        # Filter columns that exist
        valid_columns = [col for col in columns_to_remove if col in df.columns]
        
        if not valid_columns:
            return jsonify({'success': False, 'error': 'Selected columns not found'}), 400
        
        # Remove columns
        df = df.drop(columns=valid_columns)
        
        # Save
        session['dataset'] = serialize_dataframe(df)
        session.modified = True
        
        preprocessor.log_step("Remove Columns", f"Removed {len(valid_columns)} columns: {', '.join(valid_columns)}")
        
        return jsonify({
            'success': True,
            'message': f'Removed {len(valid_columns)} column(s)',
            'columns': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"❌ Remove columns error: {str(e)}")
        logger.exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess/encode', methods=['POST'])
def encode_features():
    """Encode categorical features"""
    try:
        if 'dataset' not in session:
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        data = request.get_json()
        method = data.get('method', 'Label Encoding')
        columns = data.get('columns', [])
        
        df = deserialize_dataframe(session['dataset'])
        
        if not columns:
            return jsonify({'success': False, 'error': 'No columns selected'}), 400
        
        # Filter out columns that don't exist in the dataframe
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            return jsonify({'success': False, 'error': 'Selected columns not found in dataset'}), 400
        
        # Check for high cardinality columns (warn if One-Hot encoding would create too many columns)
        if method == 'One-Hot Encoding':
            high_cardinality_cols = []
            for col in valid_columns:
                unique_count = df[col].nunique()
                if unique_count > 50:  # Threshold for "too many" unique values
                    high_cardinality_cols.append(f"{col} ({unique_count} unique values)")
            
            if high_cardinality_cols:
                return jsonify({
                    'success': False,
                    'error': f"⚠️ Cannot One-Hot encode high cardinality columns:\n{', '.join(high_cardinality_cols)}\n\nOne-Hot encoding works best with <50 unique values. Consider using Label Encoding instead, or drop these columns if they're not useful (like ID or Name columns)."
                }), 400
        
        if method == 'Label Encoding':
            for col in valid_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            preprocessor.log_step("Label Encoding", f"Encoded {len(valid_columns)} columns")
        else:  # One-Hot Encoding
            df = pd.get_dummies(df, columns=valid_columns, drop_first=True)
            preprocessor.log_step("One-Hot Encoding", f"Encoded {len(valid_columns)} columns")
        
        session['dataset'] = serialize_dataframe(df)
        session.modified = True  # Ensure Flask saves the session
        
        # Calculate scalable columns (exclude binary/encoded categorical and IDs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scalable_columns = []
        total_rows = len(df)
        
        for col in numeric_cols:
            col_lower = col.lower()
            unique_vals = df[col].nunique()
            
            # Skip ID-like columns (column name contains 'id', 'number', 'index', etc.)
            if any(keyword in col_lower for keyword in ['id', 'number', 'index', 'key', 'code']):
                continue
            
            # Skip columns where unique values = total rows (likely unique identifiers)
            if unique_vals == total_rows:
                continue
            
            # Skip binary columns (0/1) from one-hot encoding
            if unique_vals == 2:
                unique_set = set(df[col].dropna().unique())
                if unique_set.issubset({0, 1, 0.0, 1.0}):
                    continue  # Skip binary encoded columns
            
            # Skip columns with very few unique values (likely categorical)
            if unique_vals <= 2:
                continue
            
            # Skip columns with high cardinality ratio (>95% unique = likely IDs)
            if unique_vals / total_rows > 0.95:
                continue
            
            scalable_columns.append(col)
        
        # Return updated column lists after encoding
        return jsonify({
            'success': True,
            'message': f'{method} applied successfully',
            'columns': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'scalable_columns': scalable_columns  # Only columns suitable for scaling
        })
        
    except Exception as e:
        logger.error(f"Encode error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess/scale', methods=['POST'])
def scale_features():
    """Scale numerical features"""
    try:
        if 'dataset' not in session:
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        data = request.get_json()
        method = data.get('method', 'StandardScaler')
        columns = data.get('columns', None)
        
        df = deserialize_dataframe(session['dataset'])
        df = preprocessor.scale_features(df, method=method, columns=columns)
        
        session['dataset'] = serialize_dataframe(df)
        
        return jsonify({
            'success': True,
            'message': f'{method} applied successfully'
        })
        
    except Exception as e:
        logger.error(f"Scale error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess/remove-duplicates', methods=['POST'])
def remove_duplicates():
    """Remove duplicate rows"""
    try:
        if 'dataset' not in session:
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        df = deserialize_dataframe(session['dataset'])
        original_shape = df.shape
        
        df = df.drop_duplicates()
        duplicates_removed = original_shape[0] - df.shape[0]
        
        preprocessor.log_step("Remove Duplicates", f"Removed {duplicates_removed} duplicate rows")
        
        session['dataset'] = serialize_dataframe(df)
        
        return jsonify({
            'success': True,
            'message': f'Removed {duplicates_removed} duplicate rows',
            'shape': df.shape
        })
        
    except Exception as e:
        logger.error(f"Remove duplicates error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Train ML model"""
    try:
        if 'dataset' not in session:
            return jsonify({'success': False, 'error': 'No dataset loaded'}), 400
        
        data = request.get_json()
        target_column = data.get('target')
        model_name = data.get('model', 'Random Forest')
        test_size = data.get('test_size', 0.2)
        tuning_method = data.get('tuning', 'None')
        use_cv = data.get('use_cv', True)
        
        df = deserialize_dataframe(session['dataset'])
        
        if target_column not in df.columns:
            return jsonify({'success': False, 'error': f'Target column "{target_column}" not found'}), 400
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Detect task type
        task_type = detect_task_type(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        results = train_model_with_tuning(
            X_train, X_test, y_train, y_test,
            model_name=model_name,
            task_type=task_type,
            tuning_method=tuning_method,
            use_cv=use_cv
        )
        
        # Store results
        session['training_results'] = {
            'model_name': results['model_name'],
            'task_type': results['task_type'],
            'train_score': results['train_score'],
            'test_score': results['test_score'],
            'cv_scores': results['cv_scores'],
            'best_params': results['best_params'],
            'feature_names': X.columns.tolist()
        }
        
        # Get feature importance
        feature_importance = None
        if hasattr(results['model'], 'feature_importances_'):
            importance = results['model'].feature_importances_
            feature_importance = dict(zip(X.columns, importance.tolist()))
        
        # Evaluate
        metrics = evaluate_model(results['model'], X_test, y_test, task_type)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'results': {
                'model_name': model_name,
                'task_type': task_type,
                'train_score': round(results['train_score'], 4),
                'test_score': round(results['test_score'], 4),
                'cv_scores': [round(s, 4) for s in results['cv_scores']],
                'cv_mean': round(np.mean(results['cv_scores']), 4) if results['cv_scores'] else None,
                'best_params': results['best_params'],
                'feature_importance': feature_importance,
                'metrics': metrics
            }
        })
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evaluate', methods=['GET'])
def get_evaluation():
    """Get evaluation results"""
    try:
        if 'training_results' not in session:
            return jsonify({'success': False, 'error': 'No trained model'}), 400
        
        results = session['training_results']
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-preprocessing-log', methods=['GET'])
def get_preprocessing_log():
    """Get preprocessing log"""
    try:
        log = preprocessor.get_preprocessing_summary()
        return jsonify({'success': True, 'log': log})
    except Exception as e:
        logger.error(f"Get log error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session"""
    try:
        global preprocessor
        
        # Cleanup old dataset file if exists
        if 'dataset' in session:
            cleanup_old_dataset(session['dataset'])
        
        session.clear()
        session.modified = True
        preprocessor = MLPreprocessor()  # Reset preprocessor instance
        logger.info("Session and preprocessor reset successfully")
        return jsonify({'success': True, 'message': 'Session reset successfully'})
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
