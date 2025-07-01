import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_param_grid(estimator, param_grid):
    """Validate parameter grid against estimator's valid parameters."""
    valid_params = estimator.get_params().keys()
    for param, values in param_grid.items():
        if param not in valid_params:
            raise ValueError(f"Parameter '{param}' not recognized by {estimator.__class__.__name__}")
        if param == 'penalty':
            valid_penalties = {'l1', 'l2', 'elasticnet', None}
            if not all(val in valid_penalties for val in values):
                raise ValueError(f"Invalid 'penalty' values in {values}. Must be among {valid_penalties}")

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate model performance with detailed reporting."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError as e:
        metrics['roc_auc'] = float('nan')
        logger.warning(f"ROC AUC not computed for {model_name} due to {str(e)}.")
    logger.info(f"\nClassification Report for {model_name}:\n{classification_report(y_true, y_pred, zero_division=0)}")
    return metrics

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot and save ROC curve with enhanced visualization."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred_proba):.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        os.makedirs('../reports/figures/', exist_ok=True)
        plt.savefig(f'../reports/figures/roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except ValueError as e:
        logger.warning(f"ROC curve not plotted for {model_name} due to {str(e)}.")

def preprocess_data(df):
    """Preprocess features with robust handling."""
    numeric_features = ['TotalAmount', 'AvgAmount', 'StdAmount', 'TransactionCount', 'TransactionSpan', 'Recency', 'Frequency', 'Monetary']
    categorical_features = ['ProductCategory', 'ChannelId']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']
    return X, y, preprocessor

def train_models(df):
    """Train and evaluate models with MLflow tracking and fallback."""
    X, y, preprocessor = preprocess_data(df)
    
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Set up MLflow tracking with fallback to file store
    tracking_uri = "http://localhost:5000"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.tracking.get_tracking_uri()
        logger.info(f"Connected to MLflow tracking server at {tracking_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {tracking_uri}: {str(e)}. Falling back to local file store.")
        mlflow.set_tracking_uri("file:///" + os.path.abspath("./mlruns"))
        logger.info("Using local file store for MLflow tracking.")
    
    mlflow.set_experiment("Credit_Risk_Model")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weight = {0: class_weights[0], 1: class_weights[1]}
    logger.info(f"Class weights: {sample_weight}")
    
    # Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(
            class_weight=sample_weight,
            max_iter=1000,
            solver='saga',  # Changed to 'saga' to support l1 penalty
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']  # Removed None to focus on penalized models
        }
        logger.info(f"Logistic Regression param_grid: {param_grid}")
        validate_param_grid(lr, param_grid)
        grid = GridSearchCV(lr, param_grid, cv=cv, scoring='f1', error_score='raise', n_jobs=-1)
        try:
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            y_pred_proba = grid.predict_proba(X_test)[:, 1]
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(grid.best_estimator_, "model")
            plot_roc_curve(y_test, y_pred_proba, "Logistic Regression")
            logger.info(f"Best Logistic Regression parameters: {grid.best_params_}")
            logger.info(f"Logistic Regression metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error in Logistic Regression training: {str(e)}")
    
    # XGBoost
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model = xgb.XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=class_weights[1]/class_weights[0],
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }
        logger.info(f"XGBoost param_grid: {param_grid}")
        validate_param_grid(xgb_model, param_grid)
        grid = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='f1', error_score='raise', n_jobs=-1)
        try:
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            y_pred_proba = grid.predict_proba(X_test)[:, 1]
            
            # Log feature importance for XGBoost
            feature_importance = grid.best_estimator_.feature_importances_
            mlflow.log_dict(dict(zip(preprocessor.get_feature_names_out(), feature_importance)), "feature_importance.json")
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(grid.best_estimator_, "model", registered_model_name="xgboost_credit_risk", format="json")  # Specify JSON format
            plot_roc_curve(y_test, y_pred_proba, "XGBoost")
            logger.info(f"Best XGBoost parameters: {grid.best_params_}")
            logger.info(f"XGBoost metrics: {metrics}")
            logger.info(f"Feature importance logged: {dict(zip(preprocessor.get_feature_names_out(), feature_importance))}")
        except Exception as e:
            logger.error(f"Error in XGBoost training: {str(e)}")
    
    return preprocessor

if __name__ == "__main__":
    df = pd.read_csv('data/processed/processed_features.csv')
    preprocessor = train_models(df)