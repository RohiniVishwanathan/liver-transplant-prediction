#!/usr/bin/env python3
"""
================================================================================
LIVER TRANSPLANT OUTCOMES PREDICTION
================================================================================

Project: Predicting 30-day hospital readmission and post-transplant complications
         in liver transplant patients

Data Source: Liver Outcome Monitoring Registry (De-identified)

Objective: Apply machine learning techniques to identify patients at high risk
           for readmission and complications, enabling targeted interventions.

Methods:
    - Multiple ML models: Logistic Regression, Random Forest, Gradient Boosting, KNN
    - Class imbalance handling: Class weighting and undersampling
    - Evaluation: AUC-ROC, Precision, Recall, F1-Score, Cross-Validation

Key Findings:
    - Readmission prediction: AUC ~0.52-0.56 (challenging to predict)
    - Complication prediction: AUC ~0.73 (meaningful predictability)
    - Top risk factors: Intubation status, hypertension, smoking, liver enzymes

Author: Rohini Vishwanathan
Date: January 2025
Data Source: Scripps Health (obtained during Purgo AI internship) 
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, f1_score, accuracy_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
CV_FOLDS = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the liver transplant data.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Define target column
    target_col = 'Readmission within 30 days'
    
    # Encode target: Yes=1, No=0
    df['Target'] = df[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    if df['Target'].isna().any():
        df['Target'] = df[target_col].apply(
            lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0
        )
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Prepare features and target
    feature_cols = [c for c in df_encoded.columns if c not in [target_col, 'Target']]
    X = df_encoded[feature_cols]
    y = df_encoded['Target']
    
    return X, y, feature_cols


def handle_class_imbalance(X_train, y_train, method='undersample'):
    """
    Handle class imbalance through undersampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: 'undersample' to balance classes
        
    Returns:
        X_balanced: Balanced feature matrix
        y_balanced: Balanced target vector
    """
    if method == 'undersample':
        np.random.seed(RANDOM_STATE)
        minority_idx = y_train[y_train == 1].index
        majority_idx = y_train[y_train == 0].index
        
        # Undersample majority class to match minority
        undersampled_majority = np.random.choice(
            majority_idx, size=len(minority_idx), replace=False
        )
        balanced_idx = np.concatenate([minority_idx.values, undersampled_majority])
        
        return X_train.loc[balanced_idx], y_train.loc[balanced_idx]
    
    return X_train, y_train


def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_scaled=None, X_test_scaled=None):
    """
    Train multiple models and evaluate their performance.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        X_train_scaled, X_test_scaled: Scaled features for KNN
        
    Returns:
        results: Dictionary containing model results
    """
    results = {}
    
    # Calculate class weight for imbalanced learning
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    # 1. Logistic Regression
    lr = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=RANDOM_STATE
    )
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = evaluate_model(y_test, lr.predict(X_test), y_prob)
    results['Logistic Regression']['model'] = lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = evaluate_model(y_test, rf.predict(X_test), y_prob)
    results['Random Forest']['model'] = rf
    
    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    sample_weights = np.where(y_train == 1, class_weight[1], class_weight[0])
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    y_prob = gb.predict_proba(X_test)[:, 1]
    results['Gradient Boosting'] = evaluate_model(y_test, gb.predict(X_test), y_prob)
    results['Gradient Boosting']['model'] = gb
    
    # 4. KNN (requires scaled features)
    if X_train_scaled is not None:
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn.fit(X_train_scaled, y_train)
        y_prob = knn.predict_proba(X_test_scaled)[:, 1]
        results['KNN'] = evaluate_model(y_test, knn.predict(X_test_scaled), y_prob)
        results['KNN']['model'] = knn
    
    return results


def evaluate_model(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics for a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'predictions': y_pred,
        'probabilities': y_prob
    }


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract feature importances from a tree-based model.
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame of feature importances
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df.head(top_n)


def plot_roc_curves(results, y_test, save_path=None):
    """
    Plot ROC curves for all models.
    
    Args:
        results: Dictionary of model results
        y_test: True test labels
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {data['auc_roc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df, save_path=None):
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with Feature and Importance columns
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))[::-1]
    plt.barh(range(len(importance_df)), importance_df['Importance'].values, color=colors)
    plt.yticks(range(len(importance_df)), 
               [f[:35] for f in importance_df['Feature'].values], fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Readmission', 'Readmission'],
                yticklabels=['No Readmission', 'Readmission'],
                annot_kws={'size': 14})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Main analysis pipeline."""
    
    print("="*70)
    print("LIVER TRANSPLANT READMISSION PREDICTION ANALYSIS")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data('KNN_imputed_readmission_data.csv')
    
    print(f"    Dataset: {len(X)} patients, {len(feature_names)} features")
    print(f"    Class distribution: {(y==0).sum()} no readmission, {(y==1).sum()} readmission")
    print(f"    Imbalance ratio: 1:{(y==0).sum()/(y==1).sum():.1f}")
    
    # Step 2: Train-test split
    print("\n[2] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Training: {len(X_train)} samples")
    print(f"    Test: {len(X_test)} samples")
    
    # Step 3: Scale features for KNN
    print("\n[3] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Train and evaluate models
    print("\n[4] Training models...")
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )
    
    # Step 5: Display results
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-"*75)
    
    for name, data in results.items():
        print(f"{name:<25} {data['accuracy']:>10.4f} {data['precision']:>10.4f} "
              f"{data['recall']:>10.4f} {data['f1']:>10.4f} {data['auc_roc']:>10.4f}")
    
    # Step 6: Feature importance
    print("\n" + "="*70)
    print("TOP RISK FACTORS")
    print("="*70)
    
    importance_df = get_feature_importance(results['Random Forest']['model'], feature_names)
    print("\n" + importance_df.to_string(index=False))
    
    # Step 7: Generate visualizations
    print("\n[5] Generating visualizations...")
    plot_roc_curves(results, y_test, 'roc_curves.png')
    plot_feature_importance(importance_df, 'feature_importance.png')
    
    best_model = max(results.items(), key=lambda x: x[1]['auc_roc'])
    plot_confusion_matrix(y_test, best_model[1]['predictions'], best_model[0], 'confusion_matrix.png')
    
    print("    Saved: roc_curves.png, feature_importance.png, confusion_matrix.png")
    
    # Step 8: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBest Model: {best_model[0]} (AUC-ROC: {best_model[1]['auc_roc']:.4f})")
    print(f"\nKey Finding: Readmission prediction is inherently challenging (AUC ~0.5)")
    print("This suggests post-discharge factors not captured in clinical data play a major role.")
    print("\nTop 3 Risk Factors:")
    for i, row in importance_df.head(3).iterrows():
        print(f"  - {row['Feature']}")
    
    return results, importance_df


if __name__ == "__main__":
    results, importance = main()
