import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)

from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')
dataset_folder_path = "D:/university/LEVEL 3/semster 1/coding/Work-based/CIC DDOS/01-12"
output_folder = "D:/university/LEVEL 3/semster 1/coding/Work-based/final output"
folder_name = "01-12"
process_recursive = True
chunksize = 50000
test_size = 0.2
sample_fraction = 0.7

xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
}

def create_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def find_csv_files(folder_path, recursive=False):
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = list(Path(folder_path).glob(pattern))
    return csv_files

def load_csvs_from_folder(folder_path, recursive=False, chunksize=None, sample_frac=1.0):
    csv_files = find_csv_files(folder_path, recursive)
    if not csv_files:
        return None
    
    print(f"\n Found {len(csv_files)} CSV file(s)")
    
    dfs = []
    
    for csv_file in csv_files:
        try:
            if chunksize:
                chunks = pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(csv_file, low_memory=False)
            
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
                print(f"  ✓ Loaded: {csv_file.name} ({len(df):,} rows) - Sampled {int(sample_frac*100)}%")
            else:
                print(f"  ✓ Loaded: {csv_file.name} ({len(df):,} rows)")
            
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ Skipped {csv_file.name}: {str(e)}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True, sort=False)
    return None

def detect_label_column(df):
    possible_names = ['Label', 'label', 'Attack', 'attack', 'class', 'Class', 'target', 'Target',
                    ' Label', ' label', ' Attack', ' attack', ' class', ' Class', ' target', ' Target']
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"Could not detect label column. Available columns: {list(df.columns)}")

def preprocess_data(df, label_col):
    print(f"   Separating features and labels...")
    y = df[label_col].copy()
    
    print(f"   Dropping label column...")
    df.drop(columns=[label_col], inplace=True)
    
    print(f"   Dropping empty columns...")
    df.dropna(axis=1, how='all', inplace=True)
    
    print(f"   Replacing infinite values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"   Selecting numeric columns...")
    X = df.select_dtypes(include=[np.number])
    
    del df
    
    print(f"   Filling missing values...")
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    print(f"   Removing rows with NaN...")
    mask = X.isnull().any(axis=1)
    if mask.any():
        X = X[~mask].copy()
        y = y[~mask].copy()
    
    return X, y

def train_model(X_train, X_test, y_train, y_test, num_classes):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        **xgb_params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model

def compute_metrics(y_true, y_pred, y_pred_proba, num_classes):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        y_bin = label_binarize(y_true, classes=range(num_classes))
        auc_macro = roc_auc_score(y_bin, y_pred_proba, average='macro', multi_class='ovr')
    except:
        auc_macro = np.nan
    
    return {
        'Accuracy': acc,
        'Precision_Macro': prec_macro,
        'Precision_Weighted': prec_weighted,
        'Recall_Macro': rec_macro,
        'Recall_Weighted': rec_weighted,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'AUC_ROC_Macro': auc_macro
    }

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_confusion_matrix_csv(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=[f'True_{l}' for l in labels], 
                         columns=[f'Pred_{l}' for l in labels])
    cm_df.to_csv(output_path)

def plot_roc_curves(y_true, y_pred_proba, num_classes, labels, output_path):
    y_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{labels[i]} (AUC={auc_score:.3f})')
        except:
            pass
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Multi-Class)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(y_true, y_pred_proba, num_classes, labels, output_path):
    y_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        try:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, color=colors[i], lw=2, label=f'{labels[i]}')
        except:
            pass
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Per Class)')
    plt.legend(loc="best")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_metrics_by_class(y_true, y_pred, num_classes, labels, output_path):
    f1_per_class = []
    prec_per_class = []
    rec_per_class = []
    
    for i in range(num_classes):
        y_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        f1_per_class.append(f1_score(y_bin, y_pred_bin, zero_division=0))
        prec_per_class.append(precision_score(y_bin, y_pred_bin, zero_division=0))
        rec_per_class.append(recall_score(y_bin, y_pred_bin, zero_division=0))
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, f1_per_class, width, label='F1-Score', alpha=0.8)
    plt.bar(x, prec_per_class, width, label='Precision', alpha=0.8)
    plt.bar(x + width, rec_per_class, width, label='Recall', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Metrics per Class')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    print(f"\n{'#'*70}")
    print(f"# XGBoost Multi-Class Classification Trainer")
    print(f"# Dataset: {folder_name}")
    print(f"{'#'*70}")
    
    if not os.path.exists(dataset_folder_path):
        print(f" Dataset path not found: {dataset_folder_path}")
        sys.exit(1)
    
    create_directories([output_folder])
    
    print(f"\n{'='*70}")
    print(f"Processing: {folder_name}")
    print(f"{'='*70}")
    
    print(f"\n📂 Loading CSV files...")
    df = load_csvs_from_folder(dataset_folder_path, recursive=process_recursive, chunksize=chunksize, sample_frac=sample_fraction)
    
    if df is None or len(df) == 0:
        print(f"❌ No data found. Exiting.\n")
        sys.exit(1)
    
    print(f"✅ Total rows after loading: {len(df):,}\n")
    
    try:
        label_col = detect_label_column(df)
        print(f"🏷️  Label column: {label_col}\n")
    except ValueError as e:
        print(f"❌ Error: {e}\n")
        sys.exit(1)
    
    print(f"🔧 Preprocessing data...")
    X, y = preprocess_data(df, label_col)
    print(f"   ✅ Features: {X.shape[1]}, Samples: {X.shape[0]:,}\n")
    
    del df
    
    le_labels = LabelEncoder()
    y_encoded = le_labels.fit_transform(y)
    num_classes = len(le_labels.classes_)
    label_mapping = {int(i): str(label) for i, label in enumerate(le_labels.classes_)}
    print(f"📊 Classes: {num_classes} - {list(le_labels.classes_)}\n")
    
    print(f"✂️  Splitting data ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}\n")
    
    print(f"💾 Saving Test Set...")
    y_test_original = le_labels.inverse_transform(y_test)
    test_set_df = X_test.copy()
    test_set_df[label_col] = y_test_original
    test_set_path = os.path.join(output_folder, f"test_set_{folder_name}.csv")
    test_set_df.to_csv(test_set_path, index=False)
    print(f"   ✓ Test set saved: {test_set_path}")
    print(f"   ✓ Test set shape: {test_set_df.shape} ({len(test_set_df):,} rows, {len(test_set_df.columns)} columns)\n")
    
    del test_set_df
    del X
    del y
    del y_encoded
    
    print(f"📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✅ Features scaled\n")
    
    del X_train
    del X_test
    
    print(f"🤖 Training XGBoost model...")
    model = train_model(X_train_scaled, X_test_scaled, y_train, y_test, num_classes)
    print(f"   ✅ Model trained\n")
    
    print(f"🔮 Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    print(f"   ✅ Predictions complete\n")
    
    print(f"📈 Computing metrics...")
    metrics = compute_metrics(y_test, y_pred, y_pred_proba, num_classes)
    print(f"   Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"   Precision:   {metrics['Precision_Macro']:.4f}")
    print(f"   Recall:      {metrics['Recall_Macro']:.4f}")
    print(f"   F1-Score:    {metrics['F1_Macro']:.4f}")
    print(f"   AUC-ROC:     {metrics['AUC_ROC_Macro']:.4f}\n")
    
    print(f"💾 Saving artifacts...")
    model_path = os.path.join(output_folder, f"model_{folder_name}.pht")
    joblib.dump(model, model_path)
    print(f"   ✓ Model: {model_path}")
    
    scaler_path = os.path.join(output_folder, f"scaler_{folder_name}.pht")
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler: {scaler_path}")
    
    label_map_path = os.path.join(output_folder, f"label_mapping_{folder_name}.json")
    with open(label_map_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"   ✓ Label mapping: {label_map_path}")
    
    metrics_path = os.path.join(output_folder, f"metrics_{folder_name}.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   ✓ Metrics: {metrics_path}\n")
    
    print(f"🎨 Generating plots...")
    plot_confusion_matrix(y_test, y_pred, le_labels.classes_, 
                         os.path.join(output_folder, f"confusion_matrix_{folder_name}.png"))
    save_confusion_matrix_csv(y_test, y_pred, le_labels.classes_,
                             os.path.join(output_folder, f"confusion_matrix_{folder_name}.csv"))
    plot_roc_curves(y_test, y_pred_proba, num_classes, le_labels.classes_,
                   os.path.join(output_folder, f"roc_curves_{folder_name}.png"))
    plot_precision_recall_curves(y_test, y_pred_proba, num_classes, le_labels.classes_,
                                os.path.join(output_folder, f"pr_curves_{folder_name}.png"))
    plot_metrics_by_class(y_test, y_pred, num_classes, le_labels.classes_,
                         os.path.join(output_folder, f"metrics_by_class_{folder_name}.png"))
    print(f"   ✓ All plots saved\n")
    
    print(f"\n{'#'*70}")
    print(f"# ✅ TRAINING COMPLETE!")
    print(f"# Output folder: {output_folder}")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()