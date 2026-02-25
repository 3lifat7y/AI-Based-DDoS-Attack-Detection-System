import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

DATA_FILE = r"D:/university/LEVEL 3/semster 1/coding/Work-based/final output/test_set_01-12.csv"
OUTPUT_JSON = r"D:\university\LEVEL 3\semster 1\coding\Work-based\features_by_label.json"
ROWS_TO_READ = None
SAMPLE_PER_LABEL = 2000

def read_data_file(data_file: str, rows_to_read: int = None) -> pd.DataFrame:
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file, nrows=rows_to_read, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df):,} rows")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([float('inf'), float('-inf')], float('nan')).fillna(0)
    print("Handled missing values and infinities")
    return df

def build_features_by_label(df: pd.DataFrame, label_column: str = "Label") -> Dict:
    label_col = None
    for col in df.columns:
        if col.lower() == "label" or "label" in col.lower():
            label_col = col
            break
    if not label_col:
        raise ValueError("Label column not found in the dataset!")

    print(f"Using column '{label_col}' as Label")

    exclude_cols = [label_col]  # Only exclude the original label column to avoid duplication
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    all_data = []
    print("\nCollecting samples and attaching Label to each packet...\n")

    def to_native(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
        return v

    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        if SAMPLE_PER_LABEL > 0:
            n = min(SAMPLE_PER_LABEL, len(subset))
            subset = subset.sample(n=n, random_state=42)

        records = subset[feature_cols].to_dict(orient='records')
        # Convert numpy/pandas scalars to native Python types and add Label
        converted = []
        for r in records:
            nr = {k: to_native(v) for k, v in r.items()}
            nr["Label"] = str(label)
            converted.append(nr)

        all_data.extend(converted)
        print(f"{str(label):20} -> {len(converted):4} samples")

    return {"all_data": all_data}

def save_to_json(data: Dict, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data['all_data']):,} packets to: {path}")
    print("Ready to be used with send_traffic.py")

def parse_args():
    p = argparse.ArgumentParser(description='Build features_by_label.json from a CSV test set')
    p.add_argument('--input', '-i', type=str, help='CSV input file path (overrides DATA_FILE)')
    p.add_argument('--output', '-o', type=str, help='Output JSON file to write (overrides OUTPUT_JSON)')
    p.add_argument('--rows', '-r', type=int, help='Number of rows to read (overrides ROWS_TO_READ)')
    p.add_argument('--sample', '-s', type=int, help='Samples per label (overrides SAMPLE_PER_LABEL)')
    return p.parse_args()


def main(input_file: str = None, output_file: str = None, rows: int = None, sample_per_label: int = None):
    print("="*80)
    print("            Generating features_by_label.json with correct Labels")
    print("="*80)

    in_file = input_file or DATA_FILE
    out_file = output_file or OUTPUT_JSON

    df = read_data_file(in_file, rows if rows is not None else ROWS_TO_READ)
    df = handle_missing_values(df)

    global SAMPLE_PER_LABEL
    if sample_per_label is not None:
        SAMPLE_PER_LABEL = sample_per_label

    result = build_features_by_label(df)
    save_to_json(result, out_file)

    print("="*80)
    print("Done successfully! Run send_traffic.py now – all attack types will be ACTIVE")
    print("="*80)

if __name__ == "__main__":
    args = parse_args()
    main(input_file=args.input, output_file=args.output, rows=args.rows, sample_per_label=args.sample)