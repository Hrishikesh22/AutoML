import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def print_log(message, log_file=None):
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

def data_summary(df, log_file=None):
    # Log the data types directly
    print_log("Data Types:", log_file)
    print_log(df.dtypes, log_file)
    
    # Missing values summary
    print_log("\nMissing values per column:", log_file)
    print_log(df.isnull().sum(), log_file)
    
    # Numeric and categorical columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    
    if len(numeric_cols) > 0:
        print_log("\nNumeric columns summary:", log_file)
        print_log(df[numeric_cols].describe().T, log_file)
    else:
        print_log("\nNo numeric columns to describe.", log_file)
    
    if len(cat_cols) > 0:
        print_log("\nCategorical columns summary:", log_file)
        for col in cat_cols:
            unique_count = df[col].nunique(dropna=True)
            missing_count = df[col].isnull().sum()
            print_log(f"  - {col}: unique categories = {unique_count}, missing = {missing_count}", log_file)
    else:
        print_log("\nNo categorical columns to describe.", log_file)

def quick_missing_check(df, name="DataFrame", log_file=None):
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print_log(f"{name}: No missing values detected.", log_file)
    else:
        print_log(f"{name}: Missing values detected - total count: {missing_count}", log_file)

def explore_parquet_folder(folder_path, log_file=None):
    folder = Path(folder_path)
    try:
        X_train = pd.read_parquet(folder / 'X_train.parquet')
        X_test = pd.read_parquet(folder / 'X_test.parquet')
        y_train = pd.read_parquet(folder / 'y_train.parquet')
        y_test = pd.read_parquet(folder / 'y_test.parquet')
       
    except Exception as e:
        print_log(f"Error loading files in {folder}: {e}", log_file)
        return
    
    #print head of each dataset
    print_log(f"\nFolder: {folder.name}", log_file)
    print_log(f"  - X_train head:\n{X_train[:5]}", log_file)
    print_log(f"  - X_test head:\n{X_test[:5]}", log_file)
    print_log(f"  - y_train head:\n{y_train[:5]}", log_file)
    print_log(f"  - y_test head:\n{y_test[:5]}", log_file)
    
    # Directly log data types for each dataset (without changing anything)
    print_log(f"\nFolder: {folder.name}", log_file)
    print_log(f"  - X_train: {X_train.shape}, X_test: {X_test.shape}", log_file)
    print_log(f"  - y_train: {y_train.shape}, y_test: {y_test.shape}", log_file)

    quick_missing_check(X_train, "X_train", log_file)
    quick_missing_check(X_test, "X_test", log_file)
    quick_missing_check(y_train, "y_train", log_file)
    quick_missing_check(y_test, "y_test", log_file)
    
    # Summary for X_train
    print_log("\nSummary of X_train:", log_file)
    print_log("Data Types:", log_file)
    print_log(X_train.dtypes, log_file)
    data_summary(X_train, log_file)

    # Summary for X_test
    print_log("\nSummary of X_test:", log_file)
    print_log("Data Types:", log_file)
    print_log(X_test.dtypes, log_file)
    data_summary(X_test, log_file)
    
    # Summary for y_train
    print_log("\nSummary of y_train:", log_file)
    print_log("Data Types:", log_file)
    print_log(y_train.dtypes, log_file)
    data_summary(y_train, log_file)
    
    # Summary for y_test
    print_log("\nSummary of y_test:", log_file)
    print_log("Data Types:", log_file)
    print_log(y_test.dtypes, log_file)
    data_summary(y_test, log_file)
    

def explore_all_datasets(base_dir, log_file=None):
    base = Path(base_dir)
    dataset_count = 0
    fold_count = 0
    
    # Clear previous log file content
    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Exploration log started\n\n")
    
    for dataset_type in sorted(base.iterdir()):
        if dataset_type.is_dir():
            dataset_count += 1
            print_log(f"\nDataset Type: {dataset_type.name}", log_file)
            for fold in sorted(dataset_type.iterdir()):
                if fold.is_dir():
                    fold_count += 1
                    print_log(f"  Exploring fold: {fold.name}", log_file)
                    explore_parquet_folder(fold, log_file)
                else:
                    print_log(f"  Skipping non-folder in dataset_type: {fold.name}", log_file)
        else:
            print_log(f"Skipping non-folder in base dir: {dataset_type.name}", log_file)

    print_log(f"\nTotal datasets found: {dataset_count}", log_file)
    print_log(f"Total folds found: {fold_count}", log_file)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data"
    log_file = script_dir / "exploration_log.txt"
    
    if not data_path.exists():
        print(f"Error: data directory not found at: {data_path}")
    else:
        explore_all_datasets(data_path, log_file)
        print(f"\nExploration completed. Log saved to: {log_file}")