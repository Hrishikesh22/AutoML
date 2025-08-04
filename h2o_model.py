import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

def print_log(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

def is_target_categorical(y):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    unique_vals = np.unique(y) if isinstance(y, (np.ndarray, list)) else y.unique()
    return pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or len(unique_vals) < 20

def impute_and_encode(df, encoder_dict=None):
    if encoder_dict is None:
        encoder_dict = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = SimpleImputer(strategy='most_frequent').fit_transform(df[[col]]).ravel()
        if col in encoder_dict:
            df[col] = encoder_dict[col].transform(df[col])
        else:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoder_dict[col] = encoder
    return df, encoder_dict

def feature_selection(X, y, task_type, top_k=10):
    if task_type == 'regression':
        df = X.copy()
        df['target'] = y
        corrs = df.corr(numeric_only=True)['target'].dropna().drop('target')
        selected_features = corrs.abs().sort_values(ascending=False).head(top_k).index.tolist()
        return X[selected_features]

def run_h2o_automl_on_parquet(folder_path, log_file=None, training_time=None):
    folder = Path(folder_path)
    output_dir = Path(__file__).resolve().parent / "h2o_plots"
    output_dir.mkdir(exist_ok=True)
    dataset_name = folder.parent.name
    fold_name = folder.name
    try:
        X_train = pd.read_parquet(folder / 'X_train.parquet')
        X_test = pd.read_parquet(folder / 'X_test.parquet')
        y_train = pd.read_parquet(folder / 'y_train.parquet')
        y_test_path = folder / 'y_test.parquet'
        y_test = pd.read_parquet(y_test_path) if y_test_path.exists() else None
    except Exception as e:
        print_log(f"Error loading files in {folder}: {e}", log_file)
        return

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    encoder_dict = {}
    X_train, encoder_dict = impute_and_encode(X_train, encoder_dict)
    X_test, _ = impute_and_encode(X_test, encoder_dict)

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    num_imputer = SimpleImputer(strategy='mean')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    is_regression = not is_target_categorical(y_train)
    task_type = "regression" if is_regression else "classification"

    if not is_regression:
        label_encoder = LabelEncoder()
        y_combined = pd.concat([y_train, y_test], axis=0).squeeze()
        label_encoder.fit(y_combined)
        y_train = label_encoder.transform(y_train.squeeze())
        y_test = label_encoder.transform(y_test.squeeze())
    else:
        y_train = y_train.squeeze().astype(float)
        y_test = y_test.squeeze().astype(float)

    #Feature Selection
    task_type = 'regression' if not is_target_categorical(y_train) else 'classification'
    X_train = feature_selection(X_train, y_train, task_type, top_k=10)
    X_test = X_test[X_train.columns]
    print_log(f"Selected features: {X_train.columns.tolist()}", log_file)

    print_log(f"\nRunning H2O AutoML on: {folder.name} | Task: {task_type}", log_file)

    #h2o.init(max_mem_size="2G", nthreads=-1)

    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    aml = H2OAutoML(max_runtime_secs=300, seed=42, verbosity="info")
    start = time.time()
    aml.train(y="target", training_frame=train_h2o)
    elapsed = time.time() - start

    if training_time is not None:
        training_time["H2OAutoML"] += elapsed

    print_log(f"H2O AutoML training time: {elapsed:.2f} seconds", log_file)

    preds = aml.leader.predict(test_h2o).as_data_frame()
    if is_regression:
        pred_vals = preds.values.squeeze()
        np.save("predictions.npy", pred_vals)

        if y_test is not None:
            r2 = r2_score(y_test, pred_vals)
            print_log(f"H2O R² Score: {r2:.4f}", log_file)
    else:
        pred_labels = preds['predict'].astype(int)
        np.save("predictions.npy", pred_labels)

        if y_test is not None:
            acc = accuracy_score(y_test, pred_labels)
            print_log(f"H2O Accuracy: {acc:.4f}", log_file)

    #leaderboard
    lb = aml.leaderboard.as_data_frame()
    print_log(f"\nH2O AutoML Leaderboard:\n{lb}", log_file)
    print_log(f"\nH2O AutoML Model Details:\n{aml.leader}", log_file)
    print_log(f"Best model: {aml.leader.model_id}", log_file)

    leaderboard_models = lb['model_id'].tolist()
    all_predictions = []
    for model_id in leaderboard_models:
        model = h2o.get_model(model_id)
        preds = model.predict(test_h2o).as_data_frame()
        if is_regression:
            all_predictions.append(preds.values.squeeze())
        else:
            all_predictions.append(preds['predict'].astype(int).values)
    prediction_df = pd.DataFrame(np.column_stack(all_predictions), columns=leaderboard_models)
    correlation_matrix = prediction_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap = "coolwarm", fmt=".2f")
    plt.title("Correlation of H2O AutoML Predictions")
    plot_path = output_dir / f"h2o_predictions_correlation_{dataset_name}_fold{fold_name}.png"
    plt.savefig(plot_path)
    print_log(f"Saved correlation heatmap to: {plot_path}", log_file)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    #h2o.shutdown(prompt=False)

def explore_all_datasets_h2o(base_dir, log_file=None):
    base = Path(base_dir)
    training_times = defaultdict(float)

    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("H2O AutoML log started\n\n")
    h2o.init(max_mem_size="2G", nthreads=-1) # initialize H2O cluster
    try:
        for dataset_type in sorted(base.iterdir()):
            if dataset_type.is_dir():
                print_log(f"\nDataset Type: {dataset_type.name}", log_file)
                for fold in sorted(dataset_type.iterdir(), key=lambda x: int(x.name)):
                    if fold.is_dir():
                        print_log(f"  Exploring fold: {fold.name}", log_file)
                        run_h2o_automl_on_parquet(fold, log_file, training_times)
    finally:
        h2o.shutdown(prompt=False) #to shutdown H2O cluster after all datasets are processed

    print_log("\n=== Total H2O AutoML Training Time ===", log_file)
    for model_name, total_time in training_times.items():
        print_log(f"{model_name}: {total_time:.2f} seconds", log_file)
    

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data"
    log_file = script_dir / "h2o_log.txt"
    if not data_path.exists():
        print(f"Error: data directory not found at: {data_path}")
    else:
        explore_all_datasets_h2o(data_path, log_file)
        print(f"\nH2O AutoML exploration completed. Log saved to: {log_file}")
