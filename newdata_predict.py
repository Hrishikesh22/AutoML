import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import time
from collections import defaultdict
import random
import warnings
import argparse
import optuna

warnings.filterwarnings("ignore")

def print_log(message, log_file=None):
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')

def is_target_categorical(y):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    unique_vals = np.unique(y) if isinstance(y, np.ndarray) else y.unique()
    return pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or len(unique_vals) < 20

def impute_categorical_columns(df, log_file=None, encoder_dict=None):
    if encoder_dict is None:
        encoder_dict = {}

    for column in df.select_dtypes(include=['object', 'category']).columns:
        imputer = SimpleImputer(strategy='most_frequent')
        df[column] = imputer.fit_transform(df[[column]]).ravel()
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoder_dict[column] = encoder
        print_log(f"Imputed and encoded column: {column}\n", log_file)
    return df, encoder_dict

def feature_selection(X, y, task_type, top_k=10):
    if task_type == 'regression':
        df = X.copy()
        df['target'] = y
        corrs = df.corr(numeric_only=True)['target'].dropna().drop('target')
        selected_features = corrs.abs().sort_values(ascending=False).head(top_k).index.tolist()
        return X[selected_features]
    
    elif task_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features, index=X.index)

def fit_and_evaluate_model_manual(X_train, X_test, y_train, log_file=None, training_time=None, output_path=None):
    if training_time is None:
        training_time = defaultdict(float)

    is_regression = not is_target_categorical(y_train)

    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2, random_state=42),
        "LinearSVR": LinearSVR(max_iter=10000, C=1.0, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto'),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    } if is_regression else {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=2, random_state=42),
        "LinearSVC": LinearSVC(max_iter=10000, C=1.0, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    }

    best_score = -np.inf
    best_model = None
    best_model_name = None

    for model_name, model in models.items():
        print_log(f"\nTraining {model_name} on training data (manual)...", log_file)

        start_time = time.time()
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2' if is_regression else 'accuracy').mean()
        cv_time = time.time() - start_time
        training_time[model_name] += cv_time
        print_log(f"{model_name} CV score (manual): {cv_score:.4f}", log_file)
        print_log(f"{model_name} CV time (manual): {cv_time:.2f} seconds", log_file)

        if cv_score > best_score:
            best_score = cv_score
            best_model = model
            best_model_name = model_name

    print_log(f"\nBest model (manual): {best_model_name} with CV score: {best_score:.4f}", log_file)
    start_time = time.time()
    best_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    training_time[best_model_name] += train_time
    print_log(f"{best_model_name} retraining time: {train_time:.2f} seconds", log_file)

    y_pred = best_model.predict(X_test)
    np.save(output_path, y_pred)
    print_log(f"Saved predictions to {output_path}", log_file)


def get_param_space(model_name, trial=None):
    if model_name == "KNeighborsClassifier":
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15) if trial else random.choice([5, 7, 10]),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']) if trial else random.choice(['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2) if trial else random.choice([1, 2]),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']) if trial else random.choice(['auto', 'ball_tree', 'kd_tree']),
        }

    elif model_name == "DecisionTreeClassifier":
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 20) if trial else random.choice([5, 10, 15]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10) if trial else random.choice([2, 5, 8]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10) if trial else random.choice([1, 2, 4]),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) if trial else random.choice(['sqrt']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']) if trial else random.choice(['gini']),
        }
    elif model_name == "GradientBoostingClassifier":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 15) if trial else random.choice([100, 200]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2) if trial else random.uniform(0.05, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10) if trial else random.choice([3, 5, 7]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10) if trial else random.choice([2, 5, 8]),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0) if trial else random.uniform(0.8, 1.0),
        }

    elif model_name == "RandomForestClassifier":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150) if trial else random.choice([100, 200, 300]),
            'max_depth': trial.suggest_int('max_depth', 5, 20) if trial else random.choice([5, 10, 15]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10) if trial else random.choice([2, 5]),
        }

    elif model_name == "XGBClassifier":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150) if trial else random.choice([100, 200]),
            'max_depth': trial.suggest_int('max_depth', 3, 10) if trial else random.choice([3, 6, 9]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3) if trial else random.uniform(0.05, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0) if trial else random.uniform(0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0) if trial else random.uniform(0.7, 1.0),
        }

    elif model_name == "RandomForestRegressor":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150) if trial else random.choice([100, 200, 300]),
            'max_depth': trial.suggest_int('max_depth', 5, 20) if trial else random.choice([5, 10, 15]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10) if trial else random.choice([2, 5]),
        }

    elif model_name == "XGBRegressor":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150) if trial else random.choice([100, 200]),
            'max_depth': trial.suggest_int('max_depth', 3, 10) if trial else random.choice([3, 6, 9]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3) if trial else random.uniform(0.05, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0) if trial else random.uniform(0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0) if trial else random.uniform(0.7, 1.0),
        }
    
    elif model_name == "KNeighborsRegressor":
        return {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 15) if trial else random.choice([5, 7, 10]),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']) if trial else random.choice(['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2) if trial else random.choice([1, 2]),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']) if trial else random.choice(['auto', 'ball_tree', 'kd_tree']),
    }

    elif model_name == "DecisionTreeRegressor":
        return {
        'max_depth': trial.suggest_int('max_depth', 3, 20) if trial else random.choice([5, 10, 15]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10) if trial else random.choice([2, 5, 8]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10) if trial else random.choice([1, 2, 4]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) if trial else random.choice(['sqrt']),
        'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error']) if trial else random.choice(['squared_error']),
    }

    elif model_name == "GradientBoostingRegressor":
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 15) if trial else random.choice([100, 200]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2) if trial else random.uniform(0.05, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10) if trial else random.choice([3, 5, 7]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10) if trial else random.choice([2, 5, 8]),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0) if trial else random.uniform(0.8, 1.0),
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


def bayes_opt_model(model_cls, model_name, X, y, is_regression, n_trials=10, log_file=None):
    def objective(trial):
        params = get_param_space(model_name, trial)
        model = model_cls(**params)
        score = cross_val_score(model, X, y, cv=5, scoring='r2' if is_regression else 'accuracy').mean()
        return score

    direction = 'maximize'
    study = optuna.create_study(direction=direction)
    print_log(f"Starting Bayesian Optimization for {model_name} with {n_trials} trials...", log_file)
    study.optimize(objective, n_trials=n_trials)
    print_log(f"Best params for {model_name} found by Bayesian Optimization: {study.best_params}", log_file)
    print_log(f"Best CV score: {study.best_value:.4f}", log_file)
    return study.best_params, study.best_value


def random_search_model(model_cls, model_name, X, y, is_regression, n_iter=10, log_file=None):
    best_score = -np.inf
    best_params = None
    print_log(f"Starting Random Search for {model_name} with {n_iter} iterations...", log_file)

    for i in range(n_iter):
        params = get_param_space(model_name)
        model = model_cls(**params)
        score = cross_val_score(model, X, y, cv=5, scoring='r2' if is_regression else 'accuracy').mean()
        print_log(f"Iteration {i+1}: params={params}, CV score={score:.4f}", log_file)
        if score > best_score:
            best_score = score
            best_params = params

    print_log(f"Best params for {model_name} found by Random Search: {best_params}", log_file)
    print_log(f"Best CV score: {best_score:.4f}", log_file)
    return best_params, best_score

from hyperopt import hp
from hyperopt.pyll.base import scope

def get_hyperopt_space(model_name):
    if model_name == "RandomForestClassifier":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 25, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 20, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
        }
    elif model_name == "XGBClassifier":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 25, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }
    elif model_name == "RandomForestRegressor":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 25, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 20, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
        }
    elif model_name == "XGBRegressor":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 25, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }
    elif model_name == "KNeighborsClassifier" or model_name == "KNeighborsRegressor":
        return {
            'n_neighbors': scope.int(hp.quniform('n_neighbors', 1, 30, 1)),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'p': hp.choice('p', [1, 2])  # 1=Manhattan, 2=Euclidean
        }
    elif model_name == "DecisionTreeClassifier" or model_name == "DecisionTreeRegressor":
        return {
            'max_depth': scope.int(hp.quniform('max_depth', 2, 20, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))
        }
    
    elif model_name == "GradientBoostingClassifier" or model_name == "GradientBoostingRegressor":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 25, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'subsample': hp.uniform('subsample', 0.6, 1.0)
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def fit_and_evaluate_model_hyperopt(X_train, X_test, y_train, log_file=None, training_time=None, output_path=None):
    if training_time is None:
        training_time = defaultdict(float)

    is_regression = not is_target_categorical(y_train)

    model_map = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
    "XGBClassifier": XGBClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor
}

    # Select models to tune based on regression/classification
    models_to_tune = [
    "RandomForestRegressor", "XGBRegressor", "KNeighborsRegressor", 
    "DecisionTreeRegressor", "GradientBoostingRegressor"] if is_regression else [
    "RandomForestClassifier", "XGBClassifier", "KNeighborsClassifier", 
    "DecisionTreeClassifier", "GradientBoostingClassifier"]

    best_score = -np.inf
    best_model = None
    best_model_name = None

    for model_name in models_to_tune:
        model_cls = model_map[model_name]
        print_log(f"\nStarting Hyperopt TPE for {model_name}...", log_file)

        start_time = time.time()
        best_params, score = hyperopt_search_model(model_cls, model_name, X_train, y_train, max_evals=20, log_file=log_file)
        tuning_time = time.time() - start_time
        training_time[model_name] += tuning_time
        print_log(f"Tuning time for {model_name}: {tuning_time:.2f} seconds", log_file)

        if score > best_score:
            best_score = score
            best_model = model_cls(**best_params)
            best_model_name = model_name

    print_log(f"\nBest model (hyperopt): {best_model_name} with score: {best_score:.4f}", log_file)
    start_time = time.time()
    best_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    training_time[best_model_name] += train_time
    print_log(f"Training time for best {best_model_name}: {train_time:.2f} seconds", log_file)

    y_pred = best_model.predict(X_test)
    np.save(output_path, y_pred)
    print_log(f"Saved predictions to {output_path}", log_file)
    
def hyperopt_search_model(model_cls, model_name, X, y, max_evals=20, log_file=None, training_time=None):
    if training_time is None:
        training_time = defaultdict(float)

    is_regression = not is_target_categorical(y)

    space = get_hyperopt_space(model_name)

    def objective(params):
        model = model_cls(**params)
        score = cross_val_score(
            model, X, y, cv=5,
            scoring='r2' if is_regression else 'accuracy'
        ).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()

    print_log(f"Starting Hyperopt TPE for {model_name} with {max_evals} evaluations...", log_file)
    start_time = time.time()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.RandomState(42)
    )

    tuning_time = time.time() - start_time
    training_time[model_name] += tuning_time
    print_log(f"Tuning time for {model_name}: {tuning_time:.2f} seconds", log_file)

    int_params = [
        'n_estimators', 'max_depth', 'min_samples_split',
        'n_neighbors', 'min_samples_leaf']
    best = {k: int(v) if k in int_params else v for k, v in best.items()}

    if 'p' in best:
        best['p'] = [1, 2][best['p']]
    if 'weights' in best:
        best['weights'] = ['uniform', 'distance'][best['weights']]

    best_score = -min(t['result']['loss'] for t in trials.trials)

    print_log(f"Best params found by Hyperopt for {model_name}: {best}", log_file)
    print_log(f"Best CV score: {best_score:.4f}", log_file)

    return best, best_score, training_time

def fit_and_evaluate_model_tuned(X_train, X_test, y_train, optimization="random", log_file=None, training_time=None, output_path=None):
    if training_time is None:
        training_time = defaultdict(float)

    is_regression = not is_target_categorical(y_train)

    model_map = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
    "XGBClassifier": XGBClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor
}

    # Select models to tune based on regression/classification
    models_to_tune = [
    "RandomForestRegressor", "XGBRegressor", "KNeighborsRegressor", 
    "DecisionTreeRegressor", "GradientBoostingRegressor"] if is_regression else [
    "RandomForestClassifier", "XGBClassifier", "KNeighborsClassifier", 
    "DecisionTreeClassifier", "GradientBoostingClassifier"]

    best_score = -np.inf
    best_model = None
    best_model_name = None

    for model_name in models_to_tune:
        model_cls = model_map[model_name]
        print_log(f"\nStarting tuning for {model_name} using {optimization} optimization...", log_file)

        start_time = time.time()
        if optimization == "random":
            best_params, score = random_search_model(model_cls, model_name, X_train, y_train, is_regression, n_iter=10, log_file=log_file)
        elif optimization in ("bayesian", "bo"):
            best_params, score = bayes_opt_model(model_cls, model_name, X_train, y_train, is_regression, n_trials=10, log_file=log_file)
        else:
            raise ValueError(f"Unknown optimization method: {optimization}")
        tuning_time = time.time() - start_time
        training_time[model_name] += tuning_time
        print_log(f"Tuning time for {model_name}: {tuning_time:.2f} seconds", log_file)

        if score > best_score:
            best_score = score
            best_model = model_cls(**best_params)
            best_model_name = model_name

    print_log(f"\nBest model (tuned): {best_model_name} with score: {best_score:.4f}", log_file)
    start_time = time.time()
    best_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    training_time[best_model_name] += train_time
    print_log(f"Training time for best {best_model_name}: {train_time:.2f} seconds", log_file)

    y_pred = best_model.predict(X_test)
    np.save(output_path, y_pred)
    print_log(f"Saved predictions to {output_path}", log_file)

def explore_parquet_folder(folder_path, log_file=None, training_time=None, optimization="manual"):
    folder = Path(folder_path)
    try:
        X_train = pd.read_parquet(folder / 'X_train.parquet')
        y_train = pd.read_parquet(folder / 'y_train.parquet')
        X_test = pd.read_parquet(folder / 'X_test.parquet')
    except Exception as e:
        print_log(f"Error loading files in {folder}: {e}", log_file)
        return

    # Impute categorical columns
    encoder_dict = {}
    X_train, encoder_dict = impute_categorical_columns(X_train, log_file=log_file, encoder_dict=encoder_dict)
    X_test, _ = impute_categorical_columns(X_test, log_file=log_file, encoder_dict=encoder_dict)

    # Impute numeric columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    num_imputer = SimpleImputer(strategy='mean')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # Encode target if classification
    if not is_target_categorical(y_train):
        y_train_enc = y_train.squeeze()
        #y_test_enc = y_test.squeeze()
    else:
        label_encoder = LabelEncoder()
        #all_labels = pd.concat([y_train, y_test], axis=0).squeeze()
        #label_encoder.fit(all_labels)
        y_train_enc = label_encoder.transform(y_train.squeeze())
        #y_test_enc = label_encoder.transform(y_test.squeeze())

    # Feature selection
    task_type = 'regression' if not is_target_categorical(y_train) else 'classification'
    X_train = feature_selection(X_train, y_train_enc, task_type, top_k=10)
    X_test = X_test[X_train.columns]
    print_log(f"Selected features: {X_train.columns.tolist()}", log_file)
    
    print_log(f"\nFolder: {folder.name} | Optimization: {optimization}", log_file)
    if optimization == "manual":
        fit_and_evaluate_model_manual(X_train, X_test, y_train_enc, log_file=log_file, training_time=training_time, output_path=args.output_path)
    elif optimization in ("random", "bayesian", "bo"):
        fit_and_evaluate_model_tuned(X_train, X_test, y_train_enc, optimization=optimization, log_file=log_file, training_time=training_time, output_path=args.output_path)
    elif optimization == "hyperopt":
        fit_and_evaluate_model_hyperopt(X_train, X_test, y_train_enc, log_file=log_file, training_time=training_time, output_path=args.output_path)
    else:
        print_log(f"Unknown optimization method: {optimization}", log_file)

def explore_all_datasets(base_dir, log_file=None, optimization="manual"):
    base = Path(base_dir)
    dataset_count = 0
    fold_count = 0
    training_times = defaultdict(float)

    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Model training log started with optimization={optimization}\n\n")

    for dataset_type in sorted(base.iterdir()):
        if dataset_type.is_dir():
            dataset_count += 1
            print_log(f"\nDataset Type: {dataset_type.name}", log_file)
            for fold in sorted(dataset_type.iterdir(), key=lambda x: int(x.name)):
                if fold.is_dir():
                    fold_count += 1
                    print_log(f"\nExploring fold: {fold.name}", log_file)
                    explore_parquet_folder(fold, log_file, training_times, optimization=optimization)
                else:
                    print_log(f"  Skipping non-folder in dataset_type: {fold.name}", log_file)
        else:
            print_log(f"Skipping non-folder in base dir: {dataset_type.name}", log_file)

    print_log("\n=== Total Training Time per Model (seconds) ===", log_file)
    for model_name, total_time in training_times.items():
        print_log(f"{model_name}: {total_time:.2f} seconds", log_file)
    total_training_time_all_models = sum(training_times.values())
    print_log(f"\nTotal training time for all models: {total_training_time_all_models:.2f} seconds", log_file)




if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Name of the dataset folder within the data directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--optimization", default="manual", choices=["manual", "random", "bayesian", "bo", "hyperopt"],
    help="Optimization method to use (manual, random, bayesian, bo, hyperopt).")
    parser.add_argument("--output-path", required=True, help="Path to save the predictions as .npy.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    log_file = script_dir / f"log_{args.task}.txt"
    dataset_folder = data_path / args.task

    if not dataset_folder.exists():
        print(f"Error: Dataset folder '{args.task}' not found at: {dataset_folder}")
    else:
        explore_all_datasets(dataset_folder, log_file=log_file, optimization=args.optimization)
        print(f"\nPredictions completed. Log saved to: {log_file}")

#run this file as:
# python newdata_predict.py --task my_dataset --output-path results.npy --optimization random/manual/bayesian/bo/hyperopt
# Example usage:
# python newdata_predict.py --task my_dataset --output-path results.npy --optimization bayesian