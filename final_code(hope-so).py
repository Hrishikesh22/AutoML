import pandas as pd
import numpy as np
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from hyperopt.pyll.base import scope
import optuna
#from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
#import torch
import time
from collections import defaultdict
import random
import warnings
import argparse

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



def fit_and_evaluate_model_manual(X_train, X_test, y_train, y_test, log_file=None, training_time=None, dataset_model_scores=None):
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

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if not is_regression else None
    )
    
    #X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    def get_param_grid(name):
        if name in ["RandomForestRegressor", "RandomForestClassifier"]:
            return {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        if name in ["XGBoost Regressor", "XGBoost Classifier"]:
            return {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.01]
            }
        if name in ["KNeighborsRegressor", "KNeighborsClassifier"]:
            return {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        if name in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            return {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        if name in ["GradientBoostingRegressor", "GradientBoostingClassifier"]:
            return {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5]
            }
        if name in ["LinearSVR", "LinearSVC"]:
            return {
                'C': [0.1, 1.0, 10.0]
            }
        return {}

    if dataset_model_scores is not None:
        if model_name not in dataset_model_scores:
            dataset_model_scores[model_name] = {'scores': [], 'params': []}
    for model_name, model in models.items():
        print_log(f"Training {model_name} on all features (manual)...", log_file)

        start_time = time.time()
        
        if "XGB" in model_name:
            # Use internal validation set for early stopping
            model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        elif "GradientBoosting" in model_name:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        time_taken = time.time() - start_time
        training_time[model_name] += time_taken
        print_log(f"{model_name} training time (manual): {time_taken:.2f} seconds", log_file)

        y_pred = model.predict(X_test)
        pd.DataFrame(y_pred, columns=["prediction"]).to_csv(f"{model_name}_predictions.csv", index=False)
        print_log(f"{model_name} predictions saved to file.", log_file)

        '''
        if is_regression:
            r2 = r2_score(y_test, y_pred)
            print_log(f"{model_name} R² Score (manual): {r2:.4f}", log_file)
        else:
            acc = accuracy_score(y_test, y_pred)
            print_log(f"{model_name} Accuracy (manual): {acc:.4f}", log_file)
        '''
        print_log(f"Cross-validating {model_name} (manual)...", log_file)
        start_time = time.time()
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2' if is_regression else 'accuracy').mean()
        cv_time = time.time() - start_time
        training_time[model_name] += cv_time
        print_log(f"{model_name} CV score (manual): {cv_score:.4f}", log_file)
        print_log(f"{model_name} CV time (manual): {cv_time:.2f} seconds", log_file)

        param_grid = get_param_grid(model_name)
        if param_grid:
            print_log(f"Starting GridSearchCV for {model_name}...", log_file)
            start_time = time.time()

            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2' if is_regression else 'accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            grid_search_time = time.time() - start_time
            training_time[model_name] += grid_search_time
            print_log(f"GridSearchCV time for {model_name}: {grid_search_time:.2f} seconds", log_file)
            print_log(f"Best params found by GridSearchCV for {model_name}: {grid_search.best_params_}", log_file)
            print_log(f"Best CV score from GridSearchCV: {grid_search.best_score_:.4f}", log_file)
            if dataset_model_scores is not None:
                if dataset_model_scores is not None:
                    dataset_model_scores[model_name]['scores'].append(grid_search.best_score_)
                    dataset_model_scores[model_name]['params'].append(grid_search.best_params_)  
            print_log(f"Retraining {model_name} with best params on entire training set (manual)...", log_file)
            best_model = model.__class__(**grid_search.best_params_)

            start_time = time.time()
            if "XGB" in model_name:
                best_model.fit(
                    X_train_sub, y_train_sub,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            elif "GradientBoosting" in model_name:
                best_model.fit(X_train, y_train)
            else:
                best_model.fit(X_train, y_train)
            retrain_time = time.time() - start_time
            training_time[model_name] += retrain_time
            print_log(f"{model_name} retrain time (manual): {retrain_time:.2f} seconds", log_file)

            y_pred = best_model.predict(X_test)
            pd.DataFrame(y_pred, columns=["prediction"]).to_csv(f"{model_name}_predictions.csv", index=False)
            print_log(f"{model_name} predictions saved to file.", log_file)
            '''
            if is_regression:
                r2 = r2_score(y_test, y_pred)
                print_log(f"{model_name} R² Score (manual retrained): {r2:.4f}", log_file)
            else:
                acc = accuracy_score(y_test, y_pred)
                print_log(f"{model_name} Accuracy (manual retrained): {acc:.4f}", log_file)
            '''
        else:
            print_log(f"No param grid defined for {model_name}, skipping GridSearchCV.", log_file)

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

def fit_and_evaluate_model_hyperopt(X_train, X_test, y_train, y_test, max_evals=20, log_file=None, training_time=None, dataset_model_scores=None):
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

    for model_name in models_to_tune:
        model_cls = model_map[model_name]

        print_log(f"\nStarting tuning for {model_name} using Hyperopt...", log_file)

        # Run Hyperopt tuning
        best_params, best_cv_score, training_time = hyperopt_search_model(
            model_cls, model_name, X_train, y_train,
            max_evals=max_evals, log_file=log_file, training_time=training_time
        )

        # Train best model on full training data
        best_model = model_cls(**best_params)
        print_log(f"Training {model_name} with best hyperparameters on full training data...", log_file)
        start_train = time.time()
        if "XGB" in model_name:
            best_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      early_stopping_rounds=10,
                      verbose=False)
        elif "GradientBoosting" in model_name:
            best_model.fit(X_train, y_train)
        else:
            best_model.fit(X_train, y_train)

        train_duration = time.time() - start_train
        training_time[model_name] += train_duration
        print_log(f"Training time for best {model_name}: {train_duration:.2f} seconds", log_file)

        # Predict on test set
        y_pred = best_model.predict(X_test)
        if is_regression:
            score = r2_score(y_test, y_pred)
            print_log(f"{model_name} Test R² Score: {score:.4f}\n", log_file)
        else:
            score = accuracy_score(y_test, y_pred)
            print_log(f"{model_name} Test Accuracy: {score:.4f}\n", log_file)

    return training_time

def get_hyperopt_space(model_name):
    if model_name == "RandomForestClassifier":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 20, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
        }
    elif model_name == "XGBClassifier":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }
    elif model_name == "RandomForestRegressor":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 20, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
        }
    elif model_name == "XGBRegressor":
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
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
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'subsample': hp.uniform('subsample', 0.6, 1.0)
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

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
        rstate=np.random.default_rng(42)
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

def fit_and_evaluate_model_gpbo(X_train, X_test, y_train, y_test, acq_func, log_file=None, training_time=None, dataset_model_scores=None):
    if training_time is None:
        training_time = defaultdict(float)

    is_regression = not is_target_categorical(y_train)

    model_map = {
        "RandomForestRegressor": RandomForestRegressor,
        "XGBRegressor": XGBRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "XGBClassifier": XGBClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }

    models_to_tune = [
        "RandomForestRegressor", "XGBRegressor", "KNeighborsRegressor"
    ] if is_regression else [
        "RandomForestClassifier", "XGBClassifier", "KNeighborsClassifier"
    ]

    acquisition_functions = ["EI", "PI", "LCB"]

    for model_name in models_to_tune:
        model_cls = model_map[model_name]
        print_log(f"\nStarting tuning for {model_name} using GP-based Bayesian Optimization...", log_file)

        start_time = time()
        best_params, best_score, best_acq_func = None, float('-inf'), None

        for acq_func in acquisition_functions:
            params, score = gp_bo_model(
                model_cls, model_name, X_train, y_train,
                is_regression=is_regression, n_calls=10,
                acq_func=acq_func, log_file=log_file
            )
            if score > best_score:
                best_params, best_score, best_acq_func = params, score, acq_func

        tuning_time = time() - start_time
        training_time[model_name] += tuning_time
        print_log(f"Tuning time for {model_name} (best acq={best_acq_func}): {tuning_time:.2f} seconds", log_file)

        best_model = model_cls(**best_params)
        print_log(f"Training {model_name} with best GP-BO hyperparameters (acq={best_acq_func})...", log_file)

        start_train = time()
        if "XGB" in model_name:
            best_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      early_stopping_rounds=10,
                      verbose=False)
        elif "GradientBoosting" in model_name:
            best_model.fit(X_train, y_train)
        else:
            best_model.fit(X_train, y_train)

        train_time = time() - start_train
        training_time[model_name] += train_time
        print_log(f"Training time for best {model_name}: {train_time:.2f} seconds", log_file)

        y_pred = best_model.predict(X_test)
        if is_regression:
            score = r2_score(y_test, y_pred)
            print_log(f"{model_name} Test R² Score (acq={best_acq_func}): {score:.4f}\n", log_file)
        else:
            score = accuracy_score(y_test, y_pred)
            print_log(f"{model_name} Test Accuracy (acq={best_acq_func}): {score:.4f}\n", log_file)

    return training_time
    
def gp_bo_model(model_cls, model_name, X, y, is_regression, acq_func, n_calls=20, log_file=None):
    space_definitions = {
        "RandomForestClassifier": [
            Integer(10, 150, name='n_estimators'),
            Integer(5, 20, name='max_depth'),
            Integer(2, 10, name='min_samples_split')
        ],
        "XGBClassifier": [
            Integer(50, 150, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree')
        ],
        "KNeighborsClassifier": [
            Integer(3, 15, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Integer(1, 2, name='p'),
            Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm')
        ],
        "RandomForestRegressor": [
            Integer(10, 150, name='n_estimators'),
            Integer(5, 20, name='max_depth'),
            Integer(2, 10, name='min_samples_split')
        ],
        "XGBRegressor": [
            Integer(50, 150, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree')
        ],
        "KNeighborsRegressor": [
            Integer(3, 15, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Integer(1, 2, name='p'),
            Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm')
        ]
    }

    if model_name not in space_definitions:
        raise ValueError(f"GP-BO not supported for model: {model_name}")

    space = space_definitions[model_name]

    @use_named_args(space)
    def objective(**params):
        model = model_cls(**params)
        score = cross_val_score(
            model, X, y, cv=5,
            scoring='r2' if is_regression else 'accuracy'
        ).mean()
        return -score  # Minimize negative score for maximization

    print_log(f"Running GP-BO for {model_name} with acquisition function: {acq_func}", log_file)
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        acq_func=acq_func,
        random_state=42
    )

    best_params = dict(zip([dim.name for dim in space], result.x))
    best_score = -result.fun

    print_log(f"Best params found by GP-BO for {model_name}: {best_params}", log_file)
    print_log(f"Best CV score: {best_score:.4f}", log_file)

    return best_params, best_score

def fit_and_evaluate_model_tuned(X_train, X_test, y_train, y_test, optimization="random", log_file=None, training_time=None, dataset_model_scores=None):
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

    for model_name in models_to_tune:
        model_cls = model_map[model_name]

        print_log(f"\nStarting tuning for {model_name} using {optimization} optimization...", log_file)

        start_time = time.time()
        if optimization == "random":
            best_params, best_score = random_search_model(model_cls, model_name, X_train, y_train, is_regression, n_iter=10, log_file=log_file)
        elif optimization in ("bayesian", "bo"):
            best_params, best_score = bayes_opt_model(model_cls, model_name, X_train, y_train, is_regression, n_trials=10, log_file=log_file)
        else:
            raise ValueError(f"Unknown optimization method: {optimization}")
        tuning_time = time.time() - start_time
        training_time[model_name] += tuning_time

        print_log(f"Tuning time for {model_name}: {tuning_time:.2f} seconds", log_file)

        # Train best model on full training data
        best_model = model_cls(**best_params)
        print_log(f"Training {model_name} with best hyperparameters on full training data...", log_file)
        start_time = time.time()
        if "XGB" in model_name:
            best_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      early_stopping_rounds=10,
                      verbose=False)
        elif "GradientBoosting" in model_name:
            best_model.fit(X_train, y_train)
        else:
            best_model.fit(X_train, y_train)

        train_time = time.time() - start_time
        training_time[model_name] += train_time
        print_log(f"Training time for best {model_name}: {train_time:.2f} seconds", log_file)

        # Evaluate on test data
        y_pred = best_model.predict(X_test)
        if is_regression:
            score = r2_score(y_test, y_pred)
            print_log(f"{model_name} Test R² Score: {score:.4f}\n", log_file)
        else:
            score = accuracy_score(y_test, y_pred)
            print_log(f"{model_name} Test Accuracy: {score:.4f}\n", log_file)


def explore_parquet_folder(folder_path, log_file=None, training_time=None, optimization="manual"):
    folder = Path(folder_path)
    try:
        X_train = pd.read_parquet(folder / 'X_train.parquet')
        X_test = pd.read_parquet(folder / 'X_test.parquet')
        y_train = pd.read_parquet(folder / 'y_train.parquet')
        y_test = pd.read_parquet(folder / 'y_test.parquet')
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
        y_test_enc = y_test.squeeze()
    else:
        label_encoder = LabelEncoder()
        all_labels = pd.concat([y_train, y_test], axis=0).squeeze()
        label_encoder.fit(all_labels)
        y_train_enc = label_encoder.transform(y_train.squeeze())
        y_test_enc = label_encoder.transform(y_test.squeeze())

    # Feature selection
    task_type = 'regression' if not is_target_categorical(y_train) else 'classification'
    X_train = feature_selection(X_train, y_train_enc, task_type, top_k=10)
    X_test = X_test[X_train.columns]
    print_log(f"Selected features: {X_train.columns.tolist()}", log_file)
    
    print_log(f"\nFolder: {folder.name} | Optimization: {optimization}", log_file)
    if optimization == "manual":
        fit_and_evaluate_model_manual(X_train, X_test, y_train_enc, y_test_enc, log_file, training_time)
    elif optimization in ("random", "bayesian", "bo"):
        fit_and_evaluate_model_tuned(X_train, X_test, y_train_enc, y_test_enc, optimization, log_file, training_time)
    elif optimization == "hyperopt":
        fit_and_evaluate_model_hyperopt(X_train, X_test, y_train_enc, y_test_enc, log_file=log_file, training_time=training_time)
    elif optimization == "GPBO":
        for acq_fn in ["EI", "LCB", "PI"]:
            print_log(f"=== Running GP-BO with acq_fun {acq_fn} ->", log_file)
            fit_and_evaluate_model_gpbo(X_train, X_test, y_train_enc, y_test_enc, acq_func=acq_fn, log_file=log_file, training_time=training_time)
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
        dataset_model_scores = {}
        if dataset_type.is_dir():
            dataset_count += 1
            print_log(f"\nDataset Type: {dataset_type.name}", log_file)
            for fold in sorted(dataset_type.iterdir(), key=lambda x: int(x.name)):
                fold_scores = []
                fold_params = []
                if fold.is_dir():
                    fold_count += 1
                    print_log(f"\n[Dataset Summary: {dataset_type.name} | Optimization: tuned]", log_file)
                    for model_name in dataset_model_scores:
                        scores = dataset_model_scores[model_name]['scores']
                        params = dataset_model_scores[model_name]['params']
                        if scores:
                            avg_score = np.mean(scores)
                            best_fold = np.argmax(scores)
                            final_best_params = params[best_fold]
                            print_log(f"Model: {model_name}", log_file)
                            print_log(f"Best Parameters: {final_best_params}", log_file)
                            print_log(f"Average CV Score: {avg_score:.4f}\\n", log_file)
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

'''
def tabnet_nas_optuna(X_train, X_test, y_train, y_test, is_regression, n_trials=15, log_file=None):
    def objective(trial):
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64, step=8),
            "n_a": trial.suggest_int("n_a", 8, 64, step=8),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "optimizer_params": dict(lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True)),
            "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
        }

        model_cls = TabNetRegressor if is_regression else TabNetClassifier
        model = model_cls(**params, device_name='cuda' if torch.cuda.is_available() else 'cpu')

        best_model.fit(
        fold_scores.append(score)
        fold_params.append(grid_search.best_params_)
            X_train.values, y_train if is_regression else y_train.reshape(-1, 1),
            eval_set=[(X_test.values, y_test if is_regression else y_test.reshape(-1, 1))],
            eval_metric=['r2'] if is_regression else ['accuracy'],
            max_epochs=200,
            patience=15,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            verbose=0
        )

        preds = model.predict(X_test.values).squeeze()
        if not is_regression:
            preds = preds.argmax(axis=1)
            score = accuracy_score(y_test, preds)
        else:
            score = r2_score(y_test, preds)

        return score

    study = optuna.create_study(direction="maximize")
    print_log(f"Starting NAS (TabNet + Optuna) with {n_trials} trials...", log_file)
    study.optimize(objective, n_trials=n_trials)

    print_log(f"Best TabNet params: {study.best_params}", log_file)
    print_log(f"Best NAS score: {study.best_value:.4f}", log_file)
'''

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    #data_path = script_dir / "tabular-phase2/exam_dataset/1"
    data_path = script_dir/"data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", default="manual", choices=["manual", "random", "bayesian", "hyperopt", "GPBO"])
    args = parser.parse_args()
    log_file = script_dir / f"log_{args.optimization}.txt"
    if not data_path.exists():
        print(f"Error: data directory not found at: {data_path}")
    else:
        explore_all_datasets(data_path, log_file=log_file, optimization=args.optimization)
        print(f"\nOptimization completed. Log saved to: {log_file}")
