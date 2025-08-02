import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


from final_code_opt import (
    fit_and_evaluate_model_manual,
    fit_and_evaluate_model_hyperopt,
    fit_and_evaluate_model_gpbo,
    fit_and_evaluate_model_tuned,
    is_target_categorical
)
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

xtrain = pd.read_parquet("X_train.parquet")
ytrain = pd.read_parquet("y_train.parquet").values.ravel()
xtest = pd.read_parquet("X_test.parquet")

imputer = SimpleImputer(strategy='mean')
xtrain_imputed = imputer.fit_transform(xtrain)
xtest_imputed = imputer.transform(xtest)

scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain_imputed)
xtest_scaled = scaler.transform(xtest_imputed)

log_file = "new_data_pipeline_log.txt"
def print_log(msg, log_path):
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)

dataset_model_scores = defaultdict(lambda: {'scores': [], 'params': [], 'acq_funcs' : {}})

# Run all optimization techniques
print_log("\n=== Running Manual Optimization ===", log_file)
fit_and_evaluate_model_manual(xtrain_scaled, xtest_scaled, ytrain, None, dataset_model_scores=dataset_model_scores, log_file=log_file)

print_log("\n=== Running Hyperopt Optimization ===", log_file)
fit_and_evaluate_model_hyperopt(xtrain_scaled, xtest_scaled, ytrain, None, dataset_model_scores=dataset_model_scores, log_file=log_file)

print_log("\n=== Running GPBO Optimization ===", log_file)
fit_and_evaluate_model_gpbo(xtrain_scaled, xtest_scaled, ytrain, None, dataset_model_scores=dataset_model_scores, log_file=log_file)

print_log("\n=== Running Bayesian Tuning (Optuna) ===", log_file)
fit_and_evaluate_model_tuned(xtrain_scaled, xtest_scaled, ytrain, None, optimization="bayesian", dataset_model_scores=dataset_model_scores, log_file=log_file)

# === Select best model across all techniques ===
best_model_name = None
best_score = -np.inf
best_params = None
best_acq_func = None
for model_name, info in dataset_model_scores.items():
    if info['scores']:
        avg_score = np.mean(info['scores'])
        if avg_score > best_score:
            best_score = avg_score
            best_model_name = model_name
            best_params = info['params'][np.argmax(info['scores'])]
            best_acq_func = info.get('acq_funcs', [None])[np.argmax(info['scores'])]

print_log(f"\nBest Model: {best_model_name}", log_file)
print_log(f"Best Parameters: {best_params}", log_file)
print_log(f"Best Average CV Score: {best_score:.4f}", log_file)

if best_acq_func:
        print_log(f"Best Acquisition Function (for GPBO): {best_acq_func}", log_file)

model_dict = {
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'LinearSVR': LinearSVR,
    'KNeighborsRegressor': KNeighborsRegressor,
    'XGBoost Regressor': XGBRegressor,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'LinearSVC': LinearSVC,
    'KNeighborsClassifier': KNeighborsClassifier,
    'XGBoost Classifier': XGBClassifier
}

if len(np.unique(ytrain)) <= 20 and ytrain.dtype in [int, bool]:
    is_regression = False
else:
    is_regression = True

ModelClass = model_dict[best_model_name]
final_model = ModelClass(**best_params)
final_model.fit(xtrain_scaled, ytrain)
ypred = final_model.predict(xtest_scaled)
pd.DataFrame(ypred, columns=["prediction"]).to_csv("ypred.csv", index=False)
print_log("Predictions saved to ypred.csv", log_file)

