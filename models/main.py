import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial
from typing import List, Any, Tuple
from sklearn.preprocessing import MinMaxScaler

# Construct date processors

def daterange_between_month(prev_month: int, length: int, prefix: str = "") -> List[str]:
    dates = []
    for i in range(length):
        dates.append(prefix + f"20220{prev_month}{pd.Timestamp(year=2022, month=prev_month, day=1).days_in_month - i}")
    dates.reverse()
    for i in range(length):
        dates.append(prefix + f"20220{prev_month + 1}0{i + 1}")
    return dates

# Data Loading
print("Loading data...")
X_model = pd.read_csv('data/X_model.csv')
Y_model = pd.read_csv('data/Y_model.csv')

scaler = MinMaxScaler(feature_range=(0,1))

# Define preprocessors
print("Defining preprocessors...")
def column(colnames: List[str]):
    def _column(X: pd.DataFrame):
        X = X.fillna(0)
        return [
            [colname, X[colname].values] for colname in colnames
        ]
    return _column

def rangesum(
    name:str, 
    regex: str, 
    prefixes: str, 
    dist: np.ndarray
):
    def _rangesum(X: pd.DataFrame):
        X = X.fillna(0)
        return [
            [
                prefix + name, 
                X.filter(regex=(prefix + regex), axis=1).values.dot(dist)
            ] for prefix in prefixes
        ]
    return _rangesum

def rangesum_from_list(
    name: str, 
    namelist: List[str], 
    prefix: str,
    dist: np.ndarray,    
):
    def _rangesum_from_list(X: pd.DataFrame):
        X = X.fillna(0)
        return [
            [
                prefix + name, 
                X[namelist].values.dot(dist)
            ]
        ]
    return _rangesum_from_list

def _fillna(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, copy=True, nan=0)

def array_divide(
    numerator: List[Tuple[str, np.ndarray]], 
    denominator: List[Tuple[str, np.ndarray]]
) -> List[Any]:
    assert len(numerator) == len(denominator)
    return [
        [
            "r" + numerator_colname, 
            _fillna(np.divide(numerator_col, denominator_col))
        ] for [numerator_colname, numerator_col], [_, denominator_col] in zip(numerator, denominator)
    ]

def one_hot_encode(column: str) -> pd.DataFrame:
    def _one_hot_encode(X: pd.DataFrame):
        X = X.fillna(0)
        df_dummies = pd.get_dummies(X[column], prefix=column)
        return [
            [colname, df_dummies[colname].values] for colname in df_dummies.columns
        ]
    return _one_hot_encode

def preprocess(X: pd.DataFrame, processors: List[Any]) -> pd.DataFrame:
    X_new = pd.DataFrame()

    for processor in processors:
        for colname, col in processor if type(processor) == type([]) else processor(X):
            X_new[colname] = col

    X_new = X_new.fillna(0)

    X_new = pd.DataFrame(scaler.fit_transform(X_new), columns=X_new.columns)

    return X_new

def equal_dist(length: int) -> np.ndarray:
    return np.ones(length)

def linear_dist(length: int) -> np.ndarray:
    return np.arange(start=0, stop=1, step=1/length)

def triangle_dist(length: int) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(start=0, stop=1, step=1/length),
            np.arange(start=1, stop=0, step=-1/length)
        ]
    )

entire_days = 31 + 29 + 31 + 30 + 31 + 30 + 31 + 25
entire_c = rangesum(
    'Entire', 
    r"2022[0-9]{4}", 
    "c", 
    equal_dist(entire_days)
)(X_model)
entire_t = rangesum(
    'Entire', 
    r"2022[0-9]{4}", 
    "t", 
    equal_dist(entire_days)
)(X_model)
entire_s = rangesum(
    'Entire', 
    r"2022[0-9]{4}", 
    "s", 
    equal_dist(entire_days)
)(X_model)

entire = {
    "c": entire_c,
    "t": entire_t,
    "s": entire_s
}


bs = []

bs_weights = [
    0, # 1-2
    0.9, # 2-3
    1.1, # 3-4
    1, # 4-5
    3, # 5-6
    2, # 6-7
    1.5, # 7-8
]

for i in range(7):
    for prefix in ["c", "t", "s"]:
        numerator = X_model[daterange_between_month(i + 1, 3, prefix)].fillna(0)
        _, denominator = entire[prefix][0]
        base = numerator.T / denominator
        bs.append(
            [
                f"b{prefix}{i + 1}{i + 2}",
                base.T.dot(triangle_dist(3)) * bs_weights[i]
            ]
        )

X_processed = preprocess(
    X_model, 
    [
        column(['age_code']),
        one_hot_encode('gender'),
        one_hot_encode('region_code'),
        bs,
        entire_c,
        entire_t,
        entire_s,
    ]
)

print("Preparing for hyperparameter tuning...")
def _construct_and_cross_validate(**kwargs):
    classifier = DecisionTreeClassifier(
        criterion="gini",
        splitter=kwargs['splitter'],
        max_depth=kwargs['max_depth'],
        min_samples_split=kwargs['min_samples_split'],
        min_samples_leaf=kwargs['min_samples_leaf'],
        min_weight_fraction_leaf=kwargs['min_weight_fraction_leaf'],
        random_state=100,
        min_impurity_decrease=kwargs['min_impurity_decrease'],
        class_weight={0: 1, 1: 14.291397}, # Super imbalanced data
    )

    scores = cross_val_score(
        classifier,
        X_processed,
        Y_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=100),
        scoring='roc_auc' # for binary classification
    )

    return scores

# Task: Hyperparameter tuning with Optuna
def objective(trial: Trial):
    # Construct a DecisionTreeClassifier object
    scores = _construct_and_cross_validate(
        splitter=trial.suggest_categorical('splitter', ['best', 'random']),
        max_depth=trial.suggest_int('max_depth', 1, 10),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 40),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        min_weight_fraction_leaf=trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        min_impurity_decrease=trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
    )

    return scores.mean()

print("Hyperparameter tuning started...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the best parameters
print("Best params")
print(study.best_params)

print("Finalizing model...")
scores = _construct_and_cross_validate(
    splitter=study.best_params['splitter'],
    max_depth=study.best_params['max_depth'],
    min_samples_split=study.best_params['min_samples_split'],
    min_samples_leaf=study.best_params['min_samples_leaf'],
    min_weight_fraction_leaf=study.best_params['min_weight_fraction_leaf'],
    min_impurity_decrease=study.best_params['min_impurity_decrease'],
)

print("Average ROC AUC Score", np.mean(scores))
print("Standard Deviation of ROC AUC Score", np.std(scores))
