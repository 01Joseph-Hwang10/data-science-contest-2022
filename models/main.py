import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial
from typing import List, Any

# Data Loading
print("Loading data...")
X_model = pd.read_csv('../data/X_model.csv')
Y_model = pd.read_csv('../data/Y_model.csv')

# Define preprocessors
print("Defining preprocessors...")
def column(colnames: List[str]):
    def _column(X: pd.DataFrame):
        X = X.fillna(0)
        return [
            [colname, X[colname].values] for colname in colnames
        ]
    return _column

def rangesum(name:str, regex: str, prefixes: str):
    def _rangesum(X: pd.DataFrame):
        X = X.fillna(0)
        return [
            [
                prefix + name, 
                np.sum(X.filter(regex=(prefix + regex), axis=1).values, axis=1)
            ] for prefix in prefixes
        ]
    return _rangesum

def array_divide(numerator: List[Any], denominator: List[Any]) -> List[Any]:
    assert len(numerator) == len(denominator)
    return [
        [
            "r" + numerator_colname, np.divide(numerator_col, denominator_col)
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

    return X_new

print("Data preprocessing...")
abs_GIT = rangesum('GIT', r"202205[0-9]{2}", "cts")(X_model)
abs_VAT = rangesum('VAT', r"20220[17](?:[01][0-9]|2[0-5])", "ts")(X_model)
entire = rangesum('Entire', r"2022[0-9]{4}", "cts")(X_model)

X_processed = preprocess(
    X_model, 
    [
        column(['age_code']),
        one_hot_encode('gender'),
        one_hot_encode('region_code'),
        abs_GIT,
        abs_VAT,
        entire,
        array_divide(abs_GIT, entire), # rel_GIT
        array_divide(abs_VAT, entire[1:]), # rel_VAT
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
