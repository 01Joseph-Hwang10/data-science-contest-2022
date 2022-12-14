import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from optuna.trial import Trial
from typing import List, Any, Tuple
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

# Data Loading
print("Loading data...")
X_model = pd.read_csv('data/X_model.csv')
Y_model = pd.read_csv('data/Y_model.csv')

# Filter outliers by "entire"
# Ref: https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
print("Filtering outliers...")
df_base = pd.concat([X_model, Y_model], axis=1)
df_processed = pd.DataFrame(data=df_base, columns=['business'])
df_processed['cEntire'] = df_base.filter(regex="c" + r"2022[0-9]*", axis=1).fillna(0).sum(axis=1)
df_processed['tEntire'] = df_base.filter(regex="t" + r"2022[0-9]*", axis=1).fillna(0).sum(axis=1)
df_processed['sEntire'] = df_base.filter(regex="s" + r"2022[0-9]*", axis=1).fillna(0).sum(axis=1)

outliers = []

def collect_outliers(business: int, key: str):
    df_target = df_processed[df_processed['business'] == business]
    q1, q3 = df_target[key].quantile([0.25, 0.75])
    iqr = q3 - q1
    cutoff = iqr * 1.5
    lower, upper = q1 - cutoff, q3 + cutoff
    _outliers = df_target[(df_target[key] < lower) | (df_target[key] > upper)].index.tolist()
    # print(len(_outliers))
    outliers.extend(_outliers)

collect_outliers(0, 'cEntire')
collect_outliers(1, 'cEntire')
collect_outliers(0, 'tEntire')
collect_outliers(1, 'tEntire')
collect_outliers(0, 'sEntire')
collect_outliers(1, 'sEntire')

outliers = list(set(outliers))

# Filter outliers from df
def filter_outliers_from_df(df: pd.DataFrame, outliers):
    return df.drop(outliers)

X_model = filter_outliers_from_df(X_model, outliers)
Y_model = filter_outliers_from_df(Y_model, outliers)

# Get columns of "c" prefix of last 5 days of month
last_5_days = [
    *[f"c202201{i + 27}" for i in range(5)],
    *[f"c202202{i + 24}" for i in range(5)],
    *[f"c202203{i + 27}" for i in range(5)],
    *[f"c202204{i + 26}" for i in range(5)],
    *[f"c202205{i + 27}" for i in range(5)],
    *[f"c202206{i + 26}" for i in range(5)],
    *[f"c202207{i + 27}" for i in range(5)],
]
last_5_days_sum = X_model.filter(last_5_days, axis=1).fillna(0).sum(axis=1)
# print(last_5_days_sum.head())
last_5_days_sum = last_5_days_sum.sort_values(ascending=False)
# print(last_5_days_sum.index)

# Define scaler
print("Defining scaler...")
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

print("Data preprocessing...")
dist_GIT = rangesum(
    'GIT', 
    r"202205[0-9]{2}", 
    "cts", 
    equal_dist(31)
)(X_model)
dist_VAT = rangesum(
    'VAT', 
    r"20220[17](?:[01][0-9]|2[0-5])", 
    "ts", 
    np.concatenate((equal_dist(25), equal_dist(25)))
)(X_model)
entire_days = 31 + 29 + 31 + 30 + 31 + 30 + 31 + 25
entire = rangesum(
    'Entire', 
    r"2022[0-9]{4}", 
    "cts", 
    equal_dist(entire_days)
)(X_model)

X_processed = preprocess(
    X_model, 
    [
        column(['age_code']),
        one_hot_encode('gender'),
        one_hot_encode('region_code'),
        dist_GIT,
        dist_VAT,
        entire,
        # array_divide(dist_GIT, entire), # rel_GIT
        # array_divide(dist_VAT, entire[1:]), # rel_VAT
    ]
)

print("Preparing for hyperparameter tuning...")
def _construct_and_cross_validate(**kwargs):

    lgbm = LGBMClassifier(
        task = "train",
        objective = "binary", #cross-entropy
        metric = "auc",
        tree_learner = "data",
        random_state=100,
        categorical_feature = [0,1,2],
        class_weight={0: 1, 1: 14.291397},
        n_estimators=kwargs['n_estimators'],
        # to deal with overfitting, very important param
        max_depth=kwargs['max_depth'],
        learning_rate=kwargs['learning_rate'],
        num_leaves=kwargs['num_leaves'],
        min_data_in_leaf=kwargs['min_data_in_leaf'],
        #if max_bin becomes small, the accuracy goes up
        max_bin=kwargs['max_bin'],
        lambda_l1=kwargs['lambda_l1'],
        lambda_l2=kwargs['lambda_l2'],
        # to deal with overfitting
        min_child_weight=kwargs['min_child_weight'],
        #for bagging imbalanced
        bagging_fraction=kwargs['bagging_fraction'],
        pos_bagging_fraction=kwargs['pos_bagging_fraction'],
        neg_bagging_fraction=kwargs['neg_bagging_fraction'],
    )
    #cross validation K=5
    scores = cross_val_score(
        lgbm, 
        X_processed, 
        Y_model, 
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        scoring="roc_auc"
    )
    return scores

# Task: Hyperparameter tuning with Optuna
def objective(trial: Trial):
    # Construct a DecisionTreeClassifier object
    scores = _construct_and_cross_validate(
        n_estimators=trial.suggest_int('n_estimators',100,500),
        # to deal with overfitting, very important param
        max_depth = trial.suggest_int('max_depth',10,20),
        learning_rate = trial.suggest_float('learning_rate',0.02,0.1),
        num_leaves = trial.suggest_int('num_leaves',500,1000),
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf',100,1000),
        #if max_bin becomes small, the accuracy goes up
        max_bin = trial.suggest_int('max_bin',255,350),
        lambda_l1 = trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
        lambda_l2 = trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
        # to deal with overfitting
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        #for bagging imbalanced
        bagging_fraction = trial.suggest_float('bagging_fraction', 0,1),
        pos_bagging_fraction = trial.suggest_float('pos_bagging_fraction', 0,1),
        neg_bagging_fraction = trial.suggest_float('neg_bagging_fraction', 0,1),
    )

    return scores.mean()

print("Hyperparameter tuning started...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Print the best parameters
print("Best params")
print(study.best_params)

print("Finalizing model...")
scores = _construct_and_cross_validate(
    n_estimators=study.best_params['n_estimators'],
    # to deal with overfitting, very important param
    max_depth=study.best_params['max_depth'],
    learning_rate=study.best_params['learning_rate'],
    num_leaves=study.best_params['num_leaves'],
    min_data_in_leaf=study.best_params['min_data_in_leaf'],
    #if max_bin becomes small, the accuracy goes up
    max_bin=study.best_params['max_bin'],
    lambda_l1=study.best_params['lambda_l1'],
    lambda_l2=study.best_params['lambda_l2'],
    # to deal with overfitting
    min_child_weight=study.best_params['min_child_weight'],
    #for bagging imbalanced
    bagging_fraction=study.best_params['bagging_fraction'],
    pos_bagging_fraction=study.best_params['pos_bagging_fraction'],
    neg_bagging_fraction=study.best_params['neg_bagging_fraction'],
)

print("Average ROC AUC Score", np.mean(scores))
print("Standard Deviation of ROC AUC Score", np.std(scores))

"""
## Trial 13

### Included

Every Feature from Trial 7. LGBM was used. Outliers are filtered by IQR range method.

### Result

- Best params
```python
{'n_estimators': 359, 'max_depth': 15, 'learning_rate': 0.02807880125926495, 'num_leaves': 740, 'min_data_in_leaf': 784, 'max_bin': 297, 'lambda_l1': 8.2976573625914, 'lambda_l2': 1.9569776049993248, 'min_child_weight': 8, 'bagging_fraction': 0.7597392077506506, 'pos_bagging_fraction': 0.9634842653813701, 'neg_bagging_fraction': 0.02784978528281623}
```
- Average ROC AUC Score 0.9193422348722748
- Standard Deviation of ROC AUC Score 0.00040388011342969876

### Note

IQR로 아웃라이어를 제거했을때 표면적인 성능은 좋아졌으나, 아웃라이어로 필터링한 데이터의 갯수가 80만개 중 12만개임을 고려해 이러한 outlier 제거가 맞는지는 고민이 필요함.

데이터 필터링은 다음과 같이 프로세스 되었음.
cEntire && business == 0 : 62317
cEntire && business == 1 : 3595
cEntire && business == 0 : 85519
cEntire && business == 1 : 3461
cEntire && business == 0 : 63063
cEntire && business == 1 : 3548

위에서 중복되는 index를 제거하면 12만개의 데이터가 필터링됨.
"""