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

mean_business = df_processed[df_processed['business'] == 1].mean()
mean_nonbusiness = df_processed[df_processed['business'] == 0].mean()
std_business = df_processed[df_processed['business'] == 1].std()
std_nonbusiness = df_processed[df_processed['business'] == 0].std()

def collect_outliers(key):
    outliers.extend(df_processed.loc[(df_processed[key] > mean_business[key] + 3 * std_business[key]) & (df_processed['business'] == 1)].index)
    outliers.extend(df_processed.loc[(df_processed[key] < mean_business[key] - 3 * std_business[key]) & (df_processed['business'] == 1)].index)
    outliers.extend(df_processed.loc[(df_processed[key] > mean_nonbusiness[key] + 3 * std_nonbusiness[key]) & (df_processed['business'] == 0)].index)
    outliers.extend(df_processed.loc[(df_processed[key] < mean_nonbusiness[key] - 3 * std_nonbusiness[key]) & (df_processed['business'] == 0)].index)

collect_outliers('cEntire')
collect_outliers('tEntire')
collect_outliers('sEntire')

outliers = list(set(outliers))

# Filter outliers from df
def filter_outliers_from_df(df: pd.DataFrame, outliers):
    return df.drop(outliers)

X_model = filter_outliers_from_df(X_model, outliers)
Y_model = filter_outliers_from_df(Y_model, outliers)

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

# Construct date processors
def daterange_between_month(prev_month: int, length: int, prefix: str = "") -> List[str]:
    dates = []
    for i in range(length):
        dates.append(prefix + f"20220{prev_month}{pd.Timestamp(year=2022, month=prev_month, day=1).days_in_month - i}")
    dates.reverse()
    for i in range(length):
        dates.append(prefix + f"20220{prev_month + 1}0{i + 1}")
    return dates

print("Data preprocessing...")
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
    4, # 5-6
    2, # 6-7
    1.5, # 7-8
]

for i in range(7):
    for prefix in ["c", "t", "s"]:
        numerator = X_model[daterange_between_month(i + 1, 3, prefix)].fillna(0)
        _, denominator = entire[prefix][0]
        base = numerator.T / denominator
        if bs_weights[i] == 0:
            continue
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
## Trial 11

### Included

Same as Trial 10. But used LGBM.

### Result

- Best params
```python
{'n_estimators': 127, 'max_depth': 11, 'learning_rate': 0.03178782439774268, 'num_leaves': 555, 'min_data_in_leaf': 216, 'max_bin': 338, 'lambda_l1': 0.03193633496303601, 'lambda_l2': 0.7759951111876493, 'min_child_weight': 6, 'bagging_fraction': 0.2642872551619603, 'pos_bagging_fraction': 0.7331443462867194, 'neg_bagging_fraction': 0.48314405761757573}
```
- Average ROC AUC Score 0.8615170495612592
- Standard Deviation of ROC AUC Score 0.0006766505564950517
"""