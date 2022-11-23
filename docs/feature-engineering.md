# Feature Engineering

## Reference

https://www.kaggle.com/learn/feature-engineering

## Hyphtheses

### 가설1: 세금과 공과 및 사업비 처리기간 관련

#### 주요 세금과 공과 및 사업비 처리 기간 정리

아래 기간들 사이에 다른 모종의 이벤트가 있을 수 있으므로, 이를 고려하여 데이터를 살펴본다.

- 종합소득세(GIT: Global Income Tax): 5/1 ~ 5/31
- 부가가치세(VAT): 7/1 ~ 7/25 || 1/1 ~ 1/25
- 원천세(WT: Withholding Tax): 지급 후 익월 10일까지
- 4대 보험(IPO4: Insurance Premium Of 4): 익월 10일까지 (e.g. 자동이체 신청 가능)
- 지급명세서(POA: Particulars Of Account)
  - 일용근로소득, 간이지급명세서(거주자의 사업소득) (POACAT1: POA Category 1): 매월 6일 ~ 말일
  - 간이지급명세서(근로소득) (POACAT2: POA Category 2): 7/6 ~ 7/31 || 1/6 ~ 1/31
- 월급날(payday): 보통 매월 25일

- 소상공인의 대목
  - 명절 기간
  - 2주전 ~ 휴가철

#### Data Derivation

- 단순 평균?
- weighted average?
  - linear increase? `v.T.dot(np.arange(0, 1, 1 / 31))`
  - quadratic increase? `v.T.dot(np.array(map(lambda x: (x ** 2) / (30 ** 2), np.arange(0, 30, 1))))`
- or something else?

# Trials

현재까지 Trial 1이 베스트

## Trial 1

### Included

- gender
- age_code
- region_code
- abs_GIT ("cts")
- abs_VAT ("ts")
- entire ("cts")
- rel_GIT ("cts")
- rel_VAT ("ts")

### Result

- Best Params
```python
{'splitter': 'best', 'max_depth': 10, 'min_samples_split': 40, 'min_samples_leaf': 16, 'min_weight_fraction_leaf': 0.03601779703372169, 'min_impurity_decrease': 0.00015304860044375827}
```
- Average ROC AUC Score 0.8561500587409817
- Standard Deviation of ROC AUC Score 0.000608925236088104

## Trial 2

### Included

- gender
- age_code
- region_code
- rel_GIT ("cts")
- rel_VAT ("ts")

### Result

- best params
```python
{'splitter': 'best', 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 9, 'min_weight_fraction_leaf': 0.0008085208598893039, 'min_impurity_decrease': 7.502109018302336e-05}
```
- Average ROC AUC Score 0.831186252781977
- Standard Deviation of ROC AUC Score 0.0012018703259274038

## Trial 3

### Included

- gender
- age_code
- region_code
- abs_GIT ("cts")
- abs_VAT ("ts")
- entire ("cts")
- rel_GIT ("cts")
- rel_VAT ("ts")
- payday ("t")

### Result

- best params
```python
{'splitter': 'best', 'max_depth': 5, 'min_samples_split': 30, 'min_samples_leaf': 5, 'min_weight_fraction_leaf': 0.07417521496545934, 'min_impurity_decrease': 0.00019237824620779124}
```
- Average ROC AUC Score 0.8499035301971679
- Standard Deviation of ROC AUC Score 0.000662617805652769


## Trial 4

### Included

- gender (one-hot encoded)
- age_code
- region_code (one-hot encoded)
- abs_GIT ("cts")
- abs_VAT ("ts")
- entire ("cts")
- rel_GIT ("cts")
- rel_VAT ("ts")

### Result

- Best Params
```python
{'splitter': 'best', 'max_depth': 8, 'min_samples_split': 16, 'min_samples_leaf': 14, 'min_weight_fraction_leaf': 0.10577426992062908, 'min_impurity_decrease': 0.0001319117122050295}
```
- Average ROC AUC Score 0.8472392136952935
- Standard Deviation of ROC AUC Score 0.0005747983800463919

## Trial 5

### Included

- gender (one-hot encoded)
- age_code
- region_code (one-hot encoded)
- rel_GIT ("cts")
- rel_VAT ("ts")

### Result

- Best params
```python
{'splitter': 'best', 'max_depth': 5, 'min_samples_split': 14, 'min_samples_leaf': 17, 'min_weight_fraction_leaf': 0.37538175204036917, 'min_impurity_decrease': 0.005138082705659464}
```
- Average ROC AUC Score 0.7286882862727536
- Standard Deviation of ROC AUC Score 0.0018038170656588717


## Trial 6

### Included

- gender (one-hot encoded)
- age_code
- region_code (one-hot encoded)
- abs_GIT ("cts")
- abs_VAT ("ts")
- entire ("cts")

### Result

- Best params
```python
{'splitter': 'best', 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 12, 'min_weight_fraction_leaf': 0.15961839044821002, 'min_impurity_decrease': 0.0006266724192157348}
```
- Average ROC AUC Score 0.8377058119253762
- Standard Deviation of ROC AUC Score 0.0006099597608247859




## Trial 7 (Best)

### Included

- gender (one-hot encoded)
- age_code (standardized)
- region_code (one-hot encoded)
- dist_GIT = abs_GIT * linear_dist ("cts", standardized)
- dist_VAT = abs_VAT * linear_dist ("ts", standardized)
- entire ("cts", standardized)

### Result

- Best params
```python
{'splitter': 'best', 'max_depth': 8, 'min_samples_split': 15, 'min_samples_leaf': 7, 'min_weight_fraction_leaf': 0.01610099364371539, 'min_impurity_decrease': 3.353260135879302e-05}
```
- Average ROC AUC Score 0.8611127338482876
- Standard Deviation of ROC AUC Score 0.00065632895467864

## Trial 8

### Included

- gender (one-hot encoded)
- age_code (standardized)
- region_code (one-hot encoded)
- abs_GIT ("cts", standardized)
- abs_VAT ("ts", standardized)
- entire ("cts", standardized)

### Result

- Best params
```python
{'splitter': 'best', 'max_depth': 4, 'min_samples_split': 34, 'min_samples_leaf': 3, 'min_weight_fraction_leaf': 0.09868801833773594, 'min_impurity_decrease': 0.0006510732133421264}
```
- Average ROC AUC Score 0.8460058027968331
- Standard Deviation of ROC AUC Score 0.0007123669765399855

## Trial 9

### Included

- gender (one-hot encoded)
- age_code (standardized)
- region_code (one-hot encoded)
- b[0-9]+ = rel_b[0-0]+ * triangle_dist ("cts", standardized)
- entire ("cts", standardized)

### Result
