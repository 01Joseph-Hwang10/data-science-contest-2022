# 데사경 관련 노트

## 데이터 정리

### 데이터 목록

- X_model.csv : Input (for modeling)
- Y_model.csv : Label (for modeling)
- X_exam.csv : Input (for testing)
- submission.csv : Prediction (if small business owner or not) + decision making (popup, survey)

### 데이터 Feature 목록

- gender: 성별
- age_code: 나이
- region_code: 지역
- /c[0-9]{8}/g: 로그인 횟수
- /t[0-9]{8}/g: Money Transfer가 있었던 로그인 횟수
- /s[0-9]{8}/g: 체류 시간

## Preliminaries

- AUC-ROC Curve: <https://bioinformaticsandme.tistory.com/328>

## Task 1: Prediction

### References

- Covid19 Infection Predicting Model: <https://link.springer.com/content/pdf/10.1007/s42979-020-00394-7.pdf> <- 크게 보아 어떤 종류의 사람인지 예측하는 모델. 여기서는 Decision Tree Prediction Model이 제일 성능이 좋았다 함. 그 외 Logistic Regression, SVM, Naive Bayes 등이 있었음.
- Hyperparameter Tuning of Decision Tree (CART)
  - <https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680>
  - <https://towardsdatascience.com/tuning-hyperparameters-with-optuna-af342facc549>
- 개인사업자 세금 정리: <https://lifelongnews.kr/%EA%B0%9C%EC%9D%B8%EC%82%AC%EC%97%85%EC%9E%90-%EC%84%B8%EA%B8%88/>
- Cross Validation
  - <https://stackoverflow.com/questions/59453363/what-is-the-difference-of-roc-auc-values-in-sklearn>
  - <https://towardsdatascience.com/tuning-hyperparameters-with-optuna-af342facc549>

### 가설1: 세금과 공과 및 사업비 처리기간 관련

가설: 개인사업자(=대개는 소상공인)은 대개 세금과 공과 및 사업비 처리를 개인이 도맡아 하는 경우가 많다. 따라서 사업비 처리 기간(e.g. 월급 지급 기간, 종소세 신고 기간 등)에 앱 접속이 늘어날 것이다.

위의 가설을 검증하기 위해 1월부터 8월까지의 기간을 주요 세금과 공과 및 사업비 처리 기간으로 쪼개어 각각의 기간에 대해 접속 횟수를 살펴본다. 이를 위한 데이터 Pre-Processing이 필요하다.

##### 주요 세금과 공과 및 사업비 처리 기간 정리

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

#### 현재까지 결과

Test train split
- plain: 0.7624288704894253
- hypothesis1: 0.8514274891702154 <- 유의미한 정확도 상승

Cross validation
- plain: `<unknown>` (계산 너무 오래걸림)
- hypothesis1: 0.8610744690217353 (std ~= 0)

#### Potential Improvements

- class_weight에 구체적인 값 넣기
- Feature Training에 집중해보는 것은 어떨까
  - 나이 성별을 추가할때 고려할것
    - 20대 남자의 경우 2년정도 공백기가 있을때 군필로 분류
    - 연령대가 높은 경우는 중요도 낮게 (오히려 반대일수도...?)
  - 다른 요소 추가
  - 성별 제외
  - 또 다른, 소상공인의 패턴일 것으로 추정되는 feature derivation
  - 그 외 feature engineering

#### 세부가설

1. 소상공인은 주요 세금과 공과 및 사업비 처리 기간에 t와 s가 늘어날 것이다. (우선 종소세, 부가세부터 고려)
2. ~~소상공인은 월급날 t와 s가 늘어날 것이다.~~
3. 소상공인은 대체로 t와 s가 높을 것이다.
4. 소상공인은 종소세 기간에 c가 높을 것이다.

## Task 2: Ad Pop-up Planning

개인사업자 중 고르는 방법

- Rule based? (사용 시간, 로그인 횟수, 나이 등)
- Model based?
- 팝업을 클릭하는 사람의 특성?
  - 뱅킹 서비스를 사용하는 횟수가 적은 사람에게 설문조사를 진행할까?

## Task 3: Survey Planning

(LP를 활용할 수 있지 않을까?)

개인사업자 중 고르는 방법 <- 어느 정도 확률이 있는 사람에게 진행

- Rule based? (사용 시간, 로그인 횟수, 나이 등) <- 빈도수가 특정 threshold 이상인 사람에게만 설문조사를 진행
- Model based? <- 로그인 접속 기록에 따라 (e.g. 월말 접속 높은 사람. 주말에 접속 높은 사람 등), 기간 고려
- 설문조사에 응답하는 사람의 특성?
  - 기프티콘: 상대적으로 어린 연령대가 많이 참여할까? <- (Cost 정도로만 생각해도 될듯하다.)
  - 가설1에서 언급하는 기간에 해당하지 않는 여유로운 기간에 설문조사를 진행할까?
  - 뱅킹 서비스를 사용하는 횟수가 많은 사람에게 설문조사를 진행할까?

