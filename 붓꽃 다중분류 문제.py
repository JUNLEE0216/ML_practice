# 붓꽃 다중분류 문제 with 트리

# 1. 데이터셋 불러와서 훈련세트와 테스트세트로 나누기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
train_input, test_input, train_target, test_target = train_test_split(iris.data, iris.target,
                                                    test_size=0.2, random_state=119)

# 2. 결정트리 모델 생성하기
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=112)

# 3. 하이퍼파라미터 튜닝을 위한 하이퍼파라미터 목록 만들기
grid_params = {'max_depth':[2, 3, 4, 5, 6, 7, 8], 'min_samples_split':[2, 3, 4, 5, 6, 7, 8]}

# 4. 그리드서치CV 를 이용한 교차 검증 모델 생성 후 학습
from sklearn.model_selection import GridSearchCV, StratifiedKFold
grid_dt = GridSearchCV(dt, grid_params, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=911), n_jobs=-1)
grid_dt.fit(train_input, train_target)

# 5. 최적의 파라미터 조합 출력
print('GridSearchCV 최적 파라미터:', grid_dt.best_params_)

# 6. 최적의 모델을 따로 뽑아내어 테스트세트에 대한 스코어 출력해보기
best_model = grid_dt.best_estimator_
print("최적 모델의 훈련세트 스코어 :", best_model.score(train_input, train_target))
print("최적 모델의 테스트세트 스코어 :", best_model.score(test_input, test_target))

# 보너스! 예측까지 해봄!
pred = best_model.predict(test_input)
print("예측 결과", pred)
print("실제 타겟", test_target)