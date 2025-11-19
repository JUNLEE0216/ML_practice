# 타이타닉 이진분류 
# 1. 데이터 가져와서 데이터프레임 만들기
import pandas as pd
titanic_df = pd.read_csv('/content/Titanic-Dataset.csv')

# 2. 결손값을 평균값으로 채우기 
import numpy as np 
titanic_df['Age'].fillna(np.mean(titanic_df['Age']),inplace=True)

# 3. 불필요한 열을 제거하기
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1, inplace=True)

# 4. 문자열 값을 숫자로 변환하기
titanic_df.loc[titanic_df['Sex'] == 'male', 'Sex'] = 1
titanic_df.loc[titanic_df['Sex'] == 'female', 'Sex'] = 0

titanic_target = titanic_df['Survived'].to_numpy()
titanic_feature = titanic_df.drop('Survived',axis=1).to_numpy()

# 5. 모델 선정 및 학습 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train_input, test_input, train_target, test_target = \
train_test_split(titanic_feature, titanic_target, test_size=0.2, random_state=11)
lr = LogisticRegression(random_state=11)
lr.fit(train_input , train_target)
print("훈련세트 점수 :", lr.score(train_input , train_target))
print("테스트세트 점수 :", lr.score(test_input , test_target))

# 6. 하이퍼파라미터 튜닝으로 마무리 
from sklearn.model_selection import GridSearchCV, StratifiedKFold

params = {
    'max_iter':[10, 100, 500, 1000],
    'C':[0.001, 0.01, 0.1, 1, 10, 100]
}

grf = GridSearchCV(lr, params, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=9), n_jobs=-1)
grf.fit(train_input , train_target)

print('GridSearchCV 최적 하이퍼 파라미터 :',grf.best_params_)
best_model = grf.best_estimator_
print("훈련세트 점수 :", best_model.score(train_input , train_target))
print("테스트세트 점수 :", best_model.score(test_input , test_target))