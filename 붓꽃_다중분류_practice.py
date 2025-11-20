from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_target = iris.target
train_input, test_input, train_target, test_target = train_test_split(iris_df, iris_target, stratify=iris_target, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][gs.best_index_])
params = {
    'max_depth': range(2, 10),
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 10),
    'min_impurity_decrease': np.arange(0.0001, 0.01, 0.0005)
}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))
from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)
np.unique(rgen.rvs(1000),return_counts=True)
ugen = uniform(0, 1)
ugen.rvs(10)
params = {'min_impurity_decrease': uniform(0.0001, 0.001),'max_depth': randint(20, 50),'min_samples_split': randint(2, 25),'min_samples_leaf': randint(1, 25),}
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=1000, n_jobs=-1, random_state=42)
rs.fit(train_input, train_target)
print(rs.best_params_)
print(np.max(rs.cv_results_['mean_test_score']))
dt = rs.best_estimator_
print(dt.score(test_input, test_target))