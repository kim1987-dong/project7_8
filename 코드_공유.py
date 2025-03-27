# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
df = pd.DataFrame(wine.data, columns=wine.feature_names)

df['target'] = wine.target

feature = df.drop(['target'], axis = 1)
target = df['target']

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2,3,4, 5],
    "min_samples_split": [2,5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,  
    scoring="accuracy",  
    n_jobs=-1 
)

grid_search.fit(X_train, y_train) 


print("Best Hyper-parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Accuracy: {accuracy}")

feature_importance = best_model.feature_importances_
plt.figure(figsize=(20, 5))
plt.bar(feature.columns, feature_importance)
plt.xticks(rotation=45)
plt.title('Feature Importances')
plt.xlabel('Feature')
plt.ylabel('importances')
plt.show()


####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

param_grid = {
    "max_depth": [3, 5, 7, 9, 15],
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [50, 100, 200, 300]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",  # Accuracy 기준
    n_jobs=-1  # 병렬 처리
)

grid_search.fit(X_train, y_train)  


print("Best parameters:", grid_search.best_params_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Accuracy: {accuracy}")


feature_importance = best_model.feature_importances_
plt.figure(figsize=(20, 5))
plt.bar(feature.columns, feature_importance)
plt.xticks(rotation=45)
plt.title('Feature Importances')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
