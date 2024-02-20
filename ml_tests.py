import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data_2022 = pd.read_csv("2022.csv")
data_2023 = pd.read_csv("2023.csv")

data_2022 = data_2022.drop(columns = ["Opp. Team", "Date", "Pitcher"])
data_2023 = data_2023.drop(columns = ["Opp. Team", "Date", "Pitcher", "Actual"])

pitchers_full = pd.concat([data_2022, data_2023])

pitchers_full = pitchers_full.dropna()

X = pitchers_full.drop(columns = "Over/Under?")
y = pitchers_full["Over/Under?"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = [
    SVC(),
    DecisionTreeClassifier(criterion='entropy'),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    LogisticRegression(max_iter=1000) 
]

param_grid = [
    {'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},  
    {'max_depth': [None] + list(range(1, 21))},  
    {'n_neighbors': list(range(1, 21))}, 
    {"n_estimators": [1, 10, 50, 100, 200], "max_depth": [None] + list(range(5, 31, 5))},  
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 
]

best_accuracy = -1
best_classifier_index = -np.Inf
best_classifier = None

for i, classifier in enumerate(classifiers):
    grid_search = GridSearchCV(classifier, param_grid[i], cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    if grid_search.best_score_ > best_accuracy:
        best_accuracy = grid_search.best_score_
        best_classifier_index = i
        best_classifier = grid_search.best_estimator_

print("Best classifier:", best_classifier)
print("Best accuracy:", best_accuracy)