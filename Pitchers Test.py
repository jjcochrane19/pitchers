#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, linear_model, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, roc_curve,
                             accuracy_score, roc_auc_score, RocCurveDisplay)
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
import time


# In[2]:


df = pd.read_csv("pitchers.csv")
df = df.drop(["Date", "Pitcher", "Opp. Team", "Actual"], axis=1)
df = df.dropna()

y = np.array(df["Over/Under?"])
X = np.array(df.drop("Over/Under?", axis=1))


# In[3]:


X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=.2,
                                                  random_state=0, stratify=y)
# of remaining 20%, split in half to get 10% validation, 10% test
X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=.5,
  random_state=0, stratify=y_tmp)


# In[4]:


classifiers = [
    svm.SVC(),
    linear_model.LogisticRegression(max_iter=5000),
    DecisionTreeClassifier(criterion='entropy'),
    KNeighborsClassifier()
]

param_grid = [
    {'kernel': ['linear', 'rbf'], 'C': [0.01, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    {'C': [0.01, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'n_neighbors': [1, 2, 3, 4]}
]

best_accuracy = -1
best_classifier_index = -np.Inf
best_classifier = None


# In[5]:


start = time.time()

for i in [0,1,2,3]:
    clf = GridSearchCV(classifiers[i], param_grid[i])  
    clf.fit(X_train, y_train)  
        
    if best_accuracy == None or clf.best_score_ > best_accuracy:
        best_accuracy = clf.best_score_
        best_classifier_index = i
        best_classifier = clf.best_estimator_
        best_params = clf.best_params_
        
    print(f"{classifiers[i]} Accuracy: {clf.score(X_valid, y_valid)}\n")
        
print(f"Best Classifier: {best_classifier} \nParameters: {best_params} \nAccuracy: {best_accuracy}") 

finish = time.time()

print(f"\nThe total time to run this was {int((finish-start)//60//60)} hour, {int((finish-start)//60)-60} minutes, and {(finish-start)%60:.5} seconds.")


# In[7]:


df2022 = pd.read_csv("pitchers.2022.csv")


# In[8]:


clf = DecisionTreeClassifier(random_state = 0, criterion = "entropy", max_depth = 3)
clf.fit(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)


# In[9]:


df2022 = df2022.drop(["Date", "Pitcher", "Opp. Team"], axis=1)
df2022 = df2022.dropna()

y = np.array(df2022["Over/Under?"])
X = np.array(df2022.drop("Over/Under?", axis=1))
test_accuracy


# In[11]:


df = pd.DataFrame()
df["Predicted"] = best_classifier.predict(X)
df["Actual"] = y
correct = []

for i in range(len(df["Predicted"])):
    if df["Predicted"][i] == df["Actual"][i]:
        correct.append(1)
    else:
        correct.append(0)
        
df["Correct"] = correct
df


# In[16]:


# Lasso

from sklearn.linear_model import Lasso

df = pd.read_csv("pitchers.csv")
df = df.drop(["Date", "Pitcher", "Opp. Team", "Actual"], axis=1)
df = df.dropna()

y = np.array(df["Over/Under?"])
X = np.array(df.drop("Over/Under?", axis=1))


# In[43]:


model = Lasso(alpha=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# In[44]:


y_test


# In[45]:


y


# In[46]:


from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(X)

pca.components_

X1 = pd.DataFrame(pca.transform(X))

X1["Result"] = y


# In[47]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot()

ax.scatter(X1[0], X1[1], c = y)

# Show the plot
plt.show()


# In[48]:


from sklearn.ensemble import RandomForestClassifier

# Train the model
clf = RandomForestClassifier(n_estimators=10000)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# In[ ]:




