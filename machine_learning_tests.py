import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.tree import export_text 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the data

df = pd.read_csv("pitchers.csv")
df = df.drop('Actual', axis=1)
df = df.dropna()

df = df.drop(["Date", "Pitcher", "Opp. Team"], axis=1)

df_a = df.drop("Over/Under?", axis=1)
X = df_a[list(df_a.columns)]

y = df["Over/Under?"]

# Splits

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')

# Additional testing dataframe

df2 = pd.read_csv("pitchers2.csv")
df2 = df2.dropna()

df2_a = df2.drop(["Date", "Pitcher", "Opp. Team", "Over/Under?"], axis=1)

# Predicted outcomes

predicted_y = model.predict(df2_a)

predicted_df = pd.DataFrame({"Date": df2["Date"], "Pitcher": df2["Pitcher"], "Predicted": predicted_y, "Real": df2["Over/Under?"]})

print(f'Logistic Regression Test 2022 Data: {(sum(predicted_df["Predicted"] == predicted_df["Real"])/len(predicted_df))}')

# Repeat but with an SVM

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear', random_state=42)  # You can change the kernel as needed
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy * 100:.2f}%')

predicted_y = svm_classifier.predict(df2_a)

predicted_df = pd.DataFrame({"Date": df2["Date"], "Pitcher": df2["Pitcher"], "Predicted": predicted_y, "Real": df2["Over/Under?"]})

print(f'SVM Test 2022 Data: {(sum(predicted_df["Predicted"] == predicted_df["Real"])/len(predicted_df))}')

# And now with a decision tree

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

decision_tree = DecisionTreeClassifier(random_state=42)  # You can adjust hyperparameters as needed
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy * 100:.2f}%')

predicted_y = decision_tree.predict(df2_a)

predicted_df = pd.DataFrame({"Date": df2["Date"], "Pitcher": df2["Pitcher"], "Predicted": predicted_y, "Real": df2["Over/Under?"]})

print(f'Decision Tree Test 2022 Data: {(sum(predicted_df["Predicted"] == predicted_df["Real"])/len(predicted_df))}')
