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

df = pd.read_csv("pitchers.csv")
df = df.drop('Actual', axis=1)
df = df.dropna()

df = df.drop(["Date", "Pitcher", "Opp. Team"], axis=1)

df_a = df.drop("Over/Under?", axis=1)
X = df_a[list(df_a.columns)]

y = df["Over/Under?"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

df2 = pd.read_csv("pitchers2.csv")
df2 = df2.dropna()

df2_a = df2.drop(["Date", "Pitcher", "Opp. Team", "Over/Under?"], axis=1)

# Predicted outcomes

predicted_y = model.predict(df2_a)

predicted_df = pd.DataFrame({"Date": df2["Date"], "Pitcher": df2["Pitcher"], "Predicted": predicted_y, "Real": df2["Over/Under?"]})

print(sum(predicted_df["Predicted"] == predicted_df["Real"])/len(predicted_df))

print(predicted_df)
