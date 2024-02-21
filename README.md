# Pitchers

## scrape.py

This code scrapes pitcher strikeout total prop bets for the day, and puts them into a [spreadsheet](https://docs.google.com/spreadsheets/d/10qq5okYIgb8XchBUVqbWQGZRv_AcuPfnct3Ah0rY0vI/edit?usp=sharing), along with relevant data about the pitcher and the opposing team *Note that you must download [the pybaseball repository](https://github.com/jldbc/pybaseball) to run this code, as well as pitchers.json

## r_test.Rmd

This code is a test regression model and PCA analysis of the data

## ml_tests.py

This file runs the machine learning tests to predict outcomes

### Current best model

RandomForestClassifier(max_depth=2, n_estimators=10), 56.32% accuracy
