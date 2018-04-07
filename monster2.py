# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 12:22:32 2018

https://www.kaggle.com/samratp/machine-learning-with-ghouls-goblins-and-ghosts

"""
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics

from sklearn.ensemble import VotingClassifier

train_df = pd.read_csv('./input/mons/train.csv')
test_df = pd.read_csv('./input/mons/test.csv')
combine = [train_df, test_df]

#-----------------------
# Mapero de Valores
#-----------------------
train_df['color'].unique()

color_mapping = {'clear':1, 'green':2, 'black':3, 'white':4, 'blue':5, 'blood':6}
for dataset in combine:
    dataset['color'] = dataset['color'].map(color_mapping)


#-----------------------
# Model Predict y Solve
#-----------------------
df = pd.get_dummies(train_df.drop('type', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(df, train_df['type'], test_size = 0.25, random_state = 0)

# Arbol Decision
dt_clf = DecisionTreeClassifier(random_state = 0)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("\nAccuracy Score is: " + str(metrics.accuracy_score(y_test, y_pred)))

# Random Forest
dt_clf = RandomForestClassifier(n_estimators=100)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("\nRAMDOM FOREST Accuracy Score is: " + str(metrics.accuracy_score(y_test, y_pred)))


#-----------------------
# NEW Model Predict y Solve
#-----------------------
accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)

X_train = pd.get_dummies(train_df.drop('type', axis = 1))
y_train = train_df['type']
X_test = pd.get_dummies(test_df)

params = {'n_estimators':[10, 20, 50, 100], 'criterion':['gini', 'entropy'], 'max_depth':[None, 5, 10, 25, 50]}
rf = RandomForestClassifier(random_state = 0)
clf = GridSearchCV(rf, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))
rf_best = RandomForestClassifier(clf.best_params_)

params = {'n_estimators':[10, 25, 50, 100], 'max_samples':[1, 3, 5, 10]}
bag = BaggingClassifier(random_state = 0)
clf = GridSearchCV(bag, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))
bag_best = BaggingClassifier(clf.best_params_)

params = {'kernel':['linear', 'rbf'], 'C':[1, 3, 5, 10], 'degree':[3, 5, 10]}
svc = SVC(probability = True, random_state = 0)
clf = GridSearchCV(svc, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))
svc_best = SVC(clf.best_params_)

# Voting Classfier
voting_clf = VotingClassifier(estimators=[('rf', rf_best), ('bag', bag_best), ('svc', svc_best)]
                              , voting='hard')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print("\nAccuracy Score for VotingClassifier is: " + str(voting_clf.score(X_train, y_train)))
