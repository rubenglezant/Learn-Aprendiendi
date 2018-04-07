# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 12:53:09 2018

https://www.kaggle.com/tianji/explore-classify-the-goblin-ghost-ghoul-dataset

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report

train_df = pd.read_csv('./input/mons/train.csv')
test_df = pd.read_csv('./input/mons/test.csv')
combine = [train_df, test_df]

#-----------------------
# Model Predict y Solve
#-----------------------

df = pd.get_dummies(train_df.drop('type', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(df, train_df['type'], test_size = 0.25, random_state = 0)

# Arbol Decision
dt_clf = RandomForestClassifier(n_estimators=100)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print("\nAccuracy Score is: " + str(metrics.accuracy_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred)
_ = sns.heatmap(cm, square = True, xticklabels = ["ghost", "ghoul", "goblin"], annot = True, annot_kws = {"fontsize": 13}, yticklabels = ["ghost", "ghoul", "goblin"], cbar = True, cbar_kws = {"orientation": "horizontal"}, cmap = "Blues").set(xlabel = "predicted type", ylabel = "true type", title = "Confusion Matrix")
