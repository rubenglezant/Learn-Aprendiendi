# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:33:34 2018

URL: https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo

id - id of the creature
bone_length - average length of bone in the creature, normalized between 0 and 1
rotting_flesh - percentage of rotting flesh in the creature
hair_length - average hair length, normalized between 0 and 1
has_soul - percentage of soul in the creature
color - dominant color of the creature: 'white','black','clear','blue','green','blood'
type - target variable: 'Ghost', 'Goblin', and 'Ghoul'

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

train_df = pd.read_csv('./input/mons/train.csv')
test_df = pd.read_csv('./input/mons/test.csv')
combine = [train_df, test_df]

train_df.head()

# Agrupamos para visualizar
train_df[['type', 'hair_length']].groupby(['type'], as_index=False).mean().sort_values(by='hair_length', ascending=False)
train_df[['type', 'hair_length']].groupby(['type'], as_index=False).min().sort_values(by='hair_length', ascending=False)
train_df[['type', 'hair_length']].groupby(['type'], as_index=False).max().sort_values(by='hair_length', ascending=False)

#-----------------------
# Analizar visualizando datos
#-----------------------
g = sns.FacetGrid(train_df, col='type')
g.map(plt.hist, 'color', bins=20)

grid = sns.FacetGrid(train_df, col='type', row='color', size=2.2, aspect=1.6)
grid.map(plt.hist, 'bone_length', alpha=.5, bins=20)
grid.add_legend();

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
X_train = train_df.drop("type", axis=1)
X_train = X_train.drop("id", axis=1).copy()

Y_train = train_df["type"]

X_test  = test_df.drop("id", axis=1).copy()

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


#-----------------------
# Send Results
#-----------------------
submission = pd.read_csv("./input/mons/sample_submission.csv")
submission["type"] = "Unknown"

Y_pred = decision_tree.predict(X_test)

submission["type"] = Y_pred
submission.to_csv("sub.csv", index=False)

"""
kaggle competitions submit -c ghouls-goblins-and-ghosts-boo -f sub.csv -m "First Submission"
"""

