# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:16:52 2018

"""

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

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
from sklearn.externals import joblib

import gc

TAM_VENTANA_SET = 60*24*3
TAM_VENTANA_NEXT = 60*24*1
indice = 'VENBTC'

print (indice, TAM_VENTANA_SET,TAM_VENTANA_NEXT)

df_read = pd.read_csv('./input/data.txt',sep="|")
#df_read = pd.read_csv('../getData/data.txt',sep="|")

df = df_read[df_read['Indice']==indice]

closes = df['Price'].values

closes = closes[-20000:]

del df_read
del df
gc.collect()

data_X = []
data_y = []

for i in range(0,len(closes)-(TAM_VENTANA_NEXT+TAM_VENTANA_SET)):
#for i in range(0,20):
    arrayNow = closes[i:i+TAM_VENTANA_SET]
    data_X.append(arrayNow)
    
    arrayNext = closes[i+TAM_VENTANA_SET:i+TAM_VENTANA_SET+TAM_VENTANA_NEXT]
    cierre = closes[i+TAM_VENTANA_SET]
    maximo = max(arrayNext)
    minimo = min(arrayNext)
    benefWin = (maximo-cierre)/cierre
    benefLoss = (minimo-cierre)/cierre
    if (benefLoss<-0.03):
        data_y.append(-1)
    else:
        if (benefWin>0.03):
            data_y.append(1)
        else:
            data_y.append(0)
    
#-----------------------
# Model Predict y Solve
#-----------------------

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.25, random_state = 1)

# Arbol Decision
dt_clf = RandomForestClassifier(n_estimators=100)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
# print("\nAccuracy Score is: " + str(metrics.accuracy_score(y_test, y_pred)))

"""
cm = confusion_matrix(y_test, y_pred)
_ = sns.heatmap(cm, square = True, xticklabels = ["-1", "0", "1"], annot = True, annot_kws = {"fontsize": 13}, yticklabels = ["-1", "0", "1"], cbar = True, cbar_kws = {"orientation": "horizontal"}, cmap = "Blues").set(xlabel = "predicted type", ylabel = "true type", title = "Confusion Matrix")
"""

contar_win = 0
contar_lost = 0
for i in range(0,len(y_test)):
    if (y_test[i]>0):
        if (y_test[i]==y_pred[i]):
            contar_win = contar_win + 1
        else:
            contar_lost = contar_lost + 1
print("\nAccuracy INVERSION is: " + str(contar_win/(contar_win+contar_lost)))
print("\nTOTAL INVERSION is: " + str((contar_win+contar_lost)/len(y_test)))

# Si estamos por encima del 95 creamos el fichero para invertir
a = contar_win/(contar_win+contar_lost)
if (a>0.95):
    joblib.dump(dt_clf, indice+'.pkl')


