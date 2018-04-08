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

from sklearn.externals import joblib

from datetime import datetime
import time

time.sleep (20);

TAM_VENTANA_SET = 60*24*3
indice = 'EOSBTC'

#df_read = pd.read_csv('./input/data.txt',sep="|")
df_read = pd.read_csv('../getData/data.txt',sep="|")

df = df_read[df_read['Indice']==indice]

closes = df['Price'].values

closes = closes[-TAM_VENTANA_SET:]

dt_clf = joblib.load(indice+'.pkl') 
y_pred = dt_clf.predict(closes.reshape(1, -1))

print(closes[-1],closes[-2],closes[-3],closes[-4],closes[-5],indice+" - Invertir: " + str(y_pred),str(datetime.now()))

