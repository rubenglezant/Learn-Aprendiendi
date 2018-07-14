# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 02:18:01 2018

@author: RGA

Show Points. http://www.hamstermap.com/quickmap.php
Data From: 
    http://www-users.cs.umn.edu/~tianhe/BIGDATA/
    https://data.gov.ie/dataset/dublin-bus-gps-sample-data-from-dublin-city-council-insight-project
"""

import pandas as pd
import numpy as np
import datetime as datetime
import time

def totalMinutes(td):
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalMin = hours*60 + minutes
    return totalMin

valores_analisis_X = []
valores_analisis_y = []

for i in range(0,10):
    # Lectura del fichero
    df3 = pd.read_csv('siri.2012111'+str(i)+'.csv', header=None)
    df3.columns = ['time', 'line_id', 'direction', 'journey_id', 'time_table', 'id_vehicle_journey', 'operator'
                   , 'congestion', 'lon', 'lat', 'delay', 'block_id', 'id_vehicle','id_vehicle2', 'id_stop']
    df3.head()
    df3['time'] = pd.to_datetime(df3['time']*int(1e3))
    
    # Creamos array para tratar un BUS
    # Bus 40037 => Ruta al aeropuerto
    # -6.2990
    dfB = df3[df3['id_vehicle']==40037]
    lat = []
    lon = []
    tiempo = []
    for index, row in dfB.iterrows():
        lat.append(row['lat'])
        lon.append(row['lon'])
        tiempo.append(row['time'])
    
    import matplotlib.pyplot as plt
    plt.plot(lat, lon, '+')
    
    # Separamos las rutas
    rutas = []
    for i in range(0,len(lat)):
        if (lon[i]<=-6.299167):
            rutas.append(i)
    j = rutas[0]
    rutas_finales = [0]
    for i in rutas:
        if ((i - j)>1):
            rutas_finales.append(j)
        j = i
    rutas_finales.append(len(lat)-1)
            
    # Preparamos los datos para analisis
    # Anadimos valor para cada ruta
    for ind_ruta in range(0,len(rutas_finales)-1):
        print (rutas_finales[ind_ruta],rutas_finales[ind_ruta+1])
        ruta_lat = lat[rutas_finales[ind_ruta]:rutas_finales[ind_ruta+1]]
        ruta_lon = lon[rutas_finales[ind_ruta]:rutas_finales[ind_ruta+1]]
        ruta_time = tiempo[rutas_finales[ind_ruta]:rutas_finales[ind_ruta+1]]
        
        tiempoFinal = ruta_time[-1]
        
        for i in range(50,len(ruta_time)):
            td = tiempoFinal-ruta_time[i] 
            if (totalMinutes(td)<=10):
                v = []
                for j in range(0,50):
                    v.append(ruta_lat[i-j])
                    v.append(ruta_lon[i-j])
                #print (v)
                valores_analisis_X.append(v)
                valores_analisis_y.append(totalMinutes(td))
                

# Algoritmo de entrenamiento
#Se importa la librería sklearn el módulo tre
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

clf = tree.DecisionTreeClassifier()

X = valores_analisis_X
Y = valores_analisis_y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

clf = clf.fit(X_train, y_train)

acc_decision_tree = round(clf.score(X_test, y_test) * 100, 2)
print(acc_decision_tree)

"""
f = open ("ruta.txt","w")
for index, row in dfB.iterrows():
    print (row['time'],",",row['lat'],",",row['lon'], file=f)
    if (row['lon']<-6.2990):
        break
f.close()
"""

# Hay que obtener los datos para entrenar el algoritmo
# lat1,lon1,lat2,lon2,lat3,lon3,v1,v2,v3,TIEMPO LLEGADA [0:10]

