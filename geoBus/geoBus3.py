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
from math import sin, cos, sqrt, atan2, radians
import os



def totalMinutes(td):
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalMin = hours*60 + minutes
    return totalMin

def totalSegunds(td):
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalSec = hours*60*60 + minutes*60 + seconds
    return totalSec

def calcDistance(la1,lo1,la2,lo2):
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(la1)
    lon1 = radians(lo1)
    lat2 = radians(la2)
    lon2 = radians(lo2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    return distance

valores_analisis_X = []
valores_analisis_y = []

lista_data = []
for fich in os.listdir():
    if (fich.find('csv')>0):
        lista_data.append(fich)
        
    
for nombrefichero in lista_data:
    # Lectura del fichero
    df3 = pd.read_csv(nombrefichero, header=None)
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
    if (len(rutas)<=0):
        continue
    j = rutas[0]
    rutas_finales = [0]
    for i in rutas:
        if ((i - j)>100):
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
        
        """
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
        """
        for i in range(20,len(ruta_time)):
            td = tiempoFinal-ruta_time[i] 
            if ((totalMinutes(td)<=20) and ((ruta_lon[i]-ruta_lon[i-20])>=0)):
                v = []
                """
                for j in range(0,3):
                    vLat = (ruta_lat[i-j] - ruta_lat[i-j-1])/(totalSegunds((ruta_time[i-j] - ruta_time[i-j-1])))
                    vLon = (ruta_lon[i-j] - ruta_lon[i-j-1])/(totalSegunds((ruta_time[i-j] - ruta_time[i-j-1])))
                    #v.append(vLat)
                    #v.append(vLon)
                    v.append(ruta_lat[i-j])
                    v.append(ruta_lon[i-j])

                v.append(ruta_lat[i])
                v.append(ruta_lon[i])
                v.append(ruta_lat[i-1]-ruta_lat[i])
                v.append(ruta_lon[i-1]-ruta_lon[i])

                v.append(ruta_lat[-1]-ruta_lat[i])
                v.append(ruta_lon[-1]-ruta_lon[i])

                """                    
                v.append(ruta_lat[i-20])
                v.append(ruta_lon[i-20])
                v.append(ruta_lat[i-10])
                v.append(ruta_lon[i-10])
                v.append(ruta_lat[i-3])
                v.append(ruta_lon[i-3])
                v.append(ruta_lat[i-2])
                v.append(ruta_lon[i-2])
                v.append(ruta_lat[i-1])
                v.append(ruta_lon[i-1])
                v.append(ruta_lat[i-0])
                v.append(ruta_lon[i-0])

                v.append(ruta_lat[i]-ruta_lat[i-20])
                v.append(ruta_lon[i]-ruta_lon[i-20])

                v.append(calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[i-20],ruta_lon[i-20]))
                v.append(totalSegunds(ruta_time[i-20]-ruta_time[i]))

                v.append(calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[i-10],ruta_lon[i-10]))
                v.append(totalSegunds(ruta_time[i-10]-ruta_time[i]))

                v.append(calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[i-3],ruta_lon[i-3]))
                v.append(totalSegunds(ruta_time[i-3]-ruta_time[i]))

                v.append(calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[i-2],ruta_lon[i-2]))
                v.append(totalSegunds(ruta_time[i-2]-ruta_time[i]))

                v.append(calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[i-1],ruta_lon[i-1]))
                v.append(totalSegunds(ruta_time[i-1]-ruta_time[i]))

                v.append(calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[-1],ruta_lon[-1]))
                #print (v)
                valores_analisis_X.append((calcDistance(ruta_lat[i],ruta_lon[i],ruta_lat[-1],ruta_lon[-1])))
                valores_analisis_y.append(totalMinutes(td))
                

# Algoritmo de entrenamiento
#Se importa la librería sklearn el módulo tree
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

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print(acc_decision_tree)

f = open ("resultados.txt","w")
error = 0
for i in range(0,len(X_test)):
    v = clf.predict([X_test[i]])[0]
    if (abs(y_test[i]-v)>1):
        error = error + 1
        print (X_test[i],y_test[i],v, file=f)
f.close()

print ("Resultados obtenidos:")
print ((1-(error/len(X_test)))*100)
print (len(X_test))


"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=5000, min_samples_leaf=10)
clf_entropy.fit(X_train, y_train)

acc_decision_tree = round(clf_entropy.score(X_test, y_test) * 100, 2)
print(acc_decision_tree)


clf_entropy = tree.DecisionTreeClassifier(criterion = "gini", random_state = 30,
                                     min_samples_leaf=20)
clf_entropy.fit(X_train, y_train)

acc_decision_tree = round(clf_entropy.score(X_test, y_test) * 100, 2)
print(acc_decision_tree)

f = open ("ruta.txt","w")
for index, row in dfB.iterrows():
    print (row['time'],",",row['lat'],",",row['lon'], file=f)
    if (row['lon']<-6.2990):
        break
f.close()

# Hay que obtener los datos para entrenar el algoritmo
# lat1,lon1,lat2,lon2,lat3,lon3,v1,v2,v3,TIEMPO LLEGADA [0:10]


    la1 = (52.2296756)
    lo1 = (21.0122287)
    la2 = (52.406374)
    lo2 = (16.9251681)

print("Result:", )
print("Should be:", 278.546, "km")
"""

dist = []
for index, row in dfB.iterrows():
    a1 = (row['lat'])
    b1 = (row['lon'])
    dist.append((calcDistance(a1,b1,ruta_lat[-1],ruta_lon[-1])))

s = pd.Series(dist)
s.index = dfB.time

s2 = s[1040:1060]

s2.plot(style='ro')
s[1000:2060].plot()
s.plot()

# Recogemos solo los de un SENTIDO
tiempo = dfB.time.values
j = 0
sentido = "baja"
cambio_ruta = []
for i in range(5,len(dist)):
    if ((dist[i]<=0.2) and (sentido == "baja")):
        sentido = "sube"
        cambio_ruta.append(i)
    if ((dist[i]>=9.30) and (sentido == "sube")):
        sentido = "baja"
        cambio_ruta.append(i)
    print (i,tiempo[i],dist[i], sentido)        


valores_analisis_X = []
valores_analisis_y = []

# Tomamos las rutas de bajada
minutos_llegada = []
lat = dfB['lat'].values
lon = dfB['lon'].values
#for i in range(157,215):
for j in range(1,int(len(cambio_ruta))):
    if ((j%2)==0):
        for i in range(cambio_ruta[j-1],cambio_ruta[j]):
            a = tiempo[i]-tiempo[cambio_ruta[j]]
            b = int(a.astype('timedelta64[m]') / np.timedelta64(1, 'm'))*(-1)-1
            if (b>10):
                b = 10
            minutos_llegada.append(b)
            print (i,tiempo[i],dist[i],lat[i],lon[i], b)
            v =[]
            v.append(lat[i-10])
            v.append(lon[i-10])
            v.append(lat[i-5])
            v.append(lon[i-5])
            v.append(lat[i-3])
            v.append(lon[i-3])
            v.append(lat[i-2])
            v.append(lon[i-2])
            v.append(lat[i-1])
            v.append(lon[i-1])
            v.append(lat[i])
            v.append(lon[i])
            valores_analisis_X.append(v)
            
            valores_analisis_y.append(b)
    
    
