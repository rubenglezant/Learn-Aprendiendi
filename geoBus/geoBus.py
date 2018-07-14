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
    if (totalMin >= 20):
        totalMin = 20
    return totalMin

df = pd.read_csv('BusData.txt', header=None)
df.columns = ['rare_ID', 'time', 'id', 'lat', 'log', 'speed']

dfB = df[df['id']=='BJ2413']

# Obtenemos un df mas pequenno
df2 = df[:5000000]
df2.id = df2.id.str.replace("�", "")
dfB = df2[df2['id']=='BX1407']


df3 = pd.read_csv('siri.20121110.csv', header=None)
df3.columns = ['time', 'line_id', 'direction', 'journey_id', 'time_table', 'id_vehicle_journey', 'operator'
               , 'congestion', 'lon', 'lat', 'delay', 'block_id', 'id_vehicle','id_vehicle2', 'id_stop']
df3.head()

df3['time'] = pd.to_datetime(df3['time']*int(1e3))

# Bus 40037 => Ruta al aeropuerto
# -6.2990
dfB = df3[df3['id_vehicle']==40037]

dfB[dfB['lon']<-6.2990]

f = open ("ruta.txt","w")
lat = []
lon = []
for index, row in dfB.iterrows():
    print (row['lat'],",",row['lon'], file=f)
    lat.append(row['lat'])
    lon.append(row['lon'])
f.close()

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
for i in rutas:
    if ((i - j)>1):
        print (j)
    j = i
        
lat2 = lat[:665]
lon2 = lon[:665]    
plt.plot(lat2, lon2, '+')

lat2 = lat[665:1473]
lon2 = lon[665:1473]    
plt.plot(lat2, lon2, '+')

lat2 = lat[1473:1942]
lon2 = lon[1473:1942]    
plt.plot(lat2, lon2, '+')

lat2 = lat[1942:]
lon2 = lon[1942:]    
plt.plot(lat2, lon2, '+')

# Preparamos los datos para analisis
valores_analisis = []
ruta_lat = lat[:665]
ruta_lon = lon[:665]
ruta_time = tiempo[:665]

tiempoFinal = ruta_time[-1]

for i in range(20,len(ruta_time)):
    td = tiempoFinal-ruta_time[i] 
    v = []
    for j in range(0,20):
        v.append(ruta_lat[i-j])
        v.append(ruta_lon[i-j])
    v.append(totalMinutes(td))
    print (v)
    valores_analisis.append(v)

# Algoritmo de entrenamiento

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

