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

df = pd.read_csv('BusData.txt', header=None)
df.columns = ['rare_ID', 'time', 'id', 'lat', 'log', 'speed']

dfB = df[df['id']=='BJ2413']

# Obtenemos un df mas pequenno
df2 = df[:5000000]
df2.id = df2.id.str.replace("�", "")
dfB = df2[df2['id']=='BX1407']


df3 = pd.read_csv('siri.20121109.csv', header=None)
df3.columns = ['time', 'line_id', 'direction', 'journey_id', 'time_table', 'id_vehicle_journey', 'operator'
               , 'congestion', 'lon', 'lat', 'delay', 'block_id', 'id_vehicle','id_vehicle2', 'id_stop']
df3.head()

