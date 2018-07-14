# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 01:27:52 2018

@author: J. Albero Gonzalez P
"""

from binance.client import Client
from binance.enums import *
import pprint
import datetime

client = Client("m5BhiGYiD8wbcHZFjKY6rEbvZkOzI1bMx72dE8aBNOgJXrQVPtQI3SeZmVWdodtE", "ZBI24QxwdKMioKha7le4l8Km3x78f7XZ9BabEFwQBMz01bS37AouvgqNKtLY64Ty")

pp = pprint.PrettyPrinter(indent=4)
candles = client.get_klines(symbol='BCPTBTC', interval=KLINE_INTERVAL_1DAY)
for i in candles:
	print(i[4])
