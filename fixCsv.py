# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 00:26:04 2020

@author: Grace
"""

import pandas as pd 
import numpy as np

LABELS=['yes','no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

myresult=pd.read_csv("AAAAA.csv").values
#print(myresult[0][1])
REPLACE=0
REPLACESSS=0
output=[]
for i in range(10500):
    if myresult[i][1] not in LABELS:
        if myresult[i][1] == '_silence_':
            output.append([i+1,'silence'])
            REPLACESSS=REPLACESSS+1
            print("REMOVE_")
        else:
            output.append([i+1,'unknown'])
            REPLACE=REPLACE+1
            print("REPLACE")
    else:
        output.append([i+1,myresult[i][1]])
            
output=pd.DataFrame(output)
output.columns=['id','word']
output.to_csv('AAAAA_mod.csv',index=False) 
       