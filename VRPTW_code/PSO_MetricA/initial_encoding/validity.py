#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 21:26:54 2018

@author: krupa
"""

#import numpy as np
#import pandas as pd
#import math as math
#import random
#from random import shuffle
#import collections
#from random import randint
#from random import sample
#import secrets
#import copy
#import operator

# =============================================================================
# CHECK VALIDITY OF ROUTE and time
# =============================================================================

#%%
def validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time):
    cust = route_list[1]
    total_dist = arr_distance_matrix[0, cust]
    readyTime = df_customers.loc[cust, 'readyTime']
    curr_cap = df_customers.loc[cust, 'demand']
    validity = True
    if readyTime- total_dist>0:
        start_time = readyTime-total_dist
    else:
        start_time = 0
    curr_time = start_time + total_dist + df_customers.loc[cust, 'serviceTime']
    completeTime = df_customers.loc[route_list[1],'completeTime']
    for i in range(1,len(route_list)-2):
#        print(i)
        cust1 = route_list[i]
        cust2 = route_list[i+1]
        dist = arr_distance_matrix[cust1, cust2]
        total_dist = total_dist + dist
        readyTime = df_customers.loc[cust2, 'readyTime']
        completeTime = df_customers.loc[cust2, 'completeTime']
        curr_cap = curr_cap + df_customers.loc[cust2, 'demand']
        curr_time = curr_time + dist
        if curr_time<readyTime:
            waiting_time = readyTime - curr_time
            curr_time = curr_time + waiting_time + df_customers.loc[cust2, 'serviceTime']
        else:
            curr_time = curr_time + df_customers.loc[cust2, 'serviceTime']
        
        if curr_time>completeTime or curr_time>total_time or curr_cap>total_capacity:
            validity = False
#            print(route_list[i])
            break
    curr_time = curr_time + arr_distance_matrix[route_list[-2],0]
    total_dist = total_dist + arr_distance_matrix[route_list[-2],0]
    if curr_time>total_time or curr_cap>total_capacity:
        validity = False
        
    duration = curr_time - start_time
    return validity, curr_time, curr_cap, duration, start_time, total_dist


        
        

