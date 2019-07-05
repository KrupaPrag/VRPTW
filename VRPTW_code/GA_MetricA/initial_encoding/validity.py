#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:46:08 2018

@author: krupa
"""

#import numpy as np
#%%
# =============================================================================
#   Given a route [depot, c1, ....cn, 0]
#       check validity of the route: time, capacity constraints. 
#       want to see if the sequence of the route is valid
# =============================================================================

#%%
def route_validity(route, arr_distance_matrix, arr_customers, total_time, total_capacity):
    route_validity_status = True
    curr_dist = arr_distance_matrix[route[0], route[1]]
    curr_capacity = arr_customers[route[1]-1, 1] #demand of customer (route[1]) -1 index postion
    ready_time = arr_customers[route[1]-1, 3]
    due_time = arr_customers[route[1]-1, 4]
    curr_time = arr_distance_matrix[route[0], route[1]] 
    waiting_time = max(0, ready_time-curr_time)
    curr_time += waiting_time + arr_customers[route[1]-1, 2] #service time
    
    route_customers = route[1:-1]
    for i in range(len(route_customers)-1):
        curr_dist += arr_distance_matrix[route_customers[i], route_customers[i+1]]
        curr_time += arr_distance_matrix[route_customers[i], route_customers[i+1]]
        ready_time = arr_customers[route_customers[i+1]-1, 3]
        due_time = arr_customers[route_customers[i+1]-1, 4]
        service_time = arr_customers[route_customers[i+1]-1, 2]
        curr_capacity += arr_customers[route_customers[i+1]-1, 1]
        if curr_time<=due_time and curr_capacity<total_capacity:
            #accept
            waiting_time = max(0, ready_time - curr_time)
            if (curr_time + waiting_time>=ready_time) and (curr_time + waiting_time + service_time <= total_time):
                #accept next customer 
                curr_time += waiting_time + service_time
            else:
                #route invalid
                route_validity_status = False
                break
        else:
            #route invalid
            route_validity_status = False
            break
    
    if route_validity_status == True:
        curr_dist += arr_distance_matrix[route[-2], route[-1]]
        curr_time += arr_distance_matrix[route[-2], route[-1]]
        if curr_time>total_time:
            route_validity_status = False
            
    return route_validity_status, curr_dist