#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:41:23 2018

@author: krupa

"""
#%%
# =============================================================================
# PHASE II ROUTING:
#   1. REMOVE LAST CUSTOMER OF R1
#   2. ADD LASTE CUSTOMER TO BE FIRST CUSTOME OF R2
#   3. IF BOTH FEASIBLE AND NEW SUM DISTANCE IS LESS OR REDUCES NUMBER OF VEHICLES THEN ACCEPT
# =============================================================================
#%%
#import copy
#import collections
from validity import route_validity
import numpy as np
from joblib import Parallel, delayed

#%%
    
def routing_phase2(alpha, beta, chromosome, chromosome_distance_list,  total_capacity, total_time, arr_distance_matrix, arr_customers):
    index_list = np.arange(0, len(chromosome)+1)
    index_list[len(chromosome)] = 0
    for i in range(len(index_list)-1):
        r1_index = index_list[i]
        r2_index = index_list[i+1]
        route_r1 = chromosome[r1_index].copy()
        route_r2 = chromosome[r2_index].copy()
        dist_r1 = chromosome_distance_list[r1_index]
        dist_r2 = chromosome_distance_list[r2_index]
        customer_r1 = route_r1[-2]
        route_r2.insert(1, customer_r1)
        route_validity_status, new_r2_dist = route_validity(route_r2, arr_distance_matrix, arr_customers, total_time, total_capacity)
        if route_validity == True:
            if len(route_r1)>3:
                new_r1_dist = dist_r1 - arr_distance_matrix[route_r1[-2],0]+ arr_distance_matrix[route_r1[-3],0]
                if (dist_r1+dist_r2)>(new_r1_dist+new_r2_dist):
                    del route_r1[-1]
                    chromosome[r1_index] = route_r1
                    chromosome[r2_index] = route_r2
                    chromosome_distance_list[r1_index] = new_r1_dist
                    chromosome_distance_list[r2_index] = new_r2_dist
            else:
                del chromosome[r1_index]
                chromosome[r2_index] = route_r2
                del chromosome_distance_list[r1_index]
                chromosome_distance_list[r2_index] = new_r2_dist
    total_distance = sum(chromosome_distance_list)
    num_vehicles = len(chromosome)
    fitness = (alpha*num_vehicles)+(beta*total_distance)
    return chromosome, chromosome_distance_list,  num_vehicles, total_distance, fitness

#%%
                
def initial_routing_phase2(num_cores,N, alpha, beta, pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_distance_matrix, arr_customers, total_time, total_capacity):
    inputs = range(0,N)
    results_list = Parallel(num_cores)(delayed(routing_phase2)(alpha, beta, pop_chromosome_routeList_array[i], pop_chromosome_distanceList_array[i],  total_capacity, total_time, arr_distance_matrix, arr_customers ) for i in inputs)
#    chromosome, chromosome_distance_list,  num_vehicles, total_distance, fitness
    unzipped = zip(*results_list) #groups the same return for each particle and returns a list of tuples each of lenth M
    results_list = list(unzipped)
    pop_chromosome_routeList_array = np.empty(N, object)
    pop_chromosome_distanceList_array = np.empty(N, object)
    pop_chromosome_routeList_array[:] = results_list[0]
    pop_chromosome_distanceList_array[:] = results_list[1]
    arr_pop_results = np.empty((N,3), float)
    arr_pop_results[:,0] = results_list[2]
    arr_pop_results[:,1] = results_list[3]
    arr_pop_results[:,2] = results_list[4]
    return pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results
    
    

    