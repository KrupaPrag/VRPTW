#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:32:53 2018

@author: krupa
"""
# =============================================================================
# GENERATE THE INITIAL POPULATION ROUTES USING THE 3 SCHEMES;
#   1. RANDOM
#   2. GREEDY
#   3. 
# =============================================================================
from initial_routing import random_neighbour, greedy_neighbour, nearest_neighbour
import concurrent.futures as cf
from itertools import chain
import numpy as np
#from initial_routing_phase2 import phase2_routing
#%%


def initial_population(N, num_cores, number_random_chromosomes, number_greedy_chromosomes, number_nearest_nn_chromosomes, num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list, radius):
    with cf.ProcessPoolExecutor(num_cores) as executor:
        results_list = []
        for i in range(number_random_chromosomes):
            results_list.append(executor.submit(random_neighbour, num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list))
        for j in range(number_greedy_chromosomes):
            results_list.append(executor.submit(greedy_neighbour, num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list, radius))
        
#    chromosome_route_list, chromosome_distance_list, num_routes, total_distance
    temp = list(chain.from_iterable(f.result() for f in cf.as_completed(results_list))) 
    
    pop_chromosome_routeList_array = np.empty(N, object)
    pop_chromosome_distanceList_array = np.empty(N, object)
    pop_num_routes = np.empty(N)
    pop_total_distance = np.empty(N)
    
    chromosome_route_list, chromosome_distance_list, num_routes, total_distance = nearest_neighbour(num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list)
    pop_chromosome_routeList_array[0] = chromosome_route_list
    pop_chromosome_distanceList_array[0] = chromosome_distance_list
    pop_num_routes[0] = num_routes
    pop_total_distance[0] = total_distance
    
    pop_chromosome_routeList_array[1::] = temp[0::4]
    pop_chromosome_distanceList_array[1::] = temp[1::4]
    pop_num_routes[1::] = temp[2::4]
    pop_total_distance[1::] = temp[3::4]
    
    return pop_chromosome_routeList_array, pop_chromosome_distanceList_array, pop_num_routes, pop_total_distance

#%%
    
