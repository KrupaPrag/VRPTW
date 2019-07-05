#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:04:34 2018

@author: krupa
"""
from feasible_neighbours import feasible_neighbours
import random
#%%
# =============================================================================
#  INITIAL ROUTING SCHEME
#   1. Random
#   2. Greedy
#   3. Phase 2
# =============================================================================

#%%
def random_neighbour(num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list):
    available_customers_list = all_customers_list.copy()

    chromosome_route_list = []
    chromosome_distance_list = []
    route_list = [0] 
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0
    
    while len(available_customers_list)!=0:
        arr_feasible_neighbours =  feasible_neighbours(num_customers, arr_customers, arr_distance_matrix, available_customers_list, total_time, total_capacity, curr_time, curr_capacity, curr_customer)    
        if arr_feasible_neighbours.shape[0]!=0:#has neighbours, append them
            feasible_neighbours_list = arr_feasible_neighbours[:,0].tolist()
            curr_customer = int(random.choice(feasible_neighbours_list))
            curr_customer_index = feasible_neighbours_list.index(curr_customer)
            route_list.append(curr_customer)
            curr_distance += arr_feasible_neighbours[curr_customer_index, 6]
            curr_capacity = arr_feasible_neighbours[curr_customer_index, 1]
            curr_time = arr_feasible_neighbours[curr_customer_index, 10]
            available_customers_list.remove(curr_customer)
        else: #end route and initialise a new route
            route_list.append(0)
            chromosome_route_list.append(route_list)
            curr_distance += arr_distance_matrix[curr_customer,0]
            chromosome_distance_list.append(curr_distance)
            #initialise new route 
            route_list = [0] 
            curr_capacity = 0
            curr_time = 0
            curr_customer = 0
            curr_distance = 0
    #end last route        
    route_list.append(0)
    chromosome_route_list.append(route_list)
    curr_distance += arr_distance_matrix[curr_customer,0]
    chromosome_distance_list.append(curr_distance)
    
    num_routes= len(chromosome_distance_list)
    total_distance = sum(chromosome_distance_list)
    
    random_route_results = (chromosome_route_list, chromosome_distance_list, num_routes, total_distance)
    return random_route_results
    
    


#%%
def greedy_neighbour(num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list, radius):
    available_customers_list = all_customers_list.copy()

    chromosome_route_list = []
    chromosome_distance_list = []
    route_list = [0] 
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0
    
    while len(available_customers_list)!=0:
        arr_feasible_neighbours =  feasible_neighbours(num_customers, arr_customers, arr_distance_matrix, available_customers_list, total_time, total_capacity, curr_time, curr_capacity, curr_customer)    
        if arr_feasible_neighbours.shape[0]!=0:#has neighbours, append them
            if arr_feasible_neighbours[arr_feasible_neighbours[:,6]<=radius].shape[0] !=0:
                arr_feasible_neighbours = arr_feasible_neighbours[arr_feasible_neighbours[:,6]<=radius] #reduce search space to feasible neighbours withn radius
            feasible_neighbours_list = arr_feasible_neighbours[:,0].tolist()
            curr_customer = int(random.choice(feasible_neighbours_list))
            curr_customer_index = feasible_neighbours_list.index(curr_customer)
            route_list.append(curr_customer)
            curr_distance += arr_feasible_neighbours[curr_customer_index, 6]
            curr_capacity = arr_feasible_neighbours[curr_customer_index, 1]
            curr_time = arr_feasible_neighbours[curr_customer_index, 10]
            available_customers_list.remove(curr_customer)

                
        else: #end route and initialise a new route
            route_list.append(0)
            chromosome_route_list.append(route_list)
            curr_distance += arr_distance_matrix[curr_customer,0]
            chromosome_distance_list.append(curr_distance)
            #initialise new route 
            route_list = [0] 
            curr_capacity = 0
            curr_time = 0
            curr_customer = 0
            curr_distance = 0
    #end last route        
    route_list.append(0)
    chromosome_route_list.append(route_list)
    curr_distance += arr_distance_matrix[curr_customer,0]
    chromosome_distance_list.append(curr_distance)
    
    num_routes= len(chromosome_distance_list)
    total_distance = sum(chromosome_distance_list)
    
    greedy_route_results = (chromosome_route_list, chromosome_distance_list, num_routes, total_distance)
    return greedy_route_results
#    

#%%
def nearest_neighbour(num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list):
    available_customers_list = all_customers_list.copy()

    chromosome_route_list = []
    chromosome_distance_list = []
    route_list = [0] 
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0
    
    while len(available_customers_list)!=0:
        arr_feasible_neighbours =  feasible_neighbours(num_customers, arr_customers, arr_distance_matrix, available_customers_list, total_time, total_capacity, curr_time, curr_capacity, curr_customer)    
        if arr_feasible_neighbours.shape[0]!=0:#has neighbours, append them
            #find minimum travel distance
            curr_customer_index = arr_feasible_neighbours[:,6].argmin()
            curr_customer = int(arr_feasible_neighbours[curr_customer_index,0] )
            route_list.append(curr_customer)
            curr_distance += arr_feasible_neighbours[curr_customer_index, 6]
            curr_capacity = arr_feasible_neighbours[curr_customer_index, 1]
            curr_time = arr_feasible_neighbours[curr_customer_index, 10]
            available_customers_list.remove(curr_customer)
        else: #end route and initialise a new route
            route_list.append(0)
            chromosome_route_list.append(route_list)
            curr_distance += arr_distance_matrix[curr_customer,0]
            chromosome_distance_list.append(curr_distance)
            #initialise new route 
            route_list = [0] 
            curr_capacity = 0
            curr_time = 0
            curr_customer = 0
            curr_distance = 0
    #end last route        
    route_list.append(0)
    chromosome_route_list.append(route_list)
    curr_distance += arr_distance_matrix[curr_customer,0]
    chromosome_distance_list.append(curr_distance)
    
    num_routes= len(chromosome_distance_list)
    total_distance = sum(chromosome_distance_list)

    return chromosome_route_list, chromosome_distance_list, num_routes, total_distance


