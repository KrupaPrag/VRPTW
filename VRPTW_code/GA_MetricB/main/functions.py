#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:57:06 2019

@author: user
"""

import numpy as np
import pandas as pd
import multiprocessing
import math
from joblib import Parallel, delayed
from numba import jit
import scipy
import time
import copy
import random
#from collections import deque
import numba
#%%
#def distance_calculator(route, arr_disatnce_matrix):
#    distance = 0
#    for i in range(len(route)-1):
#        distance+=arr_disatnce_matrix[route[i], route[i+1]]
#        
#    return distance
def distance_calculator(route, arr_distance_matrix):
    r = route[:-1]
    c = route[1::]
    distance = sum(arr_distance_matrix[r,c])
    return distance

#%%
def tournament_selection(K_val, r, arr_pop_results, N):
    rand_chromosome_list = random.sample(range(0, N), K_val)
    arr_tournament_set = arr_pop_results[rand_chromosome_list]
    if random.uniform(0,1)<r:
        parent_chromosome_no = int(rand_chromosome_list[np.argmin(arr_tournament_set[:,2])])
    else:
        parent_chromosome_no = random.choice(rand_chromosome_list)
    return parent_chromosome_no

#%%

    
def removal_crossover_customers(RP, chromosome, chromosome_distances, arr_distance_matrix):
    for i in RP:#for each customer to remove
        for j in range(len(chromosome)):#for each route to search
            if (i in chromosome[j]) == True:
                if len(chromosome[j])>3: #only delete element
                    chromosome[j].remove(i)
                    chromosome_distances[j] = distance_calculator(chromosome[j], arr_distance_matrix)               
                    break#break route search loop
                elif len(chromosome[j]) == 3:
                    del chromosome[j]
                    del chromosome_distances[j]
                    j -= 1 #check this 

                    break#break route search loop
                
    return chromosome, chromosome_distances
    
#%%

def insertion_per_customer(customer_to_insert, route, route_distance, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time):
    inserted_status = False
    arr_positions_to_insert = np.arange(1,len(route[1:-1])+1)
    update_route_distance_eval = np.inf
    updated_route = route
    updated_distance = route_distance
    
    for positions_to_insert in arr_positions_to_insert:
        new_route = route.copy()
        new_route.insert(positions_to_insert, customer_to_insert)
        route_validity_status, curr_dist = route_validity(new_route, arr_distance_matrix, arr_customers, total_time, total_capacity, service_time)
        if route_validity_status == True:
            inserted_status = True
            distance_eval = curr_dist - route_distance
            if distance_eval<update_route_distance_eval:
                updated_route = new_route
                update_route_distance_eval = distance_eval
                updated_distance = curr_dist
                
    return inserted_status, updated_route, updated_distance, update_route_distance_eval


#%%
            
            
def insertion(RP, chromosome, chromosome_distances, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time):
    for customer_to_insert in RP:#for each customer to insert
        curr_chromosome = chromosome.copy()
        curr_chromosome_distances = chromosome_distances.copy()
        arr_tracker = np.empty((len(curr_chromosome), 6), object) # 0: curr_routes, 1: curr_distances, 2: update bool, 3: new route, 4: new_dist, 5:dist_diff
        arr_tracker[:,0] = chromosome
        arr_tracker[:,1] = chromosome_distances
        arr_tracker[:,2] = False
        arr_tracker[:,4] = np.nan
        inserted_status = False
        for i in range(len(chromosome)):
            route = arr_tracker[i,0]
            route_distance = arr_tracker[i,1]
            inserted_status, updated_route, updated_distance, update_route_distance_eval = insertion_per_customer(customer_to_insert, route, route_distance, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time)
            if inserted_status ==True:
                arr_tracker[i,2] = True
                arr_tracker[i,3] = updated_route
                arr_tracker[i,4] = updated_distance
        arr_tracker[:,5] = arr_tracker[:,4] - arr_tracker[:,1]
        if len(np.where(arr_tracker[:,5]>0)[0]) >0: #not all are nan
            min_route =np.nanargmin(arr_tracker[:,5])
            curr_chromosome[min_route] = copy.deepcopy(arr_tracker[min_route,3])
            curr_chromosome_distances[min_route] = copy.deepcopy(arr_tracker[min_route,4])
            curr_chromosome_total_distance = sum(curr_chromosome_distances)
        else:
            additional_route = [0,customer_to_insert,0]
            curr_chromosome.append(additional_route)
            curr_chromosome_distances.append(distance_calculator(additional_route, arr_distance_matrix))
            curr_chromosome_total_distance = sum(curr_chromosome_distances)
        chromosome = copy.deepcopy(curr_chromosome)
        chromosome_distances = copy.deepcopy(curr_chromosome_distances)
        curr_chromosome_total_distance = sum(chromosome_distances)
        
    return curr_chromosome, curr_chromosome_distances, curr_chromosome_total_distance        
    

#%%
def recombination(  K_val, r, arr_pop_results, N, pop_chromosome_routeList_array,pop_chromosome_distanceList_array, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time ):
    #GET PARENT CHROMOSOMES:
#    num_cores = 1
    parent_chromosome_no1 = tournament_selection(K_val, r, arr_pop_results, N)
    parent_chromosome_no2 = tournament_selection(K_val, r, arr_pop_results, N)
    while parent_chromosome_no1 == parent_chromosome_no2:
        parent_chromosome_no2 = tournament_selection(K_val, r, arr_pop_results, N)
    
    
    chromosome1_routes = copy.deepcopy(pop_chromosome_routeList_array[parent_chromosome_no1])
    chromosome2_routes = copy.deepcopy(pop_chromosome_routeList_array[parent_chromosome_no2])
    chromosome1_distances = copy.deepcopy(pop_chromosome_distanceList_array[parent_chromosome_no1])
    chromosome2_distances = copy.deepcopy(pop_chromosome_distanceList_array[parent_chromosome_no2])
    chromosome1_total_distance = copy.deepcopy(arr_pop_results[parent_chromosome_no1, 1])
    chromosome2_total_distance = copy.deepcopy(arr_pop_results[parent_chromosome_no2, 1])

    rand_route_no1 = random.randint(0,len(chromosome1_routes)-1)
    rand_route_no2 = random.randint(0,len(chromosome2_routes)-1)
    

    #subroute selected customers
    RP1 = chromosome1_routes[rand_route_no1][1:-1]
    RP2 = chromosome2_routes[rand_route_no2][1:-1]
    
    #remove customers in opposite routes
    chromosome1_routes, chromosome1_distances = removal_crossover_customers(RP2, chromosome1_routes, chromosome1_distances, arr_distance_matrix)#RP, chromosome, chromosome_distances, arr_distance_matrix
    chromosome2_routes, chromosome2_distances = removal_crossover_customers(RP1, chromosome2_routes, chromosome2_distances, arr_distance_matrix)
    
    #reinsert removed customers 
    chromosome1_routes, chromosome1_distances, chromosome1_total_distance = insertion(RP2, chromosome1_routes, chromosome1_distances, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time)
    chromosome2_routes, chromosome2_distances, chromosome2_total_distance = insertion(RP1, chromosome2_routes, chromosome2_distances, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time)
    
    return   chromosome1_routes, chromosome2_routes, chromosome1_distances, chromosome2_distances, chromosome1_total_distance, chromosome2_total_distance

#%%

def full_recombination(num_cores, K_val,r, N,arr_pop_results, arr_new_pop_results, arr_customers, arr_distance_matrix, total_time, total_capacity, pop_chromosome_routeList_array, pop_chromosome_distanceList_array, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array, service_time):
    inputs = range(int(N/2))
    results_list_recomb = Parallel(num_cores)(delayed(recombination)(K_val, r, arr_pop_results, N, pop_chromosome_routeList_array,pop_chromosome_distanceList_array, arr_customers, arr_distance_matrix, total_capacity, total_time, service_time ) for i in inputs)
    unzipped = zip(*results_list_recomb) #groups the same return for each particle and returns a list of tuples each of lenth M
    results = list(unzipped)
    
    new_pop_chromosome_routeList_array[:] =  results[0][:] + results[1][:]
    new_pop_chromosome_distanceList_array[:] =  results[2][:] + results[3][:]
    arr_new_pop_results[:,0] = [len(x) for x in new_pop_chromosome_distanceList_array]
    arr_new_pop_results[:,1] = results[4]+results[5]
    arr_new_pop_results[:,2] = (arr_new_pop_results[:,0]) + (np.arctan(arr_new_pop_results[:,1])/(math.pi/2))
    
    return arr_new_pop_results, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array
#%%

def mutation_chromosome(chromosome, chromosme_distance, arr_distance_matrix, arr_customers, total_time, total_capacity, service_time):
    route_validity_status = False
    list_len_routes =list(map(lambda n: len(n),chromosome))
    feasible_indicies = np.where(np.array(list_len_routes)>=5)[0]
    if len(feasible_indicies)>0:
        rand_route_no = random.choice(feasible_indicies)
        route = copy.deepcopy(chromosome[rand_route_no])
        mutation_len = random.choice([2,3])
        len_rand_route = len(route)
        start_index = 1
        if len_rand_route-2>mutation_len:
            start_index = random.randint(1,len_rand_route-2-mutation_len)


        temp_list = copy.deepcopy(route[start_index: start_index+mutation_len])
        reverse_temp = temp_list[::-1]
        route[start_index:start_index+mutation_len] = reverse_temp 
        route_validity_status, dist =  route_validity(route, arr_distance_matrix, arr_customers, total_time, total_capacity, service_time)
        if route_validity_status == True and dist<=chromosme_distance[rand_route_no]:
            chromosome[rand_route_no] = route
           

            chromosme_distance[rand_route_no] = dist
        
    return route_validity_status, chromosome, chromosme_distance      

#%%
def mutation(N,num_cores, arr_pop_results, pop_chromosome_routeList_array, pop_chromosome_distanceList_array,arr_distance_matrix, arr_customers, total_time, total_capacity, service_time ):
    inputs = range(0,N)#don't apply mutation to elite chromosome
    results_list = Parallel(num_cores)(delayed(mutation_chromosome)(pop_chromosome_routeList_array[i], pop_chromosome_distanceList_array[i], arr_distance_matrix, arr_customers, total_time, total_capacity, service_time ) for i in inputs)
    
    unzipped = zip(*results_list) #groups the same return for each particle and returns a list of tuples each of lenth M
    results = list(unzipped)#route_validity_status, chromosome, chromosme_distance
    
    updated_indicies = np.where(results[0]==True)[0]
    if len(updated_indicies)>0:
#        print('mutation')
        
        for i in updated_indicies:
            pop_chromosome_routeList_array[i] = copy.deepcopy(results[1][i])
            pop_chromosome_distanceList_array[i] = copy.deepcopy(results[2][i])
            arr_pop_results[i, 0] = len(pop_chromosome_distanceList_array[i])#    'num_vehicles', 'distance', 'fitness']
            arr_pop_results[i, 1] = sum(pop_chromosome_distanceList_array[i])
            arr_pop_results[i, 2] = len(pop_chromosome_distanceList_array[i])+ (np.arctan(sum(pop_chromosome_distanceList_array[i]))/(math.pi/2))

    return pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results    

#%%
    
def elitism(pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results):
    min_fitness_chromosome_index = np.argmin(arr_pop_results[:,2])
    elite_chromosome_routeList = pop_chromosome_routeList_array[min_fitness_chromosome_index]
    elite_chromosome_distanceList = pop_chromosome_distanceList_array[min_fitness_chromosome_index]
    elite_result = arr_pop_results[min_fitness_chromosome_index,:]
    
    return elite_chromosome_routeList, elite_chromosome_distanceList, elite_result
#%%
    
    # 0: customer number
    # 1: Demand
    # 2: Ready Time
    # 3: Due Time

    
def route_validity(route, arr_distance_matrix, arr_customers, total_time, total_capacity, service_time):
    route_validity_status = True
    curr_dist = arr_distance_matrix[route[0], route[1]]
    curr_capacity = arr_customers[route[1]-1, 1] #demand of customer (route[1]) -1 index postion
    ready_time = arr_customers[route[1]-1, 2]
    due_time = arr_customers[route[1]-1, 3]
    curr_time = arr_distance_matrix[route[0], route[1]] 
    waiting_time = max(0, ready_time-curr_time)
    curr_time += waiting_time + arr_customers[route[1]-1, 2] #service time
    
    route_customers = route[1:-1]
    for i in range(len(route_customers)-1):
        curr_dist += arr_distance_matrix[route_customers[i], route_customers[i+1]]
        curr_time += arr_distance_matrix[route_customers[i], route_customers[i+1]]
        ready_time = arr_customers[route_customers[i+1]-1, 2]
        due_time = arr_customers[route_customers[i+1]-1, 3]
        curr_capacity += arr_customers[route_customers[i+1]-1, 1]
        if (curr_time<=due_time and curr_capacity<=total_capacity):
            #accept
            waiting_time = max(0, ready_time - curr_time)
            if ((curr_time + waiting_time>=ready_time) and (curr_time + waiting_time + service_time <= total_time)):
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
    