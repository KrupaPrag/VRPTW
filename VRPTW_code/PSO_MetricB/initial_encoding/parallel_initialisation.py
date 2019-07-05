#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:21:24 2018

@author: krupa
"""
import pandas as pd
import numpy as np
import math 
from joblib import Parallel, delayed
from initial_encoding_routes import greedy_nearest_neighbour_heauristic, nearest_neighbour_heauristic, random_neighbour_heauristic
#%%

def initial_population(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot, num_cores, num_greedy_particles, M, num_random_particles, num_nnh_particles):
    #get individual result list of tuples
    greedy_result_list = Parallel(n_jobs = num_cores)(delayed(greedy_nearest_neighbour_heauristic)(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in range(num_greedy_particles))
    nnh_result_list = Parallel(n_jobs = num_cores)(delayed(nearest_neighbour_heauristic)(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in range(num_nnh_particles))
    random_result_list = Parallel(n_jobs = num_cores)(delayed(random_neighbour_heauristic)(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in range(num_random_particles))
    #group all tuples together in one list
    result_list = greedy_result_list + nnh_result_list
    result_list += random_result_list
    
    unzipped = zip(*result_list) #groups the same return for each particle and returns a list of tuples each of lenth M
    PSO_resultlist = list(unzipped)  # route_routeList_list, route_distance_list, particle_position, particle_velocity, total_distance, route
    
    arr_population_routeList = np.empty(M, object)
    arr_population_routeList[:] = PSO_resultlist[0][:]
    arr_population_distanceList = np.empty(M, object)
    arr_population_distanceList[:] = PSO_resultlist[1][:]
    arr_population_particle_position = np.empty(M, object)
    arr_population_particle_position[:] = PSO_resultlist[2][:]
    arr_population_particle_velocity = np.empty(M, object)
    arr_population_particle_velocity[:] = PSO_resultlist[3][:]    
    df_results = pd.DataFrame(np.zeros((M,3)))
    df_results.columns = ['num_vehicles', 'distance', 'fitness']
    df_results['num_vehicles'] = PSO_resultlist[5][:]
    df_results['distance'] = PSO_resultlist[4][:]

    #GA FITNESS
    alpha = 100
    beta = 0.001
    arr_distance = np.array(PSO_resultlist[4]).copy()
    arr_distance = beta*arr_distance
    arr_vehicles = alpha*np.array(PSO_resultlist[5])
    df_results['fitness'] = arr_distance+arr_vehicles
    
    return df_results, arr_population_routeList, arr_population_distanceList, arr_population_particle_position, arr_population_particle_velocity
    

