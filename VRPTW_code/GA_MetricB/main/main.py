#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:22:42 2019

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:19:18 2019

@author: user
"""

#GENETIC ALGORITHM FUNCTIONS
# =============================================================================
# Distance calculator
# Tournament Selection
# Removal Crossover
# Insertion Per Customer
# Mutation
# Elitism
# =============================================================================


import os
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import time
import copy
from functions import full_recombination, mutation, elitism
import matplotlib.pyplot as plt
import math
#%%
# Input information
num_customers = 25

#%%
#Paths
cwd = os.getcwd()
initial_path = cwd +'/initial'
final_path = cwd + '/final'

customer_path  = cwd +'/solomon%d_csv/customers'%(num_customers)
dataInfo_path = cwd +'/solomon%d_csv/data_info'%(num_customers)
distanceMatrix_path = cwd +'/solomon%d_csv/distance_matrix'%(num_customers)
initial_path = cwd.split('/')
initial_path = '/'.join(initial_path[:-1])
initial_path = initial_path + '/initial_encoding/initial'
#%%
#FILES
dataset_list = []
for i in range(101,110):
    dataset_list.append('c'+'%s' %(i))

for i in range(101,113):
    dataset_list.append('r'+'%s' %(i))
    
for i in range(101,109):
    dataset_list.append('rc'+'%s' %(i))

for i in range(201,209):
    dataset_list.append('c'+'%s' %(i))
    
for i in range(201,212):
    dataset_list.append('r'+'%s' %(i))

for i in range(201,209):
    dataset_list.append('rc'+'%s' %(i))
    
#%%
#EXPERIMENTAL VARIABLES
num_experiments = 3
num_cores = multiprocessing.cpu_count() #number of cores to use


#GLOBAL VARIABLES:
N = 300#population size
#alpha = 100
#beta = 0.001
num_generations = 350
num_elite_chromosmes = 1
K_val = 4
r = 0.8#tournament selection


#%%

#read in comparison results 

df_results_comparative = pd.read_csv(os.path.join(cwd, 'PSO_%d_resultsOverviewComparison.csv'%(num_customers)))
df_results_comparative.index = df_results_comparative.dataset

#%%
for i in range(len(dataset_list)):#first 18 datasets
    dataSet = dataset_list[i]
    #READ IN DATA PERTAINING TO THE DATASET
    customer_file = os.path.join(customer_path, '%scustomers.csv'%(dataSet))
    dataInfo_file = os.path.join(dataInfo_path, '%sdataInfo.csv'%(dataSet))
    distanceMatrix_file = os.path.join(distanceMatrix_path, '%sdistanceMatrix.csv'%(dataSet))

    
    df_customers = pd.read_csv(customer_file)
    df_distance_matrix = pd.read_csv(distanceMatrix_file)
    df_dataInfo = pd.read_csv(dataInfo_file)
    
    num_customers = df_customers.shape[0]
    num_customers_depot = num_customers + 1
    df_customers.index = range(1,num_customers_depot)
    

    #CONVERT TO NUMPY ARRAY
    #distance array
    arr_distance_matrix = np.asarray(df_distance_matrix)
    #greedy radius 
    dataset_radius = (np.nanmax(arr_distance_matrix) - np.nanmin(arr_distance_matrix))/2


    arr_customer_attributes = ['demand','readyTime', 'dueTime']
    
    # 0: customer number
    # 1: Demand
    # 2: Ready Time
    # 3: Due Time
    
    arr_customers = np.empty((num_customers, len(arr_customer_attributes)+1),dtype=int)
    arr_customers[:,0] = np.arange(1,num_customers_depot,1,dtype=int)
    arr_customers[:,1::] = df_customers[arr_customer_attributes]
    
    #   fleet max details
    total_time = df_dataInfo.loc[0, 'fleet_max_working_time']
    total_capacity = df_dataInfo.loc[0, 'fleet_capacity']
    service_time = df_customers.loc[1,'serviceTime']#if standard for all customers
#    
    all_customers_list = list(range(1,num_customers_depot))
    experiment_results_file = os.path.join(final_path +'/%s' %(dataSet), '%s_experiment_results.csv' %(dataSet)) #save all experiments results
    experiment_route_file = os.path.join(final_path +'/%s' %(dataSet), '%s_experiment_routes.npy' %(dataSet)) #save all experiments results routes
    experiment_distance_file = os.path.join(final_path +'/%s' %(dataSet), '%s_experiment_distances.npy' %(dataSet)) #save all experiments results distances
    
    
    experimental_routes = np.empty(num_experiments, object)
    experimental_distances  = np.empty(num_experiments, object)
    
    df_global_result = pd.DataFrame(np.zeros((1, 4)))
    df_global_result.columns = ['num_vehicles', 'distance', 'fitness', 'time'] 
    
    df_experimental_results = pd.DataFrame(np.zeros((num_experiments, 4)))
    df_experimental_results.columns = ['num_vehicles', 'distance', 'fitness', 'time']

    #read in comparison results 
    comparative_best_num_vehicles = df_results_comparative.loc[dataSet,'best_nv' ]
    comparative_metric_num_vehicles = df_results_comparative.loc[dataSet,'PSO_nv' ]
    comparative_best_distance = df_results_comparative.loc[dataSet,'best_distance' ]
    comparative_metric_distance = df_results_comparative.loc[dataSet,' PSO_distance' ]
    comparative_best_fitness =  comparative_best_num_vehicles +(np.arctan(comparative_best_distance)/(math.pi/2))
    comparative_metric_fitness = comparative_metric_num_vehicles +(np.arctan(comparative_metric_distance)/(math.pi/2))
    
    
#   FOR EACH EXPERIMENT
    for experiment in range(num_experiments):
#       READ IN FILES
        initial_pop_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), '%sinitialPop.npy'%(dataSet))
        initial_pop_distance_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), '%sinitialDistance.npy'%(dataSet))
        initial_result_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), '%sinitialResults.csv'%(dataSet))

        pop_chromosome_routeList_array = np.load(initial_pop_output_file, allow_pickle=True)
        pop_chromosome_distanceList_array= np.load(initial_pop_distance_output_file, allow_pickle=True)
        df_pop_results = pd.read_csv(initial_result_output_file)
        arr_pop_results = np.asarray(df_pop_results)
        
#        READ OUT FILES
        final_results_file = os.path.join(final_path+'/%s'%(dataSet), 'final_results.csv')#global results
        gbest_route_file = os.path.join(final_path+'/%s'%(dataSet), 'final_route.npy')#global best route


            
        start_time = time.time()
#        generational_fitness = []
        for gen in range(num_generations):        
#            print('gen', gen)
#           INITIALISE NEW POPULATION
            new_pop_chromosome_routeList_array = np.empty(N, object)
            new_pop_chromosome_distanceList_array = np.empty(N, object)
            arr_new_pop_results = np.empty((N,3), float)#['num_vehicles', 'distance', 'fitness']
            

#            RECOMBINATION
        
            arr_new_pop_results, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array = full_recombination(num_cores, K_val,r, N,arr_pop_results, arr_new_pop_results, arr_customers, arr_distance_matrix, total_time, total_capacity, pop_chromosome_routeList_array, pop_chromosome_distanceList_array, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array, service_time)            

#           ELITISM
            elite_chromosome_routeList, elite_chromosome_distanceList, elite_result = elitism(pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results)            
                        


            #Replace worst chromosome with elite
            worst_chromosome_number = np.nanargmax(arr_new_pop_results[:,2])
            new_pop_chromosome_routeList_array[worst_chromosome_number] = elite_chromosome_routeList
            new_pop_chromosome_distanceList_array[worst_chromosome_number] = elite_chromosome_distanceList
            arr_new_pop_results[worst_chromosome_number,:] = elite_result[:]
   
       
            #Update
            pop_chromosome_routeList_array = copy.deepcopy(new_pop_chromosome_routeList_array)
            pop_chromosome_distanceList_array = copy.deepcopy(new_pop_chromosome_distanceList_array)
            arr_pop_results = copy.deepcopy(arr_new_pop_results)
            
#Uncommnet if terminate at best met result
#            if ((arr_pop_results[np.argmin(arr_pop_results[:,2]),0] == comparative_best_num_vehicles)and(np.round(arr_pop_results[np.argmin(arr_pop_results[:,1]),1],2)<=comparative_best_distance+1)) or ((arr_pop_results[np.argmin(arr_pop_results[:,2]),0] == comparative_metric_num_vehicles)and(np.round(arr_pop_results[np.argmin(arr_pop_results[:,1]),1],2)<=comparative_metric_distance+1)):
#                break
        end_time = time.time() 
        print(end_time - start_time)
        #save experiment results           
        df_experimental_results.iloc[experiment, 0:3] = arr_pop_results[np.nanargmin(arr_pop_results[:,2]),:]
        df_experimental_results.loc[experiment,'time'] = end_time-start_time
        df_experimental_results.to_csv(experiment_results_file, index = False)
        #save best route from experiment in the data set of experimental routes
        experimental_routes[experiment]  = copy.deepcopy(pop_chromosome_routeList_array[np.nanargmin(arr_pop_results[:,2])])
        experimental_distances[experiment] = copy.deepcopy(pop_chromosome_distanceList_array[np.nanargmin(arr_pop_results[:,2])])
        np.save(experiment_route_file, experimental_routes)
        np.save(experiment_distance_file, experimental_distances)
        print('experiment: ', experiment)

#    #save best elite chromsome from all the experiments
    df_global_result.iloc[0,:] = df_experimental_results.iloc[df_experimental_results['fitness'].idxmin(),:]
    gbest_route = np.array(experimental_routes[df_experimental_results['fitness'].idxmin()])
    gbest_route_file = os.path.join(final_path+'/%s'%(dataSet), 'final_route.npy')
    np.save(gbest_route_file,gbest_route)
    final_results_file = os.path.join(final_path+'/%s'%(dataSet), 'final_results.csv')
    df_results = pd.DataFrame(df_experimental_results.iloc[df_experimental_results['fitness'].idxmin(),:]).transpose()
    df_results.to_csv(final_results_file, index=False)
    print(df_results)
    print('Dataset;', dataSet)        

