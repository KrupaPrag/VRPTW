#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:57:44 2018

@author: krupa
"""

import numpy as np
import pandas as pd
import os
import multiprocessing
from elitism import elitism
from crossover_recombination import full_recombination
from mutation import mutation
import copy
import time
#import decimal
#%%
# =============================================================================
# MAIN:
#    1.Read in data
#    2.For num generations
#       Generate a new population
#       1. Elitsm
#       2. recombination
#       3. mutation
#       4. evaluat and update recorded results
# =============================================================================

#%%
#paths 
cwd = os.getcwd()
initial_path = cwd +'/initial'
final_path = cwd + '/final'

customer_path  = cwd +'/solomon25_csv/customers'
dataInfo_path = cwd +'/solomon25_csv/data_info'
distanceMatrix_path = cwd +'/solomon25_csv/distance_matrix'
initial_path = cwd + '/initial'

#%%

# LIST FILE NAMES
r1_list_datasets = []
c1_list_datasets = []
rc1_list_datasets = []
r2_list_datasets = []
c2_list_datasets = []
rc2_list_datasets = []


for i in range(101,110):
    c1_list_datasets.append('c'+'%s' %(i) )

for i in range(101,113):
    r1_list_datasets.append('r'+'%s' %(i) )
    
for i in range(101,109):
    rc1_list_datasets.append('rc'+'%s' %(i) )

for i in range(201,209):
    c2_list_datasets.append('c'+'%s' %(i) )
    
for i in range(201,212):
    r2_list_datasets.append('r'+'%s' %(i) )

for i in range(201,209):
    rc2_list_datasets.append('rc'+'%s' %(i) )

dataset_list = c1_list_datasets + r1_list_datasets + rc1_list_datasets + c2_list_datasets + r2_list_datasets + rc2_list_datasets

r1_list_dataInfo = []
c1_list_dataInfo = []
rc1_list_dataInfo = []
r2_list_dataInfo = []
c2_list_dataInfo = []
rc2_list_dataInfo = []

for i in range(101,110):
    c1_list_dataInfo.append('c'+'%s' %(i) + 'dataInfo.csv')

for i in range(101,113):
    r1_list_dataInfo.append('r'+'%s' %(i) + 'dataInfo.csv')
    
for i in range(101,109):
    rc1_list_dataInfo.append('rc'+'%s' %(i) + 'dataInfo.csv')

for i in range(201,209):
    c2_list_dataInfo.append('c'+'%s' %(i) + 'dataInfo.csv')
    
for i in range(201,212):
    r2_list_dataInfo.append('r'+'%s' %(i) + 'dataInfo.csv')

for i in range(201,209):
    rc2_list_dataInfo.append('rc'+'%s' %(i) + 'dataInfo.csv')

dataInfo_files_list = c1_list_dataInfo + r1_list_dataInfo + rc1_list_dataInfo + c2_list_dataInfo + r2_list_dataInfo + rc2_list_dataInfo

#customers
r1_list_customers = []
c1_list_customers = []
rc1_list_customers = []
r2_list_customers = []
c2_list_customers = []
rc2_list_customers = []

for i in range(101,110):
    c1_list_customers.append('c'+'%s' %(i) + 'customers.csv')

for i in range(101,113):
    r1_list_customers.append('r'+'%s' %(i) + 'customers.csv')
    
for i in range(101,109):
    rc1_list_customers.append('rc'+'%s' %(i) + 'customers.csv')

for i in range(201,209):
    c2_list_customers.append('c'+'%s' %(i) + 'customers.csv')
    
for i in range(201,212):
    r2_list_customers.append('r'+'%s' %(i) + 'customers.csv')

for i in range(201,209):
    rc2_list_customers.append('rc'+'%s' %(i) + 'customers.csv')

customers_files_list = c1_list_customers + r1_list_customers + rc1_list_customers + c2_list_customers + r2_list_customers + rc2_list_customers

#distance matrix
#customers
r1_list_distanceMatrix = []
c1_list_distanceMatrix = []
rc1_list_distanceMatrix = []
r2_list_distanceMatrix = []
c2_list_distanceMatrix = []
rc2_list_distanceMatrix = []

for i in range(101,110):
    c1_list_distanceMatrix.append('c'+'%s' %(i) + 'distanceMatrix.csv')

for i in range(101,113):
    r1_list_distanceMatrix.append('r'+'%s' %(i) + 'distanceMatrix.csv')
    
for i in range(101,109):
    rc1_list_distanceMatrix.append('rc'+'%s' %(i) + 'distanceMatrix.csv')

for i in range(201,209):
    c2_list_distanceMatrix.append('c'+'%s' %(i) + 'distanceMatrix.csv')
    
for i in range(201,212):
    r2_list_distanceMatrix.append('r'+'%s' %(i) + 'distanceMatrix.csv')

for i in range(201,209):
    rc2_list_distanceMatrix.append('rc'+'%s' %(i) + 'distanceMatrix.csv')

distanceMatrix_files_list = c1_list_distanceMatrix + r1_list_distanceMatrix + rc1_list_distanceMatrix + c2_list_distanceMatrix + r2_list_distanceMatrix + rc2_list_distanceMatrix


#write out files

r1_list_initialPop = []
c1_list_initialPop = []
rc1_list_initialPop = []
r2_list_initialPop = []
c2_list_initialPop = []
rc2_list_initialPop = []

for i in range(101,110):
    c1_list_initialPop.append('c'+'%s' %(i) + 'initialPop.npy')

for i in range(101,113):
    r1_list_initialPop.append('r'+'%s' %(i) + 'initialPop.npy')
    
for i in range(101,109):
    rc1_list_initialPop.append('rc'+'%s' %(i) + 'initialPop.npy')

for i in range(201,209):
    c2_list_initialPop.append('c'+'%s' %(i) + 'initialPop.npy')
    
for i in range(201,212):
    r2_list_initialPop.append('r'+'%s' %(i) + 'initialPop.npy')

for i in range(201,209):
    rc2_list_initialPop.append('rc'+'%s' %(i) + 'initialPop.npy')

initialPop_files_list = c1_list_initialPop + r1_list_initialPop + rc1_list_initialPop + c2_list_initialPop + r2_list_initialPop + rc2_list_initialPop


r1_list_initialDistance = []
c1_list_initialDistance = []
rc1_list_initialDistance = []
r2_list_initialDistance = []
c2_list_initialDistance = []
rc2_list_initialDistance = []

for i in range(101,110):
    c1_list_initialDistance.append('c'+'%s' %(i) + 'initialDistance.npy')

for i in range(101,113):
    r1_list_initialDistance.append('r'+'%s' %(i) + 'initialDistance.npy')
    
for i in range(101,109):
    rc1_list_initialDistance.append('rc'+'%s' %(i) + 'initialDistance.npy')

for i in range(201,209):
    c2_list_initialDistance.append('c'+'%s' %(i) + 'initialDistance.npy')
    
for i in range(201,212):
    r2_list_initialDistance.append('r'+'%s' %(i) + 'initialDistance.npy')

for i in range(201,209):
    rc2_list_initialDistance.append('rc'+'%s' %(i) + 'initialDistance.npy')

initialDistance_files_list = c1_list_initialDistance + r1_list_initialDistance + rc1_list_initialDistance + c2_list_initialDistance + r2_list_initialDistance + rc2_list_initialDistance


r1_list_initialResults = []
c1_list_initialResults = []
rc1_list_initialResults = []
r2_list_initialResults = []
c2_list_initialResults = []
rc2_list_initialResults = []

for i in range(101,110):
    c1_list_initialResults.append('c'+'%s' %(i) + 'initialResults.csv')

for i in range(101,113):
    r1_list_initialResults.append('r'+'%s' %(i) + 'initialResults.csv')
    
for i in range(101,109):
    rc1_list_initialResults.append('rc'+'%s' %(i) + 'initialResults.csv')

for i in range(201,209):
    c2_list_initialResults.append('c'+'%s' %(i) + 'initialResults.csv')
    
for i in range(201,212):
    r2_list_initialResults.append('r'+'%s' %(i) + 'initialResults.csv')

for i in range(201,209):
    rc2_list_initialResults.append('rc'+'%s' %(i) + 'initialResults.csv')

initialResults_files_list = c1_list_initialResults + r1_list_initialResults + rc1_list_initialResults + c2_list_initialResults + r2_list_initialResults + rc2_list_initialResults
#%%
#EXPERIMENTAL VARIABLES
num_experiments = 30
num_cores = multiprocessing.cpu_count() #number of cores to use


#GLOBAL VARIABLES:
N = 300#population size
alpha = 100
beta = 0.001
num_generations = 350
num_elite_chromosmes = 1
K_val = 4
r = 0.8#tournament selection
    
#%%
i = 0
experiment = 0
gen = 0
#%%


#FOR EACH DATASET
for i in range(len(dataset_list)):
    dataSet = dataset_list[i]
    #READ IN DATA PERTAINING TO THE DATASET
    customer_file = os.path.join(customer_path, customers_files_list[i])
    dataInfo_file = os.path.join(dataInfo_path, dataInfo_files_list[i])
    distanceMatrix_file = os.path.join(distanceMatrix_path, distanceMatrix_files_list[i])
    
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
    arr_customers = np.empty((num_customers, 6))
    arr_customers[:,0] = np.arange(1, num_customers_depot) 
    arr_customers[:,1] = df_customers.loc[:,'demand']
    arr_customers[:,2] = df_customers.loc[:,'serviceTime']
    arr_customers[:,3] = df_customers.loc[:,'readyTime']
    arr_customers[:,4] = df_customers.loc[:,'dueTime']
    arr_customers[:,5] = df_customers.loc[:,'completeTime']
    
#   fleet max details
    total_time = df_dataInfo.loc[0, 'fleet_max_working_time']
    total_capacity = df_dataInfo.loc[0, 'fleet_capacity']
    
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


#   FOR EACH EXPERIMENT
    for experiment in range(num_experiments):
#       READ IN FILES
        initial_pop_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), initialPop_files_list[i])
        initial_pop_distance_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), initialDistance_files_list[i])
        initial_result_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), initialResults_files_list[i])
        
        pop_chromosome_routeList_array = np.load(initial_pop_output_file)
        pop_chromosome_distanceList_array= np.load(initial_pop_distance_output_file)
        df_pop_results = pd.read_csv(initial_result_output_file)
        arr_pop_results = np.asarray(df_pop_results)
        
#        READ OUT FILES
        final_results_file = os.path.join(final_path+'/%s'%(dataSet), 'final_results.csv')#global results
        gbest_route_file = os.path.join(final_path+'/%s'%(dataSet), 'final_route.npy')#global best route


            
        start_time = time.time()
        for gen in range(350):        
#            print('gen', gen)
#           INITIALISE NEW POPULATION
            new_pop_chromosome_routeList_array = np.empty(N, object)
            new_pop_chromosome_distanceList_array = np.empty(N, object)
            arr_new_pop_results = np.empty((N,3), float)#['num_vehicles', 'distance', 'fitness']
            
#            RECOMBINATION
            arr_new_pop_results, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array = full_recombination(alpha, beta, num_cores, K_val,r, N,arr_pop_results, arr_new_pop_results, arr_customers, arr_distance_matrix, total_time, total_capacity, pop_chromosome_routeList_array, pop_chromosome_distanceList_array, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array)
            
#            #MUTATION
            pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results = mutation(alpha, beta, N,num_cores, arr_new_pop_results, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array,arr_distance_matrix, arr_customers, total_time, total_capacity )

            

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
            
        end_time = time.time() 
        #save experiment results           
        df_experimental_results.iloc[experiment, 0:3] = arr_pop_results[np.nanargmin(arr_pop_results[:,2]),:]
        df_experimental_results.loc[experiment,'time'] = end_time-start_time
        df_experimental_results.to_csv(experiment_results_file, index = False)
        #save best route from experiment in the data set of experimental routes
        experimental_routes[experiment]  = copy.deepcopy(pop_chromosome_routeList_array[np.nanargmin(arr_pop_results[:,2])])
        experimental_distances[experiment] = copy.deepcopy(pop_chromosome_distanceList_array[np.nanargmin(arr_pop_results[:,2])])
        np.save(experiment_route_file, experimental_routes)
        np.save(experiment_distance_file, experimental_distances)
        print(experimental_routes[experiment])
        print('experiment: ', experiment)
#        
    df_global_result.iloc[0,:] = df_experimental_results.iloc[df_experimental_results['fitness'].idxmin(),:]
    gbest_route = np.array(experimental_routes[df_experimental_results['fitness'].idxmin()])
    gbest_route_file = os.path.join(final_path+'/%s'%(dataSet), 'final_route.npy')
    np.save(gbest_route_file,gbest_route)
    final_results_file = os.path.join(final_path+'/%s'%(dataSet), 'final_results.csv')
    df_results = pd.DataFrame(df_experimental_results.iloc[df_experimental_results['fitness'].idxmin(),:]).transpose()
    df_results.to_csv(final_results_file, index=False)
    print(df_results)
    print('Dataset;', dataSet)        
