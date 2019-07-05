#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:23:01 2018

@author: krupa
"""


# =============================================================================
#   INITIAL ENCODING GA
#       1. For each particle record: 
#           a. Route list [[cust1, cust20,...]....[cust7,cust80...]]
#           b Distance list [Dr1, .......Drk]
#           c distance
#           d num_routes
#       2. record fitness table : distance, num_routes, fitness function value
#
# =============================================================================
#%%

import os
import math
import multiprocessing
import numpy as np
import pandas as pd
import random
from initial_pop_generation import initial_population
from initial_routing_phase2 import initial_routing_phase2
#%%
#Paths
cwd = os.getcwd()
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
greedy_percent = 0.10#percentage to be greedily initialised
random_percent = 0.90#percentage to be randomily initialised
alpha = 100
beta = 0.001
    
#%%
i = 0
experiment = 0
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
    
    number_random_chromosomes = math.floor(N*random_percent)-1
    number_nearest_nn_chromosomes = 1
    number_greedy_chromosomes = N-number_random_chromosomes-1#int(N*greedy_percent)
    
    #CONVERT TO NUMPY ARRAY
    #distance array
    arr_distance_matrix = np.asarray(df_distance_matrix)
    #greedy radius 
    dataset_radius = (np.nanmax(arr_distance_matrix) - np.nanmin(arr_distance_matrix))/2
    #customer array
#    customer_index = 0
#    demand = 1
#    service_time = 2
#    ready_time = 3
#    due_time = 4
#    complete_time = 5
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
    
#   FOR EACH EXPERIMENT
    for experiment in range(num_experiments):
#       READ OUT FILES
        initial_pop_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), initialPop_files_list[i])
        initial_pop_distance_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), initialDistance_files_list[i])
        initial_result_output_file = os.path.join(initial_path + '/%s/experiment%d' %(dataSet,experiment), initialResults_files_list[i])
        
        radius = dataset_radius*(random.uniform(0.5,1)) #assumed NOTE: NOT STATED IN PAPER
        #phase1
        pop_chromosome_routeList_array, pop_chromosome_distanceList_array, pop_num_routes, pop_total_distance = initial_population(N, num_cores, number_random_chromosomes, number_greedy_chromosomes, number_nearest_nn_chromosomes, num_customers, arr_customers, arr_distance_matrix,  total_capacity, total_time, all_customers_list, radius)
        #phase2
        pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results = initial_routing_phase2(num_cores,N, alpha, beta, pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_distance_matrix, arr_customers, total_time, total_capacity)
        
        np.save(initial_pop_output_file, pop_chromosome_routeList_array)
        np.save(initial_pop_distance_output_file, pop_chromosome_distanceList_array)
        
        df_initial_results = pd.DataFrame(np.zeros((N, 3)))
        df_initial_results.columns = ['num_vehicles', 'distance', 'fitness']
        df_initial_results.loc[:, 'num_vehicles'] = pop_num_routes
        df_initial_results.loc[:, 'distance'] = pop_total_distance
        df_initial_results.loc[:,'fitness'] = pop_num_routes+(np.arctan(pop_total_distance)/(math.pi/2))

        
        df_initial_results.to_csv(initial_result_output_file, index=False)
        
        print('experiment: ', experiment)
    print('Dataset:', dataSet)
        
    
    
    
    