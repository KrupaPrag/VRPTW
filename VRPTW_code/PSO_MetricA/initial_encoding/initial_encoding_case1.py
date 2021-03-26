#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:38:05 2018

@author: krupa
"""

# =============================================================================
#  INITIAL ENCODING:
#   dictionary population storage
#   dictioray global best
#   nearest neighbour heuristic CASEI (according to paper: nn_km)
# =============================================================================
import numpy as np
import pandas as pd
import os
import copy
import multiprocessing
from parallel_initialisation import initial_population
#%%
cwd = os.getcwd()
#read_in
dataInfo_path = cwd + '/solomon25_csv/data_info'
customer_path = cwd + '/solomon25_csv/customers'
distanceMatrix_path = cwd + '/solomon25_csv/distance_matrix'

#results

initialPop_path = cwd + '/initial/initial_pop'
initialFitness_path = cwd + '/initial/initial_fitness'
#%%

#read in file names
#READ In FILES: CUSTOMERS, DISTANCES, DATA INFORMATION

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


r1_list_initialPositionList = []
c1_list_initialPositionList = []
rc1_list_initialPositionList = []
r2_list_initialPositionList = []
c2_list_initialPositionList = []
rc2_list_initialPositionList = []

for i in range(101,110):
    c1_list_initialPositionList.append('c'+'%s' %(i) + 'initialPositionList.npy')

for i in range(101,113):
    r1_list_initialPositionList.append('r'+'%s' %(i) + 'initialPositionList.npy')
    
for i in range(101,109):
    rc1_list_initialPositionList.append('rc'+'%s' %(i) + 'initialPositionList.npy')

for i in range(201,209):
    c2_list_initialPositionList.append('c'+'%s' %(i) + 'initialPositionList.npy')
    
for i in range(201,212):
    r2_list_initialPositionList.append('r'+'%s' %(i) + 'initialPositionList.npy')

for i in range(201,209):
    rc2_list_initialPositionList.append('rc'+'%s' %(i) + 'initialPositionList.npy')

initialPositionList_files_list = c1_list_initialPositionList + r1_list_initialPositionList + rc1_list_initialPositionList + c2_list_initialPositionList + r2_list_initialPositionList + rc2_list_initialPositionList


r1_list_initialVelocityList = []
c1_list_initialVelocityList = []
rc1_list_initialVelocityList = []
r2_list_initialVelocityList = []
c2_list_initialVelocityList = []
rc2_list_initialVelocityList = []

for i in range(101,110):
    c1_list_initialVelocityList.append('c'+'%s' %(i) + 'initialVelocityList.npy')

for i in range(101,113):
    r1_list_initialVelocityList.append('r'+'%s' %(i) + 'initialVelocityList.npy')
    
for i in range(101,109):
    rc1_list_initialVelocityList.append('rc'+'%s' %(i) + 'initialVelocityList.npy')

for i in range(201,209):
    c2_list_initialVelocityList.append('c'+'%s' %(i) + 'initialVelocityList.npy')
    
for i in range(201,212):
    r2_list_initialVelocityList.append('r'+'%s' %(i) + 'initialVelocityList.npy')

for i in range(201,209):
    rc2_list_initialVelocityList.append('rc'+'%s' %(i) + 'initialVelocityList.npy')

initialVelocityList_files_list = c1_list_initialVelocityList + r1_list_initialVelocityList + rc1_list_initialVelocityList + c2_list_initialVelocityList + r2_list_initialVelocityList + rc2_list_initialVelocityList


r1_list_initialDistList = []
c1_list_initialDistList = []
rc1_list_initialDistList = []
r2_list_initialDistList = []
c2_list_initialDistList = []
rc2_list_initialDistList = []

for i in range(101,110):
    c1_list_initialDistList.append('c'+'%s' %(i) + 'initialDistList.npy')

for i in range(101,113):
    r1_list_initialDistList.append('r'+'%s' %(i) + 'initialDistList.npy')
    
for i in range(101,109):
    rc1_list_initialDistList.append('rc'+'%s' %(i) + 'initialDistList.npy')

for i in range(201,209):
    c2_list_initialDistList.append('c'+'%s' %(i) + 'initialDistList.npy')
    
for i in range(201,212):
    r2_list_initialDistList.append('r'+'%s' %(i) + 'initialDistList.npy')

for i in range(201,209):
    rc2_list_initialDistList.append('rc'+'%s' %(i) + 'initialDistList.npy')

initialDistList_files_list = c1_list_initialDistList + r1_list_initialDistList + rc1_list_initialDistList + c2_list_initialDistList + r2_list_initialDistList + rc2_list_initialDistList



r1_list_initialRouteList = []
c1_list_initialRouteList = []
rc1_list_initialRouteList = []
r2_list_initialRouteList = []
c2_list_initialRouteList = []
rc2_list_initialRouteList = []

for i in range(101,110):
    c1_list_initialRouteList.append('c'+'%s' %(i) + 'initialRouteList.npy')

for i in range(101,113):
    r1_list_initialRouteList.append('r'+'%s' %(i) + 'initialRouteList.npy')
    
for i in range(101,109):
    rc1_list_initialRouteList.append('rc'+'%s' %(i) + 'initialRouteList.npy')

for i in range(201,209):
    c2_list_initialRouteList.append('c'+'%s' %(i) + 'initialRouteList.npy')
    
for i in range(201,212):
    r2_list_initialRouteList.append('r'+'%s' %(i) + 'initialRouteList.npy')

for i in range(201,209):
    rc2_list_initialRouteList.append('rc'+'%s' %(i) + 'initialRouteList.npy')

initialRouteList_files_list = c1_list_initialRouteList + r1_list_initialRouteList + rc1_list_initialRouteList + c2_list_initialRouteList + r2_list_initialRouteList + rc2_list_initialRouteList



#RESULTS
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


r1_list_dataSet = []
c1_list_dataSet = []
rc1_list_dataSet = []
r2_list_dataSet = []
c2_list_dataSet = []
rc2_list_dataSet = []

for i in range(101,110):
    c1_list_dataSet.append('c'+'%s' %(i))

for i in range(101,113):
    r1_list_dataSet.append('r'+'%s' %(i) )
    
for i in range(101,109):
    rc1_list_dataSet.append('rc'+'%s' %(i) )

for i in range(201,209):
    c2_list_dataSet.append('c'+'%s' %(i) )
    
for i in range(201,212):
    r2_list_dataSet.append('r'+'%s' %(i) )

for i in range(201,209):
    rc2_list_dataSet.append('rc'+'%s' %(i) )

dataSet_files_list = c1_list_dataSet + r1_list_dataSet + rc1_list_dataSet + c2_list_dataSet + r2_list_dataSet + rc2_list_dataSet

#%%
#Global variables
M = 20#population size
phi = 0.3#posibility of greedy initialisation
#GA weight parameters
#alpha = 100
#beta = 0.001

no_experiments = 30

neighbour_selection = 'nn_km'

num_greedy_particles = 1 #assumed parameter
#neighbour_selection = 'b_km'
#neighbour_selection = 'dist_curr_next'

#fitness to be used
#fitness = 'ga_fitness'
fitness = 'fitness'
#fitness = 'std_ditness'
#%%
#result columns numbebers
distance_col = 0
num_vehicles_col = 1
norm_distance_col = 2
fitness_col = 3

#customer array number of columns
arr_customer_cols = 15

num_cores = multiprocessing.cpu_count()



#%%
# experiment = 0
# i = 0
for experiment in range(no_experiments):
    #data = [0,9,21,45,29]
    for i in range(len(dataSet_files_list)):#len(dataSet_files_list) for each dataset (r101...c101...)
#        i = data[data_index]
        #filenames
        dataSet = dataSet_files_list[i]
#        initialTopology_file0 = initialTopology_files_list[i]
#        initialTopology_file = os.path.join(initialTopologies_path, initialTopology_file0)
        dataInfo_name0 = dataInfo_files_list[i]
        dataInfo_name = os.path.join(dataInfo_path, dataInfo_name0)
        customer_file0 = customers_files_list[i]
        customer_file = os.path.join(customer_path, customer_file0)
        distanceMatrix_file0 = distanceMatrix_files_list[i]
        distanceMatrix_file = os.path.join(distanceMatrix_path, distanceMatrix_file0)
        results_file0 = initialResults_files_list[i]
        results_file = os.path.join(initialFitness_path + '/experiment' + '%d' %(experiment), results_file0)
        

        #read in dataframes
        df_customers = pd.read_csv(customer_file)
        df_distance_matrix = pd.read_csv(distanceMatrix_file)
        df_data_information = pd.read_csv(dataInfo_name)
        
        num_cust = df_customers.shape[0]
        df_customers.index = [x for x in range(1,num_cust+1)]
        num_customers_depot = num_cust+1
    
        df_distance_matrix.columns = df_distance_matrix.columns.map(int)
        
             
        
        arr_distance_matrix = copy.deepcopy(df_distance_matrix)
        arr_distance_matrix = arr_distance_matrix.values
        arr_customers = np.empty([num_cust, arr_customer_cols ])
        arr_customers[:,0] = df_customers.index# 0=indicies
        arr_customers[:,[1,2,3,4,13]] = df_customers.loc[:,['demand', 'readyTime', 'dueTime', 'serviceTime', 'completeTime']]
        
    
        
     
      
        
            #data set values
        total_time = df_data_information.loc[0,'fleet_max_working_time']
        total_capacity = df_data_information.loc[0,'fleet_capacity']
        fleet_size = df_data_information.loc[0,'fleet_size']
        
        #initial population
        initialRouteList_file0 = initialRouteList_files_list[i]
        initialRouteList_file = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment)  + '/%s' %(dataSet_files_list[i]) , initialRouteList_file0) 
        initialDistList_file0 = initialDistList_files_list[i]
        initialDistList_file = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment) + '/%s' %(dataSet_files_list[i]), initialDistList_file0) 
        initialPositionList_files0 = initialPositionList_files_list[i]
        initialPositionList_files = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment) + '/%s' %(dataSet_files_list[i]), initialPositionList_files0)         
        initialVelocityList_files0 = initialVelocityList_files_list[i]
        initialVelocityList_files = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment) + '/%s' %(dataSet_files_list[i]), initialVelocityList_files0)         
    
        ###################################################################################################
        
        # =============================================================================
        #        #Generate population Randomly or using Nearest Neighbour heuristic. 
        #        #For each population we generate a position array(used arcs are indicated with a vehicle number)
        #        #Sequence array (corresponds with the seq of visits)
        #        #Velocity array (corresponds tot probability of arc, this is initially set to 1)
        #        #fitness result is recorded for each population of the dataset (so we record pop1 to pop20 fitness for dataset[i] )
        #        #note: we include deopy as ---->FROM v TO
        # =============================================================================
        
        
        #INITIALISE POPULATION DICTIONARY
#        dict_population = {}
#        pop_distance_list = []
#        pop_num_route_list = []
#        for j in range(M):#for each population set (1,2,....,m=20)
        rand_val_array = np.random.uniform(0,1,M-num_greedy_particles) #generate 
        
        #GENERATE POPULATION
        #number of nearest nearest greediest particle
        
        #numbe of nearest randome particles
        num_nnh_particles = len(np.where(rand_val_array<phi)[0])
        #number of random particles
        num_random_particles = M-num_greedy_particles-num_nnh_particles
        
        df_results, arr_population_routeList, arr_population_distanceList, arr_population_particle_position, arr_population_particle_velocity = initial_population(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot, num_cores, num_greedy_particles, M, num_random_particles, num_nnh_particles)
        
        np.save(initialRouteList_file, arr_population_routeList)
        np.save(initialDistList_file, arr_population_distanceList)
        np.save(initialPositionList_files, arr_population_particle_position)
        np.save(initialVelocityList_files, arr_population_particle_velocity)
        df_results.to_csv(results_file, index=False)

        print('dataset:', dataSet_files_list[i])            
    print('experiment:', experiment)


    
    
        

    
    
    
