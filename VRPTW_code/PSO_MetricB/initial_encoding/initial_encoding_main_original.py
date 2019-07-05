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
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from initial_encoding_routes import nearest_neighbour_heauristic, random_neighbour_heauristic
from initial_encoding_fitness_results import initial_fitness
#%%
cwd = os.getcwd()
#read_in
dataInfo_path = cwd + '/solomon25_csv/data_info'
customer_path = cwd + '/solomon25_csv/customers'
distanceMatrix_path = cwd + '/solomon25_csv/distance_matrix'

#results
initialPop_path = cwd + '/initial/initial_pop'
initialFitness_path = cwd + '/initial/initial_fitness'
initialTopologies_path = cwd + '/initial/initial_topologies'

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


r1_list_initialTopology = []
c1_list_initialTopology = []
rc1_list_initialTopology = []
r2_list_initialTopology = []
c2_list_initialTopology = []
rc2_list_initialTopology = []

for i in range(101,110):
    c1_list_initialTopology.append('c'+'%s' %(i) + 'initialTopology.pdf')

for i in range(101,113):
    r1_list_initialTopology.append('r'+'%s' %(i) + 'initialTopology.pdf')
    
for i in range(101,109):
    rc1_list_initialTopology.append('rc'+'%s' %(i) + 'initialTopology.pdf')

for i in range(201,209):
    c2_list_initialTopology.append('c'+'%s' %(i) + 'initialTopology.pdf')
    
for i in range(201,212):
    r2_list_initialTopology.append('r'+'%s' %(i) + 'initialTopology.pdf')

for i in range(201,209):
    rc2_list_initialTopology.append('rc'+'%s' %(i) + 'initialTopology.pdf')

initialTopology_files_list = c1_list_initialTopology + r1_list_initialTopology + rc1_list_initialTopology + c2_list_initialTopology + r2_list_initialTopology + rc2_list_initialTopology


#%%
#Global variables
M = 20#population size
phi = 0.3#posibility of greedy initialisation
#GA weight parameters
alpha = 100
beta = 0.001

no_experiments = 10

#fitness to be used
#fitness = 'ga_fitness'
fitness = 'fitness'
#fitness = 'std_ditness'
#%%
for i in range(len(dataSet_files_list)):#for each dataset (r101...c101...)
    #filenames
    dataSet = dataSet_files_list[i]
    initialTopology_file0 = initialTopology_files_list[i]
    initialTopology_file = os.path.join(initialTopologies_path, initialTopology_file0)
    dataInfo_name0 = dataInfo_files_list[i]
    dataInfo_name = os.path.join(dataInfo_path, dataInfo_name0)
    customer_file0 = customers_files_list[i]
    customer_file = os.path.join(customer_path, customer_file0)
    distanceMatrix_file0 = distanceMatrix_files_list[i]
    distanceMatrix_file = os.path.join(distanceMatrix_path, distanceMatrix_file0)
    results_file0 = initialResults_files_list[i]
    results_file = os.path.join(initialFitness_path, results_file0)
    
    #read in dataframes
    df_customers = pd.read_csv(customer_file)
    df_distance_matrix = pd.read_csv(distanceMatrix_file)
    df_data_information = pd.read_csv(dataInfo_name)
 
    num_customers = df_customers.shape[0]
    df_customers.index = [x for x in range(1,num_customers+1)]
    num_customers_depot = num_customers+1

    df_distance_matrix.columns = df_distance_matrix.columns.map(int)
    
    
    
        #data set values
    total_time = df_data_information.loc[0,'fleet_max_working_time']
    total_capacity = df_data_information.loc[0,'fleet_capacity']
    fleet_size = df_data_information.loc[0,'fleet_size']
    
    #initial population
    initialPop_file0 = initialPop_files_list[i]
    initialPop_file = os.path.join(initialPop_path , initialPop_file0)    

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
    dict_population = {}
    pop_distance_list = []
    pop_num_route_list = []
    for j in range(M):#for each population set (1,2,....,m=20)
        #GENERATE POPULATION
        if random.uniform(0,1)<phi:           
            route_routeList_list, route_velocity_list, route_position_list, route_capacity_list, route_distance_list, route_time_list, route_validity_list = nearest_neighbour_heauristic(df_customers, df_distance_matrix, total_time, total_capacity, num_customers_depot)
            particle_list = [route_routeList_list, route_velocity_list, route_position_list, route_capacity_list, route_distance_list, route_time_list, route_validity_list]
            particle_distance = sum(route_distance_list)
            particle_validity = all(route_validity_list)
            
        else:
            route_routeList_list, route_velocity_list, route_position_list, route_capacity_list, route_distance_list, route_time_list, route_validity_list = random_neighbour_heauristic(df_customers, df_distance_matrix, total_time, total_capacity, num_customers_depot)        
            particle_list = [route_routeList_list, route_velocity_list, route_position_list, route_capacity_list, route_distance_list, route_time_list, route_validity_list]
            particle_distance = sum(route_distance_list)
            particle_validity = all(route_validity_list)
            
        pop_distance_list.append(particle_distance)
        pop_num_route_list.append(len(route_routeList_list))
        dict_population.update({j:[particle_list, particle_distance, particle_validity]})
    
    #calculate population fitness
    df_results = initial_fitness(M, pop_distance_list, pop_num_route_list, alpha, beta)
    
    np.save(initialPop_file, dict_population)
    df_results.to_csv(results_file, index=False)


    
        

    
    
    
