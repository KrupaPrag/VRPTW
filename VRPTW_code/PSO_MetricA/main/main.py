#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRPTW: PSO Metric A 

@author: krupa prag 

"""


import numpy as np
import pandas as pd
import copy
import os
import time
import multiprocessing
from functions import omega, pbest_particle_pC_list, PSO, CLPSO_velocity_update, update_route_position, PSO_result_updater, CLPSO_result_updater, local_search_result_updater, global_result_from_experiments, local_search
#%%
# Global variables (change for different number of customers)
num_customers = 25
num_experiments = 4

#%%
#GLOBAL VARIABLES:
#VARIABLES STIPULATED IN PAPER

M = 20#population size 
sg = 1000# stopping gap (number of iterations that gbest must not change)
rg = 7 #refreshing gap
c = 2#constant c
phi = 0.3#phi
w0 = 0.9 #omega #decreases linearly through training to 0.4
w1 = 0.4



alpha = 100
beta = 0.001
#see parameter calculations for omega and learning probability functions

#Assummed variables:
max_gen = sg#equal to sg .... so omega changes respective to the new cycle with a max_gen threshold



#%%
#Paths
cwd = os.getcwd()
initial_path = cwd +'/initial'
final_path = cwd + '/final'#finalResults_path

customer_path  = cwd +'/solomon%d_csv/customers'%(num_customers)
dataInfo_path = cwd +'/solomon%d_csv/data_info'%(num_customers)
distanceMatrix_path = cwd +'/solomon%d_csv/distance_matrix'%(num_customers)
initial_path = cwd.split('/')
initial_path = '/'.join(initial_path[:-1])
initialPop_path = initial_path +'/initial_encoding/initial/initial_pop'
initialFitness_path = initial_path +'/initial_encoding/initial/initial_fitness'
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
#FILE NAMES AND FOLDERS


##READ In FILES: CUSTOMERS, DISTANCES, DATA INFORMATION


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

#INITIAL READ IN
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


#
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



#FINAL WRITE OUT

r1_list_finalPositionList = []
c1_list_finalPositionList = []
rc1_list_finalPositionList = []
r2_list_finalPositionList = []
c2_list_finalPositionList = []
rc2_list_finalPositionList = []

for i in range(101,110):
    c1_list_finalPositionList.append('c'+'%s' %(i) + 'finalPositionList.npy')

for i in range(101,113):
    r1_list_finalPositionList.append('r'+'%s' %(i) + 'finalPositionList.npy')
    
for i in range(101,109):
    rc1_list_finalPositionList.append('rc'+'%s' %(i) + 'finalPositionList.npy')

for i in range(201,209):
    c2_list_finalPositionList.append('c'+'%s' %(i) + 'finalPositionList.npy')
    
for i in range(201,212):
    r2_list_finalPositionList.append('r'+'%s' %(i) + 'finalPositionList.npy')

for i in range(201,209):
    rc2_list_finalPositionList.append('rc'+'%s' %(i) + 'finalPositionList.npy')

finalPositionList_files_list = c1_list_finalPositionList + r1_list_finalPositionList + rc1_list_finalPositionList + c2_list_finalPositionList + r2_list_finalPositionList + rc2_list_finalPositionList


r1_list_finalVelocityList = []
c1_list_finalVelocityList = []
rc1_list_finalVelocityList = []
r2_list_finalVelocityList = []
c2_list_finalVelocityList = []
rc2_list_finalVelocityList = []

for i in range(101,110):
    c1_list_finalVelocityList.append('c'+'%s' %(i) + 'finalVelocityList.npy')

for i in range(101,113):
    r1_list_finalVelocityList.append('r'+'%s' %(i) + 'finalVelocityList.npy')
    
for i in range(101,109):
    rc1_list_finalVelocityList.append('rc'+'%s' %(i) + 'finalVelocityList.npy')

for i in range(201,209):
    c2_list_finalVelocityList.append('c'+'%s' %(i) + 'finalVelocityList.npy')
    
for i in range(201,212):
    r2_list_finalVelocityList.append('r'+'%s' %(i) + 'finalVelocityList.npy')

for i in range(201,209):
    rc2_list_finalVelocityList.append('rc'+'%s' %(i) + 'finalVelocityList.npy')

finalVelocityList_files_list = c1_list_finalVelocityList + r1_list_finalVelocityList + rc1_list_finalVelocityList + c2_list_finalVelocityList + r2_list_finalVelocityList + rc2_list_finalVelocityList


r1_list_finalDistList = []
c1_list_finalDistList = []
rc1_list_finalDistList = []
r2_list_finalDistList = []
c2_list_finalDistList = []
rc2_list_finalDistList = []

for i in range(101,110):
    c1_list_finalDistList.append('c'+'%s' %(i) + 'finalDistList.npy')

for i in range(101,113):
    r1_list_finalDistList.append('r'+'%s' %(i) + 'finalDistList.npy')
    
for i in range(101,109):
    rc1_list_finalDistList.append('rc'+'%s' %(i) + 'finalDistList.npy')

for i in range(201,209):
    c2_list_finalDistList.append('c'+'%s' %(i) + 'finalDistList.npy')
    
for i in range(201,212):
    r2_list_finalDistList.append('r'+'%s' %(i) + 'finalDistList.npy')

for i in range(201,209):
    rc2_list_finalDistList.append('rc'+'%s' %(i) + 'finalDistList.npy')

finalDistList_files_list = c1_list_finalDistList + r1_list_finalDistList + rc1_list_finalDistList + c2_list_finalDistList + r2_list_finalDistList + rc2_list_finalDistList



r1_list_finalRouteList = []
c1_list_finalRouteList = []
rc1_list_finalRouteList = []
r2_list_finalRouteList = []
c2_list_finalRouteList = []
rc2_list_finalRouteList = []

for i in range(101,110):
    c1_list_finalRouteList.append('c'+'%s' %(i) + 'finalRouteList.npy')

for i in range(101,113):
    r1_list_finalRouteList.append('r'+'%s' %(i) + 'finalRouteList.npy')
    
for i in range(101,109):
    rc1_list_finalRouteList.append('rc'+'%s' %(i) + 'finalRouteList.npy')

for i in range(201,209):
    c2_list_finalRouteList.append('c'+'%s' %(i) + 'finalRouteList.npy')
    
for i in range(201,212):
    r2_list_finalRouteList.append('r'+'%s' %(i) + 'finalRouteList.npy')

for i in range(201,209):
    rc2_list_finalRouteList.append('rc'+'%s' %(i) + 'finalRouteList.npy')

finalRouteList_files_list = c1_list_finalRouteList + r1_list_finalRouteList + rc1_list_finalRouteList + c2_list_finalRouteList + r2_list_finalRouteList + rc2_list_finalRouteList

#
#RESULTS: population result of the final generation
r1_list_finalResults = []
c1_list_finalResults = []
rc1_list_finalResults = []
r2_list_finalResults = []
c2_list_finalResults = []
rc2_list_finalResults = []

for i in range(101,110):
    c1_list_finalResults.append('c'+'%s' %(i) + 'finalResults.csv')

for i in range(101,113):
    r1_list_finalResults.append('r'+'%s' %(i) + 'finalResults.csv')
    
for i in range(101,109):
    rc1_list_finalResults.append('rc'+'%s' %(i) + 'finalResults.csv')

for i in range(201,209):
    c2_list_finalResults.append('c'+'%s' %(i) + 'finalResults.csv')
    
for i in range(201,212):
    r2_list_finalResults.append('r'+'%s' %(i) + 'finalResults.csv')

for i in range(201,209):
    rc2_list_finalResults.append('rc'+'%s' %(i) + 'finalResults.csv')

finalResults_files_list = c1_list_finalResults + r1_list_finalResults + rc1_list_finalResults + c2_list_finalResults + r2_list_finalResults + rc2_list_finalResults



r1_list_generationTracker = []
c1_list_generationTracker = []
rc1_list_generationTracker = []
r2_list_generationTracker = []
c2_list_generationTracker = []
rc2_list_generationTracker = []

for i in range(101,110):
    c1_list_generationTracker.append('c'+'%s' %(i) + 'generationTracker.csv')

for i in range(101,113):
    r1_list_generationTracker.append('r'+'%s' %(i) + 'generationTracker.csv')
    
for i in range(101,109):
    rc1_list_generationTracker.append('rc'+'%s' %(i) + 'generationTracker.csv')

for i in range(201,209):
    c2_list_generationTracker.append('c'+'%s' %(i) + 'generationTracker.csv')
    
for i in range(201,212):
    r2_list_generationTracker.append('r'+'%s' %(i) + 'generationTracker.csv')

for i in range(201,209):
    rc2_list_generationTracker.append('rc'+'%s' %(i) + 'generationTracker.csv')

generationTracker_files_list = c1_list_generationTracker + r1_list_generationTracker + rc1_list_generationTracker + c2_list_generationTracker + r2_list_generationTracker + rc2_list_generationTracker



r1_list_pbestGenerationTracker = []
c1_list_pbestGenerationTracker = []
rc1_list_pbestGenerationTracker = []
r2_list_pbestGenerationTracker = []
c2_list_pbestGenerationTracker = []
rc2_list_pbestGenerationTracker = []

for i in range(101,110):
    c1_list_pbestGenerationTracker.append('c'+'%s' %(i) + 'pbestGenerationTracker.csv')

for i in range(101,113):
    r1_list_pbestGenerationTracker.append('r'+'%s' %(i) + 'pbestGenerationTracker.csv')
    
for i in range(101,109):
    rc1_list_pbestGenerationTracker.append('rc'+'%s' %(i) + 'pbestGenerationTracker.csv')

for i in range(201,209):
    c2_list_pbestGenerationTracker.append('c'+'%s' %(i) + 'pbestGenerationTracker.csv')
    
for i in range(201,212):
    r2_list_pbestGenerationTracker.append('r'+'%s' %(i) + 'pbestGenerationTracker.csv')

for i in range(201,209):
    rc2_list_pbestGenerationTracker.append('rc'+'%s' %(i) + 'pbestGenerationTracker.csv')

pbestGenerationTracker_files_list = c1_list_pbestGenerationTracker + r1_list_pbestGenerationTracker + rc1_list_pbestGenerationTracker + c2_list_pbestGenerationTracker + r2_list_pbestGenerationTracker + rc2_list_pbestGenerationTracker


r1_list_gbestGenerationTracker = []
c1_list_gbestGenerationTracker = []
rc1_list_gbestGenerationTracker = []
r2_list_gbestGenerationTracker = []
c2_list_gbestGenerationTracker = []
rc2_list_gbestGenerationTracker = []

for i in range(101,110):
    c1_list_gbestGenerationTracker.append('c'+'%s' %(i) + 'gbestGenerationTracker.csv')

for i in range(101,113):
    r1_list_gbestGenerationTracker.append('r'+'%s' %(i) + 'gbestGenerationTracker.csv')
    
for i in range(101,109):
    rc1_list_gbestGenerationTracker.append('rc'+'%s' %(i) + 'gbestGenerationTracker.csv')

for i in range(201,209):
    c2_list_gbestGenerationTracker.append('c'+'%s' %(i) + 'gbestGenerationTracker.csv')
    
for i in range(201,212):
    r2_list_gbestGenerationTracker.append('r'+'%s' %(i) + 'gbestGenerationTracker.csv')

for i in range(201,209):
    rc2_list_gbestGenerationTracker.append('rc'+'%s' %(i) + 'gbestGenerationTracker.csv')

gbestGenerationTracker_files_list = c1_list_gbestGenerationTracker + r1_list_gbestGenerationTracker + rc1_list_gbestGenerationTracker + c2_list_gbestGenerationTracker + r2_list_gbestGenerationTracker + rc2_list_gbestGenerationTracker



r1_list_pbestResults = []
c1_list_pbestResults = []
rc1_list_pbestResults = []
r2_list_pbestResults = []
c2_list_pbestResults = []
rc2_list_pbestResults = []

for i in range(101,110):
    c1_list_pbestResults.append('c'+'%s' %(i) + 'pbestResults.csv')

for i in range(101,113):
    r1_list_pbestResults.append('r'+'%s' %(i) + 'pbestResults.csv')
    
for i in range(101,109):
    rc1_list_pbestResults.append('rc'+'%s' %(i) + 'pbestResults.csv')

for i in range(201,209):
    c2_list_pbestResults.append('c'+'%s' %(i) + 'pbestResults.csv')
    
for i in range(201,212):
    r2_list_pbestResults.append('r'+'%s' %(i) + 'pbestResults.csv')

for i in range(201,209):
    rc2_list_pbestResults.append('rc'+'%s' %(i) + 'pbestResults.csv')

pbestResults_files_list = c1_list_pbestResults + r1_list_pbestResults + rc1_list_pbestResults + c2_list_pbestResults + r2_list_pbestResults + rc2_list_pbestResults



r1_list_gbestResults = []
c1_list_gbestResults = []
rc1_list_gbestResults = []
r2_list_gbestResults = []
c2_list_gbestResults = []
rc2_list_gbestResults = []

for i in range(101,110):
    c1_list_gbestResults.append('c'+'%s' %(i) + 'gbestResults.csv')

for i in range(101,113):
    r1_list_gbestResults.append('r'+'%s' %(i) + 'gbestResults.csv')
    
for i in range(101,109):
    rc1_list_gbestResults.append('rc'+'%s' %(i) + 'gbestResults.csv')

for i in range(201,209):
    c2_list_gbestResults.append('c'+'%s' %(i) + 'gbestResults.csv')
    
for i in range(201,212):
    r2_list_gbestResults.append('r'+'%s' %(i) + 'gbestResults.csv')

for i in range(201,209):
    rc2_list_gbestResults.append('rc'+'%s' %(i) + 'gbestResults.csv')

gbestResults_files_list = c1_list_gbestResults + r1_list_gbestResults + rc1_list_gbestResults + c2_list_gbestResults + r2_list_gbestResults + rc2_list_gbestResults


#%%
fitness_type = 'fitness'

#num cores in parallel
num_cores = multiprocessing.cpu_count()

#%%
#INITIALISE FINAL RESULTS TABLE: TEMPORARY AS CAPTURES ALL EXPERIMENT RESULTS
df_timer = pd.DataFrame(np.zeros((len(dataset_list), num_experiments)))
df_distance = pd.DataFrame(np.zeros((len(dataset_list), num_experiments)))
df_numVehicles = pd.DataFrame(np.zeros((len(dataset_list), num_experiments)))
df_fitness = pd.DataFrame(np.zeros((len(dataset_list), num_experiments)))
df_particle = pd.DataFrame(np.zeros((len(dataset_list),num_experiments)))


#%%
#SUMMARY OF RESULTS: FINAL RESULTS TO SAVE
df_overall_experimental_results = pd.DataFrame(np.zeros((len(dataset_list), 6)))#dataset, num_V, distance, fitness, average_timer, min time
df_overall_experimental_results.columns = ['dataset','num_vehicles', 'distance', 'fitness', 'average_time', 'min_time']
#%%

#read in comparison results 

df_results_comparative = pd.read_csv(os.path.join(cwd, 'PSO_%d_resultsOverviewComparison.csv'%(num_customers)))
df_results_comparative.index = df_results_comparative.dataset

#%%        
#result columns numbebers
distance_col = 1
num_vehicles_col = 0
fitness_col = 2

#customer array number of columns
arr_customer_cols = 15

#customer array number of columns
# 0: customer
# 1: demand
# 2: readtTime
# 3: dueTime
# 4: serviceTime
# 5: distanceCurrNext
# 6: ArrivalTime
# 7: WaitingTime
# 8: ExecStartTime
# 9; rkm = due - arrival
# 10 nn_km = rkm + waiting + disting
# 11: new_curr_time = execStart + serviceTime
# 12: total_time_check = new_curr_time + distance[curr,0]
# 13: completeTime
# 14: startTime = readyTime - depot distance
#%%
for  i in range(len(dataset_list)):
    #file names
    dataSet = dataset_list[i]
    
    PSO_benchmark_NV = df_results_comparative.loc[dataSet, 'PSO_nv']
    PSO_benchmark_dist = df_results_comparative.loc[dataSet, ' PSO_distance']
    PSO_benchmark_val = round(alpha*PSO_benchmark_NV + beta*PSO_benchmark_dist,2)
    
    best_benchmark_NV = df_results_comparative.loc[dataSet, 'best_nv']
    best_benchmark_dist = df_results_comparative.loc[dataSet, 'best_distance']
    best_benchmark_val = round(alpha*best_benchmark_NV + beta*best_benchmark_dist, 2)
    
    
    #READ IN DATA PERTAINING TO THE DATASET
    customer_file = os.path.join(customer_path, '%scustomers.csv'%(dataSet))
    dataInfo_file = os.path.join(dataInfo_path, '%sdataInfo.csv'%(dataSet))
    distanceMatrix_file = os.path.join(distanceMatrix_path, '%sdistanceMatrix.csv'%(dataSet))

    

    #CUSTOMER DATASET DATA
    #dataset files    
    df_customers = pd.read_csv(customer_file)
    df_distance_matrix = pd.read_csv(distanceMatrix_file)
    arr_distance_matrix = copy.deepcopy(df_distance_matrix)
    arr_distance_matrix = arr_distance_matrix.values
    df_data_information = pd.read_csv(dataInfo_file)
    num_cust = df_customers.shape[0]
    num_customers_depot = num_cust +1
    df_customers.index = [x for x in range(1,num_cust+1)]#reset indicies
    df_distance_matrix.columns = df_distance_matrix.columns.map(int)
    
    #df_customers-> array: only keeping [Demand, readyTime, dueTime, serviceTime]
    #columns of array: CUSTOMER, DEMAND-CURRCAPACITY, READYTIME, DUETIME, SERVICETIME,DISTANCECURRNEXT, ARRIVALTIME, WAITINGTIME, EXECSTARTTIME, RKM,NKM, NEWCURRTIME, TOTALTIMECHECK
    arr_customers = np.empty([num_cust, arr_customer_cols ])
    arr_customers[:,0] = df_customers.index# 0=indicies
    arr_customers[:,[1,2,3,4,13]] = df_customers.loc[:,['demand', 'readyTime', 'dueTime', 'serviceTime', 'completeTime']]
    
    
# =============================================================================
#     Updated
# =============================================================================
    arr_customer_info = copy.deepcopy(arr_customers[:, [0,1,2,3]]) #0customer, 1demand, 2ready, 3due
    
    #global variables
    total_capacity = df_data_information.loc[0,'fleet_capacity']
    total_time = df_data_information.loc[0, 'fleet_max_working_time']
    service_time = df_customers.loc[1, 'serviceTime']
    
    dict_routes = {}
    
    
    #final results files

    final_result_file = os.path.join(final_path + '/%s' %(dataSet), 'final_result.csv') 
    final_route_file = os.path.join(final_path + '/%s' %(dataSet), 'final_route.npy') 

    
    #experiment results recorder
    df_experiment_result_recorder = pd.DataFrame(np.zeros((num_experiments, 4)))
    experiment_result_file = os.path.join(final_path +'/%s' %(dataSet), 'experimentRecorder.csv' )
    experiment_gbest_route_file = os.path.join(final_path +'/%s' %(dataSet), 'experimentRoute.npy' )

    
    
    for experiment in range(num_experiments):

        
        #read in initial file names
        results_file0 = initialResults_files_list[i]
        results_file = os.path.join(initialFitness_path+ '/experiment' + '%d' %(experiment), results_file0)
        
        initialRouteList_file0 = initialRouteList_files_list[i]
        initialRouteList_file = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment)  + '/%s' %(dataSet_files_list[i]) , initialRouteList_file0) 
        initialDistList_file0 = initialDistList_files_list[i]
        initialDistList_file = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment) + '/%s' %(dataSet_files_list[i]), initialDistList_file0) 
        initialPositionList_files0 = initialPositionList_files_list[i]
        initialPositionList_files = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment) + '/%s' %(dataSet_files_list[i]), initialPositionList_files0)         
        initialVelocityList_files0 = initialVelocityList_files_list[i]
        initialVelocityList_files = os.path.join(initialPop_path+ '/experiment' + '%d' %(experiment) + '/%s' %(dataSet_files_list[i]), initialVelocityList_files0)   
       
        
  
        #READ IN INITIAL DICTIONARIES
        #dictionary form of initial population
        #READ IN INITIAL DICTIONARIES
        pop_particle_routeList_list =np.load(initialRouteList_file, allow_pickle=True)
        pop_particle_distance_list = np.load(initialDistList_file, allow_pickle=True)
        pop_particle_position_list =  np.array(list(np.load(initialPositionList_files,allow_pickle=True)))
        pop_particle_velocity_list =  np.array(list(np.load(initialVelocityList_files, allow_pickle=True)))
        df_results = pd.read_csv(results_file)
        arr_results = df_results.values# ['num_vehicles', 'distance', 'fitness']
        pop_distance_list = arr_results[:,1]
        pop_num_route_list = arr_results[:,0]
        
        #pbest is = curr
        pbest_particle_routeList_list = copy.deepcopy(pop_particle_routeList_list)
        pbest_particle_distance_list = copy.deepcopy(pop_particle_distance_list)
        pbest_particle_position_list = copy.deepcopy(pop_particle_position_list)
        pbest_particle_velocity_list = copy.deepcopy(pop_particle_velocity_list)
        arr_pbest_results = copy.deepcopy(arr_results)
        pbest_distance_list = copy.deepcopy(pop_distance_list)
        pbest_num_route_list = copy.deepcopy(pop_num_route_list)
        
        
        #gbest 
        gbest_particle_no = np.argmin(arr_pbest_results[:,fitness_col]) #df_pbest_results[fitness_type].idxmin()#min value in the fitness
        gbest_routeList_list = pop_particle_routeList_list[gbest_particle_no]
        gbest_distance_list = pop_particle_distance_list[gbest_particle_no]
        gbest_position_list = pbest_particle_position_list[gbest_particle_no]
        gbest_velocity_list = pbest_particle_velocity_list[gbest_particle_no]
        gbest_fitness = arr_pbest_results[gbest_particle_no, fitness_col]
        arr_gbest_result = arr_pbest_results[gbest_particle_no, :]
        gbest_fitness = round(arr_gbest_result[2],2)

    
    # =============================================================================
    #     Population is created for a particular dataset R...C...RC...
    #           Apply updates to the particles (various solutions) of the population
    # =============================================================================
        #initialise tracking variables: 
        k = 0 #tracker of num iterations
        flag = np.zeros(M)#[0]*M #tracker of num iterations that the pbest value hasn't changed for each particle
        gbest_tracker = 0 # tracker of num of iterations that the gbest value hasn't changed
        iteration_no = 0
    
      
    
    # =============================================================================
    # FULL PSO UPDATE
    #    while gbest stopping condition not met:
    #       if pbest value for any particle violates refreshing gap then update particle using PSO
    #       Update all particles using CLPS
    #       Perform local search on all particles
    #       Update and record results (increment gbest tracker, flag value update)
    #    
    # =============================================================================
        start_timer = time.time()#uncomment to terminate
        while gbest_tracker<sg: #or gbest_fitness>PSO_benchmark_val==True or gbest_fitness>best_benchmark_val==True:
            w = omega(w0,w1, gbest_tracker, max_gen)
            arr_pbest_particle = pbest_particle_pC_list(M)


##            PSO
            if np.max(flag)>=rg:
                PSO_resultlist, flagged_indicies  = PSO(flag, rg, w, c, num_cores, num_customers_depot, arr_distance_matrix, arr_customers,  total_time, total_capacity, gbest_position_list, pbest_particle_position_list, pop_particle_position_list, pop_particle_velocity_list)

                flag, arr_results, arr_pbest_results, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_distance_list,  pop_distance_list, pop_num_route_list, pop_particle_velocity_list, arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list, gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list = PSO_result_updater(rg,PSO_resultlist, arr_results, arr_pbest_results,  flag, flagged_indicies, pop_particle_routeList_list, pop_particle_position_list, pop_particle_distance_list,  pop_distance_list, pop_num_route_list, pop_particle_velocity_list,  arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list,  gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list, pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list)
#
#            
            #CLPSO
            #velocity update
            pop_particle_velocity_list = CLPSO_velocity_update(M,w,c,num_customers_depot, pop_particle_position_list, pbest_particle_position_list, pop_particle_velocity_list)
            
            #position update
            position_resultslist =  update_route_position(num_cores,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot, pop_particle_velocity_list, pop_particle_position_list, M)
            
            flag, arr_results, arr_pbest_results, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_distance_list,  pop_distance_list, pop_num_route_list,  arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list,   gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list = CLPSO_result_updater(M,arr_results, arr_pbest_results, flag, pop_particle_velocity_list, position_resultslist, arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list, gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list, pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list)         
         
            #LOCAL SEARCH
            local_search_results_list = local_search(M, num_cores, df_customers,arr_distance_matrix, total_time, total_capacity, pop_distance_list, pop_num_route_list, pop_particle_routeList_list, pop_particle_distance_list, pop_particle_position_list, pop_particle_velocity_list,arr_customer_info, service_time)
 
#            time4 = time.time()            
            flag, arr_results, arr_pbest_results, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_velocity_list, pop_particle_distance_list,  pop_distance_list, pop_num_route_list,  arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list,   gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list = local_search_result_updater(local_search_results_list, flag, arr_results, arr_pbest_results, pop_particle_velocity_list, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_distance_list,  pop_distance_list, pop_num_route_list,  arr_gbest_result,  gbest_velocity_list, gbest_position_list, gbest_routeList_list,   gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list)

#            

            k = k +1
            gbest_fitness = round(arr_gbest_result[2],2)
               
        end_timer = time.time()

        dict_routes[experiment] = gbest_routeList_list#best route to be appended
        print(arr_gbest_result)
        print('experiment: ', experiment)
        #save update after each experiment 
        np.save(experiment_gbest_route_file, dict_routes)
        df_experiment_result_recorder.iloc[experiment,0:3] = arr_gbest_result[0:3]
        df_experiment_result_recorder.iloc[experiment, 3] = end_timer-start_timer
        df_experiment_result_recorder.to_csv(experiment_result_file, index=False)
        
        
#    #Record  best result of all experiments for a particular dataset   
        
    df_results, dict_final_route = global_result_from_experiments(df_experiment_result_recorder, dict_routes, num_experiments)   
    df_results.to_csv(final_result_file, index=False)
    np.save(final_route_file, dict_final_route)
     
 
    
    #evaluate best solution from all the experiments
    print('dataset completed: ' ,dataSet)
    

