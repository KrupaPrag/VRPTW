#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:57:50 2019

@author: user
"""

import numpy as np
import pandas as pd
import math as math
import random
import copy
from joblib import Parallel,delayed
#%%
# =============================================================================
# CHECK VALIDITY OF ROUTE and time
# =============================================================================
# DISTANCE CALCULATOR
def distance_calculator(route, arr_distance_matrix):
    r = route[:-1]
    c = route[1::]
    distance = sum(arr_distance_matrix[r,c])
    return distance
#%%
def validity_time(route_list, arr_distance_matrix, arr_customer_info,total_capacity, total_time, service_time):
    cust = route_list[1]
    total_dist = arr_distance_matrix[0, cust]
    readyTime = arr_customer_info[cust-1, 2]
    curr_cap = arr_customer_info[cust-1, 1]
    validity = True
    if readyTime- total_dist>0:
        start_time = readyTime-total_dist
    else:
        start_time = 0
    curr_time = start_time + total_dist + service_time#df_customers.loc[cust, 'serviceTime']
    completeTime = arr_customer_info[route_list[1]-1,3]+service_time
    for i in range(1,len(route_list)-2):
#        print(i)
        cust1 = route_list[i]
        cust2 = route_list[i+1]
        dist = arr_distance_matrix[cust1, cust2]
        total_dist = total_dist + dist
        readyTime = arr_customer_info[cust2-1, 2]
        completeTime = arr_customer_info[cust2-1, 3]+service_time
        curr_cap = curr_cap + arr_customer_info[cust2-1, 1]
        curr_time = curr_time + dist
        if curr_time<readyTime:
            waiting_time = readyTime - curr_time
            curr_time = curr_time + waiting_time + service_time#df_customers.loc[cust2, 'serviceTime']
        else:
            curr_time = curr_time + service_time#df_customers.loc[cust2, 'serviceTime']
        
        if curr_time>completeTime or curr_time>total_time or curr_cap>total_capacity:
            validity = False
#            print(route_list[i])
            break
    curr_time = curr_time + arr_distance_matrix[route_list[-2],0]
    total_dist = total_dist + arr_distance_matrix[route_list[-2],0]
    if curr_time>total_time or curr_cap>total_capacity:
        validity = False
        
    duration = curr_time - start_time
    return validity, curr_time, curr_cap, duration, start_time, total_dist


        
#%%
#OMEGA: weight value function calculator
def omega(w0,w1, k, max_gen):#k = iteration
    omega_val = w0-(((w0-w1)*k)/(max_gen))
    return omega_val

#%%
# =============================================================================
# #PBEST PARTICLE LIST CALCULATOR. Helper functions : routrnament selection and learning probability
# =============================================================================
    
#calculates a single pC value for particle i of a population M
def learning_probability(particle_i, M):
    pC = 0.05 + 0.45*(((math.exp((10*(particle_i-1))/(M-1)))-1)/(math.exp(10)-1))
    return pC
#%%
#from a subest of particles a particle is selected via particle selection
#def tournament_selection(particle, best_population_particle_fitness_list, M):
#    possible_list = random.sample(range(0, M), M)#shuffled list of all particles
#    possible_list.remove(particle)#removes particle i
#    tournament_particles = random.sample(set(possible_list), 2)# returns a list of two particles
#    if best_population_particle_fitness_list[tournament_particles[0]]<best_population_particle_fitness_list[tournament_particles[1]]:
#        selected_particle = tournament_particles[1]
#    else:
#        selected_particle = tournament_particles[0]
#    return selected_particle
    

def tournament_selection(particle, M, pC):
#    possible_list = random.sample(range(0, M), M)#shuffled list of all particles
#    possible_list.remove(particle)#removes particle i
#    tournament_particles = random.sample(possible_list, 2)# returns a list of two particles
#    if pC<random.random(0,1):
#        return particle
#        
#    else:
    tournament_particle = random.randint(1,M)
    while tournament_particle == particle:
        tournament_particle = random.randint(1,M)
    return tournament_particle

#%%
#returns a dataframe of corresponding pbest particle to use for each particle    
#def pbest_particle_pC_list(best_population_particle_fitness_list, M):
##    inputs = range(0,M)
##    start_time = time.time()
##    pC_list = Parallel(n_jobs=num_cores)(delayed(learning_probability)(particle_i, M) for particle_i in inputs)
##    end_time = time.time()
##    print(end_time-start_time)#0.1776106357574463
#    
##    start_time1 = time.time()
##    pC_list = []
#    pC_list = np.empty(M)
#    for i in range(M):
##        pC_list.append(learning_probability(i+1,M))#add 1 as value are 1 to M
#        pC_list[i] = learning_probability(i+1,M)
##    end_time1 = time.time()
##    print(decimal.Decimal(end_time1-start_time1))#0.00011181831359863281
#    rand_arrList = np.random.uniform(0,1,M)
#    #returns true / false 
#    bool_arr = rand_arrList<pC_list
#    indicies_false_list = np.where(bool_arr)[0]
#    
#    arr_pbest_particle = np.arange(M)
#    
##    df_pbest_particle = pd.DataFrame(index = range(0,M), columns = [0])
##    df_pbest_particle.loc[indicies_true_list,0] = indicies_true_list
##    nan_indicies = pd.isnull(df_pbest_particle).any(1).nonzero()[0].tolist()
#    
#    #Replace nan indicies with particle from tournament selection
##    start_time = time.time()
#    for particle in range(len(indicies_false_list)):
##        particle = indicies_false_list[j]
#        arr_pbest_particle[particle] = tournament_selection(particle, best_population_particle_fitness_list, M)
##        df_pbest_particle.loc[particle,0] = selected_particle
##    print(decimal.Decimal(time.time()-start_time))
#    return arr_pbest_particle


def pbest_particle_pC_list(M):
    pC_list = [learning_probability(i,M) for i in range(1,M+1)]


    rand_arrList = np.random.uniform(0,1,M)
    bool_arr = rand_arrList<pC_list
    indicies_false_list = np.where(bool_arr)[0]
    
    arr_pbest_particle = np.arange(1,M+1)
#    start_time = time.time()
    for particle in indicies_false_list:
#    for j in range(len(indicies_false_list)):
#        particle = indicies_false_list[j]
        pC = pC_list[particle]
        arr_pbest_particle[particle] = tournament_selection(particle+1, M, pC)

    return arr_pbest_particle      
#%%
# =============================================================================
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
# =============================================================================
#%%

def nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status):
    #avaialble_customer_list: all customers remaining to be inserted
    #new_route_status: if new route then must get start time
    available_customer_list_index = np.array(available_customer_list)[:]-1
#    available_customer_list_index = available_customer_list_index.astype(int)
#    if all(i<25 for i in available_customer_list_index)== False:
#        print('False')
#        print(available_customer_list_index)
#    
    if len(available_customer_list) == 0:
        arr_nn = np.array([])
    else:
    #get customer info from df_customers for the available list of customers
        arr_nn = copy.deepcopy(arr_customers)
#        print(available_customer_list_index)
        arr_nn = arr_nn[available_customer_list_index,:]#get subest of the customers index by indicie not by value
        
        #calculations:
        arr_nn[:,5] = arr_distance_matrix[curr_loc, available_customer_list] #distance curr next
        arr_nn[:,6] = arr_nn[:,5] + curr_time #arrival time
        arr_nn[:,7] = arr_nn[:,2] - arr_nn[:,6] 
        arr_nn[:,7] = arr_nn[:,7].clip(min=0) #waiting time
        arr_nn[:,8]= arr_nn[:,6] + arr_nn[:,7] #execStartTime
        arr_nn[:,9] = arr_nn[:,3] - arr_nn[:,8] #r_km
        arr_nn[:,10] = arr_nn[:,9] + arr_nn[:, 7] + arr_nn[:,5] #nn_km
        arr_nn[:,11] = arr_nn[:,8] +  + arr_nn[:,4] #new curr time
        arr_nn[:,12] = arr_nn[:,11] + arr_distance_matrix[available_customer_list, 0]#total time check
        arr_nn[:,1] = arr_nn[:,1] + curr_capacity#update current capacity 
        
        #keep only those that adhere to constraints
        arr_nn = arr_nn[(arr_nn[:,1]<=total_capacity) & (arr_nn[:,12]<=total_time) & (arr_nn[:,11]<=arr_nn[:,13]) & (arr_nn[:,8]>=arr_nn[:,2]) & (arr_nn[:,9]>0)]

#        
        #start only calculated if new route 
        if new_route_status == True:
            cust = arr_nn[:,0] -1
            arr_nn[:,14] = arr_nn[:,2] - arr_distance_matrix[0, cust.astype(int) ]
            arr_nn[:,14] = arr_nn[:,14].clip(min = 0)
#            arr_nn['startTime'] = arr_nn['readyTime'] - arr_nn['depot_dist']
#            arr_nn.loc[arr_nn['startTime']<0, 'startTime'] = 0
        
    return arr_nn

#%%
    
# =============================================================================
# POSITION UPDATING
#   1. Cut crisp set
#       For each particle's dimension, if the velocity of an arc is less than a random value keep, else discard arc        
#   2. Construct Routes based on Velocity, Position or Adjacent arcs
#       
# =============================================================================


def cut_velocity_set(pop_particle_velocity_list, num_customers_depot, M):
    #generate a random array list using uniform distribution
    random_array_list = np.random.uniform(0,1, (M, num_customers_depot, num_customers_depot))
    
    bool_arr = np.greater_equal(pop_particle_velocity_list, random_array_list) #if p(u,v)>=rand than keep p(u,v) else set to zero
    cut_V = np.multiply(pop_particle_velocity_list,bool_arr)

    return cut_V #returns particle wise cut velocity array for each particle
    
    
#%%

def hierachical_available_customers(arr_customers, arr_distance_matrix, available_customer_list, velocity_reference, position_reference, curr_loc, curr_time, curr_capacity, total_time, total_capacity, new_route_status, num_customers_depot):
    #VELOCITY: Sv, POSITION:Sx, AVAILABLE: Sa
    #lists of avaialble customers from velocity, position and adjacent arc matricies
    arr_nn_status = True
    all_customer_list = np.arange(1,num_customers_depot)#np.array(range(1,num_customers_depot))#list(g for g in range(1,len(df_customers)+1))
    used_customer_list = np.setdiff1d(all_customer_list, available_customer_list)#numpy array with all customers used
    
    
    Sv_list_original =  np.flatnonzero(velocity_reference[curr_loc,:])#returns an array of the row indicies where the column doesnt have zeros#[index for index, value in enumerate(velocity_reference[curr_loc,:].tolist()) if value != 0] #from current location any arc to another which doesn't have the probability of 0
    Sv_list = np.setdiff1d(Sv_list_original, used_customer_list)#list(set(Sv_list_original)-set(used_customer_list))
    Sv_list = np.setdiff1d(Sv_list, [0])#list(set(Sv_list)-set([0]))#removing arcs which add to the depot
    
    #nearest neighbour using Sv, Sx, Sa lists as the available customer lists
    arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, Sv_list, total_time, total_capacity, new_route_status)
    if arr_nn.size == 0:#if no velocity customers available try position set
        Sx_list_original =   np.flatnonzero(position_reference[curr_loc,:])#[index for index, value in enumerate(position_reference[curr_loc,:].tolist()) if value != 0]
        Sx_list = np.setdiff1d(Sx_list_original, used_customer_list)#list(set(Sx_list_original)-set(used_customer_list))
        Sx_list = np.setdiff1d(Sx_list, [0])#list(set(Sx_list)-set([0]))

        arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, Sx_list, total_time, total_capacity, new_route_status)
        if arr_nn.size == 0:
            arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
            if arr_nn.size == 0:
                arr_nn_status = False
                arr_nn = np.array([])#pd.DataFrame
                
    return arr_nn, arr_nn_status
    
#%%
# =============================================================================
# Update Route:
#    Take in updated velocity and constraints
#    For each particle get the velocity reference Sv = sum of cutV for a particular particle
#    For each particle get the position refenecer Sx = sum of position of a particular particle 
#    Maintain a record of avaialable customers 
#    Using hierachical arr_nn to obtain the next customer to visit
# =============================================================================
    

#%%
    

def update_route_position(num_cores,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot, pop_particle_velocity_list, pop_particle_position_list, M) :
    #Get cut velocity 
    pop_velocity_reference =  cut_velocity_set(pop_particle_velocity_list, num_customers_depot, M)
   

    inputs = range(M) 
    results = Parallel(n_jobs=num_cores)(delayed(update_route_position_updater)(pop_velocity_reference[i], pop_particle_position_list[i], arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in inputs)
    
    unzipped = zip(*results) #groups the same return for each particle and returns a list of tuples each of lenth M
    results_list = list(unzipped)
        
    return results_list
# route_routeList_list, particle_position, route_distance_list,  total_distance, num_vehicles

#%%
    
def update_route_position_updater(velocity_reference, position_reference, arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot):#, arr_neighbour_selection_col) :

    route_routeList_list = []
    route_distance_list = []
    particle_position = np.zeros([num_customers_depot, num_customers_depot])
    route_list = [0]
    times_list = []
    available_customer_list = list(range(1, num_customers_depot))#np.arange(1,num_customers_depot)#np.array(range(1,num_customers_depot))#df_customers.index.tolist()
    route = 0
    curr_time = 0
    curr_loc = 0
    curr_capacity = 0  
    new_route_status = True
    arr_nn, arr_nn_status = hierachical_available_customers(arr_customers, arr_distance_matrix, available_customer_list, velocity_reference, position_reference, curr_loc, curr_time, curr_capacity, total_time, total_capacity, new_route_status, num_customers_depot)
    nearest_neighbour_row =  np.argmin(arr_nn[:,10])#= arr_nn[arr_neighbour_selection_col].idxmin()
    nearest_neighbour = int(arr_nn[nearest_neighbour_row,0])
    route_distance = arr_nn[nearest_neighbour_row, 5]
    route_list.append(nearest_neighbour)
    particle_position[0, nearest_neighbour] = 1
    start_time = arr_nn[nearest_neighbour_row, 14]
    times_list.append(start_time)
    curr_loc = nearest_neighbour
    curr_time = arr_nn[nearest_neighbour_row, 11]#start_time + arr_nn.loc['dist', nearest_neighbour] + arr_nn.loc['serviceTime', nearest_neighbour]
    curr_capacity = arr_nn[nearest_neighbour_row,1]
    available_customer_list.remove(nearest_neighbour)#available_customer_list =np.setdiff1d(available_customer_list, [nearest_neighbour])#.remove(nearest_neighbour)
    
    
    while len(available_customer_list)!=0:
        new_route_status = False
        arr_nn, arr_nn_status = hierachical_available_customers(arr_customers, arr_distance_matrix, available_customer_list, velocity_reference, position_reference, curr_loc, curr_time, curr_capacity, total_time, total_capacity, new_route_status, num_customers_depot)
        if arr_nn_status!=False:
            #add customer 
            nearest_neighbour_row =  np.argmin(arr_nn[:,10])#= arr_nn[arr_neighbour_selection_col].idxmin()
            nearest_neighbour = int(arr_nn[nearest_neighbour_row,0])
            curr_time = arr_nn[nearest_neighbour_row, 11]
            route_distance = route_distance + arr_nn[nearest_neighbour_row, 5]
            available_customer_list.remove(nearest_neighbour)#available_customer_list =np.setdiff1d(available_customer_list, [nearest_neighbour])#.remove(nearest_neighbour)
            route_list.append(nearest_neighbour)
            particle_position[curr_loc, nearest_neighbour] = 1 #df_velocity.loc[curr_loc, temp_next_customer]+1/arr_nn.shape[1]#probability matrix
            curr_capacity = arr_nn[nearest_neighbour_row,1]
            curr_loc = nearest_neighbour
        else:
            route_list.append(0)
            particle_position[curr_loc, 0] = 1
            times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
            route_distance = route_distance + arr_distance_matrix[curr_loc,0]
            
            route_routeList_list.append(route_list) 
            route_distance_list.append(route_distance)
            
            
            #new route 
            route_list = [0]
            times_list = []
            route = route + 1
            curr_time = 0
            curr_loc = 0
            curr_capacity = 0
            new_route_status = True
            arr_nn, arr_nn_status = hierachical_available_customers(arr_customers, arr_distance_matrix, available_customer_list, velocity_reference, position_reference, curr_loc, curr_time, curr_capacity, total_time, total_capacity, new_route_status, num_customers_depot)
            nearest_neighbour_row =  np.argmin(arr_nn[:,10])#= arr_nn[arr_neighbour_selection_col].idxmin()
            nearest_neighbour = int(arr_nn[nearest_neighbour_row,0])
            route_distance = arr_nn[nearest_neighbour_row, 5]
            route_list.append(nearest_neighbour)
            particle_position[0, nearest_neighbour] = 1
            start_time = arr_nn[nearest_neighbour_row, 14]
            times_list.append(start_time)
            curr_loc = nearest_neighbour
            curr_time = arr_nn[nearest_neighbour_row, 11]#start_time + arr_nn.loc['dist', nearest_neighbour] + arr_nn.loc['serviceTime', nearest_neighbour]
            curr_capacity = arr_nn[nearest_neighbour_row,1]
            available_customer_list.remove(nearest_neighbour)#available_customer_list =np.setdiff1d(available_customer_list, [nearest_neighbour])#.remove(nearest_neighbour)
            
                
    #last route will need to be ended as can't enter while loop                
    route_list.append(0)
    particle_position[curr_loc, 0] = 1
    times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
    route_distance = route_distance + arr_distance_matrix[curr_loc, 0]

    route_routeList_list.append(route_list) 
    route_distance_list.append(route_distance)
    total_distance = sum(route_distance_list)
    num_vehicles = len(route_routeList_list)
    
    return route_routeList_list, particle_position, route_distance_list,  total_distance, num_vehicles


# =============================================================================
# CLPSO VELOCITY UPDATE
## =============================================================================
# VELOCITY UPDATE:
#   V = w*V + c*rand*(pBest-X) #NOTE: PBEST MUST BE POSITION PBEST [55]
#   Selection of pBest: 
#       For each particle i generate pC:
#           For each dimension d of i, generate a rand number:
#               If rand>pC:
#                   pBest of particle i is used
#               else:
#                   Tournament selection: we select two other particles, thefitter one's pBest is used                
# =============================================================================
# =============================================================================
#%%

def CLPSO_velocity_update(M,w,c,num_customers_depot, pop_particle_position_list, pbest_particle_position_list, pop_particle_velocity_list):  
    #COMPONENTS OF THE VELOCITY UPDATE
    #M randoms arrays, stacked velocity

    random_array_list = np.random.uniform(0,1,(M,num_customers_depot,num_customers_depot))
#    random_array_list = np.array(random_array_list)
    #omega*V
    wV = np.multiply(w,pop_particle_velocity_list)
    #c*RAnd
    cRand = np.multiply(c,random_array_list)
    
    #constant * position ....in this instance cRand is the constant we thus have to follow conditions in equation 21
    np.clip(cRand,None, 1, cRand)

    
    #pbest and position difference
    pbest_X_difference = np.subtract(pbest_particle_position_list, pop_particle_position_list)
    #sets all negative elemants to zero
    np.clip(pbest_X_difference, None, 1, pbest_X_difference)

    #component 2: cRand *pbest_X_difference
    comp2 = np.multiply(cRand,pbest_X_difference)
    #take the maximum between vW and Crand difference: max(p(u,v) from vW,p(u,v) from crand)
    pop_particle_velocity_list = np.maximum(wV, comp2)
    return pop_particle_velocity_list
#%%
#NOTE REMOVED THE TIME WINDOW CONSTRAINT
# =============================================================================
# LOCAL SEARCH:
#   Find the shortest route r
#       If all customers in r can be inserted into other vehicle routes, then accept update. (reducing number of vehicles)
#       Else r remains unchanged
# =============================================================================
#%%

#def local_search_particle_wise(particle,arr_customers,arr_distance_matrix, total_time, total_capacity, pop_distance_list, pop_num_route_list, pop_particle_routeList_list, pop_particle_velocity_list, pop_particle_position_list, pop_particle_capacity_list, pop_particle_time_list, pop_particle_distance_list, pop_particle_validity_list):
def local_search_particle_wise(df_customers,arr_distance_matrix, total_time, total_capacity,  distance,  num_routes,  particle_route_list,  particle_velocity_list,  particle_position_list,  particle_distance_list, arr_customer_info, service_time):
   
    update_status = False
    
    copy_particle_route_list = copy.deepcopy(particle_route_list)
    copy_particle_distance_list = copy.deepcopy(particle_distance_list)
    copy_particle_velocity_list = copy.deepcopy(particle_velocity_list)
    copy_particle_position_list = copy.deepcopy(particle_position_list)
    
    
    #get number of customers in each route
    num_customers = np.empty(num_routes)
    for i in range(num_routes):
        num_customers[i] = len(copy_particle_route_list[i])-2
    
    #get the shortest route and it's customers
    shortest_route_index = np.argmin(num_customers)
    r_customers = copy_particle_route_list[shortest_route_index][1:-1]
    df_r_customers = df_customers.loc[r_customers, :]
    
    #remove the shortest route from the route list/distance list / paosition / velocity
    del copy_particle_route_list[shortest_route_index]
    del copy_particle_distance_list[shortest_route_index]

    
    feasibility = True

    for i in range(len(r_customers)):
        customer = r_customers[i]
        inserted = False
        #for each available route check if can insert customer
        for vehicle_no in range(num_routes-1):# LOOP2
            vehicle_cust_list = copy_particle_route_list[vehicle_no][1:-1]
            df_vehicle_cust = pd.DataFrame(df_customers.loc[vehicle_cust_list,:])
            df_vehicle_cust['position'] = np.arange(len(vehicle_cust_list))
            feasible_cust_list = df_vehicle_cust.loc[(df_vehicle_cust['readyTime']>= df_r_customers.loc[customer,'readyTime']) & (df_vehicle_cust['readyTime']<=df_r_customers.loc[customer,'dueTime'])].index.tolist()#get the customer's index which have a ready time>=the insertable customer's start time and a start time which is less than the due time of the insertable customer
            num_feasible_vehicle_customers = len(feasible_cust_list)

    
    

            if num_feasible_vehicle_customers!=0:
            #for each feasible customer position
                for k in range(num_feasible_vehicle_customers +1):#LOOP1
                    if k<num_feasible_vehicle_customers:
                        position_to_insert = df_vehicle_cust.loc[feasible_cust_list[k], 'position']
                    else: 
                        position_to_insert = position_to_insert + 1
                    
                    vehicle_cust_list.insert(position_to_insert, customer)#insert at next customer
                    route_list = [0] + vehicle_cust_list + [0] #generate for new route
                    validity, curr_time, curr_cap, duration, temp_route_start_time, total_dist = validity_time(route_list, arr_distance_matrix, arr_customer_info,total_capacity, total_time, service_time)
                    
                    if validity == False:#remove from list 
                        vehicle_cust_list.remove(customer)
                    elif validity == True:
                        ##TIME WINDOW CONSTRAINT
                        #time window adherence
                        #UPDATE: velocity and position adjacency matrix
                        prev_customer = route_list[position_to_insert]
                        next_customer = route_list[position_to_insert+2]
                        
                        copy_particle_position_list[prev_customer, next_customer] = 0#reset position edge
                        copy_particle_position_list[prev_customer, customer] = 1#new position edge
                        copy_particle_position_list[customer, next_customer] = 1#new position edge
                        copy_particle_velocity_list[prev_customer, customer] = 1#np.random.uniform(0,1)#new velocity edge                            
                        copy_particle_velocity_list[customer, next_customer] = 1#np.random.uniform(0,1)#new velocity edge                            
                        
#                        r_customers.remove(customer)
                        inserted = True
                        
                        copy_particle_route_list[vehicle_no] = route_list.copy()
                        copy_particle_distance_list[vehicle_no]= total_dist
                    if inserted ==True:
                        break #dont keep searching the route
                    
            if inserted == True:
                break #stop searching routes to insert customer
                
        if inserted == False:
            feasibility = False
            break
    #UPDATE: IF ALL CUSTOMERS FEASIBLY INSERTED
    if feasibility == True and len(r_customers)==0:
        particle_route_list =  copy.deepcopy(copy_particle_route_list)
        particle_distance_list = copy.deepcopy(copy_particle_distance_list)
        particle_velocity_list = copy.deepcopy(copy_particle_velocity_list)
        particle_position_list = copy.deepcopy(copy_particle_position_list)
        distance = sum(particle_distance_list)
        num_routes = len(particle_route_list)
        update_status = True
        
        for i in range(len(r_customers)-1):
            particle_position_list[r_customers[i], r_customers[i+1]] = 0
            particle_velocity_list[r_customers[i], r_customers[i+1]] = 0
        particle_position_list[0, r_customers[0]] = 0
        particle_position_list[r_customers[-1],0] = 0
        particle_velocity_list[0, r_customers[0]] = 0
        particle_velocity_list[r_customers[-1],0] = 0


        
    return update_status, distance , num_routes, particle_route_list, particle_distance_list, particle_position_list, particle_velocity_list

#%%
    
def local_search(M, num_cores, df_customers,arr_distance_matrix, total_time, total_capacity, pop_distance_list, pop_num_route_list, pop_particle_routeList_list, pop_particle_distance_list, pop_particle_position_list, pop_particle_velocity_list, arr_customer_info, service_time):
    
    inputs = range(M)
    results = Parallel(n_jobs=num_cores)(delayed(local_search_particle_wise)(df_customers,arr_distance_matrix, total_time, total_capacity,  pop_distance_list[i],  pop_num_route_list[i],  pop_particle_routeList_list[i],  pop_particle_velocity_list[i],  pop_particle_position_list[i], pop_particle_distance_list[i], arr_customer_info, service_time)for i in inputs)

    
    unzipped = zip(*results) #groups the same return for each particle and returns a list of tuples each of lenth M
    local_search_results_list = list(unzipped)
    #update_status, total_distance, num_vehicles, particle_route_list, particle_distance_list, particle_position_list, particle_velocity_list

    
    return local_search_results_list
#%%
def PSO_result_updater(rg,PSO_resultlist, arr_results, arr_pbest_results,  flag, flagged_indicies, pop_particle_routeList_list, pop_particle_position_list, pop_particle_distance_list,  pop_distance_list, pop_num_route_list, pop_particle_velocity_list,  arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list,  gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list, pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list):
#   0route_routeList_list,
#    1particle_position
#    2route_distance_list
#    3total_distance
#    4num_vehicles
#    5particle_velocity
#    flagged_indicies = np.where(flag>=rg)[0]
#    flagged_indicies = flagged_indicies.astype(int)
    
    alpha = 100
    beta = 0.001
    
    flag[flagged_indicies] = 0
    for i in range(len(flagged_indicies)):
        flagged_index_val = flagged_indicies[i]
        arr_results[flagged_index_val,1] = PSO_resultlist[3][i]#distance
        arr_results[flagged_index_val, 0] = PSO_resultlist[4][i]#num vehicles
#        arr_results[flagged_index_val, 2] = np.arctan(PSO_resultlist[3][i])/(math.pi/2)#norm distance
#        arr_results[flagged_index_val,2] = (np.arctan(PSO_resultlist[3][i])/(math.pi/2)) + arr_results[i,0]#fitness        
        
        arr_results[flagged_index_val, 2] = beta*(PSO_resultlist[3][i]) + alpha*(arr_results[i,0])
        
        pop_particle_routeList_list[flagged_index_val] = PSO_resultlist[0][i]        
        pop_particle_position_list[flagged_index_val] = PSO_resultlist[1][i]
        pop_particle_distance_list[flagged_index_val] = PSO_resultlist[2][i]        
        pop_distance_list[flagged_index_val] = PSO_resultlist[3][i]
        pop_num_route_list[flagged_index_val] = PSO_resultlist[4][i]   
        pop_particle_velocity_list[flagged_index_val] = PSO_resultlist[5][i]
        
        if arr_results[flagged_index_val,2]<arr_pbest_results[flagged_index_val,2]:
            pbest_particle_position_list[flagged_index_val] = pop_particle_position_list[flagged_index_val]
            pbest_particle_distance_list[flagged_index_val] = pop_particle_distance_list[flagged_index_val]            
            pbest_distance_list[flagged_index_val] = pop_distance_list[flagged_index_val]
            pbest_num_route_list[flagged_index_val] = pop_num_route_list[flagged_index_val]
            pbest_particle_velocity_list[flagged_index_val] = pop_particle_velocity_list[flagged_index_val] 
            pbest_particle_routeList_list[flagged_index_val] = pop_particle_routeList_list[flagged_index_val]
            
            
            
# gbest update
    if np.min(arr_pbest_results[:,2])<arr_gbest_result[2]:
        gbest_particle_no = np.argmin(arr_pbest_results[:,2])#min fitness
        arr_gbest_result = arr_pbest_results[gbest_particle_no,:]
        gbest_velocity_list = pbest_particle_velocity_list[gbest_particle_no]
        gbest_position_list = pbest_particle_position_list[gbest_particle_no]
        gbest_routeList_list = pbest_particle_routeList_list[gbest_particle_no]
        gbest_distance_list = pbest_particle_distance_list[gbest_particle_no]
        gbest_tracker = 0 #rest
        
    else:
        gbest_tracker +=1

        
    return flag, arr_results, arr_pbest_results, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_distance_list,  pop_distance_list, pop_num_route_list, pop_particle_velocity_list, arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list, gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list

#%%

def CLPSO_result_updater(M,arr_results, arr_pbest_results, flag, pop_particle_velocity_list, position_resultslist, arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list, gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list):
#    0 route_routeList_list
#    1 particle_position
#    2 route_distance_list
#    3 total_distance
#    4 num_vehicles

    
    alpha = 100
    beta = 0.001
    #upddate results
    arr_results[:,1] = position_resultslist[3]#distance
    arr_results[:, 0] = position_resultslist[4]#num vehicles   
    arr_results[:,2] = beta*np.array(position_resultslist[3]) + alpha*np.array(position_resultslist[4])#fitness
    
    
    #update data
#    pop_particle_routeList_list = np.array(position_resultslist[0], object, ndmin = 0, order='F') 
    pop_particle_routeList_list = np.empty(M, object)
    pop_particle_routeList_list[:] = position_resultslist[0]
    pop_particle_position_list = np.array(position_resultslist[1]) 
    pop_particle_distance_list = np.empty(M, object)
    pop_particle_distance_list[:] = position_resultslist[2]
#    pop_particle_distance_list = np.array(position_resultslist[2], object, ndmin = 0, order = 'F') 
    pop_distance_list = np.array(position_resultslist[3])
    pop_num_route_list= np.array(position_resultslist[4]) 
    
#    print(pop_particle_routeList_list.shape)
    
    #update pbest
    update_pbest_indicies = np.where(arr_results[:,2]<arr_pbest_results[:,2])[0]
    flag = flag[:] + 1
    if update_pbest_indicies.size !=0:
        flag[update_pbest_indicies] = 0
        arr_pbest_results[update_pbest_indicies,:] = arr_results[update_pbest_indicies,:]
        if len(update_pbest_indicies)>1:
            pbest_particle_position_list[update_pbest_indicies] = pop_particle_position_list[update_pbest_indicies]
#            pbest_particle_distance_list[update_pbest_indicies,0] = pop_particle_distance_list[update_pbest_indicies,0]
            pbest_particle_distance_list[update_pbest_indicies] = pop_particle_distance_list[update_pbest_indicies]            
            pbest_distance_list[update_pbest_indicies] = pop_distance_list[update_pbest_indicies]
            pbest_num_route_list[update_pbest_indicies] = pop_num_route_list[update_pbest_indicies]
            pbest_particle_velocity_list[update_pbest_indicies] = pop_particle_velocity_list[update_pbest_indicies]
            pbest_particle_routeList_list[update_pbest_indicies] = pop_particle_routeList_list[update_pbest_indicies]

        else:
            update_pbest_index_val = update_pbest_indicies[0]
            pbest_particle_routeList_list[update_pbest_index_val] = pop_particle_routeList_list[update_pbest_index_val]
            pbest_particle_position_list[update_pbest_index_val] = pop_particle_position_list[update_pbest_index_val]
            pbest_particle_distance_list[update_pbest_index_val] = pop_particle_distance_list[update_pbest_index_val]            
            pbest_distance_list[update_pbest_index_val] = pop_distance_list[update_pbest_index_val]
            pbest_num_route_list[update_pbest_index_val] = pop_num_route_list[update_pbest_index_val]
            pbest_particle_velocity_list[update_pbest_index_val] = pop_particle_velocity_list[update_pbest_index_val] 
        
        
        
        if np.min(arr_pbest_results[:,2])<arr_gbest_result[2]:
            gbest_particle_no = np.argmin(arr_pbest_results[:,2])#min fitness
            arr_gbest_result = arr_pbest_results[gbest_particle_no,:]
            gbest_velocity_list = pbest_particle_velocity_list[gbest_particle_no]
            gbest_position_list = pbest_particle_position_list[gbest_particle_no]
            gbest_routeList_list = pbest_particle_routeList_list[gbest_particle_no]
            gbest_distance_list = pbest_particle_distance_list[gbest_particle_no]
            gbest_tracker = 0 #rest
            
        else:
            gbest_tracker +=1
    else:
        gbest_tracker +=1 #gbest not updated
        
    return flag, arr_results, arr_pbest_results, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_distance_list,  pop_distance_list, pop_num_route_list,  arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list,   gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list

#%%
    
def local_search_result_updater(local_search_results_list, flag, arr_results, arr_pbest_results, pop_particle_velocity_list, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_distance_list,  pop_distance_list, pop_num_route_list,  arr_gbest_result,  gbest_velocity_list, gbest_position_list, gbest_routeList_list,   gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list):
#   0 update_status
#    1 distance 
#    2 num_routes
#    3 particle_route_list
#    4 particle_distance_list
#    5 particle_position_list
#    6 particle_velocity_list
    
    alpha = 100
    beta = 0.001
    
    update_particle_indicies = np.where(local_search_results_list[0])[0]
    if update_particle_indicies.size !=0: #perform update
        
        pop_distance_list[update_particle_indicies] = local_search_results_list[1][update_particle_indicies]
        pop_num_route_list[update_particle_indicies] = local_search_results_list[2][update_particle_indicies]
        pop_particle_position_list[update_particle_indicies] = local_search_results_list[5][update_particle_indicies]
        pop_particle_velocity_list[update_particle_indicies] = local_search_results_list[6][update_particle_indicies]
        pop_particle_routeList_list[update_particle_indicies] = local_search_results_list[3][update_particle_indicies]
        pop_particle_distance_list[update_particle_indicies] = local_search_results_list[4][update_particle_indicies]
        
        
        #update results
        arr_results[update_particle_indicies, 1] = local_search_results_list[1][update_particle_indicies]
        arr_results[update_particle_indicies, 0] = local_search_results_list[2][update_particle_indicies]        
        arr_results[update_particle_indicies, 2] = beta*(arr_results[update_particle_indicies, 1])+ alpha*(arr_results[update_particle_indicies, 0])#fitness
        
        
        update_pbest_indicies = np.where(arr_results[:,2]<arr_pbest_results[:,2])[0]
#        flag = flag[:] + 1
        flag[update_pbest_indicies] = 0
        arr_pbest_results[update_pbest_indicies,:] = arr_results[update_pbest_indicies,:]
        pbest_particle_routeList_list[update_pbest_indicies] = pop_particle_routeList_list[update_pbest_indicies]
        pbest_particle_position_list[update_pbest_indicies] = pop_particle_position_list[update_pbest_indicies]
        pbest_particle_distance_list[update_pbest_indicies] = pop_particle_distance_list[update_pbest_indicies]
        pbest_distance_list[update_pbest_indicies] = pop_distance_list[update_pbest_indicies]
        pbest_num_route_list[update_pbest_indicies] = pop_num_route_list[update_pbest_indicies]
        pbest_particle_velocity_list[update_pbest_indicies] = pop_particle_velocity_list[update_pbest_indicies]

        if np.min(arr_pbest_results[:,2])<arr_gbest_result[2]:
            gbest_particle_no = np.argmin(arr_pbest_results[:,2])#min fitness
            arr_gbest_result = arr_pbest_results[gbest_particle_no,:]
            gbest_velocity_list = pbest_particle_velocity_list[gbest_particle_no]
            gbest_position_list = pbest_particle_position_list[gbest_particle_no]
            gbest_routeList_list = pbest_particle_routeList_list[gbest_particle_no]
            gbest_distance_list = pbest_particle_distance_list[gbest_particle_no]
            gbest_tracker = 0 #rest
            
        else:
            gbest_tracker +=1
            
    else:
        gbest_tracker +=1
            
        
        
    return flag, arr_results, arr_pbest_results, pop_particle_routeList_list, pop_particle_position_list,  pop_particle_velocity_list, pop_particle_distance_list,  pop_distance_list, pop_num_route_list,  arr_gbest_result, gbest_velocity_list, gbest_position_list, gbest_routeList_list,   gbest_distance_list, gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,  pbest_particle_distance_list,  pbest_distance_list, pbest_num_route_list, pbest_particle_velocity_list


#%%
def global_result_from_experiments(df_experiment_result_recorder, dict_routes, no_experiments):
    df_results = pd.DataFrame(np.zeros((1,5)))
    min_fitness_experiment = df_experiment_result_recorder[2].idxmin()
    df_results.iloc[0,0:4] = df_experiment_result_recorder.iloc[min_fitness_experiment, 0:4]
    df_results.iloc[0,4] = df_experiment_result_recorder[3].sum()/no_experiments
    dict_final_route = {}
    dict_final_route[0] = dict_routes[min_fitness_experiment]
    
    return df_results, dict_final_route
#%%
def PSO_particle(w, c , num_customers_depot, arr_distance_matrix, arr_customers,  total_time, total_capacity, gbest_position_list, pbest_position_list, position_reference, particle_velocity_list):
    #PSO VELOCITY UPDATE
    
    wV = np.multiply(w, particle_velocity_list)
    cRand1 = np.multiply(c,np.random.uniform(0,1,(num_customers_depot, num_customers_depot)))
    cRand2 = np.multiply(c,np.random.uniform(0,1,(num_customers_depot, num_customers_depot)))
    pbest_X_difference = np.subtract(pbest_position_list, position_reference)
    gbest_X_difference = np.subtract(gbest_position_list, position_reference)
    np.clip(cRand1, None, 1, cRand1)
    np.clip(cRand2, None, 1, cRand2)
    
    comp2 = np.multiply(cRand1, pbest_X_difference)
    comp3 = np.multiply(cRand2, gbest_X_difference)
    
    compA = np.maximum(wV,comp2)
    particle_velocity = np.maximum(compA, comp3)
    
    
    #PSO POSITION UPDATE
    #cut velocity
    rand = np.random.uniform(0,1, (num_customers_depot, num_customers_depot))
    bool_arr = np.greater(particle_velocity, rand)
    velocity_reference = np.multiply(particle_velocity, bool_arr)
    
    #CLPSO POSITION update
    route_routeList_list, particle_position, route_distance_list,  total_distance, num_vehicles = update_route_position_updater(velocity_reference, position_reference, arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)
      
    return route_routeList_list, particle_position, route_distance_list, total_distance, num_vehicles, particle_velocity
    
   
    
#%%
    
def PSO(flag, rg, w, c, num_cores, num_customers_depot, arr_distance_matrix, arr_customers,  total_time, total_capacity, gbest_position_list, pbest_particle_position_list, pop_particle_position_list, pop_particle_velocity_list):
    
    flagged_indicies = np.where(np.array(flag)>=rg)[0].astype(int)
    results = Parallel(n_jobs=num_cores)(delayed(PSO_particle)( w, c , num_customers_depot, arr_distance_matrix, arr_customers, total_time, total_capacity, gbest_position_list, pbest_position_list = pbest_particle_position_list[particle], position_reference = pop_particle_position_list[particle], particle_velocity_list = pop_particle_velocity_list[particle])for particle in flagged_indicies)

    unzipped = zip(*results) #groups the same return for each particle and returns a list of tuples each of lenth M
    PSO_resultlist = list(unzipped)
    
    return PSO_resultlist, flagged_indicies
 
