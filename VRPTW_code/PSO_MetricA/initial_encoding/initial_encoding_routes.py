#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 20:55:21 2018

@author: krupa
"""

import numpy as np
import random
from validity import validity_time
from feasible_neighbours import nn
#%%
# =============================================================================
# Generate a nearest neighbour population
#       nearest neighbour selected according to paper [55]
#       1. keep selecting the nn until constraints violated, then start new route....repeat till all customers accounted for
# =============================================================================
#set initial particle velocity to 1 not uniform distribution 
#nearest neighbour heuristic: generate a single particle

#%%

# =============================================================================
# Generate a random neighbour route for a particle
#    randomly select a neighbpir from a set of feasible neighbours
# =============================================================================
#nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)

def random_neighbour_heauristic(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot) :

    #route storage
    route_routeList_list = []
    route_distance_list = []
    
    particle_velocity = np.zeros([num_customers_depot, num_customers_depot])#initialise a array
    particle_position = np.zeros([num_customers_depot, num_customers_depot])
    route_list = [0]
    times_list = []
    available_customer_list = np.arange(1,num_customers_depot).tolist()#df_customers.index.tolist()
    route = 1
    curr_time = 0
    curr_loc = 0
    curr_capacity = 0  
    new_route_status = True
    arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
    temp_next_customer = int(random.choice(arr_nn[:,0].tolist()))
    temp_next_customer_index = np.where(arr_nn[:,0] == temp_next_customer)[0][0]
    route_list.append(temp_next_customer)
    particle_velocity[0,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
    particle_position[0, temp_next_customer] = 1
    curr_loc = temp_next_customer
    curr_time = arr_nn[temp_next_customer_index, 11]#start_time + df_nn.loc['dist', nearest_neighbour] + df_nn.loc['serviceTime', nearest_neighbour]
    curr_capacity += arr_nn[temp_next_customer_index,1]
    available_customer_list.remove(temp_next_customer)
    new_route_status = False
    
    
    while len(available_customer_list)!=0:
        arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
        if arr_nn.shape[0]!=0:
            #add customer 
            temp_next_customer = int(random.choice(arr_nn[:,0].tolist()))
            temp_next_customer_index = np.where(arr_nn[:,0] == temp_next_customer)[0][0]
            curr_time = arr_nn[temp_next_customer_index, 11]
            available_customer_list.remove(temp_next_customer)#remove customer from av list such that not included in df_next_nn
            route_list.append(temp_next_customer)
            particle_velocity[curr_loc,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
            particle_position[curr_loc, temp_next_customer] = 1 #df_velocity.loc[curr_loc, temp_next_customer]+1/df_nn.shape[1]#probability matrix
            curr_capacity += arr_nn[temp_next_customer_index,1]
            curr_loc = temp_next_customer
        else:
            route_list.append(0)
            particle_velocity[curr_loc, 0] = 1#random.uniform(0,1)#df_velocity.loc[curr_loc, 0]+1 #pobability matrix
            particle_position[curr_loc, 0] = 1
            times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
            validity, curr_time, curr_cap, duration, start_time, total_dist = validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time)
            route_routeList_list.append(route_list) 
            route_distance_list.append(total_dist)
           
            
            #new route 
            route_list = [0]
            times_list = []
            route = route + 1
            curr_time = 0
            curr_loc = 0
            curr_capacity = 0
            new_route_status = True
            arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
            temp_next_customer = int(random.choice(arr_nn[:,0].tolist()))
            temp_next_customer_index = np.where(arr_nn[:,0] == temp_next_customer)[0][0]
            route_list.append(temp_next_customer)
            particle_velocity[0,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
            particle_position[0, temp_next_customer] = 1
            curr_loc = temp_next_customer
            curr_time = arr_nn[temp_next_customer_index, 11]#df_nn.loc[temp_next_customer, 'new_curr_time']#start_time + df_nn.loc['dist', nearest_neighbour] + df_nn.loc['serviceTime', nearest_neighbour]
            curr_capacity += arr_nn[temp_next_customer_index,1]
            available_customer_list.remove(temp_next_customer)
                
    #last route will need to be ended as can't enter while loop                
    route_list.append(0)
    particle_velocity[curr_loc, 0] = 1#random.uniform(0,1)#df_velocity.loc[curr_loc, 0]+1 #probability matrix
    particle_position[curr_loc, 0] = 1
    times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
    validity, curr_time, curr_cap, duration, start_time, total_dist = validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time)
                
    route_routeList_list.append(route_list) 
    route_distance_list.append(total_dist)
    total_distance = sum(route_distance_list)
    
    return route_routeList_list, route_distance_list, particle_position, particle_velocity, total_distance, route


#%%
    

def nearest_neighbour_heauristic(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot) :

    #route storage
    route_routeList_list = []
    route_distance_list = []
    
    particle_velocity = np.zeros([num_customers_depot, num_customers_depot])#initialise a array
    particle_position = np.zeros([num_customers_depot, num_customers_depot])
    route_list = [0]
    times_list = []
    available_customer_list = np.arange(1,num_customers_depot).tolist()#df_customers.index.tolist()
    route = 1
    curr_time = 0
    curr_loc = 0
    curr_capacity = 0  
    new_route_status = True
    arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
    temp_next_customer = int(random.choice(arr_nn[:,0].tolist()))
    temp_next_customer_index = np.where(arr_nn[:,0] == temp_next_customer)[0][0]
    route_list.append(temp_next_customer)
    particle_velocity[0,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
    particle_position[0, temp_next_customer] = 1
    curr_loc = temp_next_customer
    curr_time = arr_nn[temp_next_customer_index, 11]#start_time + df_nn.loc['dist', nearest_neighbour] + df_nn.loc['serviceTime', nearest_neighbour]
    curr_capacity += arr_nn[temp_next_customer_index,1]
    available_customer_list.remove(temp_next_customer)
    new_route_status = False
    
    
    while len(available_customer_list)!=0:
        arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
        if arr_nn.shape[0]!=0:
            #add customer 
            temp_next_customer_index = np.argmin(arr_nn[:,10])
            temp_next_customer = int(arr_nn[temp_next_customer_index,0])  
            curr_time = arr_nn[temp_next_customer_index, 11]
            available_customer_list.remove(temp_next_customer)#remove customer from av list such that not included in df_next_nn
            route_list.append(temp_next_customer)
            particle_velocity[curr_loc,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
            particle_position[curr_loc, temp_next_customer] = 1 #df_velocity.loc[curr_loc, temp_next_customer]+1/df_nn.shape[1]#probability matrix
            curr_capacity += arr_nn[temp_next_customer_index,1]
            curr_loc = temp_next_customer
        else:
            route_list.append(0)
            particle_velocity[curr_loc, 0] = 1#random.uniform(0,1)#df_velocity.loc[curr_loc, 0]+1 #pobability matrix
            particle_position[curr_loc, 0] = 1
            times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
            validity, curr_time, curr_cap, duration, start_time, total_dist = validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time)
            route_routeList_list.append(route_list) 
            route_distance_list.append(total_dist)
           
            
            #new route 
            route_list = [0]
            times_list = []
            route = route + 1
            curr_time = 0
            curr_loc = 0
            curr_capacity = 0
            new_route_status = True
            arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
            temp_next_customer = int(random.choice(arr_nn[:,0].tolist()))
            temp_next_customer_index = np.where(arr_nn[:,0] == temp_next_customer)[0][0] 
            route_list.append(temp_next_customer)
            particle_velocity[0,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
            particle_position[0, temp_next_customer] = 1
            curr_loc = temp_next_customer
            curr_time = arr_nn[temp_next_customer_index, 11]#df_nn.loc[temp_next_customer, 'new_curr_time']#start_time + df_nn.loc['dist', nearest_neighbour] + df_nn.loc['serviceTime', nearest_neighbour]
            curr_capacity += arr_nn[temp_next_customer_index,1]
            available_customer_list.remove(temp_next_customer)
                
    #last route will need to be ended as can't enter while loop                
    route_list.append(0)
    particle_velocity[curr_loc, 0] = 1#random.uniform(0,1)#df_velocity.loc[curr_loc, 0]+1 #probability matrix
    particle_position[curr_loc, 0] = 1
    times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
    validity, curr_time, curr_cap, duration, start_time, total_dist = validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time)
                
    route_routeList_list.append(route_list) 
    route_distance_list.append(total_dist)
    total_distance = sum(route_distance_list)
    
    return route_routeList_list, route_distance_list, particle_position, particle_velocity, total_distance, route
#%%
    
def greedy_nearest_neighbour_heauristic(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot) :

    #route storage
    route_routeList_list = []
    route_distance_list = []
    
    particle_velocity = np.zeros([num_customers_depot, num_customers_depot])#initialise a array
    particle_position = np.zeros([num_customers_depot, num_customers_depot])
    route_list = [0]
    times_list = []
    available_customer_list = np.arange(1,num_customers_depot).tolist()#df_customers.index.tolist()
    route = 1
    curr_time = 0
    curr_loc = 0
    curr_capacity = 0  
    new_route_status = True
    arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
    temp_next_customer_index = np.argmin(arr_nn[:,10])
    temp_next_customer = int(arr_nn[temp_next_customer_index,0])    
    route_list.append(temp_next_customer)
    particle_velocity[0,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
    particle_position[0, temp_next_customer] = 1
    curr_loc = temp_next_customer
    curr_time = arr_nn[temp_next_customer_index, 11]#start_time + df_nn.loc['dist', nearest_neighbour] + df_nn.loc['serviceTime', nearest_neighbour]
    curr_capacity += arr_nn[temp_next_customer_index,1]
    available_customer_list.remove(temp_next_customer)
    new_route_status = False
    
    
    while len(available_customer_list)!=0:
        arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
        if arr_nn.shape[0]!=0:
            #add customer 
            temp_next_customer_index = np.argmin(arr_nn[:,10])
            temp_next_customer = int(arr_nn[temp_next_customer_index,0])  
            curr_time = arr_nn[temp_next_customer_index, 11]
            available_customer_list.remove(temp_next_customer)#remove customer from av list such that not included in df_next_nn
            route_list.append(temp_next_customer)
            particle_velocity[curr_loc,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
            particle_position[curr_loc, temp_next_customer] = 1 #df_velocity.loc[curr_loc, temp_next_customer]+1/df_nn.shape[1]#probability matrix
            curr_capacity += arr_nn[temp_next_customer_index,1]
            curr_loc = temp_next_customer
        else:
            route_list.append(0)
            particle_velocity[curr_loc, 0] = 1#random.uniform(0,1)#df_velocity.loc[curr_loc, 0]+1 #pobability matrix
            particle_position[curr_loc, 0] = 1
            times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
            validity, curr_time, curr_cap, duration, start_time, total_dist = validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time)
            route_routeList_list.append(route_list) 
            route_distance_list.append(total_dist)
           
            
            #new route 
            route_list = [0]
            times_list = []
            route = route + 1
            curr_time = 0
            curr_loc = 0
            curr_capacity = 0
            new_route_status = True
            arr_nn = nn(curr_loc, curr_time, curr_capacity, arr_customers, arr_distance_matrix, available_customer_list, total_time, total_capacity, new_route_status)
            temp_next_customer_index = np.argmin(arr_nn[:,10])
            temp_next_customer = int(arr_nn[temp_next_customer_index,0])  
            route_list.append(temp_next_customer)
            particle_velocity[0,temp_next_customer] = 1#random.uniform(0,1)#random uniform probability#particle_velocity[0,nearest_neighbour]+1#velocity matrix#ADD CURR VELOCITY TO IT
            particle_position[0, temp_next_customer] = 1
            curr_loc = temp_next_customer
            curr_time = arr_nn[temp_next_customer_index, 11]#df_nn.loc[temp_next_customer, 'new_curr_time']#start_time + df_nn.loc['dist', nearest_neighbour] + df_nn.loc['serviceTime', nearest_neighbour]
            curr_capacity += arr_nn[temp_next_customer_index,1]
            available_customer_list.remove(temp_next_customer)
                
    #last route will need to be ended as can't enter while loop                
    route_list.append(0)
    particle_velocity[curr_loc, 0] = 1#random.uniform(0,1)#df_velocity.loc[curr_loc, 0]+1 #probability matrix
    particle_position[curr_loc, 0] = 1
    times_list.append(curr_time + arr_distance_matrix[curr_loc,0])
    validity, curr_time, curr_cap, duration, start_time, total_dist = validity_time(route_list, arr_distance_matrix, df_customers,total_capacity, total_time)
                
    route_routeList_list.append(route_list) 
    route_distance_list.append(total_dist)
    total_distance = sum(route_distance_list)
    
    return route_routeList_list, route_distance_list, particle_position, particle_velocity, total_distance, route

#%%%