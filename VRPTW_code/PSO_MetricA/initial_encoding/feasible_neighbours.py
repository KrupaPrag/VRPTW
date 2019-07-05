#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 22:07:16 2018

@author: krupa
"""
import numpy as np
import copy
# =============================================================================
# Given Customer set, distance matrix, data information , initialised route, seq, velocity
# set of avaialble customers need to be updated, current time needs to be accounted for
# df_nn: RETURNS A DATAFRAME WITH THE CORRESPONDING DETAILS OF 
#                                                       FEASIBLE NEIGHBOURS (SUBSET OF THE AVAILABLE CUSTOMERS)
#                                                       ###################
# =============================================================================


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
    if all(i<25 for i in available_customer_list_index)== False:
        print('False')
        print(available_customer_list_index)
    
    if len(available_customer_list) == 0:
        arr_nn = np.array([])
    else:
    #get customer info from df_customers for the available list of customers
        arr_nn = copy.deepcopy(arr_customers)
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
#
#        
#    
#    
        #start only calculated if new route 
        if new_route_status == True:
            cust = arr_nn[:,0] -1
            arr_nn[:,14] = arr_nn[:,2] - arr_distance_matrix[0, cust.astype(int) ]
            arr_nn[:,14] = arr_nn[:,14].clip(min = 0)

    return arr_nn