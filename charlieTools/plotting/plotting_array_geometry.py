#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:13:38 2017
@author: hellerc
"""

# Plotting utilities for 64D Masmanidis array
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_from_mat(h_filename, ids_filename, lv):
    h = loadmat(h_filename)
    for key in h.keys():
        ht = h[key]
    h = ht
    cellids = loadmat(ids_filename)
    
    for key in cellids.keys():
        cellidst = cellids[key]
    cellids = cellidst

    plot_weights_64D(h[:,lv].squeeze(), cellids[0])

def plot_weights_64D(h, cellids, cbar=True):
    
    '''
    given a weight vector, h, plot the weights on the appropriate electrode channel
    mapped based on the cellids supplied. Weight vector must be sorted the same as
    cellids. Channels without weights will be plotted as empty dots
    '''
    

    if type(cellids) is not np.ndarray:
        cellids = np.array(cellids)
    
    if type(h) is not np.ndarray:
        h = np.array(h)
        vmin = np.min(h)
        vmax = np.max(h)
    else:
        vmin = np.min(h)
        vmax = np.max(h)
     # Make a vector for each column of electrodes
    
    # left column + right column are identical
    lr_col = np.arange(0,21*0.25,0.25)  # 25 micron vertical spacing
    left_ch_nums = np.arange(3,64,3)
    right_ch_nums = np.arange(4,65,3)
    center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)
    center_col = np.arange(-0.25,20.25*.25,0.25)-0.125
    ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
    sort_inds = np.argsort(ch_nums)
    
    
    
    l_col = np.vstack((np.ones(21)*-0.2,lr_col))
    r_col = np.vstack((np.ones(21)*0.2,lr_col))
    c_col = np.vstack((np.zeros(22),center_col))
    locations = np.hstack((l_col,c_col,r_col))[:,sort_inds]
    plt.figure()
    plt.scatter(locations[0,:],locations[1,:],facecolor='none',edgecolor='k',s=50)
    
    
   
    # Now, color appropriately
    electrodes = np.zeros(len(cellids))
    c_id = np.zeros(len(cellids))
    for i in range(0, len(cellids)):
        electrodes[i] = int(cellids[i][-4:-2])
        
    # Add locations for cases where two or greater units on an electrode
    electrodes=list(electrodes-1)  # cellids labeled 1-64, python counts 0-63
    dupes = list(set([int(x) for x in electrodes if electrodes.count(x)>1]))
    print('electrodes with duplicates:')
    print([d+1 for d in dupes])

    num_of_dupes = [electrodes.count(x) for x in electrodes]
    num_of_dupes = list(set([x for x in num_of_dupes if x>1]))
    #max_duplicates = np.max(np.array(num_of_dupes))
    dup_locations=np.empty((2,2*np.sum(num_of_dupes)))
    max_=0
    for i in np.arange(0,len(dupes)):
        loc_x = locations[0,dupes[i]]
        loc_y = locations[1,dupes[i]]
         
        dup_locations[0,i]= loc_x
        dup_locations[1,i]= loc_y
        
        n_dupes = electrodes.count(dupes[i])-1
        shift = 0
        for d in range(0,n_dupes):
            if loc_x < 0:
                shift -= 0.2
            elif loc_x == 0:
                shift += 0.4
            elif loc_x > 0:
                shift += 0.2
            m = shift
            
            if m > max_:
                max_=m
                
            dup_locations[0,i+1+d]= loc_x+shift
            dup_locations[1,i+1+d]= loc_y
    
    plt.scatter(dup_locations[0,:],dup_locations[1,:],facecolor='none',edgecolor='k',s=50)

    plt.axis('scaled')
    plt.xlim(-max_-.1,max_+.1)

    c_id = np.sort([int(x) for x in electrodes if electrodes.count(x)==1])
    electrodes = [int(x) for x in electrodes]

    # find the indexes of the unique cellids
    indexes=[]
    for c in c_id:
        indexes.append(np.argwhere(electrodes==c))
    indexes=[x[0][0] for x in list(indexes)]
    
    # make an inverse mask of the unique indexes
    mask = np.ones(len(h),np.bool)
    mask[indexes]=0


    # plot the unique ones
    import matplotlib
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h[indexes])
    #mappable.set_cmap('jet')
    colors = mappable.to_rgba(h[indexes])
    plt.scatter(locations[:,c_id][0,:],locations[:,c_id][1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    # plot the duplicates
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h[mask])
    #mappable.set_cmap('jet')
    colors = mappable.to_rgba(h[mask])
    plt.scatter(dup_locations[0,:],dup_locations[1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    if cbar is True:
        plt.colorbar(mappable)
        
    
    
        
# plotting utils fro 128ch 4-shank depth

def plot_weights_128D(h, cellids,vmin,vmax,cbar):
    # get gemoetry from Luke's baphy function probe_128D
    
    channels = np.arange(0,128,1)
    #x = loadmat('/auto/users/hellerc/nems/nems/utilities/probe_128D/x_positions.mat')['x_128']
    #y = loadmat('/auto/users/hellerc/nems/nems/utilities/probe_128D/z_positions.mat')['z_128']
    x = loadmat('/auto/users/hellerc/x_pos.mat')['x_pos']
    y = loadmat('/auto/users/hellerc/y_pos.mat')['y_pos']
    #ch_map = loadmat('/auto/users/hellerc/chan_map_128D.mat')['ch_map_128D']
    #ch_map = np.array([int(x) for x in ch_map])
    # scale to get more separtion visuallyu
    x = (x/100)*4
    y = (y/100)*2
    
    locations=np.hstack((x,y))  
    #locations=locations[np.argsort(ch_map),:]
    plt.scatter(locations[:,0],locations[:,1], s=40,facecolor='none',edgecolor='k')
    plt.axis('scaled')


    import matplotlib
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h)
    #mappable.set_cmap('jet')
    colors = mappable.to_rgba(h)
    
    # Now, color appropriately
    electrodes = np.zeros(len(cellids))
    c_id = np.zeros(len(cellids))
    for i in range(0, len(cellids)):
        ind= cellids[i][0].split('-')[1]
        electrodes[i] = int(ind)
    electrodes = np.unique(electrodes)-1
    
       # move units when there are >1 on same electrode
    for i, weight in enumerate(h):
        c_id[i] = int(cellids[i][0].split('-')[1])
        tf =1
        while tf==1:
             if (int(cellids[i][0][-1])>1 and (int(c_id[i]+1) != 32 and int(c_id[i]+1)!=64 and int(c_id[i]+1)!=96 and int(c_id[i]+1)!=128)):
                 c_id[i] = int(c_id[i]+1)
                 if sum(c_id[i] == electrodes)>0:
                     tf=1
                 else:
                     tf = 0
             elif (int(cellids[i][0][-1])>1 and (int(c_id[i]+1) == 32 and int(c_id[i]+1)==64 and int(c_id[i]+1)==96 and int(c_id[i]+1)==128)):
                 c_id[i] = int(c_id[i]-1)
                 if sum(c_id[i] == electrodes)>0:
                     tf=1
                 else:
                     tf = 0
             else:
                 tf = 0
                 
    plt.scatter(locations[(c_id.astype(int)),:][:,0],locations[c_id.astype(int),:][:,1],
                          c=colors,vmin=vmin,vmax=vmax,s=40,edgecolor='none')
    if cbar is True:
        plt.colorbar(mappable)