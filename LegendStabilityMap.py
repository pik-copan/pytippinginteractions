# -*- coding: utf-8 -*-

###################################
# LegendStabilityMap.py
###################################

# creates legends for stability maps  
# for manuscript: 
# "Emergence of cascading dynamics in interacting tipping elements of ecology and climate"

############################################################

# import packages 
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

###############################################################################


# choose M depending on range of legend in terms of number of stable equilibria 
# M = 4                   # number of stable equilibria ranging from 0 to 4
# M = 14                 # number of stable equilibria ranging from 1 to 4
M = 18                 # number of stable equilibria ranging from 1 to 8



###############################################################################

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1)
nr_stable_all = np.zeros((1,1)) 



if M == 4: 
    cMap = ListedColormap(['lightslategrey','lightgray','silver', 'darkgray', 'dimgray' ])
    plt.imshow(nr_stable_all, cmap = cMap, aspect='auto')        
    plt.xticks(())
    plt.yticks(())       
    for spine in plt.gca().spines.values():
       spine.set_visible(False)
    cbar = plt.colorbar(ticks=np.array([0,1,2,3,4]), orientation='horizontal', aspect = 5)
    cbar.ax.tick_params(size = 0, labelsize=20, pad = -40)
    cbar.set_label(label='number of stable fixed points', fontsize = 15, labelpad = 10, weight = 'bold')
    plt.clim(-0.5, 4.5)
       
if M == 14: 
    cMap = ListedColormap(['lightgray','silver', 'darkgray', 'dimgray' ])
    plt.imshow(nr_stable_all, cmap = cMap, aspect='auto')        
    plt.xticks(())
    plt.yticks(())       
    for spine in plt.gca().spines.values():
       spine.set_visible(False)
    cbar = plt.colorbar(ticks=np.array([1,2,3,4]), orientation='horizontal', aspect = 5)
    cbar.ax.tick_params(size = 0, labelsize=20, pad = -40)
    cbar.set_label(label='number of stable fixed points', fontsize = 15, labelpad = 10, weight = 'bold')
    plt.clim(0.5, 4.5)
    
if M == 18: 
    cMap = ListedColormap(['lightgray','silver', 'darkgray', 'dimgray','lightgreen','skyblue','thistle','firebrick'])
    plt.imshow(nr_stable_all, cmap = cMap, aspect='auto')        
    plt.xticks(())
    plt.yticks(())       
    for spine in plt.gca().spines.values():
       spine.set_visible(False)
    
    # cbar = plt.colorbar(ticks=range(9), orientation='horizontal')

    cbar = plt.colorbar(ticks=np.array([1,2,3,4,5,6,7,8]), orientation='horizontal', aspect = 7)
    
    cbar.ax.tick_params(size = 0, labelsize=20, pad = -40)
    cbar.set_label(label='number of stable fixed points', fontsize = 15, labelpad = 10, weight = 'bold')
    plt.clim(0.5, 8.5)
