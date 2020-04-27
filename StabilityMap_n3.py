# -*- coding: utf-8 -*-

##################################
# StabilityMap_n3.py
##################################


# analysis of three unidirectionally coupled tipping elements 
# for manuscript: 
# "Emergence of cascading dynamics in interacting tipping elements of ecology and climate"

# three undirectionally coupled tipping elements given by 
# subsystem 0
# dx0/dt = a_0*x0 - b_0*x0^3 + c_0
# subsystem 1
# dx1/dt = a_1*x1 - b_1*x1^3 + c_1 + d_1*x0
# subsystem 2
# dx2/dt = a_2*x2 - b_2*x2^3 + c_2 + d_2*x1

# computing a (2d) matrix of stability maps showing the number of stable fixed points
# depending on the control parameters of subsystem 1 and 2 
# given a fixed value for the control patameter of subsystem 0
# the position of the stability map withing the matrix is determined by the coupling strength of the tipping elements

# script generates elements of Figure 7 in manuscript and Figure 2/3 in Supplementary Material 
# for parameter settings indicated below 


###############################################################################


# Import packages 
import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.colors import ListedColormap
from numpy import roots


# definition of coefficients 
a_0 = 1.0
a_1 = 1.0
a_2 =1.0                  
b_0 =1.0
b_1 = 1.0
b_2 = 1.0

# calulcation of intrinsic tipping points 
c_1_crit = 2*np.sqrt((a_1/3)**3/(b_1))
c_2_crit = 2*np.sqrt((a_2/3)**3/(b_2))

# control parameter arrays 
anz = 500      # 500 chosen for single stability map, 100 chosen for matrix of stability maps                             
value_c1 = np.linspace(0.0, 0.8,anz)          
value_c2 = np.linspace(0.8,0.0,anz) 

# coupling stength arrays for subsystem 1 and 2        
value_d1 = np.array([0.2])              # used: 
                # for matrix of stability maps (Figure 7 in manuscript, Figure 2 in Supplementary Material): 0.0,0.2,0.3,0.5,0.7,0.9
                # for Figure 3 in Supplementary Material: 0.2
value_d2 = np.array([0.2])              # used: 
                # for matrix of stability maps (Figure 7 in manuscript, Figure 2 in Supplementary Material): 0.9,0.7,0.5,0.3,0.2,0.0
                # For Figure 3 in Supplementary Material: 0.2

# fixed control parameter of subsystem 0
c_0 = 0.4               # used: 
                        # for matrix of stability maps (Figure 7 in manuscript): 0.4
                        # for matrix of stability maps (Figure 2 in Supplementary Material): 0.2
                        # for Figure 3 in Supplementary material: 0.4


###############################################################################
# definition of functions 
###############################################################################

                  
def roots_(*params):
    return roots(list(params)) 

roots3 = np.vectorize(roots_, signature = "(),(),(),()->(n)",otypes=[complex])
roots9 = np.vectorize(roots_, signature = "(),(),(),(),(),(),(),(),(),()->(n)", otypes=[complex])
                                  
# find equilibria via roots
def find_roots(a0 = 1,b0 = 1 , c0 = 0 , a1 = 1, b1 = 1,c1 = 0,d1 = 0, a2 = 1, b2 = 1, c2 = 0, d2 = 0):
    
    x0_roots_ = roots([-b0,0,a0,c0])
    x0 = []
    x1 = []
    x2_f = []
    x1_f = []
    x0_f = []
    for x0_root in x0_roots_:
         x1 += [roots([-b1,0,a1,c1 + d1 * x0_root])]
         x0 += [x0_root]*3
         
    x0e = np.round(np.array(x0).flatten(),decimals=5)
    x1e = np.round(np.array(x1).flatten(),decimals=5)
    for i in range(0,len(x1e)): 
        x2_f += [roots([-b2,0,a2,c2 + d2 * x1e[i]])]
        x1_f += [x1e[i]]*3
        x0_f += [x0e[i]]*3
    
    return (np.round(np.array(x0_f).flatten(),decimals=5),np.round(np.array(x1_f).flatten(),decimals=5),np.round(np.array(x2_f).flatten(),decimals=5))

# determine stability via eigenvalues of Jacobian 
def eigenvalues(a,b,c):
            
    a00 = a_0 - 3.0*b_0*(a**2)                        
    a01 = 0                                           
    a02 = 0                                           
    a10 = d_1                                         
    a11 = a_1 - 3.0*b_1*(b**2)                        
    a12 = 0                                          
    a20 = 0                                           
    a21 = d_2                                         
    a22 = a_2 - 3.0*b_2*(c**2)                        

    A = np.array([[a00,a01,a02],[a10,a11,a12], [a20,a21,a22]])  
    eig = LA.eigvals(A)                                         
                                                                
    lambda1 = eig[0]                                           
    lambda2 = eig[1]                                            
    lambda3 = eig[2]                                           
    
    return lambda1, lambda2, lambda3                             

       
             
################################################################################
################################################################################


# definitions for axis ticks 
major_xticks = np.linspace(value_c1[0], value_c1[anz-1],5)          
major_yticks = np.linspace(value_c2[anz-1], value_c2[0],5)          
ya = 0

# initialization of array nr_stable_all for number of stable fixed points
nr_stable_all = np.zeros((value_c2.shape[0],value_c1.shape[0]))     
   
# initialization of counters                                                                  
count_c1 = -1                                                        
count_c2= -1                                                        
count_loop = 0                                                     

# open figure
fig = plt.figure(figsize = (10,10))                                  


###############################################################################
# determine number of stable equilibria 
###############################################################################


# loop over coupling strength d2
for d_2 in value_d2:                                                
    
    # loop over coupling strength d1
    for d_1 in value_d1: 
        
        count_loop = count_loop+ 1                                 
        count_c1 = -1
        
        # loop over control parameter c1
        for c_1 in value_c1:                                       
            
            count_c1 = count_c1+1
            count_c2 = -1                                            
            
            # loop over control parameter c2
            for c_2 in value_c2:                                   
                
                count_c2 = count_c2+1               
                
                params = {"c0" : c_0, "c1" : c_1, "c2": c_2,"d1" : d_1,"d2" : d_2}

                # find equilibira for given combination of control parameters 
                x0,x1,x2 = find_roots(**params)  
                # determine stability 
                l0 = np.zeros(len(x0),dtype = np.complex) +0j
                l1 = np.zeros(len(x0),dtype = np.complex) +0j
                l2 = np.zeros(len(x0),dtype = np.complex) +0j
                for i in range(0,len(x0)): 
                    l0[i],l1[i],l2[i] = eigenvalues(x0[i],x1[i],x2[i])
    
                # find real FP
                such_real = np.logical_and(np.logical_and(np.isreal(x0),np.isreal(x1)),np.isreal(x2))
                # find stable FP 
                such_l = np.logical_and(np.logical_and(np.real(l0[such_real])<0,np.real(l1[such_real])<0),np.real(l2[such_real])<0)
                x0_stab = np.array(x0[such_real][such_l])
                x1_stab = np.array(x1[such_real][such_l])  
                x2_stab = np.array(x2[such_real][such_l]) 
                
                # count number of stable FP and save them in result array 
                nr_stable_all[count_c2,count_c1] = len(x0_stab)

                
                

#################################################################
# Plotten 
#################################################################    

        # subplots   
        ax = fig.add_subplot(value_d2.shape[0],value_d1.shape[0],count_loop)     
        
        
        # definition of colormap 
        cmap_s = np.array(['lightslategrey','lightgray','silver', 'darkgray', 'dimgray','lightgreen','skyblue','thistle','firebrick'])
        # choose color range depending on range of number of stable equilibria 
        crange = np.arange(np.min(nr_stable_all),np.max(nr_stable_all)+1,1)
        cMap = ListedColormap(cmap_s[crange.astype(int)])
        
        # plot result array nr_stable_all       
        plt.imshow(nr_stable_all, interpolation='nearest', cmap = cMap, extent = [value_c1[0],value_c1[anz-1],value_c2[anz-1],value_c2[0]], aspect='auto')  # Plotten von nr_stable_all     
        
        # add intrinsic tipping points
        plt.plot(np.zeros(len(value_c2))+c_2_crit,value_c2,'--', color = 'black',linewidth = 1)  
        plt.plot(value_c2,np.zeros(len(value_c2))+c_2_crit,'--', color = 'black',linewidth = 1)  
       

        # axis labels 
        plt.xticks(())
        plt.yticks(())
    
        if count_loop > ((value_d2.shape[0] * value_d1.shape[0])-value_d1.shape[0]):
           plt.xticks(major_xticks, fontsize = 15)  # 15
           plt.xlabel(r"$c_2$" "\n" "\n" r"$d_{12} = %s$"%d_1 , fontsize = 15 ) 
        if count_loop == (1+(ya*value_d1.shape[0])):
           plt.yticks(major_yticks, fontsize = 15)                         
           plt.ylabel(r"$d_{23} = %s$" "\n" r"$c_3$" %d_2, fontsize =15) 
           ya = ya + 1 

      
        plt.gca().set_aspect('equal', adjustable='box')
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
            



        # add arrows (for some specific plots)
        # ax.annotate("", xy = (0.25, 0.25), xytext = (0.11, 0.11), arrowprops=dict(color = 'deeppink', width = 10, headwidth = 20, headlength = 20) )
        # ax.annotate("", xy = (0.30, 0.11), xytext = (0.11, 0.11), arrowprops=dict(color = 'yellow', width = 10, headwidth = 20, headlength = 20) )
        # ax.annotate("", xy = (0.11, 0.301), xytext = (0.11, 0.11), arrowprops=dict(color = 'yellow', width = 10, headwidth = 20, headlength = 20) )
        
            
