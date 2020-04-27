# -*- coding: utf-8 -*-

#######################################
# StabilityMap_2d.py
#######################################

# analysis of two coupled tipping elements 
# for manuscript: 
# "Emergence of cascading dynamics in interacting tipping elements of ecology and climate"

# two coupled tipping elements given by 
# subsystem 0
# dx0/dt = a_0*x0 - b_0*x0^3 + c_0 + d_0*x1
# subsystem 1
# dx1/dt = a_1*x1 - b_1*x1^3 + c_1 + d_1*x0


# computes a (2d) matrix of stability maps each giving the number of stable fixed points 
# depending on the control parameters of two tipping elements 
# the position of the stability map withing the matrix is determined by the coupling strength of the tipping elements

# script generates elements of Figure 4/5/6 in manuscript and Figure 1 in Supplementary Material 
# for parameter settings indicated below 

###############################################################################

# Import packages  
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from numpy import roots

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# definition of coefficients 
b_0 = 1.0 
b_1 = 1.0              
a_0 = 1.0
a_1 = 1.0                
 
# calulcation of intrinsic tipping points 
c_0_crit = 2*np.sqrt((a_0/3)**3/(b_0))
c_1_crit = 2*np.sqrt((a_1/3)**3/(b_1))

# control parameter arrays 
anz = 500  # 500 chosen for single stability map, 100 chosen for matrix of stability maps 
value_c0 = np.linspace(0.0,0.8,anz)       
value_c1 = np.linspace(0.8,0.0,anz)      
      

# coupling strength array                                       
value_d0 = np.array([0.0])       # used: 
                    # for matrix of stability maps (Figure 4 in manuscript): -0.9, -0.7, -0.5, -0.3, -0.2, 0.0
                    # for undirectional example (Figure 5 in manuscript): 0.0
                    # for bidirectional example (Figure 6 in manuscript): -0.2
                    # for undirectional example of high coupling strength (Figure 1 in Suppl.Material): 0.0
value_d1 = np.array([0.2])        # used: 
                    # for matrix of stability maps (Figure 4 in manuscript): 0.9,0.7,0.5,0.3,0.2,0.0
                    # for unidirectional example (Figure 5 in manuscript): 0.2
                    # for bidirectional example (Figure 6 in manuscript): 0.2
                    # for undirectional example of high coupling strength (Figure 1 in Suppl.Material): 0.9
                    
                                  
# set quiver to 1 if additional phase portraits shall be plotted onto the stability map 
# note: this is only recommended if there is one combination of coupling strength defined 
# (i.e. Figure 5,6 in manuscript and Figure 1 in Suppl. Material)
quiver = 1 
# if quiver is set to 1, control parameter values need to be provided for which 
# phase portraits are added to the stability map  
value_c0_q = np.array([0.2, 0.6])     # for example:  
                                                # for bidirectional example (Figure 6 in manuscript): 0.1, 0.33,0.5,0.72
                                                # for undirectional example (Figure 5 in manuscript): 0.2, 0.6
                                                # for unidirectional example of high coupling strength (Figure 1 in Suppl. Material): 0.2,0.6
value_c1_q = np.array([0.1,0.3,0.5,0.7])    
                                                # for bidirectional example (Figure 6 in manuscript): 0.09,0.3,0.49,0.7
                                                # for undirectional example (Figure 5 in manuscript): 0.1,0.3,0.5,0.7
                                                # for unidirectional example of high coupling strength (Figure 1 in Suppl. Material): 0.2,0.6
 
# specification of additional arrows/markers/annotations according to Figures found in manuscript 
# should be added to plot 
                                                
F_add = "Figure_5"
# F_add = "Figure_6"
# F_add = "Figure_1_Suppl"

###############################################################################
# definition of functions 
###############################################################################
                                  
def roots_(*params):
    return roots(list(params)) 

roots3 = np.vectorize(roots_, signature = "(),(),(),()->(n)",otypes=[complex])
roots9 = np.vectorize(roots_, signature = "(),(),(),(),(),(),(),(),(),()->(n)", otypes=[complex])
                                  
# find equilibria via roots
def find_roots(a0 = 1,b0 = 1 , c0 = 0 , d0 = 0 , a1 = 1, b1 = 1,c1 = 0,d1 = 0):
    if (d0 != 0) and (d1 != 0):
        α = []
        α += [- b1 * (b0/d0)**3 ]          # x^9
        α += [0]                           # x^8
        α += [3 * a0 * b0**2 * b0 / d0**3] # x^7
        α += [+ 3 * c0*b0**2*b1/d0**3]     # x^6
        α += [- 3 * b0*a0**2*b1/d0**3]     # x^5
        α += [- 6 * a0 * b0 * c0 * b1 / d0**3] # x^4
        α += [- 3 * b0 * c0**2 *b1 / d0 **3 + a0**3*b1/d0**3 + a1*b0/d0] #x^3    a1**3
        α += [3 * a0 **2 * c0 * b1 / d0**3] # x^2
        α += [3 * a0 * c0**2 * b1 / d0 **3 + d1 - a1 * a0 / d0] # x^1
        α += [-a1*c0/ d0 + c1 + b1 *(c0/d0)**3] # x^0
        x0 = roots(α)
        x1 = 1 / d0 * (b0 * x0**3 - a0*x0 - c0)
    if (d0 == 0):
        x0_roots_ = roots([-b0,0,a0,c0])
        x0 = []
        x1 = []
        for x0_root in x0_roots_:
             x1 += [roots([-b1,0,a1,c1 + d1 * x0_root])]
             x0 += [x0_root]*3
    if (d1 == 0):# and (d0 >= 0):
        x1_roots_ = roots([- b1,0,a1,c1])
        x0 = []
        x1 = []
        for x1_root in x1_roots_:
            x0 += [roots([-b0,0,a0,c0 + d0 * x1_root])]
            x1 += [x1_root]*3   
    
    return (np.round(np.array(x0).flatten(),decimals=5),np.round(np.array(x1).flatten(),decimals=5))

# determine stability of equilibria by calculating eigenvalues 
def stability(x0,x1,a0 = 1, b0 = 1,c0 = 0,d0 = 0,a1 = 1,b1 = 1,c1 = 0,d1 = 0):
 
    D = np.sqrt((a0 + a1 - 3*(b1*x1**2 + b0*x0**2))**2 -4*((a0-3*b0*x0**2)*(a1-3*b1*x1**2)-d0*d1)+1J*0)
    return ((a0 + a1 - 3*(b1*x1**2 + b0*x0**2)) - D)/2,((a0 + a1 - 3*(b1*x1**2 + b0*x0**2)) +D)/2


###############################################################################
###############################################################################


# definitions for axis ticks 
ya = 0
major_xticks = np.linspace(value_c0[0], value_c0[anz-1],5)    
major_yticks = np.linspace(value_c1[anz-1], value_c1[0],5)     

# initialization of array nr_stable_all for number of stable fixed points      
nr_stable_all = np.zeros((value_c1.shape[0],value_c0.shape[0]))     
         
# initialization of counters                                                            
count_c0 = -1                                                        
count_c1 = -1                                                        
count_loop = 0                                                      

# open figure 
fig = plt.figure(figsize = (10,10)) 

###############################################################################
# determine number of stable fixed points 
###############################################################################

# loop over coupling stength d1
for d_1 in value_d1:                                                
    
    # loop over coupling strength d0
    for d_0 in value_d0:                                           
        count_loop = count_loop+1                                  
        count_c0 = -1
        
        # loop over control parameter c0  
        for c_0 in value_c0:                                       
             
            count_c0 = count_c0+1
            count_c1 = -1                                            
            
            # loop over control parameter c1
            for c_1 in value_c1:                                   
                
                count_c1 = count_c1+1               
    
                params = {"c0" : c_0, "c1" : c_1, "d0" : d_0,"d1" : d_1}
                # find equilibira for given combination of control parameters 
                x0, x1 = find_roots(**params)     
                # determine stability / eigenvalues
                l0, l1 = stability(x0, x1, **params)         
                
                # find real FP
                such_real = np.logical_and(np.isreal(x0),np.isreal(x1))
                # find stable FP 
                such_l = np.logical_and(np.real(l0[such_real])<0,np.real(l1[such_real])<0)
                x0_stab = np.array(x0[such_real][such_l])
                x1_stab = np.array(x1[such_real][such_l])
                
                # count number of stable FP and save them in result array 
                nr_stable_all[count_c1,count_c0] = len(x0_stab)
       


#################################################################
# Plotten 
#################################################################       

        # subplots 
        ax = fig.add_subplot(value_d1.shape[0],value_d0.shape[0],count_loop)      

        # definition of color map 
        cmap_s = np.array(['lightslategrey','lightgray','silver', 'darkgray', 'dimgray' ])   # colors 
        # choose color range depending on range of number of stable equilibria 
        crange = np.arange(np.min(nr_stable_all),np.max(nr_stable_all)+1,1)     
        cMap = ListedColormap(cmap_s[crange.astype(int)])
        
    
        
        # plot result array nr_stable_all  
        plt.imshow(nr_stable_all, interpolation='nearest',cmap = cMap, extent = [value_c0[0],value_c0[anz-1],value_c1[anz-1],value_c1[0]], aspect='auto')  # Plotten von nr_stable_all     
        
        # add intrinsic tipping points 
        plt.plot(np.zeros(len(value_c1))+c_0_crit,value_c1,'--', color = 'black',linewidth = 1)  
        plt.plot(value_c0,np.zeros(len(value_c0))+c_1_crit,'--', color = 'black',linewidth = 1)  
        
        # axis labels/ ticks and other properties 
        plt.xticks(())
        plt.yticks(())
        
        if count_loop > ((value_d1.shape[0] * value_d0.shape[0])-value_d0.shape[0]):
           plt.xticks(major_xticks, fontsize = 15) # fontsize = 15 for single, fontsize = 10 for multiple 
           plt.xlabel(r"$c_1$""\n" r"$d_{21} = %s$"%d_0, fontsize = 15  ) 
        
        if count_loop == (1+(ya*value_d0.shape[0])):
           plt.yticks(major_yticks, fontsize = 15) 
           plt.ylabel(r"$d_{12} = %s$" "\n" r"$c_2$" %d_1, fontsize = 15)    
           ya = ya + 1 

        plt.gca().set_aspect('equal', adjustable='box')
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

fig.tight_layout()  


###############################################################################         
# add quiver plots
###############################################################################

if quiver == 1: 
    
    d_0 = value_d0[0]                    
    d_1 = value_d1[0]                   
   
    
    ############################################################################### 
    ###############################################################################
                      
   
    # definition of axis ticks 
    major_xticks = np.array([-2, -1, 0, 1, 2])              
    major_yticks = np.array([-2, -1,  0,  1, 2])             
    ya = 0 
  
    # initialize counter
    count_loop = 0                                          
    
    # loop over control parameter c1
    for c_1 in value_c1_q:                            
                            
        # loop over control parameter c0                        
        for c_0 in value_c0_q:          
             
            count_loop = count_loop+ 1                    
            
            #######################################################################
            # find Equilibria 
            #######################################################################
        
            params = {"c0" : c_0, "c1" : c_1, "d0" : d_0,"d1" : d_1}
            # find equilibira for given combination of control parameters 
            x0, x1 = find_roots(**params)     
            # determine stability / eigenvalues
            l0, l1 = stability(x0, x1, **params)   

            # find real FP 
            such_real = np.logical_and(np.isreal(x0),np.isreal(x1))
            # find stable FP 
            such_l = np.logical_and(np.real(l0[such_real])<0,np.real(l1[such_real])<0)
            x0_stab = np.array(x0[such_real][such_l])
            x1_stab = np.array(x1[such_real][such_l])  
            # find unstable FP / saddles 
            such_m = np.logical_or(np.real(l0[such_real]) > 0, np.real(l1[such_real]) > 0)        
            x0_ustab =  np.array(x0[such_real][such_m])
            x1_ustab =  np.array(x1[such_real][such_m])
        
           
   
            #######################################################################
            # determine flow in phase space 
            #######################################################################
        
            def fun(x):
                return [a_0*x[0] - (b_0*(x[0]**3)) + c_0 + d_0*x[1], a_1*x[1] - (b_1*(x[1]**3)) + c_1 + d_1*x[0]]
        
            raum_begin = -2                           
            raum_end = 2 
            x0_flow = np.linspace(raum_begin,raum_end,1000)        
            x1_flow = np.linspace(raum_begin,raum_end,1000)

            X0,X1 = np.meshgrid(x0_flow,x1_flow)                     

            DX0,DX1 = fun([X0,X1])                          
            speed = np.sqrt(DX0*DX0 + DX1*DX1)              
            max_value = speed.max()                         
            speed_norm = [v/max_value for v in speed]  
        
            #################################################################################
            # Plotten
            #################################################################################
            
            axins = inset_axes(ax, width="100%", height="100%",
                       bbox_to_anchor=(c_0/0.8-0.5*0.12/0.8, c_1/0.8-0.5*0.12/0.8, .12, .12),
                       bbox_transform=ax.transAxes, loc='lower left')
            
           
            
            # flow in phase space via streamplot()
            axins.streamplot(X0,X1,DX0,DX1, density=[0.6, 0.6], color = 'white', linewidth =0.4, arrowsize = 0.5 )   
            # normalised speed  speed_norm via contourf()
            CF = axins.contourf(X0,X1,speed_norm, levels = np.arange(np.min(speed_norm),np.max(speed_norm),0.025)) 
                 

         	# stable FP  
            axins.plot(np.real(x0_stab), np.real(x1_stab), "o", color = "goldenrod",markersize = 4.5)
            # unstable FP / saddle 
            axins.plot(np.real(x0_ustab), np.real(x1_ustab), "o", color = "orangered", markersize = 4.5)
                
            ################################################################################
            # additional arrows/markers/text
            # this is specific for certain parameter settings and may need to be adjusted 
            ################################################################################
            
            # for unidirectional example (Figure 5 in manuscript)
            
            if F_add == "Figure_5": 
                # add markers
                if c_0 == value_c0_q[0] and c_1 == value_c1_q[0]:
                    axins.scatter(x0_stab[x0_stab < 0], x1_stab[x0_stab < 0], s=100, marker = 'o', facecolors='none', edgecolors='lime', linewidth = 3) 
                    such00 = np.logical_and(np.real(x0_stab) < 0, np.real(x1_stab < 0))
                    axins.scatter(np.real(x0_stab[such00]), np.real(x1_stab[such00]), s=300, marker = 'p', facecolors='none', edgecolors='deeppink', linewidth = 3) 
                    such10 = np.logical_and(np.real(x0_stab)> 0, np.real(x1_stab)< 0)
                    axins.scatter(np.real(x0_stab[such10]), np.real(x1_stab[such10]), s=100, marker = 's', facecolors='none', edgecolors='yellow', linewidth = 3) 
                        
                
                # add arrows 
                if c_0 == value_c0_q[1] and c_1 == value_c1_q[0]: 
                    such10 = np.logical_and(np.real(x0_stab) > 0, np.real(x1_stab) < 0)
                    axins.annotate("", xy = (np.real(x0_stab[such10])-0.25, np.real(x1_stab[such10])-0.45), xytext = (-1,-1.35), arrowprops=dict(facecolor = 'lime', edgecolor = 'none', width = 3, headwidth = 8) )
                    such11 = np.logical_and(np.real(x0_stab) > 0, np.real(x1_stab) > 0)
                    axins.annotate("", xy = (np.real(x0_stab[such11])-0.25, np.real(x1_stab[such11])+0.2), xytext = (-1,1.25), arrowprops=dict(facecolor = 'lime', edgecolor = 'none', width = 3, headwidth = 8) )
                if c_0 == value_c0_q[1] and c_1 == value_c1_q[1]:                    
                    axins.annotate("", xy = (1, -1.25), xytext = (-1,-1.25), arrowprops=dict(facecolor = 'deeppink', edgecolor = 'none', width = 3, headwidth = 8) )
                    such11 = np.logical_and(np.real(x0_stab) > 0, np.real(x1_stab) > 0)
                    axins.annotate("", xy = (np.real(x0_stab[such11])+0.25, np.real(x1_stab[such11])-0.25), xytext = (1.5,-1), arrowprops=dict(facecolor = 'deeppink', edgecolor = 'none', width = 3, headwidth = 8) )
                if c_0 == value_c0_q[0] and c_1 == value_c1_q[1]: 
                    such11 = np.logical_and(np.real(x0_stab) > 0, np.real(x1_stab) > 0)
                    axins.annotate("", xy = (np.real(x0_stab[such11])+0.25, np.real(x1_stab[such11])-0.25), xytext = (1.35,-1), arrowprops=dict(facecolor = 'yellow', edgecolor = 'none', width = 3, headwidth = 8) )
                
                # add text
                ax.annotate("", xy = (0.475,0.1), xytext = (0.275,0.1), arrowprops=dict(color = 'lime', arrowstyle = '->',ls = 'dashed') )
                ax.annotate("", xy = (0.19,0.2), xytext = (0.19,0.16), arrowprops=dict(color = 'yellow', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.08,0.17,'Facilitated \n tipping', color = 'yellow', fontsize = 13, fontweight = 'bold')
                ax.annotate("", xy = (0.536,0.209), xytext = (0.275,0.11), arrowprops=dict(color = 'deeppink', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.290,0.15,'Tipping \n cascade', color = 'deeppink', fontsize = 13, fontweight = 'bold')
                
                # add text for number of stable FP 
                ax.text(0.02,0.02, '4', color = 'black', fontsize = 14)
                ax.text(0.02,0.22, '3', color = 'black', fontsize = 14)
                ax.text(0.02, 0.60, '2', color = 'black', fontsize = 14)
                ax.text(0.78, 0.02, '2', color = 'black', fontsize = 14)
                ax.text(0.78, 0.16, '1', color = 'black', fontsize = 14)
                          
                            
                        
                        
            # for bidirectional example (Figure 6 in manuscript)
            
            if F_add == "Figure_6":
            # add markers 
                if c_0 == value_c0_q[0] and c_1 == value_c1_q[0]:
                    such = np.logical_and(np.real(x0_stab) < 0,np.real(x1_stab) < 0)
                    axins.scatter(np.real(x0_stab[such]), np.real(x1_stab[such]), s=100, marker = 's', facecolors='none', edgecolors='yellow', linewidth = 3) 
                    axins.scatter(np.real(x0_stab[such]),np.real(x1_stab[such]), s=300, marker = 'p', facecolors='none', edgecolors='deeppink', linewidth = 3) 
                    such2 = np.logical_and(np.real(x0_stab) > 0,np.real(x1_stab) < 0)
                    axins.scatter(np.real(x0_stab[such2]), np.real(x1_stab[such2]), s=100, marker = 's', facecolors='none', edgecolors='yellow', linewidth = 3) 
                        
                if c_0 == value_c0_q[2] and c_1 == value_c1_q[0]: 
                    such3 = np.logical_and(np.real(x0_stab) < 0,np.real(x1_stab) > 0)
                    axins.scatter(np.real(x0_stab[such3]), np.real(x1_stab[such3]), s=100, marker = 's', facecolors='none', edgecolors='yellow', linewidth = 3) 
               
                if c_0 == value_c0_q[0] and c_1 == value_c1_q[2]: 
                    such4 = np.logical_and(np.real(x0_stab) < 0,np.real(x1_stab) < 0)
                    axins.scatter(np.real(x0_stab[such4]), np.real(x1_stab[such4]), s=100, marker = 's', facecolors='none', edgecolors='yellow', linewidth = 3) 
    
                # add arrows within phase space portraits
                if c_0 == value_c0_q[1] and c_1 == value_c1_q[0]: 
                    such10 = np.logical_and(np.real(x0_stab) > 0,np.real(x1_stab) < 0)
                    axins.annotate("", xy = (np.real(x0_stab[such10]) -0.25, np.real(x1_stab[such10])-0.45), xytext = (-1,-1.35), arrowprops=dict(facecolor = 'yellow', edgecolor = 'none', width = 3, headwidth = 8) )
                if c_0 == value_c0_q[3] and c_1 == value_c1_q[0]: 
                    such11 = np.logical_and(np.real(x0_stab) > 0,np.real(x1_stab) > 0)
                    axins.annotate("", xy = ( np.real(x0_stab[such11]) -0.25, np.real(x1_stab[such11])+0.35), xytext = (-1,1.35), arrowprops=dict(facecolor = 'yellow', edgecolor = 'none', width = 3, headwidth = 8) )
                if c_0 == value_c0_q[0] and c_1 == value_c1_q[1]: 
                    such11 = np.logical_and(np.real(x0_stab) > 0,np.real(x1_stab) > 0)
                    axins.annotate("", xy = (np.real(x0_stab[such11])+0.25, np.real(x1_stab[such11])-0.25), xytext = (1.35,-1), arrowprops=dict(facecolor = 'yellow', edgecolor = 'none', width = 3, headwidth = 8) )
                if c_0 == value_c0_q[0] and c_1 == value_c1_q[3]: 
                    such01 = np.logical_and(np.real(x0_stab) < 0,np.real(x1_stab) > 0)
                    axins.annotate("", xy = (np.real(x0_stab[such01]), np.real(x1_stab[such01])-0.25), xytext = (-1.35,-1), arrowprops=dict(facecolor = 'yellow', edgecolor = 'none', width = 3, headwidth = 8) )
                if c_0 == value_c0_q[1] and c_1 == value_c1_q[1]: 
                    axins.annotate("", xy = (1, -1.25), xytext = (-1,-1.25), arrowprops=dict(facecolor = 'deeppink', edgecolor = 'none', width = 3, headwidth = 8) )
                    such11 = np.logical_and(np.real(x0_stab) > 0,np.real(x1_stab) > 0)
                    axins.annotate("", xy = (np.real(x0_stab[such11])+0.25, np.real(x1_stab[such11])-0.25), xytext = (1.5,-1), arrowprops=dict(facecolor = 'deeppink', edgecolor = 'none', width = 3, headwidth = 8) )
            
                # add text 
                ax.annotate("", xy = (0.09,0.20), xytext = (0.09,0.15), arrowprops=dict(color = 'yellow', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.10,0.16,'Facilitated \n tipping', color = 'yellow', fontsize = 13, fontweight = 'bold')
                
                ax.annotate("", xy = (0.09,0.60), xytext = (0.09,0.55), arrowprops=dict(color = 'yellow', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.10,0.56,'Impeded \n tipping', color = 'yellow', fontsize = 13, fontweight = 'bold')
                
                ax.annotate("", xy = (0.22, 0.085), xytext = (0.16, 0.085), arrowprops=dict(color = 'yellow', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.16,0.095,'Facilitated \n tipping', color = 'yellow', fontsize = 13, fontweight = 'bold')
                
                ax.annotate("", xy = (0.62, 0.085), xytext = (0.56, 0.085), arrowprops=dict(color = 'yellow', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.56,0.095,'Impeded \n tipping', color = 'yellow', fontsize = 13, fontweight = 'bold')
                
                ax.annotate("", xy = (0.25, 0.22), xytext = (0.16, 0.13), arrowprops=dict(color = 'deeppink', arrowstyle = '->',ls = 'dashed') )
                ax.text(0.23,0.16,'Tipping \n cascade', color = 'deeppink', fontsize = 13, fontweight = 'bold')
                
                ax.text(0.01,0.01, '4', color = 'black', fontsize = 14)
                ax.text(0.01,0.2, '3', color = 'black', fontsize = 14)
                ax.text(0.01, 0.60, '2', color = 'black', fontsize = 14)
                ax.text(0.2, 0.01, '3', color = 'black', fontsize = 14)
                ax.text(0.78, 0.01, '2', color = 'black', fontsize = 14)
                ax.text(0.78, 0.15, '1', color = 'black', fontsize = 14)
                
            
            
            # Figure 1 in supplementary material 
            
            if F_add == "Figure_1_Suppl": 
                
                ax.text(0.01,0.01, '2', color = 'black', fontsize = 14)
                ax.text(0.01,0.53, '3', color = 'black', fontsize = 14)
                ax.text(0.78,0.01, '1', color = 'black', fontsize = 14)
            
            
            # axis labels and ticks 
            plt.xticks(())
            plt.yticks(())
 
            plt.xticks(major_xticks,fontsize = 8 )
            plt.xlabel(r"$x_1$" ) 
            plt.yticks(major_yticks, fontsize = 8)                         
            plt.ylabel(r"$x_2$" ) 
            plt.tick_params(axis='both', labelsize=7.5)   
            
            plt.gca().set_aspect('equal', adjustable='box')
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
