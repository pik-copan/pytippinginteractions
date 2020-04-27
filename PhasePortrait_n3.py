# -*- coding: utf-8 -*-

##################################
# PhasePortrait_n3.py
##################################

# analysis of three unidirectionally coupled tipping elements in 3d space 
# for manuscript: 
# "Emergence of cascading dynamics in interacting tipping elements of ecology and climate"

# three unidirectionally coupled tipping elements given by 
# subsystem 0
# dx0/dt = a_0*x0 - b_0*x0^3 + c_0
# subsystem 1
# dx1/dt = a_1*x1 - b_1*x1^3 + c_1 + d_1*x0
# subsystem 2
# dx2/dt = a_2*x2 - b_2*x2^3 + c_2 + d_2*x1

# plots '3d phase portraits' and trajectory 

# script generates elements of Figure 8 in manuscript and Figure 4/5 in Supplementary Material
# for parameter settings indicated below 


###############################################################################

# import packages 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D
from numpy import roots
from numpy import linalg as LA

# define coefficients
a_0 = 1.0
a_1 = 1.0                 
a_2 = 1.0
b_0 = 1.0
b_1 = 1.0             
b_2 = 1.0

# define coupling strength
# d_0 = 0.0  
d_1 = 0.2
d_2 = 0.2 

# fixed control parameter of subsystem 0
c_0 = 0.4                   # for Figure 8 in manucript, for Figure 4/5 in Supplementary Material: 0.4

# define control parameters of subsystem 1 and 2 
value_c1 = np.array([0.2]) # for Figure 8 in manuscript: 0.2  
                           # for Figure 4 in Supplementary Material: 0.2
                           # for Figure 5 in Supplementary Material: 0.0
value_c2 = np.array([0.2]) # for Figure 8 in manuscript: 0.2   
                           # for Figure 4 in Supplementary Material: 0.0
                           # for Figure 5 in Supplementary Material: 0.2
                           
         
# define starting value for trajectory of simulation 
fun0 = np.array([ [-1,-1,-1]])      
# for Figure 8 in manuscript and Figure 4 in Supplementary Material: (-1,-1,-1)
# for Figure 5 in Supplementary Material: (-1,1,-1)

# color of trajectory 
col_traj = ['deeppink' ,'yellow','orange']  # 'deeppink'                                   
                                            
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


############################################################################### 
###############################################################################


# definition for axis ticks 
major_xticks = np.array([-2, -1, 0, 1, 2])              
major_yticks = np.array([-2, -1,  0,  1, 2])            
ya = 0 

# open figure 
fig = plt.figure(figsize = (8,8))

# initialization of counter
count_loop = 0                                          

###############################################################################
# determine equilibria 
###############################################################################

# loop over control parameter c2
for c_2 in value_c2:        

    # loop over control parameter c1                      
    for c_1 in value_c1:                        
        
        
        count_loop = count_loop+ 1                    
       
        
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
        
        # find unstable FP / saddles 
        such_m = np.logical_or(np.logical_or(np.real(l0[such_real])>0,np.real(l1[such_real])>0),np.real(l2[such_real])>0)
        x0_ustab = np.array(x0[such_real][such_m])
        x1_ustab = np.array(x1[such_real][such_m])  
        x2_ustab = np.array(x2[such_real][such_m]) 
        
        
        #####################################################################
        # definition of system of coupled tipping elements  
        #####################################################################
        def fun(x):
            return [a_0*x[0] - (b_0*(x[0]**3)) + c_0, a_1*x[1] - (b_1*(x[1]**3)) + c_1 + d_1*x[0], a_2*x[2] - (b_2*(x[2]**3)) + c_2 + d_2*x[1]]

        def fun2(x,t = 0):
            return [a_0*x[0] - (b_0*(x[0]**3)) + c_0, a_1*x[1] - (b_1*(x[1]**3)) + c_1 + d_1*x[0], a_2*x[2] - (b_2*(x[2]**3)) + c_2 + d_2*x[1]]

        ##################################################################################
        # plot of flow and equilibria in 'phase space'
        ##################################################################################
        raum_begin = -2                      # phase space limits       
        raum_end = 2                                
            
        ax = fig.gca(projection='3d')
        col = ['black','black', 'black']
        
        # define level for x0 system 
        if len(np.unique(x0[such_real])) == 3: 
            stream_level = np.unique(np.real(x0[such_real]))
        else: 
            stream_level = np.array([-1,0,np.real(np.unique(x0[such_real]))])
            
        # loop over x0 system level 
        for l in range(0,len(stream_level)): 
            
            x0_flow = stream_level[l]   
            x1_flow = np.linspace(raum_begin,raum_end,50)
            x2_flow = np.linspace(raum_begin,raum_end,50)                    
            X1,X2,X0 = np.meshgrid(x1_flow,x2_flow,x0_flow)                            
    
            DX0,DX1,DX2 = fun([X0,X1,X2])                     
    
            # subplots 
            fig_tmp, ax_tmp = plt.subplots(figsize = (8,8))
            # flow in phase space for x1-x2 area and fixed level of x0 
            res = ax_tmp.streamplot(X1[:,:,0],X2[:,:,0],DX1[:,:,0],DX2[:,:,0], density=[0.5, 0.5], color = col[l], linewidth =2.5, arrowsize = 2.5 )   
            ax_tmp.set_xlabel(r"$x_2$",fontsize = 24, labelpad = 6)
            ax_tmp.set_ylabel(r"$x_3$",fontsize = 24, labelpad = -11)
            plt.xticks(())
            plt.yticks(())
            ax_tmp.set_xticks(major_xticks)
            ax_tmp.set_yticks([-1,0,1]) 
            ax_tmp.tick_params(axis='both', labelsize=24 ,pad = 4)
            fig_tmp.show()
            
            
            # extract streamlines 
            lines = res.lines.get_paths()       
            # plot streamlines in 3d-system as x1-x2 area 
            i = 0
            for line in lines:
                i = i + 1
                old_x = line.vertices.T[0]
                old_y = line.vertices.T[1]
                new_x = old_x
                new_y = old_y
                ax.plot(new_x, new_y, zs = x0_flow, zdir = 'z', color = col[l])
                
                if i%5 ==1:
                    ax.quiver(new_x[0],new_y[0], x0_flow, new_x[1]-new_x[0],new_y[1]-new_y[0],0, color = col[l])
            
        # add stable FP 
        for i in range(0,len(x0_stab)): 
            ax.scatter(np.real(x1_stab[i]), np.real(x2_stab[i]), s =50, zs=np.real(x0_stab[i]), zdir='z',color = "goldenrod")
        for i in range(0,len(x0_ustab)): 
            ax.scatter(np.real(x1_ustab[i]),np.real(x2_ustab[i]), s = 30, zs=np.real(x0_ustab[i]), zdir='z',color = "orangered")
        
        ##################################################################################
        # simulation starting from x = (-1,-1,-1)
        ##################################################################################
        t = np.linspace(0, 100,  1000)  
        
        for fp_start in range(0,np.size(fun0,0)):
            X = integrate.odeint(fun2, fun0[fp_start], t, full_output=True)
            ax.plot(X[0][:,1],X[0][:,2],X[0][:,0], linewidth = 3, color = col_traj[fp_start])  
        
        # set axis limits 
        ax.set_zlim3d(np.min(stream_level), np.max(stream_level))                    
        ax.set_ylim3d(-2, 2)                    
        ax.set_xlim3d(-2, 2)                    
        
        # set view
        ax.view_init(17,295)  # 255
        
        # turn off grid 
        ax.grid(False)

        # axis label and ticks 
        plt.xticks(())
        plt.yticks(())
        ax.set_xlabel(r"$x_2$", fontsize = 12, labelpad = 10)
        ax.set_ylabel(r"$x_3$", fontsize = 12, labelpad = 10)
        ax.set_zlabel(r"$x_1$", fontsize = 12, labelpad = 10)
        ax.set_xticks(major_xticks)
        ax.set_yticks(major_yticks)  
        ax.set_zticks([-1,-0.5,0,0.5,1])     
        ax.tick_params(axis='both', labelsize=12 ,pad = 5)   
        plt.gca().set_aspect('equal', adjustable='box')
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
 

        ###############################################################################
        # get lateral surfaces 
        ###############################################################################
                     
        # x2-x0 surface       
        col = ['black','black', 'black']
        stream_levelx0x2 = np.array([2, -2])
        # loop over x1-system levels
        for l in range(len(stream_levelx0x2)): 
            
            x1_flow = stream_levelx0x2[l]
            x0_flow = np.linspace(-1,1,50)                    
            x2_flow = np.linspace(raum_begin,raum_end,50)                    
            X2,X0,X1 = np.meshgrid(x2_flow,x0_flow,x1_flow)                             
        
            DX0,DX1,DX2 = fun([X0,X1,X2])                      
            
            # flow in phase space for x2-x0 area and fixed level of x1
            fig_tmp, ax_tmp = plt.subplots(figsize = (8,8))
            res = ax_tmp.streamplot(X2[:,:,0],X0[:,:,0],DX2[:,:,0],DX0[:,:,0], density=[0.5, 0.5], color = col[l], linewidth =2.5, arrowsize =2.5 )   
            ax_tmp.set_xlabel(r"$x_3$",fontsize = 24, labelpad = 6)
            ax_tmp.set_ylabel(r"$x_1$",fontsize = 24, labelpad = -11)
            plt.xticks(())
            plt.yticks(())
            ax_tmp.set_xticks(major_xticks)
            ax_tmp.set_yticks([-1,0,1]) 
            ax_tmp.tick_params(axis='both', labelsize=24 ,pad = 4)
            fig_tmp.show()
            
            
            # extract streamlines 
            lines = res.lines.get_paths()      
            # plot streamlines in 3d-system as x2-x0 area 
            i = 0
            fig = plt.figure(figsize = (8,8))
            ax = fig.gca(projection='3d')
            for line in lines:
                i = i + 1
                old_x = line.vertices.T[0]
                old_y = line.vertices.T[1]
                new_x = old_x
                new_y = old_y
                # ax.plot(new_x, new_y, new_z, 'k')
                ax.plot(new_x, new_y, zs = x1_flow, zdir = 'x', color = col[l])
                
                if i%5 ==1:
                    ax.quiver(x1_flow,new_x[0], new_y[0], 0, new_x[1]-new_x[0], new_y[1]-new_y[0], color = col[l])
                
                # set axis limits
                ax.set_zlim3d(np.min(stream_level), np.max(stream_level))                    
                ax.set_ylim3d(-2, 2)                    
                ax.set_xlim3d(-2, 2)                    
                
                # set view
                ax.view_init(17,295)  # 255
                 
                # turn off grid
                ax.grid(False)
                
                # axis labels and ticks
                plt.xticks(())
                plt.yticks(())
                ax.set_xlabel(r"$x_2$", fontsize = 12, labelpad = 10)
                ax.set_ylabel(r"$x_3$", fontsize = 12, labelpad = 10)
                ax.set_zlabel(r"$x_1$", fontsize = 12, labelpad = 10)
                ax.set_xticks(major_xticks)
                ax.set_yticks(major_yticks)  
                ax.set_zticks([-1,0,1])     
                ax.tick_params(axis='both', labelsize=12 ,pad = 5) 
            
        ###############################################################################
            
        # x1-x0 surface 
        col = ['black','black', 'black']
        stream_levelx0x1 = np.array([2, -2])
        
        # loop over x2-system levels
        for l in range(len(stream_levelx0x1)): 
            
            x2_flow = stream_levelx0x1[l]
            
            x0_flow = np.linspace(-1,1,50)                    
            x1_flow = np.linspace(raum_begin,raum_end,50)                    
        
            X1,X0,X2 = np.meshgrid(x1_flow,x0_flow,x2_flow)                             
        
            DX0,DX1,DX2 = fun([X0,X1,X2])                      
            
            # flow in phase space for x1-x0 area and fixed level of x2
            fig_tmp, ax_tmp = plt.subplots(figsize = (8,8))
            res = ax_tmp.streamplot(X1[:,:,0],X0[:,:,0],DX1[:,:,0],DX0[:,:,0], density=[0.5, 0.5], color = col[l], linewidth =2.5, arrowsize = 2.5 )   
            ax_tmp.set_xlabel(r"$x_2$",fontsize = 24, labelpad = 6)
            ax_tmp.set_ylabel(r"$x_1$",fontsize = 24, labelpad = -11)
            plt.xticks(())
            plt.yticks(())
            ax_tmp.set_xticks(major_xticks)
            ax_tmp.set_yticks([-1,0,1]) 
            ax_tmp.tick_params(axis='both', labelsize=24 ,pad = 5)
            fig_tmp.show()
            
            # extract streamlines 
            lines = res.lines.get_paths()       
            # plot streamlines in 3d-system as x1-x0 surface 
            i = 0
            fig = plt.figure(figsize = (8,8))
            ax = fig.gca(projection='3d')
            for line in lines:
                i = i + 1
                old_x = line.vertices.T[0]
                old_y = line.vertices.T[1]
                new_x = old_x
                new_y = old_y
                # ax.plot(new_x, new_y, new_z, 'k')
                ax.plot(new_x, new_y, zs = x2_flow, zdir = 'y', color = col[l])
                
                if i%5 ==1:
                    ax.quiver(new_x[0], x2_flow,new_y[0],  new_x[1]-new_x[0], 0,new_y[1]-new_y[0], color = col[l])
                
                # set axis limits
                ax.set_zlim3d(np.min(stream_level), np.max(stream_level))                    
                ax.set_ylim3d(-2, 2)                    
                ax.set_xlim3d(-2, 2)                    
                
                # set view
                ax.view_init(17,295)  # 255
                 
                # turn grid off 
                ax.grid(False)
                
                # axis labels and ticks 
                plt.xticks(())
                plt.yticks(())
                ax.set_xlabel(r"$x_2$", fontsize = 12, labelpad = 10)
                ax.set_ylabel(r"$x_3$", fontsize = 12, labelpad = 10)
                ax.set_zlabel(r"$x_1$", fontsize = 12, labelpad = 10)
                ax.set_xticks(major_xticks)
                ax.set_yticks(major_yticks)  
                ax.set_zticks([-1,0,1])     
                ax.tick_params(axis='both', labelsize=12 ,pad = 4)   
            
            