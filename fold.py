# -*- coding: utf-8 -*-


###################################
# fold.py
###################################

# plots bifurcation diagram of isolated tipping elements
# for manuscript: 
# "Emergence of cascading dynamics in interacting tipping elements of ecology and climate"

# tipping element is given by 
# dx/dt = ax - bx^3 + c

# script generates elements of Figure 1 in manuscript 
# and is used for illustration of tipping rules (Figure 2/3 in manuscript)


###############################################################################

# import packages 
import numpy as np 
import matplotlib.pylab as plt
from numpy import roots

# definition of coefficients 
b_eq  = 1
a_eq = 1

# definition of control parameter range 
c_eq= np.linspace(-1, 1, 5000)

###############################################################################

fig = plt.figure(figsize = (9,7))
ax = fig.add_subplot(1,1,1)

###########################################################
# determine equilibria and their stability
###########################################################

Erg_stable_lower = []
Erg_stable_upper = []
Erg_unstable = []
    
# loop over control paraeter 
for c in c_eq:
    
     def fx(x_prime):
         return -b_eq*3*(x_prime**2) +a_eq
     
     Erg = roots([-b_eq,0,a_eq,c])   
     
     for i in range(0,len(Erg)): 
         stab = fx(Erg[i])
         # stable real equilibrium, lower brancg 
         if stab < 0 and np.real(Erg[i]) < 0 and np.isreal(Erg[i]): 
             Erg_stable_lower.append((c,Erg[i]))
        # stable real equilibrium, upper branch 
         if stab < 0 and Erg[i] > 0 and np.isreal(Erg[i]): 
             Erg_stable_upper.append((c,Erg[i]))
        # unstable equilibrium / saddle 
         if stab > 0 and np.isreal(Erg[i]): 
             Erg_unstable.append((c,Erg[i]))

Erg_Stable_Lower = np.asarray(Erg_stable_lower)
Erg_Stable_Upper = np.asarray(Erg_stable_upper)
Erg_Unstable = np.asarray(Erg_unstable)        
       
###############################################################################      
# Plotting 
###############################################################################

# Unstable Equilibria 
plt.plot(np.real(Erg_Unstable[:,0]),np.real(Erg_Unstable[:,1]),'--',color = 'grey',linewidth = 6.5)
# Lower stable Equilibria
plt.plot(np.real(Erg_Stable_Lower[:,0]),np.real(Erg_Stable_Lower[:,1]),'-',color = 'black',linewidth = 6.5)
# Upper stable Equilibria 
plt.plot(np.real(Erg_Stable_Upper[:,0]),np.real(Erg_Stable_Upper[:,1]),'-',color = 'black',linewidth = 6.5)

# add lines for orientation 
plt.plot(c_eq, np.zeros(len(c_eq)), '--',color = 'grey', linewidth=0.5)
plt.plot(np.zeros(50), np.linspace(-2, 2, 50), '--',color = 'grey', linewidth=0.5)

# add text - may have to be adjusted according to subsystem which is plotted 
plt.text(-0.92,-1.6,r"$x_{1^-}^*$", fontsize = 19)
plt.text(0.82,1.5,r"$x_{1^+}^*$", fontsize = 19)        # , color = (round(68/255,2), round(114/255,2), round(196/255,2))
plt.text(-0.9,1.7,r"$X_{1}$", fontsize = 22)
plt.text(np.sqrt(4/27),-2.21,r"$c_{{1}_{crit}}$ ", fontsize = 19)
plt.text(-np.sqrt(4/27)-0.15,-2.21,r"$-c_{{1}_{crit}}$ ", fontsize = 19)

# axis properties and labels 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(())
plt.xticks(np.array([0.0]),fontsize = 19)
plt.yticks(())
plt.yticks(np.array([-1.0,0.0,1.0]),fontsize = 19 )    
ax.set_ylim([-2,2])
ax.set_xlim([-1,1])
plt.xlabel("tipping parameter $c_{1}$", fontsize = 19, labelpad = 8)
plt.ylabel("equilibrium $x_{1}^*$", fontsize = 19)