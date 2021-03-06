import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import cos,arccos,sin,arcsin,tan,arctan,exp,pi,log,sqrt
from mpl_toolkits.mplot3d import Axes3D

def IndArgmax(Narray):
    return  np.unravel_index(np.argmax(Narray, axis=None), a.shape)

def Pss_SMART(a,a0,b):
    F1 = lambda x,y : sqrt(np.maximum(1-(x**2)-(y**2),0))
    Num1 = a**3 + a*(a0**2) - 2*(a**3)*(a0**2) + a*(b**2) + (a0)*(b**2) - 2*a*(a0**2)*(b**2)
    Num2 = 2*(a**2)*a0*F1(a0,0)*F1(a,b) + a0*F1(a0,0)*b*F1(a,b)
    Denom1 = ((a**2) + (b**2))
    Denom2 = (a**2 + a0**2 + b**2 + (2*a*a0*F1(a0,0)*F1(a,b)) - (a0**2) * ((2*a**2)+(b**2)))**2
    #return [ ((Num1+Num2)**2)/(Denom1*Denom2) , ((Num1+Num2)**2) , (Denom1*Denom2) ]
    return ((Num1+Num2)**2)/(Denom1*Denom2) 

def Pss_BRUT(a,a0,b):
    return ((a**3 + a*(a0**2) - 2*(a**3)*(a0**2) + a*(b**2) + (a0)*(b**2) - 2*a*(a0**2)*(b**2)+ 2*(a**2)*a0*(sqrt(1-(a0**2)))*(sqrt(1-(a**2)-(b**2))) + a0*b*(sqrt(1-(a0**2)))*(sqrt(1-(a**2)-(b**2))))**2)/(((a**2) + (b**2))*(a**2 + a0**2 + b**2 + (2*a*a0* (sqrt(1-(a0**2))) * (sqrt(1-(a**2)-(b**2))) ) - (a0**2) * ((2*(a**2))+(b**2)))**2) 

def Pss(a,a0,b):
    Num1 = a**3 + a*(a0**2) - 2*(a**3)*(a0**2) + a*(b**2) + (a0)*(b**2) - 2*a*(a0**2)*(b**2)
    Num2 = 2*(a**2)*a0*(sqrt(1-(a0**2)))*(sqrt(1-(a**2)-(b**2))) + a0*b*(sqrt(1-(a0**2)))*(sqrt(1-(a**2)-(b**2)))
    Denom1 = ((a**2) + (b**2))
    Denom2 = (a**2 + a0**2 + b**2 + (2*a*a0* (sqrt(1-(a0**2))) * (sqrt(1-(a**2)-(b**2))) ) - (a0**2) * ((2*(a**2))+(b**2)))**2
    #return [ ((Num1+Num2)**2)/(Denom1*Denom2) , ((Num1+Num2)**2) , (Denom1*Denom2) ]
    return ((Num1 + Num2)**2)/(Denom1 * Denom2) 

N = 1000
eps=1e-1
a1d = np.linspace(-1,1,N) # Explose si a = - a0 et B proche de 0 mais non nul (>>1e-1)
b1d = np.linspace(0,1,N) #
a0 = 0.5

a,b = np.meshgrid(a1d, b1d) # x ,y
Pss_matrix_v1 = Pss_SMART(a,a0,b)
Pss_matrix_v2= np.minimum(Pss_matrix_v1,10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X=a,Y=b,Z=Pss_matrix_v1,cmap='CMRmap')
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel(r'$\beta$', fontsize=18)
ax.set_zlabel('Z', fontsize=18)
