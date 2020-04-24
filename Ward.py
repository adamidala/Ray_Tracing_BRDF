# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:22:50 2020

@author: omen
"""

from numpy import pi,exp,linspace,tan,sqrt,cos,sin,array,meshgrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rho(kd,ks,theta,delta,sigma,gamma,phi):
    G = kd/pi
    
    gamma = f1(theta,delta)
    phi = f2(theta,delta)
    
    parenthese = ((cos(phi)**2) / (sigma[0])**2) + ((sin(phi)**2) / (sigma[1])**2)
    N1 = ks*exp( -1*(tan(gamma)**2) * (parenthese))
    D1 = sqrt(cos(theta)*cos(delta)) * 4 * pi * sigma[0]*sigma[1]
    return G + N1/D1

def f1(t,d):
    return np.arcsin(t+d+1)*0.2

def  f2(t,d):
    return 1/(t**3 + 1 + d**2)

kd = 0.1
ks = 0.330
sigma = array([0.050 , 0.160])



N = 1000
eps=1e-1
theta1d = np.linspace(-pi/2,pi/2,N) # Explose si a = - a0 et B proche de 0 mais non nul (>>1e-1)
delta1d = np.linspace(-pi/2,pi/2,N) #
a0 = 0.5

theta,delta = np.meshgrid(theta1d, delta1d) # x ,y
phi = pi/5
gamma = pi/5
Stock = rho(kd,ks,theta,delta,sigma,gamma,phi)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X = theta, Y = delta, Z = Stock ,cmap='CMRmap')
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel(r'$\beta$', fontsize=18)
ax.set_zlabel('Z', fontsize=18)
