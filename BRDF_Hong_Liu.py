import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import cos,arccos,sin,arcsin,tan,arctan,exp,pi,sqrt

def theta(th_i, th_r, phi):
	co_th = cos(th_i)*cos(th_r) + sin(th_i)*sin(th_r)*cos(phi)
	return arccos(co_th)/2


def alpha(th_i,th_r,th): # cos(a) =  (cos(thi) + cos(thr))/(2cos(th))
	co = (cos(th_i)+cos(th_r))/(2*cos(th))
	return arccos(co)

def G(thi,thr,phi) :
    th  = theta(thi, thr, phi)
    a = alpha(thi,thr,th)
    
    return np.minimum(cos(th), 2*cos(a)*cos(thr) , 2*cos(a)*cos(thi)) /cos(th)

def f(th_i,th_r,phi): 
    #Define Theta and Alpha
    th  = theta(th_i, th_r, phi)
    a = alpha(th_i,th_r,th)
    #Compute f
    f = exp( ( -( (tan(a)) **2) ) / (2*(sigma**2)) ) * (G(thi,thr,phi)) / (2*pi*4*(sigma**2) * (cos(a)**4) * cos(th_i)*cos(th_r))
    return f

DegToRad = lambda th : th * pi / 180
#Supposition de notre part :
#________________________________________
phi = DegToRad(90)
#________________________________________

sigma = 0.25
N = 900
thr = np.linspace(DegToRad(-90),DegToRad(90),N) 
plt.figure("Blinn's G")
for thi in [DegToRad(30),DegToRad(45),DegToRad(60)]:
    plt.plot(thr,G(thi,thr,phi))
plt.show()
plt.xlabel(r"${\theta_r}$ (deg)",fontsize = 14)
plt.ylabel(" Blinn's G ",fontsize = 14)
plt.title("Blinn's G function curves")
plt.xticks(np.arange(-pi/2,pi/2,pi/9),labels = np.arange(-90,90,20) )



plt.figure("T-S fs curves with Blinn’s G")
for thi in [DegToRad(30),DegToRad(45),DegToRad(60)]:
    plt.plot(thr,f(thi,thr,phi))
plt.show()

plt.xlabel(r"${\theta_r}$ (deg)",fontsize = 14)
plt.ylabel(r" fs $[sr^{−1}]$",fontsize = 14)
plt.title("T-S fs curves with Blinn’s G")
plt.xticks(np.arange(-pi/2,pi/2,pi/9),labels = np.arange(-90,90,20) )




