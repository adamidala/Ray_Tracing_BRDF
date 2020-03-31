import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import cos,arccos,sin,arcsin,tan,arctan,exp,pi,sqrt

#______________   PAGE 6  ___________________
# Sample : Pressed PTFE
n = 1.45                            # +/-0.04
rhol = 0.74                         # +/- 0.07
gamma = 0.049                       # +/- 0.015
K = 1.7                             # +/- 0.2 

RAP = n/n0
# Supposition de notre part
#___________________________________________
a = 1
n0 = 1
th = pi/2
G = lambda th : 2/  ( 1 + sqrt( 1 + (gamma**2) * (tan(th)**2) ) )  # Page 6 (10)
#___________________________________________


C = lambda thi,thr : exp( -K/2*(cos(thi)+cos(thr))  ) # Page 5 (7)

F = lambda th_i,n_t,n_i : (n_t*cos(th_i)-n_i/n_t*(np.sqrt(n_t**2-n_i**2*sin(th_i)**2)))/\
    (n_i/n_t*np.sqrt(n_t**2-n_i**2*sin(th_i)**2) + n_t*cos(th_i))


P = lambda a : (gamma**2) / ( pi * (cos(a)**4) * ( (gamma**2) + (tan(a)**2))**2 ) # Ptr (8)


qd = lambda thi,thr,n_t,n_i : rhol/pi * cos(thr) * (1 - F(thi,n_t,n_i)) * (1 - F(thr,n_i,n_t)) 
qs = lambda thi,thr,a,th,n_t,n_i : (1 - C(thi,thr)) * (F(thi,n_t,n_i) * P(a) * G(th))
qc = lambda thi,thr,th,n_t,n_i : C(thi,thr) * F(thi,n_t,n_i) * G(th)

q = lambda qc,qs,qd : qc + qs + qd

N = 90
th_r = np.linspace(-40*pi/180,100*pi/180,N) # -40 - 100 degrés
thi = 30*pi/180

QD = qd(thi,th_r,n,n0)
QS = qs(thi,th_r,a,th,n,n0)
QC = qc(thi,th_r,th,n,n0)
Q = QD + QS + QC

plt.plot(th_r,Q)
