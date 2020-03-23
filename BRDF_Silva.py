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

C = lambda thi,thr : exp( -K/2*(cos(thi)+cos(thr))  ) # Page 5 (7)

F = lambda thi,RAP : # RAP 

G = lambda th : 2/  ( 1 + sqrt( 1 + (gamma**2) * (tan(th)**2) ) )  # Page 6 (10)

P = lambda a : (gamma**2) / ( pi * (cos(a)**4) * ( (gamma**2) + (tan(a)**2))**2 ) # Ptr (8)


qd = lambda thi,thr,RAP : rhol/pi * cos(thr) * (1 - F(thi,RAP)) * (1 - F(thr,1/RAP)) 
qs = lambda thi,thr,a,th,RAP : (1 - C(thi,thr)) * (F(...,RAP) * P(a) * G(th))
qc = lambda thi,thr,th,RAP : C(thi,thr) * F(...,RAP) * G(th)

q = lambda qc,qs,qd : qc + qs + qd

N = 90
th_r = np.linspace(0,np.pi/2,N) # 0 - 90 degrés
