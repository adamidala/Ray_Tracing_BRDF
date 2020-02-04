import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import cos,arccos,sin,arcsin,tan,arctan,exp,pi

#MATH OR NUMPY test speed.
#On peut tester performance si T est un dictionnaire T = {"ss" : ..., "sp" : ...,"ps" : ...,"pp" : ...} où ... = complex
def a(n_i,n_t,th_i):
    a_p = (n_t*cos(th_i)-n_i/n_t*(np.sqrt(n_t**2-n_i**2*sin(th_i)**2)))/\
    (n_i/n_t*np.sqrt(n_t**2-n_i**2*sin(th_i)**2) + n_t*cos(th_i))
	
    a_s = (n_i*cos(th_i)-np.sqrt(n_t**2-n_i**2*sin(th_i)**2))/\
    (n_i*cos(th_i) + np.sqrt(n_t**2-n_i**2*sin(th_i)**2))
	
    return np.array(a_p), np.array(a_s)

def Jonex_Matrix(th_i, th_r, phi):
	b = beta(th_i, th_r, phi)
	et_i,et_r = eta(th_i,th_r,b)
	a_p, a_s = a(n_i,n_t,th_i)
	T_0 = a_s*cos(et_i)*cos(et_r) + a_p * sin(et_i)*sin(et_r)
	T_1 = -1*a_s*sin(et_i)*cos(et_r) + a_p * cos(et_i)*sin(et_r)
	T_2 = -1*a_s*cos(et_i)*sin(et_r) + a_p * sin(et_i)*cos(et_r)
	T_3 = a_s*sin(et_i)*sin(et_r) + a_p * cos(et_i)*cos(et_r)
	return np.array([T_0,T_1,T_2,T_3])

def Mueller(T):
	M_00 = np.absolute(T[0])**2 + np.absolute(T[1])**2 + np.absolute(T[2])**2 + np.absolute(T[3])**2
	M_01 = np.absolute(T[0])**2 + np.absolute(T[1])**2 - np.absolute(T[2])**2 - np.absolute(T[3])**2
	M_02 = T[0]*np.conj(T[2]) + np.conj(T[0]*np.conj(T[2])) + T[1]*np.conj(T[3]) + np.conj(T[1]*np.conj(T[3]))
	M_03 = 1j*(T[2]*np.conj(T[0]) - np.conj(T[2]*np.conj(T[0]))) + 1j*(T[3]*np.conj(T[1]) + np.conj(T[3]*np.conj(T[1])))

	M_10 = np.absolute(T[0])**2 - np.absolute(T[1])**2 + np.absolute(T[2])**2 - np.absolute(T[3])**2
	M_11  = np.absolute(T[0])**2 + np.absolute(T[1])**2 - np.absolute(T[2])**2 + np.absolute(T[3])**2
	M_12  = T[0]*np.conj(T[2]) + np.conj(T[0]*np.conj(T[2])) - T[1]*np.conj(T[3]) - np.conj(T[1]*np.conj(T[3]))
	M_13  = 1j*(T[2]*np.conj(T[0]) - np.conj(T[2]*np.conj(T[0]))) - 1j*(T[3]*np.conj(T[1]) + np.conj(T[3]*np.conj(T[1])))

	M_20 = T[0]*np.conj(T[1]) + np.conj(T[0]*np.conj(T[1])) + T[2]*np.conj(T[3]) + np.conj(T[2]*np.conj(T[3]))
	M_21 = T[0]*np.conj(T[1]) + np.conj(T[0]*np.conj(T[1])) - T[2]*np.conj(T[3]) - np.conj(T[2]*np.conj(T[3]))
	M_22 = T[0]*np.conj(T[3]) + np.conj(T[0]*np.conj(T[3])) + T[2]*np.conj(T[1]) + np.conj(T[2]*np.conj(T[1]))
	M_23 = 1j*(T[2]*np.conj(T[1]) - np.conj(T[2]*np.conj(T[1]))) - 1j*(T[0]*np.conj(T[3]) - np.conj(T[0]*np.conj(T[3])))

	M_30 = 1j*(T[0]*np.conj(T[1]) - np.conj(T[0]*np.conj(T[1]))) + 1j*(T[2]*np.conj(T[3]) - np.conj(T[2]*np.conj(T[3])))
	M_31 = 1j*(T[0]*np.conj(T[1]) - np.conj(T[0]*np.conj(T[1]))) - 1j*(T[2]*np.conj(T[3]) - np.conj(T[2]*np.conj(T[3])))
	M_32 = 1j*(T[0]*np.conj(T[3]) - np.conj(T[2]*np.conj(T[1]))) + 1j*(T[0]*np.conj(T[3]) - np.conj(T[0]*np.conj(T[3])))
	M_33 = 1j*(T[0]*np.conj(T[1]) - np.conj(T[0]*np.conj(T[1]))) - 1j*(T[2]*np.conj(T[3]) + np.conj(T[2]*np.conj(T[3])))
	M = np.array( [ [M_00,M_01,M_02,M_03]  ,  [M_10,M_11,M_12,M_13]  ,  [M_20,M_21,M_22,M_23]  , [M_30,M_31,M_32,M_33] ])
	return M/2
 
def beta(th_i, th_r, phi):
	co_b=cos(th_i)*cos(th_r)+sin(th_i)*sin(th_r)*cos(phi)
	return arccos(co_b)/2

def eta(th_i,th_r,b):
	co_et_i = ((cos(th_i)+cos(th_r))/(2*cos(b))-cos(th_i)*cos(b))/(sin(th_i)*sin(b))
	co_et_r = ((cos(th_i)+cos(th_r))/(2*cos(b))-cos(th_r)*cos(b))/(sin(th_r)*sin(b))
	return arccos(co_et_i),arccos(co_et_r)

def theta(th_i,th_r,b):
	co = (cos(th_i)+cos(th_r))/(2*cos(b))
	return arccos(co)
    
def f(th_i,th_r,phi):
	#f = np.ones((4,4),dtype=complex)
	b = beta(th_i, th_r, phi)
	th = theta(th_i,th_r,b)
	f = exp(-(tan(th)**2)/(2*sigma**2))\
	/(2*pi*4*(sigma**2)*(cos(th)**4)*cos(th_r)*cos(th_i))\
	*Mueller(Jonex_Matrix(th_i, th_r, phi))
	return f

N = 90
th_r1d = np.linspace(0,np.pi/2,N) # 0 - 90 degrés
phi1d = np.linspace(0,np.pi/2,N) # 0 - 90 degrés
th_i = np.pi/3 # 60 degrés

th_r,phi = np.meshgrid(th_r1d, phi1d) # x ,y
sigma = 0.15
n_t = 1.57
n_i = 1
F = f(th_i,th_r,phi)

f00 = F[0,0]*cos(th_r)
plt.contour(th_r,phi,f00) # x , y , z
plt.xlabel("\thetar")
plt.show()
