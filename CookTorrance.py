import numpy as np
from numpy import exp
from sklearn import preprocessing

m = np.array([1e-3,3e-1,1e-1])
f0 = np.array([0,1,1e-1])
m = np.array([1e-3,3e-1,1e-1])
m = np.array([1e-3,3e-1,1e-1])

include_F = True
include_G = True

sqr = lambda x : x*x

def Beckmann(m,t):
    M = m*m
    T = t*t
    return exp((T-1)/(M*T)) / (M*T*T)

def Fresnel(f0,u):
    return f0 + (1-f0) * pow(1-u, 5)

'''def normalize(v):

   length_of_v = sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
   return vec3(v.x / length_of_v, v.y / length_of_v, v.z / length_of_v)
'''

def BRDF (L,V,N,X,Y):
    H = preprocessing.normalize(L+V, norm='l2')
    NdotH  = N@H
    VdotH  = V@H
    NdotL  = N@L
    NdotV  = N@V
    oneOverNdotV =  1.0/NdotV
 
    D = Beckmann(m, NdotH)
    F = Fresnel(f0, VdotH)
 
    NdotH = NdotH + NdotH
 
    if (NdotV < NdotL):
        if (NdotV*NdotH < VdotH) :
            G = NdotH / VdotH
        else :
            G = oneOverNdotV
    else:
        if (NdotL*NdotH < VdotH):
            G = NdotH*NdotL / (VdotH*NdotV)
        else :
            G = oneOverNdotV

    if (include_G):
        G = oneOverNdotV
    val = 1.0*(NdotH > 0)*D * G

    if (include_F):
        val = F

    val = val / NdotL
    print(np.shape(val))
    return val

L = np.array([[1]])
V = np.array([[1]])
N = np.array([[1]])
X = np.array([[1]])
Y = np.array([[1]])
Z = np.array([[1]])
import matplotlib.pyplot as plt
plt.plot(BRDF(L, V, N, X, Y))


    
