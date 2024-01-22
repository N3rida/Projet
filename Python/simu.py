import numpy as np 
from matplotlib import pyplot as plt 

def Lorenz(t,X,sigma,rho,beta):
    t,
    Xout = np.ndarray(3)
    Xout[0] = sigma*(X[0]-X[1])
    Xout[1] = rho*X[0] - X[1] - X[0]*X[2]
    Xout[2] = X[0]*X[1] - beta*X[2]
    return Xout

def RK4(t,h,Y,f):
    k1 = f(t,Y)
    k2 = f(t + (h/2.),Y + (h/2.)*k1)
    k3 = f(t + (h/2.),Y + (h/2.)*k2)
    k4 = f(t + h, Y + h*k3)
    Yout = Y + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
    return Yout