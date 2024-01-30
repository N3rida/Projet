import numpy as np
from matplotlib import pyplot as plt

sigma = 10
rho = 28
beta = 8/3

def Lorenz(t,X:np.ndarray(3)):
    t = 0
    Xout = np.zeros(3)
    Xout[0] = sigma*(X[1] - X[0])
    Xout[1] = rho*X[0] - X[1] - X[0]*X[2]
    Xout[2] = X[0]*X[1] - beta*X[2]
    return Xout

def RK4(tn,h,Y:np.array,f):
    k1 = f(tn,Y)
    k2 = f(tn + (h/2),Y + (h/2)*k1)
    k3 = f(tn + (h/2),Y + (h/2)*k2)
    k4 = f(tn + h, Y + h*k3)
    Yout = Y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(Yout)

T = 10000
h = .1
epsilon = .1

nbInterv = int(T/h)

Y = np.zeros((nbInterv,3))
Y[0] = [6,4,2]

Y_e = np.zeros((nbInterv,3))
Y_e[0] = Y[0] + epsilon

delta = np.zeros(nbInterv)
delta[0] = np.sqrt(3)*epsilon

for i in range(1,nbInterv):
    Yout = RK4(i*h,h,Y[i-1],Lorenz)
    Y[i] = Yout
    # Yout_e = RK4(i*h,h,Y_e[i-1],Lorenz)
    # Y_e[i] = Yout_e

# for i in range(nbInterv):
    # delta[i] = NEuclid(Y[i] - Y_e[i])
    
# plt.plot(t,delta)
# plt.show()