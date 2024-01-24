import numpy as np 
from matplotlib import pyplot as plt 
from scipy.integrate import solve_ivp

sigma,rho,beta = 10,28,8/3

def Lorenz(t,X:np.ndarray(3)):
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

Total = 30
h = .01
epsilon = .01
nbInterv = int(Total/h)
t = np.linspace(0,10,nbInterv)
Y = np.zeros((nbInterv,3))
Y[0] = [6,4,2]

Ye = np.zeros((nbInterv,3))
Ye[0] = Y[0] + epsilon

for i in range(1,nbInterv):
    Yout = RK4(i*h,h,Y[i-1],Lorenz)
    Yeout = RK4(i*h,h,Ye[i-1],Lorenz)
    Y[i] = Yout
    Ye[i] = Yeout

ax = plt.figure().add_subplot(projection='3d')
ax.plot3D(Y[:,0],Y[:,1],Y[:,2], color = 'red')
ax.plot3D(Ye[:,0],Ye[:,1],Ye[:,2], color = 'green')
ax.plot(Y[-1,0],Y[-1,1],Y[-1,2], color = 'red', marker = 'o')
ax.plot(Ye[-1,0],Ye[-1,1],Ye[-1,2], color = 'green', marker = 'o')
ax.plot(6,4,2,color = 'grey' , marker = 'o')
plt.show()