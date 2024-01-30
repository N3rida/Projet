import numpy as np 
from matplotlib import pyplot as plt 

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
nbInterv = int(Total/h)
Y = np.zeros((nbInterv,3))
Y[0] = [6,4,2]

t = np.linspace(0,Total,nbInterv)

for i in range(1,nbInterv):
    Yout = RK4(i*h,h,Y[i-1],Lorenz)
    Y[i] = Yout

ax = plt.figure().add_subplot(projection='3d')
ax.plot3D(Y[:,0],Y[:,1],Y[:,2], color = 'red') #Graphe de la solution aprochée
ax.plot(6,4,2,color = 'grey' , marker = 'o') #pts de départ
plt.show()