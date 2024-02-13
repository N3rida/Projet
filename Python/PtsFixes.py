import numpy as np 
from matplotlib import pyplot as plt 
from scipy import linalg

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

Total = 100
h = .01

Ptfixe = np.array([np.sqrt(beta*(rho-1)),np.sqrt(beta*(rho-1)),rho - 1])
Ptfixe = np.array([0,0,0])

# nbInterv = int(Total/h)
# Y = np.zeros((nbInterv,3))
# Y[0] = Ptfixe + .01 

# t = np.linspace(0,Total,nbInterv)

# for i in range(1,nbInterv):
#     Yout = RK4(i*h,h,Y[i-1],Lorenz)
#     Y[i] = Yout

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot3D(Y[:,0],Y[:,1],Y[:,2], color = 'red') #Graphe de la solution aprochée
# ax.plot(Ptfixe[0],Ptfixe[1],Ptfixe[2],color = 'grey' , marker = 'o') #pts de départ
# plt.show()

#differentielle
def Diff(x,y,z):
    return np.array([[- sigma,sigma,0],[rho-z, -1, -x],[ y, x, -beta]])

A = Diff(Ptfixe[0],Ptfixe[1],Ptfixe[2])

Y0 = np.array([1,1,1])
Y = np.zeros((int(10/.1),3))
t = np.linspace(0,10,int(10/.1))

for i in t:
    Y[int(i)] = linalg.expm(A*i).dot(Y0)

print(Y)
#plt.plot(Y[:,0],Y[:,1],Y[:,2])
#plt.show()