import numpy as np 
from matplotlib import pyplot as plt 
plt.rcParams.update({
    'text.usetex' : True
})

sigma,rho,beta = 10,1,1 #Lorenz 10,28,8/3

def Lorenz(t,X):
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

Total = 40
h = .01
nbInterv = int(Total/h)
Y = np.zeros((nbInterv,3))
Y[0] = [0,1,1.05]

t = np.linspace(0,Total,nbInterv)

for i in range(1,nbInterv):
    Yout = RK4(i*h,h,Y[i-1],Lorenz)
    Y[i] = Yout

#context
plt.style.use('ggplot')
fig = plt.figure(layout = 'constrained', facecolor= '#E5E5E5')
ax = fig.add_subplot(projection='3d')
ax.grid(True)

#camera
ax.azim = -50
ax.elev = 10

#legend 
ax.set_xlabel(r'Axe $x$')
ax.set_ylabel(r'Axe $y$')
ax.set_zlabel(r'Axe $z$')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_axis_off()
StrTitle = f"Trajectoire de la solution de condition initiale $x_0=(6,4,2)$  \n avec $(\\sigma,\\rho,\\beta)=({sigma},{rho},{beta})$"
ax.set_title(StrTitle, wrap = True)

#solution
ax.plot3D(Y[:,0],Y[:,1],Y[:,2], color = 'blue') #Graphe de la solution aprochée

#condition initiales
ax.plot(6,4,2,color = 'grey' , marker = 'o') #pts de départ de la sol° 
ax.text(8,4,0,r'$(6,4,2)$', color = 'grey')
# plt.savefig('pic.png', format='png', dpi=600, transparent=True)
plt.show()