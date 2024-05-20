import numpy as np
from matplotlib import pyplot as plt
import matplotlib 

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plt.rcParams.update({
    'text.usetex' : True
})

#Params
sigma = 10
rho = 28
beta = 2.666
C1 = [6,4,2]
epsilon = .1
C2 = [i + epsilon for i in C1]

#DATA
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

Total = 30
h = .01
nbInterv = int(Total/h)
Y1 = np.zeros((nbInterv,3))
Y2 = np.zeros((nbInterv,3))
Y1[0] = C1
Y2[0] = C2

X = np.linspace(0,Total,nbInterv)

for i in range(1,nbInterv):
    Yout1 = RK4(i*h,h,Y1[i-1],Lorenz)
    Y1[i] = Yout1
    Yout2 = RK4(i*h,h,Y2[i-1],Lorenz)
    Y2[i] = Yout2

distance = [np.linalg.norm(Y1[i]-Y2[i]) for i in range(len(Y1))]

#Graph param & init
plt.style.use('seaborn-v0_8-notebook')
plot_settings = {'ytick.labelsize': 16,
                        'xtick.labelsize': 16,
                        'font.size': 26,
                        'axes.titlesize': 26,
                        'axes.labelsize': 16,
                        'legend.fontsize': 22,
                        'mathtext.fontset': 'stix',
                        'font.family': 'STIXGeneral'}
plt.style.use(plot_settings)

fig = plt.figure(figsize= (12,6),layout="constrained")
ax = fig.add_subplot(1,2,1,projection='3d')
norm = fig.add_subplot(1,2,2)
ax.grid(True)

#camera (ax)
ax.azim = -50
ax.elev = 10

#legend 
ax.set_xlabel(r'Axe $x$')
ax.set_ylabel(r'Axe $y$')
ax.set_zlabel(r'Axe $z$')

norm.set_xlabel(r'Temps $t$')

StrTitle = f"Trajectoire de la solution du système de Lorenz\n avec $(\\sigma,\\rho,\\beta)=({sigma},{rho},{beta})$"
ax.set_title(StrTitle, wrap = True)
norm.set_title('Distance des solutions en fonction du temps')

#plot 
ax.plot(Y1[:,0],Y1[:,1],Y1[:,2], color = '#0000FF', label ='Solution sans perturbation')
ax.plot(Y2[:,0],Y2[:,1],Y2[:,2], color = '#FF0000',label = f"Solution avec perturbation $\\varepsilon = $ {epsilon}")
ax.plot([], [], ' ', label=f"Condtion initiale $x_0 = ${C1}")
norm.plot(X,distance, color = '#0000FF', label ='Distance entre les solutions à l\'instant $t$')
#norm.plot(X,epsilon*np.ones(len(X)), color = '#FF0000', marker = "-", alpha = .8, label= f"$\\varepsilon = $ {epsilon}")


ax.legend()
norm.legend()
plt.show()