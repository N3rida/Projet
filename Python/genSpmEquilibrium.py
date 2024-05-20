import numpy as np
from matplotlib import pyplot as plt


sigma,beta = 10,3
rhoStar =  sigma*((sigma + beta + 3)/(sigma - beta -1))
rhol = [.5,10,20,30,40,50]
C1 = [6,4,2]

Splus = lambda rho : [np.sqrt(beta*(rho - 1)),np.sqrt(beta*(rho - 1)),rho-1]
Smoins = lambda rho : [-np.sqrt(beta*(rho - 1)),-np.sqrt(beta*(rho - 1)),rho-1]

#data
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
DATA = np.array([np.zeros((nbInterv,3)) for i in range(len(rhol))])
DATA[:,0] = [C1,C1,C1,C1,C1,C1]

for r in range(len(rhol)):
    rho = rhol[r]
    for i in range(1,nbInterv):
        Yout = RK4(i*h,h,DATA[r,i-1],Lorenz)
        DATA[r,i] = Yout

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
# fig = plt.figure(figsize=(9,6),  facecolor= '#E5E5E5')
# axs = fig.add_subplot((3,2))
fig,axs = plt.subplots(3,2,figsize=(12,18),layout = 'constrained', subplot_kw={'projection': '3d'}, linewidth = .5)

i = 0
for ax in axs.flat:
    rho = rhol[i]
    #camera (ax)
    ax.azim = -50
    ax.elev = 10

    #plot lims
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    ax.set_zlim(0,50)
    #
    fig.suptitle(f"Solution du système de Lorenz en fonction de $\\rho$ \n avec $(\\sigma,\\beta)=$({sigma},{beta}) fixé, et $\\rho^*=${((rhoStar*100)//1)/100} ")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.set_title(f"$\\rho=${rho}")
    #equilibrium
    try:
        ax.plot(Splus(rho)[0],Splus(rho)[1],Splus(rho)[2],'ro')
        ax.plot(Smoins(rho)[0],Smoins(rho)[1],Smoins(rho)[2],'ro')
    except:
        print('Root')
    ax.plot(DATA[i,:,0],DATA[i,:,1],DATA[i,:,2],color = '#0000FF', alpha=.75)
    i += 1
plt.savefig("spm.png")
plt.show()