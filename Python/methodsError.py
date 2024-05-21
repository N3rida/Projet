import numpy as np
from matplotlib import pyplot as plt 
plt.rcParams.update({
    'text.usetex': True
})

#Euler method
def EulerStep(f,tn,yn,h):
    tnp1 = tn + h 
    ynp1 = yn + h*f(tn,yn)
    return tnp1,ynp1

def Euler(f,t0,tf,y0,N):
    h = (tf - t0)/ N
    Y = np.zeros(N+1)
    T = np.zeros(N+1)
    Y[0],T[0] = y0,t0
    for i in range(1,N+1):
        T[i],Y[i] = EulerStep(f,T[i-1],Y[i-1],h)
    return T,Y

#RK4 method
def RK4Step(f,tn,yn,h):
    tnp1 = tn + h
    k1 = f(tn,yn)
    k2 = f(tn + (h/2),yn + (h/2)*k1)
    k3 = f(tn + (h/2),yn + (h/2)*k2)
    k4 = f(tn + h, yn + h*k3)
    Yout = yn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return tnp1,Yout

def RK4(f,t0,tf,y0,N):
    h = (tf - t0)/ N
    Y = np.zeros(N+1)
    T = np.zeros(N+1)
    Y[0],T[0] = y0,t0
    for i in range(1,N+1):
        T[i],Y[i] = RK4Step(f,T[i-1],Y[i-1],h)
    return T,Y

# on considère u' = -u avec u(0) = 1, la solution est u(t) = e^{-t}
y = lambda t : np.exp(-t)
f = lambda t,y : -y
t0,tf = 0,1
y0 = 1
N = np.logspace(1,6)

EulerError = []
RK4Error = []

#calcul des erreurs
for n in N:
    TEuler,YEuler = Euler(f,t0,tf,y0,int(n))
    EulerError.append(abs(YEuler-y(TEuler)).max())    
    TRK4,YRK4 = RK4(f,t0,tf,y0,int(n))
    RK4Error.append(abs(YRK4-y(TRK4)).max())

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

fig = plt.figure(figsize= (12,6))
ax = fig.add_subplot(1,1,1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([10,N[-1]])
ax.set_ylim([10**(-16),.1])
ax.set_title("Comparaison de l'erreur des méthodes en fonction \n du nombre de subdivisions $N$ (échelle logarithmique)")
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$e^N$")

ax.plot(N,EulerError, color ='red', label = 'Erreur de la méthode \n d\'Euler explicite' )
ax.plot([10,N[-1]],[.1,N[-1]**(-1)], label = "Droite de pente 1", color = 'orange')

ax.plot(N,RK4Error, color = 'blue', label = " Erreur de la méthode \n de Runge-Kutta 4")
ax.plot([10,10**5],[10**(-4),10**(-5*4)], label = "Droite de pente 4", color = 'cyan')

box = ax.get_position()
ax.set_position([box.x0,box.y0,box.width*.7,box.height])

plt.legend(loc='center left', bbox_to_anchor = (1,.5))
plt.show()