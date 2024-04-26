import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plt.rcParams.update({
    'text.usetex' : True
})

#roots
Rplus = lambda x: 1-2*x + np.sqrt(((4*x-2)**2)/4 -1)
Rminus = lambda x: 1-2*x - np.sqrt(((4*x-2)**2)/4 -1)
#domains
D1 = np.linspace(-3,0,90)
D2 = np.linspace(1,3,60)

plt.style.use('ggplot')
fig = plt.figure(figsize= (5,3),layout = 'compressed', facecolor= '#E5E5E5')
ax = fig.add_subplot(1, 1, 1)
ax.grid(True)

#spine placement data centered
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

RPD1 =Rplus(D1)
RPD2 = Rplus(D2)
RMD1 = Rminus(D1)
RMD2 = Rminus(D2)
limsupD2 = [10 for i in D2]
limsupD1 = [10 for i in D1]
liminfD1= [-6 for i in D1]
liminfD2= [-6 for i in D2]

#Delta = 0
ax.plot(D1,RPD1,linewidth = 2, color = '#7F007F', label= r'$\Delta = 0$')
ax.plot(D2,RPD2,linewidth = 2, color = '#7F007F')

ax.plot(D1,RMD1,linewidth = 2, color = '#7F007F')
ax.plot(D2,RMD2,linewidth = 2, color = '#7F007F')
#Delta < 0
ax.fill_between(D1, RPD1,RMD1,facecolor = '#0000FF',alpha=.2)
ax.fill_between(D2, RPD2,RMD2,facecolor = '#0000FF',alpha=.2)
#Delta > 0
ax.fill_between(D1, RMD1,liminfD1 , facecolor = '#FF0000' ,alpha=.2)
ax.fill_between(D2, RPD2,limsupD2 , facecolor = '#FF0000' ,alpha=.2)
ax.fill_between(D1, RPD1,limsupD1 , facecolor = '#FF0000' ,alpha=.2)
ax.fill_between(D2, RMD2,liminfD2 , facecolor = '#FF0000' ,alpha=.2)
ax.fill_between([0,1],[-6,-6],[10,10], facecolor = '#FF0000' ,alpha=.2)
#text
ax.text(-2,2,r"$\Delta < 0$",fontsize = 15)
ax.text(2,-3.2,r"$\Delta < 0$",fontsize = 15)
ax.text(1,2,r"$\Delta > 0$",fontsize = 15)

ax.set_xlim([-3,3])
ax.set_ylim([-5,5])

#plt.title(r"Signe de $\Delta$ en fonction de $\rho$ en abscisse et de $\sigma$ en ordonné", fontsize = 33)
ax.legend(fontsize = 15)
plt.show()