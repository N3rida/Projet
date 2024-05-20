import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    'text.usetex' : True
})

#DATA
X = np.linspace(-.2,4.4,150)
f = lambda x : -.3*(x-1)**3 + (x-1)**2 -.1*(x-1) + .6
Y = f(X)

plt.style.use('seaborn-v0_8-notebook')
plot_settings = {'ytick.labelsize': 16,
                        'xtick.labelsize': 16,
                        'font.size': 22,
                        'axes.titlesize': 26,
                        'axes.labelsize': 16,
                        'legend.fontsize': 22,
                        'mathtext.fontset': 'stix',
                        'font.family': 'STIXGeneral'}
plt.style.use(plot_settings)
fig = plt.figure(figsize=(15,6),layout='constrained')
fig.suptitle('Example d\'estimation d\'aire par les méthodes des rectangles')

axG = plt.subplot(1,3,1)
axG.set_xlim([-.5,5])
axG.set_ylim([-.5,3])
axG.set_xticks([])
axG.set_yticks([])
axG.grid()
axG.set_title('Méthode des rectangles à gauche')
axG.spines['left'].set_position('zero')
axG.spines['bottom'].set_position('zero')
axG.spines['right'].set_color('none')
axG.spines['top'].set_color('none')
axG.plot(X,Y, color = 'blue')
axG.plot([3,3],[0,f(3)],color='red')
axG.text(3,-.1,r"$a$", horizontalalignment='center', verticalalignment='top')
axG.plot([4,4],[0,f(3)],color='red')
axG.text(4,-.1,r"$b$", horizontalalignment='center', verticalalignment='top')
axG.plot([3,4],[f(3),f(3)],color='red')
axG.plot([0,3],[f(3),f(3)],'r--')
axG.text(-0.1,f(3),r"$f(a)$", horizontalalignment='right', verticalalignment='center')
axG.fill_between([3,4],[0,0],[f(3),f(3)],color = 'none',hatch = '..',edgecolor='red',label='approximation de l\'aire')


axD = plt.subplot(1,3,3)
axD.set_xlim([-.5,5])
axD.set_ylim([-.5,3])
axD.set_xticks([])
axD.set_yticks([])
axD.grid()
axD.set_title('Méthode des rectangles à droite')
axD.spines['left'].set_position('zero')
axD.spines['bottom'].set_position('zero')
axD.spines['right'].set_color('none')
axD.spines['top'].set_color('none')
axD.plot(X,Y, color = 'blue')
axD.plot([3,3],[0,f(4)],color='red')
axD.text(3,-.1,r"$a$", horizontalalignment='center', verticalalignment='top')
axD.plot([4,4],[0,f(4)],color='red')
axD.text(4,-.1,r"$b$", horizontalalignment='center', verticalalignment='top')
axD.plot([3,4],[f(4),f(4)],color='red')
axD.plot([0,3],[f(4),f(4)],'r--')
axD.text(-.1,f(4),r"$f(b)$", horizontalalignment='right', verticalalignment='center')
axD.fill_between([3,4],[0,0],[f(4),f(4)],color = 'none',hatch = '..',edgecolor='red',label='approximation de l\'aire')

axM = plt.subplot(1,3,2)
axM.set_xlim([-.5,5])
axM.set_ylim([-.5,3])
axM.set_xticks([])
axM.set_yticks([])
axM.grid()
axM.set_title('Méthode des rectangles millieu')
axM.spines['left'].set_position('zero')
axM.spines['bottom'].set_position('zero')
axM.spines['right'].set_color('none')
axM.spines['top'].set_color('none')
axM.plot(X,Y, color = 'blue')
axM.plot([3,3],[0,f(3.5)],color='red')
axM.text(3,-.1,r"$a$", horizontalalignment='center', verticalalignment='top')
axM.plot([4,4],[0,f(3.5)],color='red')
axM.text(4,-.1,r"$b$", horizontalalignment='center', verticalalignment='top')
axM.plot([3.5,3.5],[0,f(3.5)],'r--')
axM.text(3.5,-.1,r"$\frac{a+b}{2}$", horizontalalignment='center', verticalalignment='top')
axM.plot([3,4],[f(3.5),f(3.5)],color='red')
axM.plot([0,3],[f(3.5),f(3.5)],'r--')
axM.text(-.1,f(3.5),r"$f(\frac{a+b}{2})$", horizontalalignment='right', verticalalignment='center')
axM.fill_between([3,4],[0,0],[f(3.5),f(3.5)],color = 'none',hatch = '..',edgecolor='red',label='approximation de l\'aire')

plt.legend()
plt.show()