import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.colors
import cv2
import os

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

sigma,rho,beta = 10,28,8/3
#champs vectoriel
def Lorenz(t,X):
    Xout = np.zeros(3)
    Xout[0] = sigma*(X[1] - X[0])
    Xout[1] = rho*X[0] - X[1] - X[0]*X[2]
    Xout[2] = X[0]*X[1] - beta*X[2]
    return Xout
#méthode de résolutions
def RK4(tn,h,Y:np.array,f):
    k1 = f(tn,Y)
    k2 = f(tn + (h/2),Y + (h/2)*k1)
    k3 = f(tn + (h/2),Y + (h/2)*k2)
    k4 = f(tn + h, Y + h*k3)
    Yout = Y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(Yout)

def tailGradient(colorA,colorB,length,tailLength):
    core = np.zeros(length-tailLength if length > tailLength else 0)
    tail = np.ones(tailLength if length > tailLength else length)
    step = 1/len(tail) if length != 0 else 0
    for i in range(len(tail)):
        tail[i] = i*step
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("tail{}{}".format(colorA,colorB),[colorA,colorB])
    classes = np.concatenate((core,tail))
    return(classes,cmap)

Total = 30                                  #Temps total de la simulation
h = .01                                     # \Delta t


nbInterv = int(Total/h)                     #nombres d'intervalles de temps \Delta t

Y = np.zeros((nbInterv,3))                  #initialisation de la trajectoire
Y1,Y2,Y3 =np.zeros((nbInterv,3)),np.zeros((nbInterv,3)),np.zeros((nbInterv,3))

Y[0] = [6,4,2]                              #Condition initiale
Y1[0],Y2[0],Y3[0] = [6.1,4.1,3.1],[6.01,4.01,2.01],[6.001,4.001,2.001]

t = np.linspace(0,Total,nbInterv)

for i in range(1,nbInterv):                 #calcul de la trajectoire
    Yout = RK4(i*h,h,Y[i-1],Lorenz)
    Y1out = RK4(i*h,h,Y1[i-1],Lorenz)
    Y2out = RK4(i*h,h,Y2[i-1],Lorenz)
    Y3out = RK4(i*h,h,Y3[i-1],Lorenz)
    Y[i] = Yout
    Y1[i],Y2[i],Y3[i] = Y1out,Y2out,Y3out

#couleurs
A= 'blue'
B= '#FF0000'

### PARTIE ANIMATION
FPS = 60
step = 1
#Endroit de sortie de la video
path_out_video = r"C:\Users\chabe\Documents\L3\PROJET\Projet\Python\Lorenz2.mp4"
#Source des images
path_IMG = r"C:\Users\chabe\Documents\L3\PROJET\Projet\Python\IMG"
#Type d'encodage de video voir OpenCV doc pour + d'infos
vid_fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
#on construit l'objet video
video = cv2.VideoWriter(path_out_video,fourcc=vid_fourcc,fps=FPS, frameSize= [1000,1000])

# Genere chaque image de la video
for n in range(int(nbInterv/step)):
    #met a jour les points a tracer dans la video
    x = Y[:FPS*n,0]  
    y = Y[:FPS*n,1]  
    z = Y[:FPS*n,2]  
    
    ### Dessine les points
    fig = plt.figure(figsize=(10, 10))              #initialise la figure (taille Width, height in inches.)
    ax = fig.add_subplot(111, projection='3d')      #cree les axes de même proportion en 3D
    ax.set_xticks([])                               #etablie la facon de numeroter les axes
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-30,30)                             #Domaine de definition du graphe des solutions
    ax.set_ylim(-30,30)
    ax.set_zlim(0,50)
    ax.plot( Y[step*n,0],Y[step*n,1] ,Y[step*n,2], marker = 'o', color = 'blue' )               #Trace les points dans la configuration etablie
    ax.plot(Y1[step*n,0],Y1[step*n,1] ,Y1[step*n,2], color = 'red', marker = 'o')
    ax.plot(Y2[step*n,0],Y2[step*n,1] ,Y2[step*n,2], color = 'orange', marker = 'o')
    ax.plot(Y3[step*n,0],Y3[step*n,1] ,Y3[step*n,2], color = 'yellow', marker = 'o')
    ax.plot(Y[:,0],Y[:,1],Y[:,2], color = 'blue', alpha = .2)


    plt.savefig(f"Python/IMG/{n}.png")              #Enregistre les images une a une 
    plt.close()                                     #Ferme l interface graphique
    #encode l image
    if os.path.exists(path_IMG + f"\\{n}.png"):     #si l image existe on la rajoute a la prochaine frame de la video
        video.write(cv2.imread(path_IMG + f"\\{n}.png"))
    
    if os.path.exists( path_IMG + f"\\{n-1}.png"):  #supprime la n-1 image si elle existe
        os.remove(path_IMG + f"\\{n-1}.png")

    #Compteur de temps restent
    progress = n/(nbInterv/step) *100
    if round(progress)%5 == 0:
        print(f"{progress}% done")

### ferme l objet video
video.release()

if os.path.exists(path_IMG + f"\\{nbInterv-1}.png"): #supprime la dernière image
        os.remove(path_IMG + f"\\{nbInterv-1}.png")