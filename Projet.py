import numpy as np
import matplotlib.gridspec as gridspec
from Crank_Nicolson import WaveFunction
import Util as ut
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import time, sys, os
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import Slider


plt.rcParams.update({'font.size': 7})



#####################################
#       Creer le systeme        #
#####################################
# specifier temps, steps et duration
dt = 0.005
wavelength = 500e-9
FPS=30
start_time = 0
compteur = 0
duration = 10

# constants
hbar = 6.62607015e-34   # planck's constant
m = 9.10938356e-31      # particle mass

# spécifier un axe en coordonnées x
x_min = -12
x_max = 12
dx = 0.08
x = np.arange(x_min, x_max+dx, dx)

# spécifier un axe en coordonnées y
y_min = x_min
y_max = x_max
dy = dx
y = np.arange(y_min, y_max+dy, dy)

######## Créer des points à toutes les coordonnées x y dans la maille ########
N = 300   #dimension en nombre de points de la simulation
x_axis = np.linspace(x.min(),x.max(),N)
y_axis = np.linspace(y.min(),y.max(),N)
X, Y = np.meshgrid(x_axis, y_axis)

#Creer une barriere avec fentes
#Potentiel de la barriere
V_Wall = 1e10

#Partie basse de la barriere 
x01 = 0
xf1 = 0.3 # épaisseur de la barrière
y01 = y.min()
yf1 = -1.5 # taille de la fente = 2*|yf1|

#Pour créer une deuxième fente
x0m = x01
xfm = xf1
y0m = -0.5
yfm = 0.5

#Partie haute de la barriere
x02 = x01
xf2 = xf1
y02 = -yf1
yf2 = y.max()

# en cas d'une deuxième fente décommenter (la partie commentée du potentiel V)

V = ut.potentiel_heaviside(V_Wall,x01,xf1,y01,yf1,x,y) + ut.potentiel_heaviside(V_Wall,x02,xf2,y02,yf2,x,y) + ut.potentiel_heaviside(V_Wall,x0m,xfm,y0m,yfm,x,y)

#V = np.zeros(len(x)*len(y))

#Spécifier le paramètre du paquet gaussien initial
x0 = -5  # position du paquet d'onde sur l'axe X
y0 = 0
kx0 = 20  #Nombre d'onde du paquet sur l'axe X
#kx0 = 20
ky0 = 0
delta_x = 1.5
delta_y = 1.5

######## Creer et initialiser le paquet d'ondes ########
size_x = len(x)
size_y = len(y)
xx, yy = np.meshgrid(x,y)
psi_0 = ut.gauss_pqt(xx, yy, delta_x, delta_y, x0, y0, kx0, ky0).transpose().reshape(size_x*size_y)

# Définir les parametres de l'equation de Schrodinger pour effectuer les calculs
wave_fct = WaveFunction(x=x, y=y, psi_0=psi_0, V=V, dt=dt, hbar=hbar,m=m)
wave_fct.psi = wave_fct.psi/wave_fct.compute_norm()

######################################
#    Mise en place de graphiques     #
######################################

#
nb_frame = 300
nbr_level = 200

#Crer la figure
fig = plt.figure(figsize=(11,8))
gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1.5], height_ratios=[1,0.1,1])
ax1 = plt.subplot(gs[:,:-1])
ax2 = plt.subplot(gs[0,-1],projection='3d')
ax3 = plt.subplot(gs[2,-1])
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', '3%', '3%')

#Composantes du graphique
ax1.set_aspect(1)
ax1.set_xlim([x_min,x_max])
ax1.set_ylim([y_min,y_max])
ax1.set_xlabel(r"x ($a_0$)", fontsize = 16)
ax1.set_ylabel(r"y ($a_0$)", fontsize = 16)

ax2.view_init(elev=40., azim=-25.)
ax2.set_aspect('auto')
ax2.set_xlim([x_min,x_max])
ax2.set_ylim([y_min,y_max])
ax2.set_xlabel(r"x ($a_0$)", fontsize = 9)
ax2.set_ylabel(r"y ($a_0$)", fontsize = 9)

ax3.set_xlim([y_min, y_max])
ax3.set_xlabel(r"y ($a_0$)", fontsize = 9)
ax3.set_ylabel(r"$|\psi(y,t)|^2$", fontsize = 9)

#graphique initial
t = 0
proba = wave_fct.get_prob().reshape(size_x,size_y).transpose()

level = np.linspace(0,proba.max(),nbr_level)
cset = ax1.contourf(xx, yy, proba, levels=level, cmap=plt.cm.jet, zorder=1)

#Dessiner la barriere des fentes
ax1.text(0.02, 0.92, r"t = 0.0000 s ".format(wave_fct.t), color='white', transform=ax1.transAxes, fontsize=12)
ax1.vlines(x01, y01, yf1, colors='white', zorder=2)
ax1.vlines(xf1, y01, yf1, colors='white', zorder=2)
ax1.vlines(x02, y02, yf2, colors='white', zorder=2)
ax1.vlines(xf2, y02, yf2, colors='white', zorder=2)
ax1.hlines(yf1, x01, xf1, colors='white', zorder=2)
ax1.hlines(y02, x01, xf1, colors='white', zorder=2)
ax1.hlines(y0m, x0m, xfm, colors='white', zorder=2)
ax1.hlines(yfm, x0m, xfm, colors='white', zorder=2)
ax1.vlines(x0m, y0m, yfm, colors='white', zorder=2)
ax1.vlines(xfm, y0m, yfm, colors='white', zorder=2)

# Deuxieme graphique
zi = griddata((xx.reshape(size_x*size_y), yy.reshape(size_x*size_y)), proba.reshape(size_x*size_y), (x_axis[None,:], y_axis[:,None]), method='cubic')
ax2.plot_surface(X, Y, zi, cmap=plt.cm.jet, rcount=N, ccount=N, alpha=0.95)
#ax2.grid(False)
#ax2.plot_surface(xx, yy, proba, cmap=plt.cm.jet, zorder=1,rcount=75,ccount=75,antialiased=False)
z_i = 0.0
ax2.plot([x01,xf1,xf1,x01,x01], [y01,y01,yf1,yf1,y01], z_i*np.ones(5), color='k', linewidth=2, zorder=2, alpha=1.)
ax2.plot([x02,xf2,xf2,x02,x02], [y02,y02,yf2,yf2,y02], z_i*np.ones(5), color='k', linewidth=2, zorder=2, alpha=1.)
ax2.plot([x0m,xfm,xfm,x0m,x0m], [y0m,y0m,yfm,yfm,y0m], z_i*np.ones(5), color='k', linewidth=2, zorder=2, alpha=1.)

#Troisieme graphique
scr_distance = x_max/2
k = abs(x-scr_distance).argmin()
ax3.plot(yy[:,k],proba[:,k])
#ax3.set_ylim([0, proba[:,k].max()+0.01])
ax3.set_ylim([0, 0.23])
ax1.vlines(x[k], y_min, y_max, colors='orange', linestyle='dashed', zorder=2)

#Configurer la colorbar
cbar1 = fig.colorbar(cset, cax=cax1)
major_ticks = np.linspace(0,4*proba.max(),50)
ticks = ut.intervalle(proba.max(), 0, major_ticks)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels(ticks)

t_vec = np.arange(0,nb_frame*dt,dt)
coupe = np.zeros((nb_frame,len(proba[:,k])))

#Creer l'animation
def animate(i):
    t = t_vec[i]
    wave_fct.step()
    proba = wave_fct.get_prob().reshape(size_x,size_y).transpose()
    coupe[i] = proba[:,k]

    ax1.clear()
    ax2.clear()
    ax3.clear()

    #Graphiques
    #Premier graphe
    level = np.linspace(0,proba.max(),nbr_level)
    cset = ax1.contourf(xx, yy, proba, levels=level, cmap=plt.cm.jet,zorder=1)
    ax1.set_xlabel(r"x ($a_0$)", fontsize = 16)
    ax1.set_ylabel(r"y ($a_0$)", fontsize = 16)
    #Deuxieme graphe
    zi = griddata((xx.reshape(size_x*size_y), yy.reshape(size_x*size_y)), proba.reshape(size_x*size_y), (x_axis[None,:], y_axis[:,None]), method='cubic')
    ax2.plot_surface(X, Y, zi, cmap=plt.cm.jet, rcount=N, ccount=N, alpha=0.95)
    ax2.set_zlim([0,zi.max()])
    ax2.set_xlabel(r"x ($a_0$)", fontsize = 9)
    ax2.set_ylabel(r"y ($a_0$)", fontsize = 9)
    ax2.set_xlim([x_min,x_max])
    ax2.set_ylim([y_min,y_max])
    #ax2.grid(False)
    #Troisieme grpahe
    ax3.plot(yy[:,k],proba[:,k])
    ax3.set_xlim([y_min, y_max])
    ax3.set_ylim([0, 0.23])
    ax3.set_xlabel(r"y ($a_0$)", fontsize = 9)
    ax3.set_ylabel(r"$|\psi(y,t)|^2$", fontsize = 9)

    #Dessiner la barriere des fentes
    ax1.text(0.02, 0.92, r"t = {0:.3f} s".format(wave_fct.t), color='white', transform=ax1.transAxes, fontsize=12)
    ax1.vlines(x01, y01, yf1, colors='white', zorder=2)
    ax1.vlines(xf1, y01, yf1, colors='white', zorder=2)
    ax1.vlines(x02, y02, yf2, colors='white', zorder=2)
    ax1.vlines(xf2, y02, yf2, colors='white', zorder=2)
    ax1.hlines(yf1, x01, xf1, colors='white', zorder=2)
    ax1.hlines(y02, x01, xf1, colors='white', zorder=2)
    ax1.vlines(x0m, y0m, yfm, colors='white', zorder=2)
    ax1.vlines(xfm, y0m, yfm, colors='white', zorder=2)
    ax1.hlines(y0m, x0m, xfm, colors='white', zorder=2)
    ax1.hlines(yfm, x0m, xfm, colors='white', zorder=2)
    ax2.plot([x01,xf1,xf1,x01,x01], [y01,y01,yf1,yf1,y01], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([x02,xf2,xf2,x02,x02], [y02,y02,yf2,yf2,y02], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([x0m,xfm,xfm,x0m,x0m], [y0m,y0m,yfm,yfm,y0m], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax1.vlines(x[k], y_min, y_max, colors='orange', linestyle='dashed', zorder=2)

    #Adjuster la colorbar
    cbar1 = fig.colorbar(cset, cax=cax1)
    ticks = ut.intervalle(proba.max(), 0, major_ticks)
    cbar1.set_ticks(ticks)
    cbar1.set_ticklabels(ticks)
    

    print_update()

def print_update():
    global compteur

    NORM = wave_fct.compute_norm()
    os.system('cls')
    rapport = compteur / (duration * FPS)
    M = 20
    k = int(rapport * M)
    l = M - k
    Progression = '[' + k * '#' + l * '-' + ']   {0:.3f} %'

    d_time = time.time() - start_time

    print('--- Simulation en cours ---')
    print(Progression.format(rapport * 100))
    print('Temps écoulé : {0:.1f} s'.format(d_time))
    if rapport > 0:
        print('Temps restant estimé : {0:.1f} s'.format(d_time / rapport - d_time))
    print('Norme de la fonction : {0:.3f} '.format(NORM))
    compteur += 1
    while rapport==1:
        break
start_time = time.time()
interval = 0.001
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
anim = animation.FuncAnimation(fig,animate,nb_frame,interval=interval*1e+3,blit=False, repeat=False)
writer = animation.PillowWriter(fps=FPS)
anim.save('2D_2slit_dx={0}_dt={1}_yf1={2}_k={3}.gif'.format(dx,dt,abs(yf1),kx0), writer=writer)
plt.show()

########### Cette partie du script évalue l'impact des parametres 
########### de l'experience de Young sur l'intensité reçu à l'ecran
########### !!!!!! Afin de bien simuler l'experience une mise à l'échelle et appliquée !!!!!!!

slit_width = (yf1-y0m)*(10**-7)
screen_distance = scr_distance*(10**-1)
distance_between_slits= (yfm-y0m)*10**-3

Y = ut.double_fentes_intensite(slit_width, wavelength, screen_distance, distance_between_slits, X)
plot, = plt.plot(X,Y)
plt.xlabel("Distance from center")
plt.ylabel("Intensity")

axis=(plt.axes([0.75, 0.75, 0.14, 0.05]))
axis2 = (plt.axes([0.75,0.65, 0.14, 0.05]))
axis3 = (plt.axes([0.75,0.55, 0.14, 0.05]))
axis4 = (plt.axes([0.75,0.45, 0.14, 0.05]))

wavelength_slider = Slider(axis,'Wavelength(nm)',100, 1000,valinit=wavelength*10**9)
slit_width_slider = Slider(axis2, "Slit Width(micrometers)", 10, 1000, valinit=slit_width*10**6)
screen_distance_slider = Slider(axis3, "Screen Distance(cm)", 10, 100, valinit= screen_distance*10**2)
distance_between_slits_slider = Slider(axis4, "Distance b/w slits(mm)", 0.1, 10, valinit=distance_between_slits*10**3) 

def update(val) :
  wavelength = wavelength_slider.val*(10**-9)
  slit_width = slit_width_slider.val*(10**-6)
  screen_distance = screen_distance_slider.val*(10**-2)
  distance_between_slits = distance_between_slits_slider.val*(10**-3)
  Y = ut.double_fentes_intensite(slit_width, wavelength, screen_distance, distance_between_slits, X)
  plot.set_ydata(Y)

wavelength_slider.on_changed(update)
slit_width_slider.on_changed(update)
screen_distance_slider.on_changed(update)
distance_between_slits_slider.on_changed(update)

plt.show()