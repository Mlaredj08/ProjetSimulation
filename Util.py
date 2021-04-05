import numpy as np

# Paquet d'ondes gaussien
def gauss_pqt(x, y, delta_x, delta_y, x0, y0, kx0, ky0):
    return 1/(2*delta_x**2*np.pi)**(1/4) * 1/(2*delta_y**2*np.pi)**(1/4) * np.exp(-((x-x0)/(2*delta_x)) ** 2) * np.exp(-((y-y0)/(2*delta_y)) ** 2) * np.exp( 1.j * (kx0*x + ky0*y))


#################################################
# Fonction Heaviside d'un potentiel Quadratique #
#################################################
# Un potentiel quadratique est appliqué aux bordure de l'image 
# Des conditions limites de reflexions sont donc appliqués 
# à l'onde une fois qu'elle atteint une de ces bordures
        
def potentiel_heaviside(V0, x0, xf, y0, yf, x, y):
    V = np.zeros(len(x)*len(y))
    size_y = len(y)
    for i,yi in enumerate(y):
        for j,xj in enumerate(x):
            if (xj >= x0) and (xj <= xf) and (yi >= y0) and (yi <= yf):
                V[i+j*size_y] = V0
            else:
                V[i+j*size_y] = 0
    return V

# iv) Model analytique
def analytic_model(x, y, a, x0, y0, kx0, ky0, t):
    sigma = np.sqrt(a**2 + t**2/(4*a**2))
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-x0-(kx0)*t)/sigma)**2) * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((y-y0-(ky0)*t)/sigma)**2)

def compute_err(z1,z2,x,y):
    return np.trapz(np.trapz(abs(z1-z2), x).real, y).real
    
# 
def intervalle(max_list,min_list,list_ref,n=3):
    return [round(i, -int(np.floor(np.log10(i))) + (n - 1))  for i in list_ref if (i < max_list) and (i > min_list) ]

def double_fentes_intensite (largeur_fente, longueur_onde, distance_ecran, distance_entre_fentes,  X) :
  """
    Prend en largeur de la fente, longueur d'onde, distance d'écran, distance entre les deux fentes et un tableau numpy X (un tableau de distances à partir du centre).
    Produit un tableau d'intensités normalisées correspondant à X.
  """
  return (((np.sin((np.pi*largeur_fente*X)/(longueur_onde*distance_ecran)))/((np.pi*largeur_fente*X)/(longueur_onde*distance_ecran)))**2)*((np.cos((np.pi*distance_entre_fentes*X)/(longueur_onde*distance_ecran)))**2)
