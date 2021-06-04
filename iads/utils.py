# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import cm

# ------------------------    
def plot2DSet(desc, lab):
    """ ndarray * ndarray -> affichage
    la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    positif = desc[lab == 1]
    negatif = desc[lab == -1]

    plt.scatter(positif[:,0], positif[:,1], marker='o', c='#0000FF')
    plt.scatter(negatif[:,0], negatif[:,1], marker='x', c='#FF0000')
    plt.grid(True)
    

def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])    

    
# ------------------------ 
def gener_dataset_uniform(p, n, inf, sup):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    if (n % 2 != 0):
        print("Erreur : la valeur attendue de n est paire")
    desc = np.random.uniform(inf, sup, (n, p))      # on crée le tableau de données
    labels = np.asarray([-1 for i in range(n//2)] + [1 for i in range (n - n//2)])  # on génère le tableau des labels équi-réparti
    np.random.shuffle(labels)   # on mélange ce tableau de manière aléatoire
    return desc, labels
    
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    labels = np.asarray([-1. for i in range(nb_points)] + [1. for i in range(nb_points)])
    negatifs = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    positifs = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    data_set = np.vstack((negatifs, positifs))
    return data_set, labels


# ------------------------
def create_XOR(n, sigma):
    labels = np.asarray([-1. for i in range(2*n)] + [1. for i in range(2*n)])

    centre = np.array([0, 0])
    centre_hg = centre + np.array([-100*sigma, 100*sigma])
    centre_hd = centre_hg + np.array([200*sigma, 0])
    centre_bg = centre + np.array([-100*sigma, -100*sigma])
    centre_bd = centre_bg + np.array([200*sigma, 0])

    grp1 = np.random.multivariate_normal(centre_hg, np.array([[sigma, 0], [0, sigma]]), n)
    grp2 = np.random.multivariate_normal(centre_bd, np.array([[sigma, 0], [0, sigma]]), n)
    grp3 = np.random.multivariate_normal(centre_bg, np.array([[sigma, 0], [0, sigma]]), n)
    grp4 = np.random.multivariate_normal(centre_hd, np.array([[sigma, 0], [0, sigma]]), n)
    
    data_set = np.vstack((grp1, grp2, grp3, grp4))
    return data_set, labels


 # ------------------------ 
def plot_frontiere_V3(desc_set, label_set, w, kernel, step=30, forme=1, fname="out/tmp.pdf"):
    """ desc_set * label_set * array * function * int * int * str -> NoneType
        Note: le classifieur linéaire est donné sous la forme d'un vecteur de poids pour plus de flexibilité
    """
    # ETAPE 1: construction d'une grille de points sur tout l'espace défini par les points du jeu de données
    mmax=desc_set.max(0)                                                                                            # mmax = tableau des maximum de chaque colonnes
    mmin=desc_set.min(0)                                                                                            # mmin = tableau des minimum de chaque colonnes
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))                  # on crée les différents points de la grille
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))                                   # on présente tous les points sous la forme d'un tableau 2 x n de points (x, y)

    #
    # Si vous avez du mal à saisir le concept de la grille, décommentez ci-dessous
    #plt.figure()
    #plt.scatter(grid[:,0],grid[:,1])
    #if True:
        #return
    #
    # ETAPE 2: calcul de la prediction pour chaque point de la grille
    res=np.array([kernel(grid[i,:])@w for i in range(len(grid)) ])                                                 # on calcule la prédiction pour chaque point
    # pour les affichages avancés, chaque dimension est présentée sous la forme d'une matrice
    res=res.reshape(x1grid.shape) 
    #
    # ETAPE 3: le tracé
    #
    # CHOIX A TESTER en décommentant:
    # 1. lignes de contours + niveaux
    if forme <= 2 :
        fig, ax = plt.subplots() # pour 1 et 2
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
    if forme == 1:
        CS = ax.contour(x1grid,x2grid,res)
        ax.clabel(CS, inline=1, fontsize=10)
    #
    # 2. lignes de contour 0 = frontière 
    if forme == 2:
        CS = ax.contour(x1grid,x2grid,res, levels=[0], colors='k')
    #
    # 3. fonction de décision 3D
    if forme == 3 or forme == 4:
        fig = plt.gcf()
        ax = fig.gca(projection='3d') # pour 3 et 4
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
        ax.set_zlabel('f(X)')
    # 
    if forme == 3:
        surf = ax.plot_surface(x1grid,x2grid,res, cmap=cm.coolwarm)
    #
    # 4. fonction de décision 3D contour grid + transparence
    if forme == 4:
        norm = plt.Normalize(res.min(), res.max())
        colors = cm.coolwarm(norm(res))
        rcount, ccount, _ = colors.shape
        surf = ax.plot_surface(x1grid,x2grid,res, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
        surf.set_facecolor((0,0,0,0))
    #
    # ETAPE 4: ajout des points
    negatifs = desc_set[label_set == -1]     # Ensemble des exemples de classe -1
    positifs = desc_set[label_set == +1]     # +1 
    # Affichage de l'ensemble des exemples en 2D:
    if forme <= 2:
        ax.scatter(negatifs[:,0],negatifs[:,1], marker='o', c='b') # 'o' pour la classe -1
        ax.scatter(positifs[:,0],positifs[:,1], marker='x', c='r') # 'x' pour la classe +1
    else:
        # on peut ajouter une 3ème dimension si on veut pour 3 et 4
        ax.scatter(negatifs[:,0],negatifs[:,1], -1, marker='o', c='b') # 'o' pour la classe -1
        ax.scatter(positifs[:,0],positifs[:,1], 1,  marker='x', c='r') # 'x' pour la classe +1
    #
    # ETAPE 5 en 3D: régler le point de vue caméra:
    if forme == 3 or forme == 4:
        ax.view_init(20, 70) # a régler en fonction des données
    #
    # ETAPE 6: sauvegarde
    if fname != None:
        # avec les options pour réduires les marges et mettre le fond transprent
        plt.savefig(fname,bbox_inches='tight', transparent=True,pad_inches=0)

# ---------------------------------------------------
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    shan = 0
    k = len(P)
    if 1 in P:
        return 0.0
    for i in range(k):
        if (P[i] != 0):
            shan += P[i] * math.log(P[i], k)    
    return float(-shan)


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    labels_occur = {}
    for i in range(len(Y)):
        labels_occur[Y[i]] = labels_occur.get(Y[i], 0.) + 1.
    P = [(elem / len(Y)) for elem in labels_occur.values()]
    return shannon(P)
# ----------------------------------------------
def crossval_strat(X, Y, n_iterations, iteration):
    Xtest = []
    Ytest = []
    Xapp = []
    Yapp = []
    for i in range(len(X)):
        if (i % n_iterations == iteration):
            Xtest.append(X[i])
            Ytest.append(Y[i])
        else:
            Xapp.append(X[i])
            Yapp.append(Y[i])
    
    Xapp = np.asarray(Xapp)
    Yapp = np.asarray(Yapp)
    Xtest = np.asarray(Xtest)
    Ytest = np.asarray(Ytest)

    return Xapp, Yapp, Xtest, Ytest