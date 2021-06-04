# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes
import numpy as np
import pandas as pd
import math
import copy
import sys
import graphviz as gv

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        somme = 0
        for x in range(0, len(desc_set)):
            if self.predict(desc_set[x]) == label_set[x]:
                somme += 1
        return (somme / len(label_set))


# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        w = np.zeros(input_dimension)
        for i in range(w.size):
            w[i] = np.random.rand()
        self.w = w
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur.")
    
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        prod_scal = x.dot(self.w)
        return prod_scal
   
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pred = self.score(x)
        if pred > 0:
            pred = 1
        elif pred < 0:
            pred = -1
        return pred    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k = k
        self.data_ref = np.zeros(input_dimension)       # on mettra a valeur correcte grâce à train
        self.dimension = input_dimension
        self.label_set = np.zeros(input_dimension)      # on mettra a valeur correcte grâce à train



    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        # construction du tableau des distances
        dist = np.zeros(self.data_ref[:, 0].size)         # on crée un vecteur ligne avec autant de composantes que de points dans le data_set de référence
        for i in range(dist.size):
            d = 0
            for j in range(self.dimension):
                d += (x[j] - self.data_ref[i][j])**2
            d = math.sqrt(d)
            dist[i] = d
        
        # tri du tableau du plus proche au moins proche (ordre croissant)
        ind = np.argsort(dist)
        inter = np.zeros(dist.size)
        for i in range(dist.size):
            inter[i] = dist[ind[i]]
        dist = inter

        # calcul de la proportion de +1 parmis les k-plus proches voisins et renvoie de cette valeur
        prop = 0
        for i in range(self.k):
            if self.label_set[ind[i]] == 1:
                prop += 1
        prop /= self.k
       
        return prop


    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        prop_1 = self.score(x)
        if prop_1 > 0.5:
            return 1
        if prop_1 <= 0.5:
            return -1



    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.data_ref = desc_set
        self.label_set = label_set
 # ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.epsilon = learning_rate
        w = np.concatenate([-np.random.random(input_dimension//2), np.random.random(input_dimension - input_dimension//2)])     # autant de valeurs aléatoires positives que négatives
        w = w / 10**3                                                                                                         # on s'assure que les valeurs soint très petites
        np.random.shuffle(w)                                                                                                  # on mélange aléatoirement le tableau
        self.w = w
        self.history = history
        self.allw = []
        

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        n_iter_max = 10000
        sur_place_max = 100
        cpt_sur_place = 0
        iterations = 0

        # Évaluation et éventuelle correction de w
        while iterations < n_iter_max and cpt_sur_place < sur_place_max:
            iterations += 1
            pos_alea = np.random.randint(0, label_set.size)
            x = desc_set[pos_alea]
            y = label_set[pos_alea]
            score = self.score(x)
            if not(score * y > 1):                       # si les deux solutions sont de signes opposés
                self.w += x * (y - score) * self.epsilon
                if (self.history):
                    self.allw.append(np.array(self.w))
                if (cpt_sur_place != 0):
                    cpt_sur_place = 0
            else:
                cpt_sur_place += 1

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res = x.dot(self.w)
        return res
    

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if score > 0:
            return 1
        return -1
 # ---------------------------
class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.epsilon = learning_rate
        w = np.concatenate([-np.random.random(input_dimension//2), np.random.random(input_dimension - input_dimension//2)])
        w = w / 10**3
        np.random.shuffle(w)
        self.w = w
        self.noyau = noyau
        self.w_ker = self.noyau.transform(self.w)


    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        x_ker = self.noyau.transform(x)
        #w_ker = self.noyau.transform(self.w)
        res = x_ker.dot(self.w_ker)
        return res
    


    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if score > 0:
            return 1
        return -1



    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        n_iter_max = 10000
        sur_place_max = 1000
        cpt_sur_place = 0
        iterations = 0

        # Évaluation et éventuelle correction de w
        while iterations < n_iter_max and cpt_sur_place < sur_place_max:
            iterations += 1
            pos_alea = np.random.randint(0, label_set.size)
            x = desc_set[pos_alea]
            y = label_set[pos_alea]
            score = self.score(x)
            if not(score * y > 0):                       # si les deux solutions sont de signes opposés
                self.w_ker += self.noyau.transform(x) * (y - score) * self.epsilon
                if (cpt_sur_place != 0):
                    cpt_sur_place = 0
            else:
                cpt_sur_place += 1
# ------------------------ 
class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """

    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.dim = input_dimension
        self.epsilon = learning_rate
        self.history = history
        self.iMax = niter_max
        w = np.concatenate([-np.random.random(input_dimension//2), np.random.random(input_dimension - input_dimension//2)])     # autant de valeurs aléatoires positives que négatives
        w = w / 10**3                                                                                                         # on s'assure que les valeurs soint très petites
        np.random.shuffle(w)                                                                                                  # on mélange aléatoirement le tableau
        self.w = w
        self.allw = []

        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        iterations = 0

        while iterations < self.iMax:
            iterations += 1
            pos_alea = np.random.randint(0, len(label_set))
            x = desc_set[pos_alea]
            y = label_set[pos_alea]
            w = self.w
            score = self.score(x)
            grad = x * (score - y)
            w += -self.epsilon * grad
            
            if self.history:
                self.allw.append(np.array([w[0], w[1]]))
    

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return x.dot(self.w)

    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if score > 0:
            return 1
        return -1

# ------------------------
class ClassifierMultiOAA(Classifier):

    def __init__(self, classifier, nbC):
        """
        Construit un classifieur multi-classe à partir de ses arguments.
        classifier: un objet classifier dont on va faire la copie profonde
        nbC: le nombre de classes dans notre problème
        """
        self.nbC = nbC
        self.classifs = []
        for i in range(self.nbC):
            c = copy.deepcopy(classifier)
            self.classifs.append(c)
    

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        res = []
        for i in range(self.nbC):
            res.append(self.classifs[i].score(x))
        return res
    

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.argmax(self.score(x))
    

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        for i in range(self.nbC):
           train_labels = np.where(label_set == i, 1, -1)
           self.classifs[i].train(desc_set, train_labels)

# -----------------------------------------------
def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)
    return valeurs[np.argmax(nb_fois)]


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


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
    
        entropies = np.zeros(len(X[0]))     # le tableau des entropies conditionnelles
        for i in range(len(X[0])):          # pour chaque variable Xj
            vals = np.unique(X[:, i])
            ent = 0
            for val in vals:                # pour chaque valeur possible de Xj
                descs = X[X[:, i] == val]
                labels = Y[X[:, i] == val]
                ent += (len(descs) / len(X)) * entropie(labels)
            entropies[i] = ent
        
        i_best = entropies.argmin()
        min_entropie = entropies[i_best]
        vals = np.unique(X[:, i_best])
        Xbest_valeurs = [val for val in vals]
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)

# ----------------------------------------------
def discretise(desc, labels, col):
    """ array * array * int -> tuple[float, float]
        Hypothèse: les 2 arrays sont de même taille et contiennent au moins 2 éléments
        col est le numéro de colonne à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation: (import sys doit avoir été fait)
    min_entropie = sys.float_info.max  # on met à une valeur max car on veut minimiser 
    min_seuil = 0.0     
    
    # trie des valeurs: ind contient les indices dans l'ordre croissant des valeurs pour chaque colonne
    ind = np.argsort(desc,axis=0)
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    # Dictionnaire pour compter les valeurs de classes qui restera à voir
    Avenir_nb_class = dict()
    # et son initialisation: 
    for j in range(0, len(desc)):
        if labels[j] in Avenir_nb_class:
            Avenir_nb_class[labels[j]] += 1
        else:
            Avenir_nb_class[labels[j]] = 1
    
    # Dictionnaire pour compter les valeurs de classes que l'on a déjà vues
    Vues_nb_class = dict()
    
    # Nombre total d'exemples à traiter:
    nb_total = 0  
    for c in Avenir_nb_class:
        nb_total += Avenir_nb_class[c]
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(desc)-1):
        v_ind_i = ind[i]   # vecteur d'indices de la valeur courante à traiter
        courant = desc[v_ind_i[col]][col]  # valeur courante de la colonne
        lookahead = desc[ind[i+1][col]][col] # valeur suivante de la valeur courante
        val_seuil = (courant + lookahead) / 2.0; # Seuil de coupure: entre les 2 valeurs
        
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if labels[v_ind_i[col]] in Vues_nb_class:
            Vues_nb_class[labels[v_ind_i[col]]] += 1
            
        else:
            Vues_nb_class[labels[v_ind_i[col]]] = 1
        # on retire de l'avenir:
        Avenir_nb_class[labels[v_ind_i[col]]] -= 1
        
        # construction de 2 listes: ordonnées sur les mêmes valeurs de classes
        # contenant le nb d'éléments de chaque classe
        nb_inf = [] 
        nb_sup = []
        tot_inf = 0
        tot_sup = 0
        for (c, nb_c) in Avenir_nb_class.items():
            nb_sup.append(nb_c)
            tot_sup += nb_c
            if (c in Vues_nb_class):
                nb_inf.append(Vues_nb_class[c])
                tot_inf += Vues_nb_class[c]
            else:
                nb_inf.append(0)
        
        # calcul de la distribution des classes de chaque côté du seuil:
        freq_inf = [nb/float(tot_inf) for nb in nb_inf]
        freq_sup = [nb/float(tot_sup) for nb in nb_sup]
        # calcul de l'entropie de la coupure
        val_entropie_inf = ut.shannon(freq_inf)
        val_entropie_sup = ut.shannon(freq_sup)
        
        val_entropie = (tot_inf / float(tot_inf+tot_sup)) * val_entropie_inf \
                       + (tot_sup / float(tot_inf+tot_sup)) * val_entropie_sup
        # Ajout de la valeur trouvée pour l'historique:
        liste_entropies.append(val_entropie)
        liste_coupures.append(val_seuil)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie), (liste_coupures,liste_entropies,)


# ----------------------------------------------
def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    score = 0
    for i in range(len(DS)):
        desc = copy.deepcopy(DS[0])
        lab = copy.deepcopy(DS[1])
        test_desc = desc[i]
        test_lab = lab[i]
        desc = np.delete(desc, i, axis=0)
        lab = np.delete(lab, i, axis=0)
        
        algo = copy.deepcopy(C)
        algo.train(desc, lab)
        if algo.predict(test_desc) == test_lab:
            score += 1
    
    return score / len(DS)