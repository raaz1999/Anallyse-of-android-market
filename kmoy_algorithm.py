import random
import numpy as np
import matplotlib.pyplot as plt


'''
Algorithme des k-moyennes 

version 2.0
tested : no 
 

'''


def normalisation(X):
    min_val = np.min(X)
    max_val = np.max(X)
    v = (X-min_val) / (max_val-min_val)
    return v

def dist_vect(a,b):
    return np.linalg.norm(a-b)

def affiche_resultat(d,c,m):
    
    
    for i in range(len(m)):
        l=[]
        for j in m[i]:
            l.append(j)

        l=np.array(l)
        plt.scatter(l[:,0],l[:,1],marker='x')
    
    plt.scatter(c[:,0],c[:,1],color='k',marker='v')
    return

def affiche_resultat2(d,c,m):
    
    
    for i in range(len(m)):
        l=[]
        for j in m[i]:
            l.append(d[j])

        l=np.array(l)
        plt.scatter(l[:,0],l[:,1],marker='x')
    
    plt.scatter(c[:,0],c[:,1],color='k',marker='v')
    return

def centroide(data): 
    return np.array(data).mean(axis=0)
    
def inertie_cluster(data):
    centre=centroide(data)
    inertie=0
    for i in data:
        #print("centre: ",str(centre)," 	Exemple:  ",str(i)," 	distance =",str(dist_vect(centre,i)))
        inertie+= dist_vect(centre,i)
    return inertie

def initialisation(k,data):
    l=[]
    for i in range(k):
        l.append(random.choice(data))
    l=np.array(l)
    
    return l

def plus_proche(x,kcenter):

    l=[]
    
    for i in kcenter:
        l.append(dist_vect(x,i))
    
    return np.argmin(l,axis=0)

def affecte_cluster(dn,cs):
    dicte={}

    for i in range(len(cs)):
        dicte[i]=[]
    

    for i in range(len(dn)):
        dicte[plus_proche(dn[i],cs)].append(i)

    return dicte

def nouveaux_centroides(x,dico):
    l=[]
    for i in x:
        l.append(i)
    
    return centroide(l)

def inertie_globale(dn,da):
    ig=0
    for i in range(len(da)):
        l=[]
        for j in da[i]:
            l.append(dn[j])
        
        ig+=inertie_cluster(l)
    return ig

def kmoyennes(k,db,ep,iter):
    inertg=0
    kmoy= initialisation(k,db)
    dn=db
    while iter!=0:
        m=affecte_cluster(dn,kmoy)
        ij=inertie_globale(dn,m)
        if abs(ij-inertg)<ep:
            print("inersie :",inertg)
            return np.array(kmoy),m
            break
        else:
            inertg=ij
            iter+=1
            kmoy=[]
            for i in range(len(m)):
                l=[]
                for j in m[i]:
                    l.append(db[j])

                kmoy.append(centroide(l))




'''
class KMoy():

    def __init__(self,dimension,k,epsilon):

        
        #@param(dimension type<int>) la dimension ou le nombre de colonnes de notre donnée
        #@param(k type<int> ) le nombre de groupes parmis nos données qu'on veut determine 
        
        #Constructeur pour initialiser nos variables 

        


        self.e=epsilon    #notre cueille
        self.dim=dimension
        self.k=k
        self.kmoy=[]    # cette liste contiendra les coordonnée des k centres de clusters
        self.groups=[]     # cette liste contiendra k listes tel que chaque liste représente un groupe de points qui contient les points les plus proche de notre k centre 
        self.J=0     #c'est pour calculer l'inertie de notre ensemble des k-moyenne

    def train(self,desc_set):
        
        #@param(desc_set type<np.array>)liste de donnée 

        


        print(desc_set[0])
        if len(self.kmoy)==0:   #si on n'a pas encore definit ou initialisé les k moyennes
            for i in range(self.k):   # on choisi k points aleatoire de notre base de donnée 
                self.kmoy.append(random.choice(desc_set))   # et on commence avec eux notre étude 
            
            


        while True:     # la boucle principale qui va itéré jusqu'a ce que nos k moyennes se stabilise (ne change plus de coordonnée)
            
            self.groups=[ [] for i in range(self.k)]    # vidé notre liste des points les plus proche de chaque points 
            
            inertie=0
            
            for i in desc_set :     
                dist=[]     
                for j in range(self.k):  # pour chaque exemple de notre data set on calcule la distance entre cette donnée et les k centres
                    dist.append(np.linalg.norm(i-self.kmoy[j]))   # elle sont deja ordonnée le centre numéro 1 puis 2 ....

                self.groups[np.argmin(dist,axis=0)].append(i) # on atribue notre exemple au cluster a qui il est le plus proche de lui  
                inertie += min(dist)    #


            if abs(self.J-inertie)<=self.e:
                break
            else:
                self.J=inertie



            for i in range(self.k):     # claculer la moyenne de chaque groupes 
                moy=np.zeros(self.dim)
                for j in self.groups[i]:
                    moy=moy+j
                
                moy=moy*(1/(len(self.groups[i])))
                    
                if self.kmoy[i].all()!=moy.all():   # si il y'a une difference en modifie notre k moyenne et on passe a une autre itération
                    self.kmoy[i]=moy
                    



        return 


    def predict(self,x):
        
        @param(x) a simple from our data base 
        
        dist=[]

        for i in range(self.k):  # calculer la distance entre X et chaque centre des k moyennes 
              dist.append(np.linalg.norm(x-self.kmoy[i]))
            

        return np.argmin(dist,axis=0)   # return le centre le plus proche par le numero de son indice 
    

    def test_qualite(self):
        return 

'''
        
