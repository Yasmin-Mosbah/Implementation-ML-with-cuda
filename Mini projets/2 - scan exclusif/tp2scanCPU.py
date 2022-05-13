import math
import numpy as np


#Écrivez une fonction scanCPU-special(array) qui prend en paramètre 
#un tableau numpy de taille n=2^m et effectue les phases de montée et 
#descente pour calculer le prefix exclusif.
#Vérifiez bien chaque étape du calcul en faisant afficher le tableau
a = [2,3,4,6]
def scanCPU(a):
    n = len(a)
    m = int(math.log(n,2))

    # print("MONTEE")
    for d in range(0,(m)):
        for k in range(0,(n-1),(2**(d+1))):
            a[k+(2**(d+1))-1] += a[k+(2**d)-1]

    # print("RESULTAT DE LA MONTEE : ", a)
    # print("===========================================")
    # print("DESCENTE")

    a[n-1]=0
    # print(a)
    for d in range(m-1,-1,-1):
        for k in range(0,n,(2**(d+1))):
            t= a[k+(2**d)-1]
            a[k+(2**d)-1] = a[k+(2**(d+1))-1]
            a[k+(2**(d+1))-1] += t
    #print(a)
    return a

# Écrivez une fonction scanCPU_sub(array, index_min, index_max)
# qui calcul SCAN sur le sous-tableau array[index_min, index_max]

def scanCPU_sub(a, index_min, index_max):
    n = index_max - index_min +1
    m = int(math.log(n,2))

    print("MONTEE")
    for d in range(0,(m)):
        for k in range(0,(n-1),(2**(d+1))):
            val = k+index_min
            a[val+(2**(d+1))-1] += a[val+(2**d)-1]
    
    print("RESULTAT DE LA MONTEE : ", a)

    print("_____________________________")
    print("DESCENTE")
    a[index_max] = 0
    print("Index max converti en zero :", a)
    for d in range(m-1,-1,-1):

        for k in range(0,n,(2**(d+1))):
            val = k+index_min
            t= a[val+(2**d)-1]
            a[val+(2**d)-1] = a[val+(2**(d+1))-1]
            a[val+(2**(d+1))-1] += t
    print("_____________________________")
    return a
    #print("RESULTAT DE LA DESCENTE : ", a)

#scanCPU_sub([2,3,4,6,10,2,1,5,20,14], 0, 7)

#À l'aide de la fonction précédente, implémentez le calcul SCAN 
# sur un tableau de taille arbitraire. Une stratégie possible 
# est de diviser le tableau initial en sous-tableaux de taille 
# "bien choisie" et ajouter des éléments neutres à la fin du dernier sous-tableau

a = [2,3,4,6,5,10,14,2,10]

def scanCPU_sub2(a, index_min, index_max):
    n = index_max - index_min +1
    m = int(math.log(n,2))

    print("MONTEE")
    for d in range(0,(m)):
        for k in range(0,(n-1),(2**(d+1))):
            val = k+index_min
            a[val+(2**(d+1))-1] += a[val+(2**d)-1]
    
    lastmontee = a[index_max]
    print("RESULTAT DE LA MONTEE : ", a)
    print("Valeur récupérer pour le sous tableau", lastmontee)
    print("_____________________________")
    print("DESCENTE")
    a[index_max] = 0
    print("Index max converti en zero :", a)
    for d in range(m-1,-1,-1):
        for k in range(0,n,(2**(d+1))):
            val = k+index_min
            t= a[val+(2**d)-1]
            a[val+(2**d)-1] = a[val+(2**(d+1))-1]
            a[val+(2**(d+1))-1] += t
    print("_____________________________")
    return a, lastmontee

def deuxpuissance(n):
    return [1 << i for i in range(n)]

def scanCPU_all(tab,div):
    somtab = []
    if div <= len(tab):
        if len(tab) % div == 0 :
            for i in range(0,len(tab),div):
                res = scanCPU_sub2(tab, i, i+div-1)
                tab = res[0]
                somtab.append(res[1])
        else :
            for j in range(div-1):
                tab.append(0)
            for i in range(0,len(tab),div):
                res = scanCPU_sub2(tab, i, i+div-1)
                tab = res[0]
                somtab.append(res[1])
    else :
        print("ERREUR")

    while len(somtab) not in deuxpuissance(len(somtab)):
        somtab.append(0)
    somtab = scanCPU(somtab)

    print("sous tableau :", somtab)
    for z in range(0,len(tab),div):
        for u in range(div):
            tab[z+u]=tab[z+u] v+somtab[z//div]

    print(tab)
    return tab


# def scanCPU_seq(a, index_min = 0):
#     t = 0
#     index_min = 0
#     index_max = None
#     index_max = index_max if index_max != None else len(a)
#     for i in range(index_min, index_max):
#         t += a[i]
#         a[i] = t - a[i]
#     print(a)
#     return a

a =[2,3,4,6]
scanCPU_all(a,2)

#a = [10,20,14,50,2,14,12,13]
# scanCPU_seq(a)



# def scanCPU_all(tab,div):
#     somtab = [0]*(len(tab)//div +1)
#     if div <= len(tab):
#         if len(tab) % div == 0 :
#             for i in range(0,len(tab),div):
#                 res = scanCPU_sub2(tab, i, i+div-1)
#                 tab = res[0]
#                 somtab[i] = res[1]
#         else :
#             for j in range(div-1):
#                 tab.append(0)
#             for i in range(0,len(tab),div):
#                 res = scanCPU_sub2(tab, i, i+div-1)
#                 tab = res[0]
#                 print(somtab)
#                 somtab[i//div] = res[1]
#     else :
#         print("ERREUR")

#     while len(somtab) not in problem(len(somtab)):
#         somtab.append(0)

#     somtab = scanCPU(somtab)

#     for z in range(0,len(tab),div):
#         for u in range(div):
#             tab[z+u]=tab[z+u]+somtab[z//div]

#     print(tab)
#     return tab

# a = [10,20,14,50,2,14,12,13,50]
# scanCPU_all(a,2)