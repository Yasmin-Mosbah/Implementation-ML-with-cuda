from numba import cuda, int32
import numpy as np
import math

#Var global
g_blockDim = 32
g_threadsperblock = 32,32
g_m = int(math.log(g_blockDim,2)) 

## Code de l'article "Efficient Integral Imge Computation on the GPU" adapté en python
@cuda.jit
def transpose_kernel(mat_input,mat_output):
    #Memoire partagée
    sA = cuda.shared.array(g_threadsperblock, dtype=int32)

    #Id du block * la dimension du block + l'id du thread
    idx = cuda.blockIdx.x * g_threadsperblock[0] + cuda.threadIdx.x
    idy = cuda.blockIdx.y * g_threadsperblock[0]  + cuda.threadIdx.y

    #Largeur et hauteur de la matrice
    w = mat_input.shape[0]
    h = mat_input.shape[1]

    if idx < w and idy < h:
        sA[cuda.threadIdx.y][cuda.threadIdx.x] = mat_input[idx][idy]
    cuda.syncthreads()

    idx = cuda.blockIdx.y * g_threadsperblock[0] + cuda.threadIdx.x
    idy = cuda.blockIdx.x * g_threadsperblock[0]  + cuda.threadIdx.y

    if idx < h and idy < w:
        mat_output[idx][idy] = sA[cuda.threadIdx.x][cuda.threadIdx.y]

#Adaptation de la fct "transpose_kernel" sur une matrice contenant toutes les images du dataset
@cuda.jit
def transpose_kernel_All(mat_input,mat_output):
    #Memoire partagée
    sA = cuda.shared.array(g_threadsperblock, dtype=int32)

    #Id du block * la dimension du block + l'id du thread
    idx = cuda.blockIdx.x * g_threadsperblock[0] + cuda.threadIdx.x
    idy = cuda.blockIdx.y * g_threadsperblock[0]  + cuda.threadIdx.y
    z = cuda.blockIdx.z

    #Largeur et hauteur de la matrice
    w = mat_input.shape[1]
    h = mat_input.shape[2]

    if idx < w and idy < h:
        sA[cuda.threadIdx.y][cuda.threadIdx.x] = mat_input[z][idx][idy]
    cuda.syncthreads()

    idx = cuda.blockIdx.y * g_threadsperblock[0] + cuda.threadIdx.x
    idy = cuda.blockIdx.x * g_threadsperblock[0]  + cuda.threadIdx.y

    if idx < h and idy < w:
        mat_output[z][idx][idy] = sA[cuda.threadIdx.x][cuda.threadIdx.y]

#Correspond au scan kernel "special" du tp2 en version GPU et adapté a une matrice      
@cuda.jit
def scanKernel2D(a):
    #Mémoire partagée
    sA = cuda.shared.array(g_blockDim,dtype=int32)   
    thID = cuda.threadIdx.x 
    y = cuda.blockIdx.y

    #On envoie sur la mémoire partagée les données contenu dans la matrice qu'on désire traiter
    if thID < a.shape[1]:
        sA[thID] = a[y,thID]
    else :
        sA[thID] = 0
    cuda.syncthreads()

    #Montée 
    for d in range(0,g_m):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]
    cuda.syncthreads()
    #Descente
    sA[cuda.blockDim.x - 1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            t= sA[k+pow(2,d)-1]
            sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
            sA[k+(pow(2,d+1)-1)] += t
    cuda.syncthreads()

    #On renvoie les données de la memoire partagée sur la matrice
    if thID < a.shape[1] :
        a[y,thID] = sA[thID]

#C'est le "scan kernel sub" du TP2 en version GPU et adpaté a une matrice
@cuda.jit
def scanKernel_sA2D(a,somtab):
    #Initialisation de la mémoire partagée
    sA = cuda.shared.array(g_blockDim,dtype=int32)   
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.blockIdx.y
    #Envoie des données de la matrice sur la mémoire partagée
    if thID+offset < a.shape[1]:
        sA[thID] = a[y, thID+offset]
    else :
        sA[thID] = 0
    cuda.syncthreads()
    for d in range(0,g_m):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]
    cuda.syncthreads()

    #Remplissage du tableau intermédiaire
    #somtab[cuda.blockIdx.x,y] = int(sA[cuda.blockDim.x - 1])
    if thID == 0 : 
        somtab[y,cuda.blockIdx.x] = sA[g_blockDim -1]
    #Descente
        sA[g_blockDim -1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            t= sA[k+pow(2,d)-1]
            sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
            sA[k+(pow(2,d+1)-1)] += t
    cuda.syncthreads()

    #Récupération des données de la mémoire partagée dans la matrice
    if thID+offset < a.shape[1] :
        a[y,thID+offset] = sA[thID]

#Pareil que la fct précedente mais qui s'applique a toutes les images
@cuda.jit
def scanKernel2D_All(a):
    sA = cuda.shared.array(g_blockDim,dtype=int32)   
    thID = cuda.threadIdx.x 
    y = cuda.blockIdx.y
    z = cuda.blockIdx.z

    if thID < a.shape[1]:
        sA[thID] = a[z,y,thID]
    else :
        sA[thID] = 0
    cuda.syncthreads()

    for d in range(0,g_m):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]
    cuda.syncthreads()

    sA[cuda.blockDim.x - 1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            t= sA[k+pow(2,d)-1]
            sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
            sA[k+(pow(2,d+1)-1)] += t
    cuda.syncthreads()

    if thID < a.shape[1] :
        a[z,y,thID] = sA[thID]

@cuda.jit
def scanKernel_sA2D_All(a,somtab):
    sA = cuda.shared.array(g_blockDim,dtype=int32)   
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.blockIdx.y
    z = cuda.blockIdx.z

    if thID+offset < a.shape[2]:
        sA[thID] = a[z,y, thID+offset]
    else :
        sA[thID] = 0
    cuda.syncthreads()
    for d in range(0,g_m):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]
    cuda.syncthreads()

    #somtab[cuda.blockIdx.x,y] = int(sA[cuda.blockDim.x - 1])
    if thID == 0 : 
        somtab[z,y,cuda.blockIdx.x] = sA[g_blockDim -1]
        sA[g_blockDim -1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            t= sA[k+pow(2,d)-1]
            sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
            sA[k+(pow(2,d+1)-1)] += t
    cuda.syncthreads()
    if thID+offset < a.shape[2] :
        a[z,y,thID+offset] = sA[thID]

#Kernel qui réalise la somme entre le tableau2D initiale et intermédiaire 
@cuda.jit
def sum_el2D(tab,somtab):
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.blockIdx.y

    if thID+offset < tab.shape[1]:
        tab[y,offset+thID] += somtab[y,cuda.blockIdx.x]

#Le meme que précédement mais il s'applique a toutes les images
@cuda.jit
def sum_el2D_All(tab,somtab):
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.blockIdx.y
    z = cuda.blockIdx.z
    if thID+offset < tab.shape[1]:
        tab[z,y,offset+thID] += somtab[z,y,cuda.blockIdx.x]

#Scan avec un tableau 2D (matrice)       
def scan_sa2D(ar):
    tab = np.array(ar)
    tab_d = cuda.to_device(tab)
    #longueur/blockdim, largeur
    gridDim = int(math.ceil(tab.shape[1]/g_blockDim)),tab.shape[0]
    if tab.shape[1] <= g_blockDim :
        scanKernel2D[gridDim,g_blockDim](tab_d)
        cuda.synchronize()
        res = tab_d.copy_to_host()
        return res
    else :
        somtab = np.zeros(gridDim[::-1],dtype=int)
        somtab_d = cuda.to_device(somtab)
        scanKernel_sA2D[gridDim ,g_blockDim](tab_d,somtab_d)
        cuda.synchronize()
        somtab = somtab_d.copy_to_host()
        recur = scan_sa2D(somtab)
        recur_d = cuda.to_device(np.array(recur))
        sum_el2D[gridDim,g_blockDim](tab_d,recur_d)
        res = tab_d.copy_to_host()
        return res

#Scan 2D avec toutes les images
def scan_sa2D_All(ar):
    tab = np.array(ar)
    tab_d = cuda.to_device(tab)
    #longueur/blockdim, largeur
    gridDim = int(math.ceil(tab.shape[2]/g_blockDim)),tab.shape[1],tab.shape[0]
    if tab.shape[2] <= g_blockDim :
        scanKernel2D_All[gridDim,g_blockDim](tab_d)
        cuda.synchronize()
        res = tab_d.copy_to_host()
        return res
    else :
        somtab = np.zeros(gridDim[::-1],dtype=int)
        somtab_d = cuda.to_device(somtab)
        scanKernel_sA2D_All[gridDim ,g_blockDim](tab_d,somtab_d)
        cuda.synchronize()
        somtab = somtab_d.copy_to_host()
        recur = scan_sa2D_All(somtab)
        recur_d = cuda.to_device(np.array(recur))
        sum_el2D_All[gridDim,g_blockDim](tab_d,recur_d)
        res = tab_d.copy_to_host()
        return res

#Invocation du kernel qui se charge de la transposée
def transpose(mat_input):
    gridDim = np.ceil(np.asarray(mat_input.shape)/ g_threadsperblock).astype(np.int32).tolist()
    mat_output = np.zeros([mat_input.shape[1],mat_input.shape[0]],dtype=np.int32)
    input_d = cuda.to_device(np.array(mat_input))
    output_d = cuda.to_device(np.array(mat_output))
    transpose_kernel[gridDim,g_threadsperblock](input_d,output_d)
    return output_d.copy_to_host()

#Invocation du kernel qui se charge de la transposée mais traitant toutes les images
def transpose_All(mat_input):
    gridDim = np.ceil(np.asarray(mat_input.shape[1:])/ g_threadsperblock).astype(np.int32).tolist()
    mat_output = np.zeros([mat_input.shape[0],mat_input.shape[2],mat_input.shape[1]],dtype=np.int32)
    input_d = cuda.to_device(np.array(mat_input))
    output_d = cuda.to_device(np.array(mat_output))
    transpose_kernel_All[(gridDim[0],gridDim[1],mat_input.shape[0]),g_threadsperblock](input_d,output_d)
    return output_d.copy_to_host()

#Fct qui réalise l'image integrale
def image_integrale(mat):
    mat1 = scan_sa2D(mat)
    mat2 = transpose(mat1)
    mat3 = scan_sa2D(mat2)
    return transpose(mat3)

#Fct qui calcule l'image integrale avec un tableau np contenant toutes les images
def image_integrale_All(mat):
    mat1 = scan_sa2D_All(mat)
    mat2 = transpose_All(mat1)
    mat3 = scan_sa2D_All(mat2)
    return transpose_All(mat3)

#Version Python du pseaudocode "Efficient Integral Imge Computation on the GPU" adapté en python
def image_integrale_CPU(mat):
    h, w = mat.shape
    ig_int = np.zeros_like(mat)
    for y in range(1, h):
        s = 0
        for x in range(w - 1):
            s += mat[y - 1][x]
            ig_int[y][x + 1] = s + ig_int[y - 1][x + 1]

    return ig_int

#Amelioration de image integrale cpu  avec une matrice contenant toutes les images (ici on ne traite plus une seule image)
def image_integrale_CPU_All(mats):
    nb, h, w = mats.shape
    ig_int = np.zeros_like(mats)
    for i in range(nb):
        min_ig = np.zeros((h,w),dtype=mats.dtype)
        for y in range(1, h):
            s = 0
            for x in range(w - 1):
                s += mats[i,y - 1][x]
                min_ig[y][x + 1] = s + min_ig[y - 1][x + 1]
        ig_int[i] = min_ig
    return ig_int


##Fct mini test unitaire pour vérifier si mon image integrale gpu donne le meme resultat que la version cpu
def test_unit(h,w,nomdutest):
    #Petit fct non nommé donné par pierre saunders pour que je puisse tester
    r = lambda w, h : np.random.randint(10, size = (w, h))
    m = r(h,w)
    assert (image_integrale_CPU(m.copy()) == image_integrale(m.copy())).all(), nomdutest



if __name__ == "__main__":
    r = lambda w, h : np.random.randint(10, size = (w, h))
    m = np.ones((10,11),dtype=np.int32)
    print(m)
    print("----------------------------------------")
    # print(scan_sa2D(m.copy()))
    print(image_integrale_CPU(m.copy()))
    print("----------------------------------------")
    print(image_integrale(m.copy()))

    test_unit(500,20000,"test")
    test_unit(50,25,"test")
    test_unit(50,2,"test")
