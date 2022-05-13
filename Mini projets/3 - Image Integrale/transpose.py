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

@cuda.jit
def scanKernel2D(a):
    sA = cuda.shared.array(g_blockDim,dtype=int32)   
    thID = cuda.threadIdx.x 
    y = cuda.blockIdx.y

    if thID < a.shape[1]:
        sA[thID] = a[y,thID]
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
        a[y,thID] = sA[thID]

@cuda.jit
def scanKernel_sA2D(a,somtab):
    sA = cuda.shared.array(g_blockDim,dtype=int32)   
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.blockIdx.y

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

    #somtab[cuda.blockIdx.x,y] = int(sA[cuda.blockDim.x - 1])
    if thID == 0 : 
        somtab[y,cuda.blockIdx.x] = sA[g_blockDim -1]
        sA[g_blockDim -1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            t= sA[k+pow(2,d)-1]
            sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
            sA[k+(pow(2,d+1)-1)] += t
    cuda.syncthreads()
    if thID+offset < a.shape[1] :
        a[y,thID+offset] = sA[thID]

@cuda.jit
def sum_el2D(tab,somtab):
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.blockIdx.y

    if thID+offset < tab.shape[1]:
        tab[y,offset+thID] += somtab[y,cuda.blockIdx.x]

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

def transpose(mat_input):
    gridDim = np.ceil(np.asarray(mat_input.shape)/ g_threadsperblock).astype(np.int32).tolist()
    mat_output = np.zeros([mat_input.shape[1],mat_input.shape[0]],dtype=np.int32)
    input_d = cuda.to_device(np.array(mat_input))
    output_d = cuda.to_device(np.array(mat_output))
    transpose_kernel[gridDim,g_threadsperblock](input_d,output_d)

    return output_d.copy_to_host()

def image_integrale(mat):
    mat1 = scan_sa2D(mat)
    mat2 = transpose(mat1)
    mat3 = scan_sa2D(mat2)
    return transpose(mat3)

#Version Python du pseaudocode de l'article 
def image_integrale_CPU(mat):
    h, w = mat.shape
    ig_int = np.zeros_like(mat)
    for y in range(1, h):
        s = 0
        for x in range(w - 1):
            s += mat[y - 1][x]
            ig_int[y][x + 1] = s + ig_int[y - 1][x + 1]

    return ig_int


##Fct mini test unitaire
def test_unit(h,w,nomdutest):
    r = lambda w, h : np.random.randint(10, size = (w, h))
    m = r(h,w)
    assert (image_integrale_CPU(m.copy()) == image_integrale(m.copy())).all(), nomdutest


if __name__ == "__main__":
    
    m = np.ones((10,11),dtype=np.int32)
    print(m)
    print("----------------------------------------")
    # print(scan_sa2D(m.copy()))
    print(image_integrale_CPU(m.copy()))
    print("----------------------------------------")
    print(image_integrale(m.copy()))

    # test_unit(500,20000,"test")
    # test_unit(50,25,"test")
    # test_unit(50,2,"test")

