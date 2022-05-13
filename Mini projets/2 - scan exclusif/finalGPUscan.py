from numba import cuda, int32
import numpy as np
import math


@cuda.jit
def scanKernel_sA(a,somtab): 
    sA = cuda.shared.array(g_nbthread,dtype=int32)   
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x

    if thID+offset < a.shape[0]:
        sA[thID] = a[thID+offset]
    else :
        sA[thID] = 0
    cuda.syncthreads()
    for d in range(0,g_m):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]
        cuda.syncthreads()

    somtab[cuda.blockIdx.x] = int(sA[cuda.blockDim.x - 1])
    sA[cuda.blockDim.x - 1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1) * thID
        if k < cuda.blockDim.x :
            t= sA[k+pow(2,d)-1]
            sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
            sA[k+(pow(2,d+1)-1)] += t
        cuda.syncthreads()
    if thID +offset < a.shape[0] :
        a[thID+offset] = sA[thID]

@cuda.jit
def scanKernel(a): 
    sA = cuda.shared.array(g_nbthread,dtype=int32)   
    thID = cuda.threadIdx.x

    if thID < a.shape[0]:
        sA[thID] = a[thID]
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

    if thID < a.shape[0] :
        a[thID] = sA[thID]

@cuda.jit
def sum_el(tab,somtab):
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    if thID+offset < tab.shape[0]:
        tab[offset+thID]=tab[offset+thID]+somtab[offset//g_nbthread]

def scan_sa(ar):
    threadsperblock = g_nbthread
    tab = np.array(ar)
    tab_d = cuda.to_device(tab)
    blocksdim = int(math.ceil(len(ar)/threadsperblock)),1,1

    if tab.shape[0] <= threadsperblock :
        scanKernel[blocksdim,threadsperblock](tab_d)
        cuda.synchronize()
        res = tab_d.copy_to_host()
        return res
    else :
        somtab = np.zeros(blocksdim[0],dtype=int)
        somtab_d = cuda.to_device(somtab)
        scanKernel_sA[blocksdim,threadsperblock](tab_d,somtab_d)
        cuda.synchronize()
        res = tab_d.copy_to_host()
        somtab = somtab_d.copy_to_host()

        recur = scan_sa(somtab)
        recur_d = cuda.to_device(np.array(recur))
        sum_el[blocksdim,threadsperblock](tab_d,recur_d)
        res = tab_d.copy_to_host()
        return res

    
# a = [2,3,4,6,2,10,15,2,5]
a = [0,1,2,3,4,5,6,7]
#a = [2,3]
#A modifier en fonciton de la taille des threads 
g_nbthread = 2
#Ne pas modifier :
g_m = int(math.log(g_nbthread,2)) 
res = scan_sa(a)

print(res)