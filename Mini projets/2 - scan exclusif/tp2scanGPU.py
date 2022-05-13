from numba import cuda, int32
import numpy as np
import math


# @cuda.jit
# def scanKernel(a,m):
#     thID = cuda.threadIdx.x
#     for d in range(0,m):
#         k = pow(2,d+1)*thID
#         print("K : ", k)
#         a[k+(pow(2,d+1)-1)] += a[k+pow(2,d)-1]
#     if thID == 0:
#         print("Resultat Montee :")
#         for n in range(len(a)):
#             print(a[n])
#     ##Descente :
#     a[len(a)-1]=0
#     for d in range(m-1,-1,-1):
#         k = pow(2,d+1)*thID
#         print("K : ", k)
#         t= a[k+pow(2,d)-1]
#         a[k+pow(2,d)-1] = a[k+(pow(2,d+1)-1)]
#         a[k+(pow(2,d+1)-1)] += t

#     if thID == 0:
#         print("Resultat descente :")
#         for n in range(len(a)):
#             print(a[n])

# a1 = [2,3,4,6,8,10,12,14]
# tab = np.array(a1)
# tab_d = cuda.to_device(tab)

# threadim = 4
# n = len(a1)
# m = int(math.log(n,2))
# scanKernel[1,threadim](tab_d,m) 
# cuda.synchronize()


# @cuda.jit
# def scanKernel_sub(a,m):
#     thID = cuda.threadIdx.x
#     offset = cuda.blockIdx.x * cuda.blockDim.x
#     # if thID == 0 :
#     #     print("off : ",offset)
#     #thID = cuda.grid(1)
#     for d in range(0,m):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             a[offset +k+(pow(2,d+1)-1)] += a[offset +k+pow(2,d)-1]
#     if thID == 0:
#         print(a[offset+ thID],a[offset+ thID+1])
        
#     ##Descente :
#     a[offset + cuda.blockDim.x - 1]=0
#     for d in range(m-1,-1,-1):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             t= a[offset +k+pow(2,d)-1]
#             a[offset +k+pow(2,d)-1] = a[offset +k+(pow(2,d+1)-1)]
#             a[offset +k+(pow(2,d+1)-1)] += t

#     if thID == 0:
#         print(a[offset+ thID],a[offset+ thID+1])

# a2 = [2,3,4,6,10,12]
# tab = np.array(a2)
# tab_d = cuda.to_device(tab)


#n = len(a2)


# threadsperblock = 2  
# m = int(math.log(threadsperblock,2))      
# blocksdim = len(a2)//threadsperblock,1
# print(blocksdim,threadsperblock)
# scanKernel_sub[blocksdim,threadsperblock](tab_d,m) 
# cuda.synchronize()
# res = tab_d.copy_to_host()
# print(res)

# @cuda.jit
# def scanKernel_sub2(a,m,somtab):   
#     thID = cuda.threadIdx.x
#     offset = cuda.blockIdx.x * cuda.blockDim.x
#     for d in range(0,m):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             a[offset +k+(pow(2,d+1)-1)] += a[offset +k+pow(2,d)-1]
#         cuda.syncthreads()
#     if thID == 0:
#         print(a[offset+ thID],a[offset+ thID+1])
        
#     ##Descente :
#     somtab[cuda.blockIdx.x] = int(a[offset + cuda.blockDim.x - 1])
#     a[offset + cuda.blockDim.x - 1]=0
#     for d in range(m-1,-1,-1):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             t= a[offset +k+pow(2,d)-1]
#             a[offset +k+pow(2,d)-1] = a[offset +k+(pow(2,d+1)-1)]
#             a[offset +k+(pow(2,d+1)-1)] += t
#         cuda.syncthreads()

#     if thID == 0:
#         print(a[offset+ thID],a[offset+ thID+1])

# a2 = [2,3,4,6,10,12]
# tab = np.array(a2)
# tab_d = cuda.to_device(tab)


#n = len(a2)


# threadsperblock = 2  
# m = int(math.log(threadsperblock,2))      
# blocksdim = len(a2)//threadsperblock,1
# print(blocksdim,threadsperblock)

# somtab = np.zeros(int(len(a2)//threadsperblock))
# somtab_d = cuda.to_device(somtab)
# print(somtab)

# scanKernel_sub2[blocksdim,threadsperblock](tab_d,m,somtab_d) 
# cuda.synchronize()

# res = tab_d.copy_to_host()
# res2 = somtab_d.copy_to_host()
# print(res)
# print(res2)


# @cuda.jit
# def scanKernel_montee(a,m):   
#     thID = cuda.threadIdx.x
#     offset = cuda.blockIdx.x * cuda.blockDim.x
#     for d in range(0,m):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             a[offset +k+(pow(2,d+1)-1)] += a[offset +k+pow(2,d)-1]
#         cuda.syncthreads()
#     if thID == 0:
#         print(a[offset+ thID],a[offset+ thID+1])

# @cuda.jit
# def scanKernel_descente(a,m,somtab): 
#     thID = cuda.threadIdx.x
#     offset = cuda.blockIdx.x * cuda.blockDim.x

#     somtab[cuda.blockIdx.x] = int(a[offset + cuda.blockDim.x - 1])
#     a[offset + cuda.blockDim.x - 1]=0
#     for d in range(m-1,-1,-1):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             t= a[offset +k+pow(2,d)-1]
#             a[offset +k+pow(2,d)-1] = a[offset +k+(pow(2,d+1)-1)]
#             a[offset +k+(pow(2,d+1)-1)] += t
#         cuda.syncthreads()

#     if thID == 0:
#         print(a[offset+ thID],a[offset+ thID+1])




# @cuda.jit
# def scanKernel_montee_sA(a):
   
#     sA = cuda.shared.array(g_nbthread,dtype=int32)   
#     thID = cuda.threadIdx.x
#     offset = cuda.blockIdx.x * cuda.blockDim.x
#     #TODO ca va planter : marche seulement parce que j'ai un block
#     if thID+offset < a.shape[0]:
#         sA[thID] = a[thID+offset]
#     else :
#         sA[thID] = 0
#     cuda.syncthreads()

#     for d in range(0,g_m):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]
#         cuda.syncthreads()

#     if thID < a.shape[0] :
#         a[thID+offset] = sA[thID]

# @cuda.jit
# def scanKernel_descente_sA(a,somtab):
#     sA = cuda.shared.array(g_nbthread,dtype=int32)   

#     thID = cuda.threadIdx.x
#     offset = cuda.blockIdx.x * cuda.blockDim.x
#     if thID+offset< a.shape[0]:
#         sA[thID] = a[thID+offset]
#     else :
#         sA[thID] = 0
#     cuda.syncthreads()

#     somtab[cuda.blockIdx.x] = int(sA[cuda.blockDim.x - 1])
#     sA[cuda.blockDim.x - 1]=0
#     for d in range(g_m-1,-1,-1):
#         k = pow(2,d+1) * thID
#         if k < cuda.blockDim.x :
#             t= sA[k+pow(2,d)-1]
#             sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
#             sA[k+(pow(2,d+1)-1)] += t
#         cuda.syncthreads()
#     if thID < a.shape[0] :
#         a[thID+offset] = sA[thID]





# def scan(ar):
#     threadsperblock = g_nbthread
#     tab = np.array(ar)
#     tab_d = cuda.to_device(tab)
#     blocksdim = len(ar)//threadsperblock,1

#     scanKernel_montee_sA[blocksdim,threadsperblock](tab_d)
#     cuda.synchronize()

#     somtab = np.zeros(int(len(ar)//threadsperblock),dtype=int)
#     somtab_d = cuda.to_device(somtab)
#     # somtab_inter = somtab_d.copy_to_host()
#     # somtab_inter_d = cuda.to_device(somtab_inter)

#     scanKernel_descente_sA[blocksdim,threadsperblock](tab_d,somtab_d)
#     cuda.synchronize()

#     res = tab_d.copy_to_host()
#     somtab = somtab_d.copy_to_host()

#     print(res)
#     print(res2)





    # if somtab_inter_d > puiss:
    #     scan(somtab_inter_d,puiss)
    # while len(somtab_intr_d) not in deuxpuissance(len(somtab_intr_d)):
    #     somtab_intr_d.append(0)
    # somtab_intr_d = scanCPU(somtab_intr_d)
    # kernel_somme(tab_inter_d,m,somtab_intr_d)


#Tableau :
# a = [2,3,4,6]
# #A modifier en fonciton de la taille des threads 
# g_nbthread = 2
# #Ne pas modifier :
# g_m = int(math.log(g_nbthread,2)) 
# scan(a)



@cuda.jit
def scanKernel_sA(a,somtab):
   
    sA = cuda.shared.array(g_nbthread,dtype=int32)   
    thID = cuda.threadIdx.x
    offset = cuda.blockIdx.x * cuda.blockDim.x
    #TODO ca va planter : marche seulement parce que j'ai un block
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
    if thID < a.shape[0] :
        a[thID+offset] = sA[thID]

def deuxpuissance(n):
    return [1 << i for i in range(n)]

# def scan_sa(ar):
#     threadsperblock = g_nbthread
#     tab = np.array(ar)
#     tab_d = cuda.to_device(tab)
#     #blocksdim = len(ar)//threadsperblock,1
#     blocksdim = int(math.ceil(len(ar)/threadsperblock)),1,1

#     #somtab = np.zeros(int(len(ar)//threadsperblock),dtype=int)
#     somtab = np.zeros(blocksdim[0],dtype=int)
#     somtab_d = cuda.to_device(somtab)

#     scanKernel_sA[blocksdim,threadsperblock](tab_d,somtab_d)
#     cuda.synchronize()

#     res = tab_d.copy_to_host()
#     somtab = somtab_d.copy_to_host()

#     print(res)
#     print(somtab)

    # if len(somtab) > threadsperblock :
    #     scanKernel_sA[blocksdim,threadsperblock](somtab,somtab_d)
    # else :
    #     somtab_scan = scan_sa(somtab)

#     print("sous tableau :", somtab)
#     for z in range(0,len(res),threadsperblock):
#         for u in range(threadsperblock):
#             res[z+u]=tab[z+u]+somtab_scan[z//div]


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
        k = pow(2,d+1)*thID
        sA[k+(pow(2,d+1)-1)] += sA[k+pow(2,d)-1]

    ##Descente :
    sA[len(a)-1]=0
    for d in range(g_m-1,-1,-1):
        k = pow(2,d+1)*thID
        t= sA[k+pow(2,d)-1]
        sA[k+pow(2,d)-1] = sA[k+(pow(2,d+1)-1)]
        sA[k+(pow(2,d+1)-1)] += t
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
    #blocksdim = len(ar)//threadsperblock,1
    blocksdim = int(math.ceil(len(ar)/threadsperblock)),1,1

    if tab.shape[0] <= threadsperblock :
        scanKernel[blocksdim,threadsperblock](tab_d)
        cuda.synchronize()
        res = tab_d.copy_to_host()
        return res
    else :
        #somtab = np.zeros(int(len(ar)//threadsperblock),dtype=int)
        somtab = np.zeros(blocksdim[0],dtype=int)
        somtab_d = cuda.to_device(somtab)
        scanKernel_sA[blocksdim,threadsperblock](tab_d,somtab_d)
        cuda.synchronize()
        res = tab_d.copy_to_host()
        somtab = somtab_d.copy_to_host()

        # print(res)
        # print(somtab)
        recur = scan_sa(somtab)
        recur_d = cuda.to_device(np.array(recur))
        #print(recur,res)
        sum_el[blocksdim,threadsperblock](tab_d,recur_d)
        # for z in range(0,len(res),threadsperblock):
        #     for u in range(threadsperblock):
        #         res[z+u]=res[z+u]+recur[z//threadsperblock]
        #print(res)
        res = tab_d.copy_to_host()
        return res

    
a = [2,3,4,6,2]
#a = [2,3]
#A modifier en fonciton de la taille des threads 
g_nbthread = 2
#Ne pas modifier :
g_m = int(math.log(g_nbthread,2)) 
res = scan_sa(a)
print(res)
# cuda.kernel.transpose import transpose
# Modif dimmension des block
