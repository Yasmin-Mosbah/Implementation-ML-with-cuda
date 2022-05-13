from numba import cuda
import numba as nb

@cuda.jit
def print_coord_cuda():
    #Cordonnées locale 
    print("block - thread :",cuda.blockIdx.x,cuda.blockIdx.y,cuda.blockIdx.z,cuda.threadIdx.x,cuda.threadIdx.y,cuda.threadIdx.z)

@cuda.jit
def print_coord_grid():
    #Cordonnées locale 
    print("block - thread :",cuda.blockIdx.x,cuda.blockIdx.y,cuda.blockIdx.z,cuda.threadIdx.x,cuda.threadIdx.y,cuda.threadIdx.z)
    print("Global id : " , cuda.grid(1))


# print("1 bloc dans grille, 1 thread dans bloc 1D ")
# print_coord_cuda[1, 1]()
# cuda.synchronize()


# print("==============================")
# print("1 bloc dans grille, 16 threads dans bloc 1D ")
# print_coord_cuda[1, 16]()
# cuda.synchronize()


# print("==============================")
# print("2 blocs dans grille 1D, 1 thread dans bloc 1D ")
# print_coord_cuda[2, 1]()
# cuda.synchronize()


# print("==============================")
# print("1 bloc dans grille, 16 threads dans bloc 3D")
# threadsperblock = 16           
# threadim = 4,2,2  
# blockspergrid = 1
# blocksdim = blockspergrid,1,1
# print_coord_cuda[1, threadim]()
# cuda.synchronize()


# print("==============================")
# print("Grille 1D de 2 blocs, 16 threads dans bloc 1D")
# print_coord_grid[2,16]()
# cuda.synchronize()

print("==============================")
print("Grille 2D de 4 blocs, 14 threads dans bloc 2D")
threadsperblock = 500          
threadim = 10,50
blockspergrid = 1
blocksdim = blockspergrid,1,1
print_coord_grid[blocksdim,threadim]()
cuda.synchronize()