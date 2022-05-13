from numba import cuda, int32
import numpy as np
import math


#Var global
g_blockDim = 32
g_threadsperblock = 32,32
g_m = int(math.log(g_blockDim,2)) 


@cuda.jit(device=True)   
def compute_feature(x,y,w,h,ii):
    return ii[y+h][x+w] + ii[y][x] - (ii[y+h][x]+ii[y][x+w])   

@cuda.jit
def apply_features_kernel(training_data1_d,features_d,output_d):
    idx, idy = cuda.grid(2)
    if  idx < features_d.shape[0] and idy < training_data1_d.shape[0]:
        #Les 2 rectangles positifs
        p1 = compute_feature(features_d[idx][0][0][0],features_d[idx][0][0][1],features_d[idx][0][0][2],features_d[idx][0][0][3],training_data1_d[idy])
        p2 = compute_feature(features_d[idx][0][1][0],features_d[idx][0][1][1],features_d[idx][0][1][2],features_d[idx][0][1][3],training_data1_d[idy])
        #Les deux rectangles negatifs
        n1 = compute_feature(features_d[idx][1][0][0],features_d[idx][1][0][1],features_d[idx][1][0][2],features_d[idx][1][0][3],training_data1_d[idy])
        n2 = compute_feature(features_d[idx][1][1][0],features_d[idx][1][1][1],features_d[idx][1][1][2],features_d[idx][1][1][3],training_data1_d[idy])
        #La somme des pixels des rectangles positifs est soustraite a la somme des rectangles négatifs
        output_d[idx,idy] = ( p1 + p2 ) - ( n1 + n2 )


#Invocation de apply feature kernel 
def apply_features_GPU(training_data1, features):
    #On envoie sur le device
    training_data1_d = cuda.to_device(training_data1)
    output_int = np.empty((features.shape[0],training_data1.shape[0]),dtype=training_data1.dtype)
    output_d = cuda.to_device(output_int)
    features_d = cuda.to_device(features)
    #On calcule la dim de la grid
    gridDim = int(math.ceil(features.shape[0]/g_blockDim)),int(math.ceil(training_data1.shape[0]/g_blockDim))
    #On invoque le kernel
    apply_features_kernel[gridDim,g_threadsperblock](training_data1_d,features_d,output_d)
    cuda.synchronize()
    res = output_d.copy_to_host()
    return res
    

