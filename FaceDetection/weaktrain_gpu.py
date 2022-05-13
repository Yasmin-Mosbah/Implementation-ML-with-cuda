from numba import cuda, int32, float32
import numpy as np
import math


#Var global
g_blockDim = 32
g_threadsperblock = 32,32
g_m = int(math.log(g_blockDim,2))

#!!! Pierre saunders m'a enormement aidé pour cette partie !!!!!

@cuda.jit
def weak_train(features,classifiers,weights,applied_feature,labels, total_pos, total_neg):
# for index, feature in enumerate(X):
    pos_seen, neg_seen = 0, 0
    pos_weights, neg_weights = 0, 0
    min_error, best_threshold, best_polarity = 1e9,0.0,0.0
    index = cuda.grid(1)
    if index >= features.shape[0]:
        return 
    for indice_feature in applied_feature[index]:
        error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
        if error < min_error:
            min_error = error
            #Threshold est definie par la valeur d'une feature 
            best_threshold = float32(features[index,indice_feature])
            best_polarity = 1.0 if pos_seen > neg_seen else -1.0

        if labels[indice_feature] == 1:
            pos_seen += 1
            pos_weights += weights[indice_feature]
        else:
            neg_seen += 1
            neg_weights +=  weights[indice_feature]
    
    classifiers[index] = (best_threshold, best_polarity)

def gpu_weak_train(X, weights, applied_feature, labels, total_pos, total_neg):
    #Envoie sur le device
    d_features = cuda.to_device(X)
    classifiers = np.empty((X.shape[0],2),dtype=np.float32)
    d_classifiers = cuda.to_device(classifiers)
    d_weights = cuda.to_device(weights)
    d_appli_feat = cuda.to_device(applied_feature)
    d_labels = cuda.to_device(labels)
    #Calcul de la dim de grid (nb de block)
    gridDim = int(math.ceil(X.shape[0]/g_blockDim))
    #Invocation du kernel
    weak_train[gridDim ,g_blockDim](d_features,d_classifiers,d_weights,d_appli_feat,d_labels,total_pos, total_neg)
    cuda.synchronize()
    return d_classifiers.copy_to_host()