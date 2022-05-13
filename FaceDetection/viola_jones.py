"""
A Python implementation of the Viola-Jones ensemble classification method described in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
Works in both Python2 and Python3
"""
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
import ii_scan_gpu as gpu_scan 
from apply_features_gpu import apply_features_GPU
from weaktrain_gpu import gpu_weak_train
import time
from tqdm import tqdm
class ViolaJones:
    def __init__(self, T = 10):
        """
          Args:
            T: The number of weak classifiers which should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []
        self.features = []

    def train(self, training, pos_num, neg_num):
        """
        Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
          Args:
            training: An array of tuples. The first element is the numpy array of shape (m, n) representing the image. The second element is its classification (1 or 0)
            pos_num: the number of positive samples
            neg_num: the number of negative samples
        """
        weights = np.zeros(len(training))
        print("Computing integral images")

        #Tableau numpy de toutes les images a traiter
        images = np.array(list(map(lambda data: data[0], training))).astype(np.int32)
        labels = np.array(list(map(lambda data: data[1], training))).astype(np.int32)
        imagesii = gpu_scan.image_integrale_All(images)
        for x in range(len(imagesii)):
            if labels[x] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)

        print("Building features")
        features = self.build_features(imagesii[0].shape)
        print("Applying features to training examples")
        X = self.apply_features(features, imagesii)
        print("Selecting best features")
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, labels).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        self.features = features
        print("Selected %d potential features" % len(X))

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, labels, weights)
            clf, error, accuracy = self.select_best(X,weak_classifiers, weights, labels)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append((clf,weak_classifiers[clf,0],weak_classifiers[clf,1]))
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), (len(accuracy) - sum(accuracy))/len(accuracy), alpha))

    def train_weak(self, applied_feature, labels, weights):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            applied_feature: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            labels: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            weights: A numpy array of shape len(training_data). The ith element is the weight assigned to the ith training example
          Returns:
            An array of weak classifiers
        """
        #Ca compte le nombre de fois ou il y a un label pos/neg
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, labels):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        #Tries une seule fois pour gagner du temps
        #sorted n'existe pas en numba donc il est remplacé par agrsort
        #Argsort trie les indices des elements
        id_sort_applied_feature = np.argsort(applied_feature)
        classifiers = gpu_weak_train(applied_feature, weights, id_sort_applied_feature, labels, total_pos, total_neg)
        #applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

        return classifiers
                
    def build_features(self, image_shape):
        """
        Builds the possible features given an image shape
          Args:
            image_shape: a tuple of form (height, width)
          Returns:
            an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
        """
        height, width = image_shape
        features = []
        #Un rectangle ayant pour coordonnées x =0 et y= 0 avec une largeur et une longueur de 0 a une air nulle.
        rectvide = (0,0,0,0)
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #Construction des 4 features
                        #2 rectangle features
                        immediate = (i, j, w, h)
                        right = (i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right,rectvide], [immediate,rectvide]))

                        bottom = (i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate,rectvide], [bottom,rectvide]))
                        
                        right_2 = (i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right,rectvide], [right_2, immediate]))

                        bottom_2 = (i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom,rectvide], [bottom_2, immediate]))
                        #4 rectangle features
                        bottom_right = (i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

#!!! Pierre saunders m'a enormement aidé pour cette partie !!!!!
    def select_best(self, applied_features,classifiers, weights, labels):

        best_clf, best_error, best_accuracy = None, float('inf'), None
        for indice, (threshold,polarity) in enumerate(tqdm(classifiers,leave=False,desc="Train select_best")):
            error, accuracy = 0, []
            for applied_feature ,label,w in zip(applied_features[indice],labels, weights):
                correctness = abs(weak_classify_app_feat(applied_feature,threshold,polarity) - label)
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(applied_features)
            if error < best_error:
                best_clf, best_error, best_accuracy = indice, error, accuracy
        return best_clf, best_error, best_accuracy

 #VERSION GPU 
    def apply_features(self, features, imagesii):
        appli_feat = np.zeros((len(features), imagesii.shape[0]))
        start_time = time.time_ns()
        appli_feat = apply_features_GPU(imagesii ,features)
        ns = (time.time_ns() - start_time)
        ms = ns/1000000
        s = ms*0.001
        print("Temps de l'apply_features GPU:" ,s, "secondes")
        return appli_feat

#Ancienne verison CPU
    # def apply_features(self, features, training_data):
    #     """
    #     Maps features onto the training dataset
    #       Args:
    #         features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
    #         training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
    #       Returns:
    #         X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
    #         y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
    #     """
    #     X = np.zeros((len(features), len(training_data)))
    #     y = np.array(list(map(lambda data: data[1], training_data)))
    #     i = 0

    #     start_time = time.time_ns()        

    #     for positive_regions, negative_regions in features:
    #         feature = lambda ii: sum([compute_feature(pos[0],pos[1],pos[2],pos[3],ii) for pos in positive_regions]) - sum([compute_feature(neg[0],neg[1],neg[2],neg[3],ii) for neg in negative_regions])
    #         X[i] = list(map(lambda data: feature(data[0]), training_data))
    #         i += 1

    #     print("chrono de l'apply_features CPU :" ,time.time_ns() - start_time)

    #     return X, y


    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = gpu_scan.image_integrale(image)
        for alpha, (clf,threshold,polarity) in zip(self.alphas, self.clfs):
            total += alpha * weak_classify(ii,self.features[clf,0],self.features[clf,1],threshold,polarity)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

def weak_classify_app_feat(applied_feature, threshold, polarity):

    return 1 if polarity * applied_feature < polarity * threshold else 0
          
def weak_classify(x, positive_regions, negative_regions, threshold, polarity):
    """
    Classifies an integral image based on a feature f and the classifiers threshold and polarity
        Args:
        x: A 2D numpy array of shape (m, n) representing the integral image
        Returns:
        1 if polarity * feature(x) < polarity * threshold
        0 otherwise
    """
    #    feature = lambda ii: sum([compute_feature(pos[0],pos[1],pos[2],pos[3],ii) for pos in positive_regions]) - sum([compute_feature(neg[0],neg[1],neg[2],neg[3],ii) for neg in negative_regions])

    feature = lambda ii: sum([compute_feature(*pos,ii) for pos in positive_regions]) - sum([compute_feature(*neg,ii) for neg in negative_regions])
    return 1 if polarity * feature(x) < polarity * threshold else 0
    

def compute_feature(x,y,width,height,ii):
    return ii[y+height][x+width] + ii[y][x] - (ii[y+height][x]+ii[y][x+width]) 

 