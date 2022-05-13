import numpy as np
import pickle
from viola_jones import ViolaJones
from face_detection import test_viola,train_viola
import time


"""
Pierre Saunders m'a énormément aidé à comprendre le code, 
et à réalisation de ce projet. Sans son aide, je pense que j'aurais été totalement perdue.

J'ai également beaucoup travaillé avec Alexis Vighi, Rémi Felin et Benjamin Molinet, 
donc il est fort possible que vous distinguiez des similarités dans les codes.

"""

def bench_train(a):
    print("-----Train-----")
    return train_viola(a)

def bench_accuracy(a):
    print("-----Test------")
    return test_viola(a)


bench_accuracy("10")
print("------------------------------")
#bench_train(10)