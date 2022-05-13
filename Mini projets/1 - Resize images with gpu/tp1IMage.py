from numba import cuda
import numpy as np
from PIL import Image
import math

#ssh user2@192.168.0.1
#rsync -avh TP1 user2@192.168.0.1:~/
#scp user2@192.168.0.1:/home/user2/TP1/newImg.jpg .

@cuda.jit
#Fct qui permet de convertir une image en N&B
def convertImage(arr, imgNB):
    #Id du block * la dimension du block + l'id du thread
    #idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    #Cette ligne remplace les deux précédentes 
    idx, idy = cuda.grid(2)

    if idx < arr.shape[1] and idy < arr.shape[0]:
    #Conversion de l'image en noir&blanc 
        imgNB[idy,idx] = np.uint8((0.3 * arr[idy,idx][0]) + (0.59 * arr[idy,idx][1]) + (0.11 * arr[idy,idx][2]))



#Ouverture de l'image qu'on veut convertir
img = Image.open("kai.jpg")

#On converti l'image en une np.array
pixels = np.array(img)
#Aloue de la place en fonction de la taille du np.array (donc du nombre de pixels)
pixels_d = cuda.to_device(pixels)

#Le nombre de threads va dépendre de la taille de l'image ici on fait le produit
#du nombre de pixels en colonne et le nombre de pixels par ligne
nbThreads = img.size[0] * img.size[1]

imgNB_d = cuda.device_array((img.size[1],img.size[0]),dtype=np.uint8)


threadim = 32,32,1  #Nombre de threads

blocksdim = math.ceil(pixels.shape[1]/32),math.ceil(pixels.shape[0]/32),1
convertImage[blocksdim, threadim](pixels_d, imgNB_d) #Entre crochet [la dimension de la grille, la dimension du block]
cuda.synchronize()
imgNB = imgNB_d.copy_to_host()

#print(pixels)

newImg = Image.fromarray(np.array(imgNB))
newImg.save("kaibnw.jpg")