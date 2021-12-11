import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def transfer_PCA(color_layer):
    pca = PCA(n_components=50)
    pca.fit(color_layer)
    trans_pca = pca.transform(color_layer)
    return pca, trans_pca

img_origin = cv2.imread("stone.jpg")
img_origin = cv2.resize(img_origin, (500,500))

img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

_img = img_origin

pca_img = transfer_PCA(_img)

#Reconstruct the image and visualize
img_arr = pca_img[0].inverse_transform(pca_img[1])

'''cv2.imshow('Origin', img_origin)
cv2.imshow('Reconstruct', img_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
plt.imshow(img_arr)
plt.show()