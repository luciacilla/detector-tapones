"""
Modulo para importar las imagenes y almacenarlas en una lista


"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ClaseDetectarTapones import DetectarTapones

images = []

images.append(cv2.imread("1.jpg")) 
images.append(cv2.imread("2.jpg"))
images.append(cv2.imread("3.jpg"))
images.append(cv2.imread("4.jpg"))
images.append(cv2.imread("5.jpg"))
images.append(cv2.imread("6.jpg"))
images.append(cv2.imread("7.jpg"))
images.append(cv2.imread("8.jpg"))

# Comprobar imagenes bien guardadas pq se muestran 
"""
fig, axes = plt.subplots(1, 8, figsize=(48, 24))
for i in range(8):
    axes[i].imshow(images[i], cmap='Blues', vmin=0, vmax=255)
    axes[i].set_title('image i')
plt.show()
"""

def main():
    detector = DetectarTapones("1.jpg")
    cnts = detector.detectar_contornos()
    print(cnts)

main()