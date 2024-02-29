"""
CLASE tapon
    · Constructor
    · Detectar contornos
    · Dibujar contornos + enseñar imagen
    · Limpiar contornos por redondez (circularity)
    · Calcular centro del contorno y area
    · Buscar color tapon
    · Dibujar centro, area y color en la imagen
    · Main

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

class DetectarTapones:
    def __init__(self, image_path) -> None:
        """
        image
        gray 
        """
        self.image = cv2.imread(image_path)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
    def detectar_contornos(self) -> tuple:
        """
        Funcion de deteccion de contornos, obtiene la imagen borrosa, la threshold y busca los coontornos en esta.

        Returns: 
            cnts: tupla con los contornos
        """
        blurred_img = cv2.GaussianBlur(self.image_gray, (5, 5), 0)
        thresh_img = cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # Busco contornos thresholded image
        cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Grab the appropriate tuple value based on whether we are using OpenCV 2.4, 3, or 4.
        cnts = imutils.grab_contours(cnts)  
        return cnts
    
    def areas_contornos(self, cnts):
        """
        """
        areas = []
        for contour in cnts:
            area = cv2.contourArea(contour)
            areas.append(area)
            #cv2.putText(self.image, f'Area: {area}', (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return areas
    
    def calc_circularity(area, perimeter):
        return (4 * np.pi * area) / (perimeter * perimeter)

    def clean_contours_by_circularity(self, cnts, area, circularity, min_circularity):
        """
        filter all contours by size and circularity to prevent noise
        Valores redondez: 
            circulo perfecto = 1
            formas no circulares << 1

        Atributos:
            cnts: contornos
            area: area de cada objeto
            min_circularity: valor minimo / de corte para saber si es circular

        Returns:
            filtered_contours: lista con los contornos de los objetos circulares
        """
        filtered_cnts  = []
        for contour in cnts:
            perimeter = cv2.arcLength(contour, True)
            # Center of the contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Redondez: {circularity}")
            if circularity > min_circularity:
                filtered_cnts.append(contour)
        return filtered_cnts

    def color_tapon(self, cnts):
        """
        cambiar a hsv
        mirar para cada tapon en que rango/numero cae h
            lo puedo mirar en el punto central
        mascaras de cada color??

        """
        img_hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
        colores = []
        for i, contour in enumerate(cnts):
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            x1, y1 = cX-10, cY+10 # esquina sup izq
            x2, y2 = cX+10, cY-10  # esquina inf drcha
            roi = img_hsv[y1:y2,x1:x2]
            h,s,v = cv2.split(roi)
            color_medio = cv2.mean(h)
            valor_medio = cv2.mean(v)
            # Rojo 175-10
            if 175<=color_medio<10:
                color = 'Rojo'
                colores.append(color)
                break
            # Naranja 10-30
            elif 10<=color_medio<30:
                color = 'Naranja'
                colores.append(color)
                break
            # Amarillo 30-60 
            elif 30<=color_medio<60:
                color = 'Amarillo'
                colores.append(color)
                break
            # Verde 60-100
            elif 60<=color_medio<100:
                color = 'Verde'
                colores.append(color)
                break
            # Azul 100-140
            elif 100<=color_medio<140:
                color = 'Azul'
                colores.append(color)
                break
            # Violeta 140-165
            elif 140<=color_medio<165:
                color = 'Violeta'
                colores.append(color)
                break
            # Rosa 165-175
            elif 165<=color_medio<175:
                color = 'Rosa'
                colores.append(color)
                break
            # Blanco v:200-255
            elif 200<=valor_medio<255:
                color = 'Blanco'
                colores.append(color)
                break
            # Negro v:0-50
            elif 0<=valor_medio<50:
                color = 'Negro'
                colores.append(color)
                break
            else:
                color = 'None'
        return colores

    
    def dibujar_contornos_centro_areas(self, cnts, areas, colores):
        """
        Función que dibuja y muestra los contornos
        """
        image = self.image.copy()
        # Contornos
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        # Centros y areas
        for i, contour in enumerate(cnts):
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 4, (255, 255, 255), -1)
            cv2.putText(image, f'Area: {areas[i]}', (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, f'Color: {colores[i]}', (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # show thresholded image
        fig, axes = plt.subplots(1, 1, figsize=(48, 24))
        axes.imshow(image, cmap='gray', vmin=0, vmax=255)
        axes.set_title('Image contoured')
        plt.show()

    