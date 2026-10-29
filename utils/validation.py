import cv2
import numpy as np
from PIL import Image

# Detector de rostros de OpenCV
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def contiene_rostro(imagen_pil: Image.Image):
    """
    Verifica si la imagen contiene al menos un rostro humano.

    Retorna:
        True  -> se detectó un rostro
        False -> no se detectó ningún rostro
    """

    imagen = np.array(imagen_pil.convert("RGB"))

    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

    rostros = face_detector.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return len(rostros) > 0