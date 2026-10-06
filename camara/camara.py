import cv2


# Clase: gestiona la cámara del sistema
class Camara:

    # Método: inicializa la cámara
    def __init__(self, indice=0):

        self.camara = cv2.VideoCapture(indice)

        self.camara.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camara.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

    # Método: obtiene un frame de la cámara
    def leer(self):

        ret, frame = self.camara.read()

        if ret:
            frame = cv2.flip(frame, 1)

        return ret, frame
    # Método: muestra las dimensiones de la cámara
    def mostrar_dimensiones(self):

        print(
            "Ancho:",
            self.camara.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        print(
            "Alto:",
            self.camara.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    # Método: libera la cámara
    def cerrar(self):

        self.camara.release()