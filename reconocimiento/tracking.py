
import numpy as np
import time

DISTANCIA_TRACKING = 100


def distancia_puntos(p1, p2):
    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


class Tracker:

    def __init__(self):
        self.personas = {}
        self.siguiente_id = 1
        self.tiempo_perdido = 2

    def crear_persona(self, centro):

        id_persona = self.siguiente_id

        self.siguiente_id += 1

        ahora = time.time()

        self.personas[id_persona] = {
            "id":id_persona,
            "centro": centro,
            "identidad": "Desconocido",
            "reconocido": False,
            "atencion_registrada": False,
            "ultima_vez_visto": ahora,

            # Nuevos datos para reconocimiento temporal
            "inicio_tracking": ahora,
            "embeddings_temporales": [],
            "ultimo_embedding": 0
        }

        return id_persona

    def buscar_persona(self, centro_actual, ids_usados):

        mejor_id = None
        menor_distancia = float("inf")

        for id_persona, datos in self.personas.items():

            if id_persona in ids_usados:
                continue

            distancia = distancia_puntos(
                centro_actual,
                datos["centro"]
            )

            if distancia < menor_distancia:
                menor_distancia = distancia
                mejor_id = id_persona

        if (
            mejor_id is not None
            and menor_distancia < DISTANCIA_TRACKING
        ):
            return mejor_id

        return None

    def actualizar_persona(self, id_persona, centro):

        self.personas[id_persona]["centro"] = centro
        self.personas[id_persona]["ultima_vez_visto"] = time.time()

    def eliminar_personas_perdidas(self):

        ahora = time.time()

        ids_eliminados = []

        for id_persona, datos in self.personas.items():

            tiempo_perdido = (
                ahora - datos["ultima_vez_visto"]
            )

            if tiempo_perdido >= self.tiempo_perdido:
                ids_eliminados.append(id_persona)

        for id_persona in ids_eliminados:
            del self.personas[id_persona]