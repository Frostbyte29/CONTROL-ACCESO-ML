import time
import numpy as np


class GestorReconocimiento:

    def __init__(
        self,
        numero_embeddings=5,
        intervalo_embedding=0.25,
        tiempo_estabilizacion=1,
        umbral=0.40
    ):

        self.numero_embeddings = numero_embeddings
        self.intervalo_embedding = intervalo_embedding
        self.tiempo_estabilizacion = tiempo_estabilizacion
        self.umbral = umbral

    def procesar(
        self,
        datos,
        frame,
        landmarks,
        obtener_embedding_rostro,
        reconocer_usuario
    ):

        # ==========================================
        # YA FUE RECONOCIDA
        # ==========================================

        if datos["reconocido"]:
            return None
        
        ahora = time.time()

        # ==========================================
        # TIEMPO DE ESTABILIZACIÓN
        # ==========================================

        tiempo_tracking = (
            ahora - datos["inicio_tracking"]
        )

        if tiempo_tracking < self.tiempo_estabilizacion:
            return None

        # ==========================================
        # INTERVALO ENTRE EMBEDDINGS
        # ==========================================

        if (
            ahora - datos["ultimo_embedding"]
            < self.intervalo_embedding
        ):
            return None

        # ==========================================
        # GENERAR EMBEDDING
        # ==========================================

        embedding = obtener_embedding_rostro(
            frame,
            landmarks
        )

        if embedding is None:
            return None

        datos[
            "embeddings_temporales"
        ].append(embedding)

        datos[
            "ultimo_embedding"
        ] = ahora

        cantidad = len(
            datos["embeddings_temporales"]
        )

        # ==========================================
        # TODAVÍA FALTAN EMBEDDINGS
        # ==========================================

        if cantidad < self.numero_embeddings:
            return None

        # ==========================================
        # PROMEDIAR EMBEDDINGS
        # ==========================================

        embeddings = np.array(
            datos["embeddings_temporales"],
            dtype=np.float32
        )

        embedding_promedio = np.mean(
            embeddings,
            axis=0
        )

        norma = np.linalg.norm(
            embedding_promedio
        )

        if norma == 0:

            datos[
                "embeddings_temporales"
            ].clear()

            return None

        embedding_promedio /= norma

        # ==========================================
        # RECONOCER
        # ==========================================

        (
            mejor_usuario,
            mejor_distancia
        ) = reconocer_usuario(
            embedding_promedio,
            self.umbral
        )

        datos["identidad"] = mejor_usuario
        datos["reconocido"] = True

        datos[
            "embeddings_temporales"
        ].clear()

        return (
            mejor_usuario,
            mejor_distancia
        )