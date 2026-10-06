import os
import cv2
import numpy as np

from reconocimiento.reconocimiento import obtener_embedding


def generar_embedding(usuario_rostro):

    carpeta = os.path.join(
        "rostros",
        usuario_rostro
    )

    embeddings = []

    for numero in range(1, 11):

        ruta = None

        for extension in [".jpg", ".jpeg", ".png"]:

            posible = os.path.join(
                carpeta,
                f"foto{numero}{extension}"
            )

            if os.path.exists(posible):
                ruta = posible
                break

        if ruta is None:
            print(
                f"No se encontró foto{numero}"
            )
            continue

        imagen = cv2.imread(ruta)

        embedding = obtener_embedding(imagen)

        if embedding is None:
            print(
                f"No se pudo generar embedding de foto{numero}"
            )
            continue

        embeddings.append(embedding)

        print(
            f"Foto {numero}: embedding generado"
        )

    if len(embeddings) == 0:
        raise ValueError(
            "No se pudo generar ningún embedding."
        )

    embedding_promedio = np.mean(
        embeddings,
        axis=0
    )

    embedding_promedio = (
        embedding_promedio /
        np.linalg.norm(embedding_promedio)
    )

    print(
        f"Embeddings generados: {len(embeddings)}"
    )

    print(
        "Dimensión:",
        embedding_promedio.shape
    )

    return embedding_promedio