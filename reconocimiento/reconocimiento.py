import numpy as np

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from base_datos.conexion import conectar


# =============================
# Cargar InsightFace
# =============================

app = FaceAnalysis(
    name="buffalo_m"
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

modelo_deteccion = app.models["detection"]
modelo_reconocimiento = app.models["recognition"]


# =============================
# Obtener embedding
# =============================

def obtener_embedding(imagen):

    if imagen is None or imagen.size == 0:
        return None

    try:

        # Detectar rostro
        bboxes, kpss = modelo_deteccion.detect(
            imagen
        )

        if bboxes is None or len(bboxes) == 0:
            return None

        # Para registro debe existir
        # solamente un rostro
        if len(bboxes) > 1:
            return None

        # Obtener landmarks
        kps = kpss[0]

        # Alinear rostro
        rostro_alineado = face_align.norm_crop(
            imagen,
            landmark=kps
        )

        # Generar embedding
        embedding = modelo_reconocimiento.get_feat(
            rostro_alineado
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        norma = np.linalg.norm(
            embedding
        )

        if norma == 0:
            return None

        return embedding / norma

    except Exception as e:

        print(
            "Error generando embedding:",
            e
        )

        return None


# =============================
# Detectar caras
# =============================

def detectar_caras(frame):

    bboxes, kpss = modelo_deteccion.detect(
        frame
    )

    rostros = []

    if bboxes is None:
        return rostros

    for i, bbox in enumerate(bboxes):

        x1, y1, x2, y2 = (
            bbox[:4].astype(int)
        )

        centro_x = int(
            (x1 + x2) / 2
        )

        centro_y = int(
            (y1 + y2) / 2
        )

        rostros.append({

            "x": x1,
            "y": y1,
            "x2": x2,
            "y2": y2,

            "centro": (
                centro_x,
                centro_y
            ),

            "landmarks": kpss[i]
        })

    return rostros


# =============================
# Generar embedding de un rostro
# =============================

def obtener_embedding_rostro(
    imagen,
    landmarks
):

    if imagen is None or imagen.size == 0:
        return None

    try:

        rostro_alineado = face_align.norm_crop(
            imagen,
            landmark=landmarks
        )

        embedding = modelo_reconocimiento.get_feat(
            rostro_alineado
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        norma = np.linalg.norm(
            embedding
        )

        if norma == 0:
            return None

        return embedding / norma

    except Exception as e:

        print(
            "Error generando embedding:",
            e
        )

        return None


# =============================
# Cargar embeddings desde SQLite
# =============================

def obtener_embeddings_guardados():

    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT usuario_rostro, embedding
        FROM clientes
        WHERE embedding IS NOT NULL
    """)

    registros = cursor.fetchall()

    conexion.close()

    embeddings = {}

    for usuario, blob in registros:

        embedding = np.frombuffer(
            blob,
            dtype=np.float32
        )

        if embedding.shape == (512,):

            norma = np.linalg.norm(
                embedding
            )

            if norma != 0:

                embedding = (
                    embedding / norma
                )

                embeddings[usuario] = embedding

    return embeddings


# =============================
# Reconocer usuario
# =============================

def reconocer_usuario(
    embedding_actual,
    umbral=0.40
):

    if embedding_actual is None:

        return (
            "Desconocido",
            float("inf")
        )

    embeddings_guardados = (
        obtener_embeddings_guardados()
    )

    if not embeddings_guardados:

        return (
            "Desconocido",
            float("inf")
        )

    mejor_usuario = "Desconocido"
    mejor_distancia = float("inf")

    for usuario, embedding_guardado in (
        embeddings_guardados.items()
    ):

        distancia = 1 - np.dot(
            embedding_actual,
            embedding_guardado
        )

        if distancia < mejor_distancia:

            mejor_distancia = distancia
            mejor_usuario = usuario

    if mejor_distancia <= umbral:

        return (
            mejor_usuario,
            float(mejor_distancia)
        )

    return (
        "Desconocido",
        float(mejor_distancia)
    )