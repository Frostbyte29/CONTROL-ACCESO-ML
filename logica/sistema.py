import cv2

from logica.atencion import determinar_atencion
from reconocimiento.tracking import Tracker
from camara.camara import Camara
from logica.zona import obtener_zona, dentro_de_zona

from base_datos import (
    registrar_atencion,
    obtener_cliente_por_rostro
)

from reconocimiento.reconocimiento import (
    detectar_caras,
    reconocer_usuario,
    obtener_embedding_rostro
)

from logica.reconocimiento import GestorReconocimiento


class Sistema:

    def __init__(self):

        self.camara = Camara()

        self.tracker = Tracker()

        self.gestor_reconocimiento = GestorReconocimiento(
            numero_embeddings=5,
            intervalo_embedding=0.25,
            tiempo_estabilizacion=1,
            umbral=0.40
        )

    def procesar_frame(self):

        ret, frame = self.camara.leer()

        if not ret:
            return None

        zona = obtener_zona(frame)

        rostros_actuales = detectar_caras(frame)

        ids_usados = set()

        for rostro in rostros_actuales:

            centro_actual = rostro["centro"]

            en_zona = dentro_de_zona(
                centro_actual,
                zona
            )

            mejor_id = self.tracker.buscar_persona(
                centro_actual,
                ids_usados
            )

            if mejor_id is not None:
                id_persona = mejor_id
            else:
                id_persona = self.tracker.crear_persona(
                    centro_actual
                )

            ids_usados.add(id_persona)

            self.tracker.actualizar_persona(
                id_persona,
                centro_actual
            )

            datos = self.tracker.personas[id_persona]

            self.gestor_reconocimiento.procesar(
                datos,
                frame,
                rostro["landmarks"],
                obtener_embedding_rostro,
                reconocer_usuario
            )
            
            if (
                en_zona
                and datos["reconocido"]
                and not datos["atencion_registrada"]
            ):

                cliente = obtener_cliente_por_rostro(
                    datos["identidad"]
                )

                atencion = determinar_atencion(
                    cliente
                )

                registrar_atencion(
                    atencion["id_cliente"]
                )
                print(
                    "COBRO REGISTRADO |",
                    "Usuario:", datos["identidad"],
                    "| Monto: S/",
                    atencion["monto"]
                )
                datos["atencion_registrada"] = True

            identidad = datos["identidad"]

            cv2.rectangle(
                frame,
                (rostro["x"], rostro["y"]),
                (rostro["x2"], rostro["y2"]),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {id_persona}: {identidad}",
                (
                    rostro["x"],
                    rostro["y"] - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        self.tracker.eliminar_personas_perdidas()

        x1, y1, x2, y2 = zona

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        return frame

    def cerrar(self):

        self.camara.cerrar()