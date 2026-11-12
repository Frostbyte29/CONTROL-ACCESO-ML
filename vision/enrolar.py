"""Enrolamiento de personas: captura rostros con la cámara y los guarda en la base.

Uso: python -m vision.enrolar --nombre "Carlos" [--fuente 0] [--n 25] [--listar]
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from datos.db import conectar, crear_perfil
from vision.rostro import ExtractorRostro

_INSTRUCCIONES = [
    "De frente, mirando al centro",
    "Gira un poco a la izquierda",
    "Gira un poco a la derecha",
    "Levanta un poco la cabeza",
    "Baja un poco la cabeza",
    "Con lentes si los usas",
    "Sin lentes",
]

_MODELO = "sface_2021dec"
_INTERVALO_S = 0.5


def enrolar(
    nombre: str,
    fuente: int | str,
    n_objetivo: int,
    ruta_yunet: str,
    ruta_sface: str,
    ruta_db: str,
) -> None:
    """Captura n_objetivo rostros de calidad y guarda el perfil en la base."""
    extractor = ExtractorRostro(ruta_yunet, ruta_sface)
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"No se pudo abrir la fuente: {fuente}", file=sys.stderr)
        sys.exit(1)

    vectores: list[np.ndarray] = []
    calidades: list[float] = []
    ultimo_captura = 0.0

    try:
        while len(vectores) < n_objetivo:
            ok, imagen = cap.read()
            if not ok:
                break

            rostros = extractor.buscar_en_cuadro(imagen)
            ahora = time.monotonic()
            instruccion = _INSTRUCCIONES[len(vectores) % len(_INSTRUCCIONES)]
            progreso = f"{len(vectores)}/{n_objetivo}"

            if len(rostros) == 0:
                aviso = "Acercate a la camara"
            elif len(rostros) > 1:
                aviso = "Mas de una persona: esperando..."
            else:
                aviso = instruccion
                if ahora - ultimo_captura >= _INTERVALO_S:
                    vectores.append(rostros[0].vector)
                    calidades.append(rostros[0].puntaje)
                    ultimo_captura = ahora

            _dibujar_hud(imagen, progreso, aviso)
            cv2.imshow("Enrolamiento", imagen)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                print("Enrolamiento cancelado.")
                return

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(vectores) < n_objetivo:
        print("No se completó el enrolamiento.")
        return

    con = conectar(ruta_db)
    perfil_id = crear_perfil(con, nombre, vectores, _MODELO, calidades)
    print(f"Perfil guardado: {nombre!r} (id={perfil_id}, {len(vectores)} vectores)")


def listar_perfiles(ruta_db: str) -> None:
    """Imprime los perfiles activos en la base."""
    con = conectar(ruta_db)
    filas = con.execute(
        "SELECT id, nombre, modelo FROM perfil WHERE activo = 1"
    ).fetchall()
    if not filas:
        print("No hay perfiles enrolados.")
        return
    for f in filas:
        print(f"  id={f['id']}  nombre={f['nombre']!r}  modelo={f['modelo']}")


def _dibujar_hud(imagen: np.ndarray, progreso: str, aviso: str) -> None:
    """Dibuja el progreso y el aviso en la imagen para guiar al usuario."""
    h = imagen.shape[0]
    cv2.putText(imagen, progreso, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(imagen, aviso, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrola una persona en la base")
    parser.add_argument("--nombre", type=str, default=None)
    parser.add_argument("--fuente", default=0)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--listar", action="store_true")
    parser.add_argument("--db", default="data/control.db")
    parser.add_argument("--yunet", default="data/modelos/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--sface", default="data/modelos/face_recognition_sface_2021dec.onnx")
    args = parser.parse_args()

    if args.listar:
        listar_perfiles(args.db)
        sys.exit(0)

    if not args.nombre:
        parser.error("--nombre es obligatorio para enrolar")

    fuente = int(args.fuente) if str(args.fuente).isdigit() else args.fuente
    enrolar(args.nombre, fuente, args.n, args.yunet, args.sface, args.db)
