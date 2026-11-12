"""Enumeración y selección de cámaras disponibles."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np


def listar(max_indice: int = 5) -> list[tuple[int, np.ndarray]]:
    """Prueba índices de 0 a max_indice y devuelve [(indice, miniatura)] para los que abren."""
    disponibles = []
    usar_dshow = sys.platform == "win32"

    for idx in range(max_indice):
        cap = _abrir_camara(idx, usar_dshow)
        if cap is None:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            miniatura = cv2.resize(frame, (160, 120))
            disponibles.append((idx, miniatura))

    return disponibles


def elegir() -> int:
    """Devuelve el índice de la cámara a usar; pregunta si hay más de una."""
    camaras = listar()
    if not camaras:
        raise RuntimeError("No se encontró ninguna cámara disponible")
    if len(camaras) == 1:
        return camaras[0][0]
    return _seleccionar_en_ventana(camaras)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _abrir_camara(idx: int, preferir_dshow: bool) -> cv2.VideoCapture | None:
    """Intenta abrir la cámara con el backend más rápido disponible."""
    if preferir_dshow:
        t0 = time.monotonic()
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        dt_dshow = time.monotonic() - t0
        cap.release()

        t0 = time.monotonic()
        cap2 = cv2.VideoCapture(idx)
        if cap2.isOpened():
            dt_normal = time.monotonic() - t0
            if dt_normal <= dt_dshow:
                return cap2
            cap2.release()
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _seleccionar_en_ventana(
    camaras: list[tuple[int, np.ndarray]],
) -> int:
    """Muestra miniaturas numeradas y espera que el usuario presione 1-9."""
    n = len(camaras)
    alto = 120
    ancho_total = 160 * n
    panel = np.zeros((alto + 30, ancho_total, 3), dtype=np.uint8)

    for i, (idx, mini) in enumerate(camaras):
        x = i * 160
        panel[:alto, x : x + 160] = mini
        cv2.putText(
            panel,
            f"{i + 1}: cam {idx}",
            (x + 5, alto + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imshow("Elegir camara (tecla 1-9)", panel)
    while True:
        tecla = cv2.waitKey(0) & 0xFF
        num = tecla - ord("1")
        if 0 <= num < n:
            cv2.destroyAllWindows()
            return camaras[num][0]
