"""Tests de vision.captura."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from vision.captura import Captura


def crear_video(ruta, n_cuadros: int = 30, ancho: int = 64, alto: int = 48, fps: int = 15):
    """Genera un video .avi de prueba con cv2.VideoWriter."""
    writer = cv2.VideoWriter(
        str(ruta),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (ancho, alto),
    )
    for i in range(n_cuadros):
        frame = np.full((alto, ancho, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return ruta


def test_01_lee_en_orden(tmp_path):
    """30 cuadros en orden, t=indice/15, luego None y terminado True."""
    video = crear_video(tmp_path / "test.avi")
    cap = Captura(str(video))
    cap.iniciar()

    cuadros = []
    while True:
        c = cap.leer(timeout_s=2.0)
        if c is None:
            break
        cuadros.append(c)

    assert len(cuadros) == 30
    for i, c in enumerate(cuadros):
        assert c.indice == i
        assert abs(c.t - i / 15) < 1e-6
    assert cap.terminado


def test_02_consumidor_lento(tmp_path):
    """Consumidor que duerme 50 ms: archivo no descarta cuadros."""
    video = crear_video(tmp_path / "test.avi")
    cap = Captura(str(video))
    cap.iniciar()

    recibidos = []
    while True:
        c = cap.leer(timeout_s=5.0)
        if c is None:
            break
        recibidos.append(c.indice)
        time.sleep(0.05)

    assert recibidos == list(range(30))


def test_03_ruta_inexistente():
    """Ruta inexistente: iniciar() lanza ValueError."""
    cap = Captura("/no/existe/video.avi")
    with pytest.raises(ValueError):
        cap.iniciar()


def test_04_detener_con_cola_llena(tmp_path):
    """detener() termina en < 2 s aunque nadie lea y la cola esté llena."""
    video = crear_video(tmp_path / "test.avi")
    cap = Captura(str(video))
    cap.iniciar()

    # Espera suficiente para que el hilo llene la cola (maxsize=16) y quede bloqueado
    time.sleep(0.3)

    t0 = time.monotonic()
    cap.detener()
    assert time.monotonic() - t0 < 2.0


def test_05_es_archivo():
    """es_archivo True para ruta de archivo, False para índice entero."""
    assert Captura("video.mp4").es_archivo is True
    assert Captura("otro/path/clip.avi").es_archivo is True
    assert Captura(0).es_archivo is False
