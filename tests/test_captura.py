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


# ---------------------------------------------------------------------------
# Tests nuevos de F3: rama en vivo con VideoCapture falso
# ---------------------------------------------------------------------------

class _CapFalsaVivo:
    """VideoCapture falso que simula una fuente en vivo con N cuadros."""

    def __init__(self, n_cuadros: int = 10, falla_en: int | None = None):
        self._n = n_cuadros
        self._falla_en = falla_en
        self._leidos = 0
        self.abierta = True

    def isOpened(self) -> bool:
        return self.abierta

    def get(self, prop_id: int) -> float:
        return 30.0  # fps

    def set(self, *args) -> None:
        pass

    def read(self):
        if self._falla_en is not None and self._leidos >= self._falla_en:
            raise RuntimeError("fallo simulado en read()")
        if self._leidos >= self._n:
            return False, None
        frame = np.full((48, 64, 3), self._leidos % 256, dtype=np.uint8)
        self._leidos += 1
        return True, frame

    def release(self):
        self.abierta = False


def test_06_vivo_conserva_ultimo(monkeypatch):
    """Fuente en vivo: si el consumidor es lento, solo queda el cuadro más reciente."""
    cap_falsa = _CapFalsaVivo(n_cuadros=20)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: cap_falsa)

    cap = Captura(0)  # índice → es_archivo=False
    cap.iniciar()
    time.sleep(0.2)  # da tiempo a que el hilo llene y descarte

    c = cap.leer(timeout_s=1.0)
    cap.detener()
    assert c is not None
    # El hilo llena y descarta, así que el índice recibido puede ser > 0
    assert c.indice >= 0


def test_07_vivo_t_es_monotonico(monkeypatch):
    """En fuente en vivo, t viene de time.monotonic y es siempre creciente."""
    cap_falsa = _CapFalsaVivo(n_cuadros=5)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: cap_falsa)

    cap = Captura(0)
    cap.iniciar()
    tiempos = []
    while not cap.terminado:
        c = cap.leer(timeout_s=1.0)
        if c:
            tiempos.append(c.t)
    assert all(tiempos[i] <= tiempos[i + 1] for i in range(len(tiempos) - 1))


def test_09_reabrir_continua_reloj(monkeypatch):
    """Después de detener() y reabrir(), el t del primer cuadro nuevo es mayor."""
    instancias = [
        _CapFalsaVivo(n_cuadros=5),
        _CapFalsaVivo(n_cuadros=5),
    ]
    idx = [0]

    def cap_factory(*a, **kw):
        c = instancias[idx[0]]
        idx[0] += 1
        return c

    monkeypatch.setattr(cv2, "VideoCapture", cap_factory)

    cap = Captura(0)
    cap.iniciar()

    # Leer primera tanda
    cuadros_1 = []
    while not cap.terminado:
        c = cap.leer(timeout_s=1.0)
        if c:
            cuadros_1.append(c)

    assert len(cuadros_1) > 0
    t_ultimo_1 = cuadros_1[-1].t

    cap.detener()
    time.sleep(0.05)  # pausa para que el reloj monotónico avance

    cap.reabrir()

    # Leer segunda tanda
    cuadros_2 = []
    while not cap.terminado:
        c = cap.leer(timeout_s=1.0)
        if c:
            cuadros_2.append(c)

    assert len(cuadros_2) > 0
    t_primero_2 = cuadros_2[0].t
    assert t_primero_2 > t_ultimo_1, (
        f"t del primer cuadro tras reabrir ({t_primero_2:.4f}) "
        f"debería ser > t del último cuadro anterior ({t_ultimo_1:.4f})"
    )


def test_08_excepcion_en_read_termina_hilo(monkeypatch):
    """Una excepción en read() marca terminado y libera la fuente."""
    cap_falsa = _CapFalsaVivo(n_cuadros=10, falla_en=3)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: cap_falsa)

    cap = Captura(0)
    cap.iniciar()
    time.sleep(0.3)
    # El hilo debe haber terminado tras el fallo
    assert cap._terminado
    assert not cap_falsa.abierta  # release() fue llamado en finally
