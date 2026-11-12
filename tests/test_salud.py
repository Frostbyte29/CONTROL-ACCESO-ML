"""Tests de vision.salud con imágenes sintéticas."""

from __future__ import annotations

import numpy as np
import pytest

from vision.salud import MonitorSalud


def _oscura(h=64, w=64) -> np.ndarray:
    """Imagen negra: brillo=0, variación=0 → imagen mala."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _clara(h=64, w=64) -> np.ndarray:
    """Imagen gris uniforme con variación cero: brillo alto pero variación baja."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _normal(h=64, w=64) -> np.ndarray:
    """Imagen con brillo y variación suficientes."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h//2, :w//2] = 200
    img[h//2:, w//2:] = 50
    return img


# ---------------------------------------------------------------------------
# test_01: CAMARA_TAPADA se emite una sola vez tras segundos de imagen mala
# ---------------------------------------------------------------------------

def test_01_tapada_emitida_tras_segundos():
    """Imagen oscura 3 s con segundos=2: emite CAMARA_TAPADA exactamente una vez."""
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    img = _oscura()
    resultados = []
    for i in range(30):
        ev = monitor.actualizar(img, t=float(i) * 0.1)
        if ev:
            resultados.append(ev[0])
    assert resultados == ["CAMARA_TAPADA"]


def test_02_tapada_no_emite_antes_del_umbral():
    """Imagen oscura durante menos de segundos: no emite CAMARA_TAPADA."""
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    img = _oscura()
    resultados = [monitor.actualizar(img, t=float(i) * 0.1) for i in range(10)]
    assert all(r is None for r in resultados)


# ---------------------------------------------------------------------------
# test_03: CAMARA_OK al recuperarse
# ---------------------------------------------------------------------------

def test_03_camara_ok_tras_recuperacion():
    """Tapada → imagen normal 1.5 s: emite CAMARA_TAPADA y luego CAMARA_OK."""
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    eventos = []

    # 25 cuadros oscuros × 0.1 s = 2.5 s → CAMARA_TAPADA
    for i in range(25):
        ev = monitor.actualizar(_oscura(), t=float(i) * 0.1)
        if ev:
            eventos.append(ev[0])

    # 15 cuadros normales × 0.1 s = 1.5 s → CAMARA_OK
    base = 2.5
    for j in range(15):
        ev = monitor.actualizar(_normal(), t=base + float(j) * 0.1)
        if ev:
            eventos.append(ev[0])

    assert "CAMARA_TAPADA" in eventos
    assert "CAMARA_OK" in eventos
    assert eventos.index("CAMARA_TAPADA") < eventos.index("CAMARA_OK")


# ---------------------------------------------------------------------------
# test_04: CAMARA_CAIDA se emite una sola vez
# ---------------------------------------------------------------------------

def test_04_caida_una_sola_vez():
    """sin_senal emite CAMARA_CAIDA la primera vez y None las siguientes."""
    monitor = MonitorSalud()
    assert monitor.sin_senal(0.0) == "CAMARA_CAIDA"
    assert monitor.sin_senal(1.0) is None
    assert monitor.sin_senal(2.0) is None


# ---------------------------------------------------------------------------
# test_05: imagen normal nunca emite nada
# ---------------------------------------------------------------------------

def test_05_imagen_normal_sin_eventos():
    """Imagen siempre normal: no emite ningún evento."""
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    img = _normal()
    resultados = [monitor.actualizar(img, t=float(i) * 0.1) for i in range(50)]
    assert all(r is None for r in resultados)


# ---------------------------------------------------------------------------
# test_06: variación baja también activa CAMARA_TAPADA
# ---------------------------------------------------------------------------

def test_06_variacion_baja_activa_tapada():
    """Imagen uniforme (variación=0) activa CAMARA_TAPADA aunque el brillo sea alto."""
    monitor = MonitorSalud(brillo_min=10, variacion_min=8, segundos=2.0)
    img = _clara()  # brillo=128 pero variación≈0
    resultados = []
    for i in range(30):
        ev = monitor.actualizar(img, t=float(i) * 0.1)
        if ev:
            resultados.append(ev[0])
    assert "CAMARA_TAPADA" in resultados
