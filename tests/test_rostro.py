"""Tests de vision.rostro: funciones puras y humo con modelos."""

from __future__ import annotations

import os

import numpy as np
import pytest

from vision.rostro import ExtractorRostro, centroide, normalizar

_YUNET = "data/modelos/face_detection_yunet_2023mar.onnx"
_SFACE = "data/modelos/face_recognition_sface_2021dec.onnx"
_MODELOS_PRESENTES = os.path.exists(_YUNET) and os.path.exists(_SFACE)


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------

def test_01_normalizar():
    """normalizar devuelve un vector unitario."""
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = normalizar(v)
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-6)


def test_02_normalizar_cero():
    """normalizar no falla con vector cero."""
    v = np.zeros(4, dtype=np.float32)
    n = normalizar(v)
    assert np.all(n == 0.0)


def test_03_centroide():
    """centroide de dos vectores opuestos es cero o muy pequeño."""
    v1 = normalizar(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = normalizar(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    c = centroide([v1, v2])
    assert np.linalg.norm(c) == pytest.approx(0.0, abs=1e-6)


def test_04_centroide_normalizado():
    """centroide de vectores en la misma dirección es unitario."""
    v1 = normalizar(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v2 = normalizar(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    c = centroide([v1, v2])
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests con modelos reales (se saltan si faltan los .onnx)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MODELOS_PRESENTES, reason="modelos onnx ausentes")
def test_05_imagen_negra_no_detecta():
    """Una imagen negra no produce ningún rostro."""
    extractor = ExtractorRostro(_YUNET, _SFACE)
    imagen = np.zeros((480, 640, 3), dtype=np.uint8)
    caja = (100.0, 100.0, 300.0, 400.0)
    resultado = extractor.buscar(imagen, caja)
    assert resultado is None


@pytest.mark.skipif(not _MODELOS_PRESENTES, reason="modelos onnx ausentes")
def test_06_buscar_en_cuadro_negro():
    """buscar_en_cuadro sobre imagen negra devuelve lista vacía."""
    extractor = ExtractorRostro(_YUNET, _SFACE)
    imagen = np.zeros((480, 640, 3), dtype=np.uint8)
    resultados = extractor.buscar_en_cuadro(imagen)
    assert resultados == []
