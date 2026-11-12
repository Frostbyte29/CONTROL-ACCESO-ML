"""Tests de vision.identidad con vectores sintéticos."""

from __future__ import annotations

import numpy as np
import pytest

from vision.identidad import (
    NO_IDENTIFICADO,
    PAGANTE,
    PERSONAL,
    Identificador,
)
from vision.rostro import Rostro, normalizar


def _rostro(v: np.ndarray, puntaje: float = 0.95) -> Rostro:
    """Crea un Rostro con vector normalizado para pruebas."""
    return Rostro(vector=normalizar(v.astype(np.float32)), puntaje=puntaje, ancho_px=60)


def _perfiles_ejemplo():
    """Dos perfiles con centroides ortogonales para pruebas limpias."""
    c1 = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    c2 = normalizar(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    return {1: ("Ana", c1), 2: ("Luis", c2)}


# ---------------------------------------------------------------------------
# test_01: reconoce PERSONAL cuando similitud >= umbral_alto
# ---------------------------------------------------------------------------

def test_01_personal(tmp_path):
    """Track con vector idéntico al perfil 1: categoría PERSONAL."""
    perfiles = _perfiles_ejemplo()
    ident = Identificador(perfiles, umbral_alto=0.8, umbral_bajo=0.3, min_rostros=1)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(3):
        ident.agregar(1, _rostro(v))
    cat, pid, sim = ident.decidir(1)
    assert cat == PERSONAL
    assert pid == 1
    assert sim >= 0.8


# ---------------------------------------------------------------------------
# test_02: PAGANTE cuando similitud baja y suficientes rostros
# ---------------------------------------------------------------------------

def test_02_pagante():
    """Track con vector distante de todos los perfiles y suficientes rostros: PAGANTE."""
    perfiles = _perfiles_ejemplo()
    ident = Identificador(perfiles, umbral_alto=0.8, umbral_bajo=0.3, min_rostros=3)
    # Vector ortogonal a ambos perfiles → similitud ≈ 0
    v = normalizar(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    for _ in range(3):
        ident.agregar(2, _rostro(v))
    cat, pid, sim = ident.decidir(2)
    assert cat == PAGANTE
    assert pid is None


# ---------------------------------------------------------------------------
# test_03: NO_IDENTIFICADO por similitud intermedia
# ---------------------------------------------------------------------------

def test_03_no_identificado_similitud_intermedia():
    """Similitud entre umbrales → NO_IDENTIFICADO aunque tenga suficientes rostros."""
    perfiles = _perfiles_ejemplo()
    ident = Identificador(perfiles, umbral_alto=0.9, umbral_bajo=0.1, min_rostros=3)
    # Vector a 45° del perfil 1: similitud coseno ≈ 0.707
    v = normalizar(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    for _ in range(5):
        ident.agregar(3, _rostro(v))
    cat, pid, sim = ident.decidir(3)
    assert cat == NO_IDENTIFICADO
    assert 0.1 <= sim <= 0.9


# ---------------------------------------------------------------------------
# test_04: NO_IDENTIFICADO por pocos rostros
# ---------------------------------------------------------------------------

def test_04_no_identificado_pocos_rostros():
    """Menos de min_rostros: NO_IDENTIFICADO aunque la similitud sea baja."""
    perfiles = _perfiles_ejemplo()
    ident = Identificador(perfiles, umbral_alto=0.8, umbral_bajo=0.3, min_rostros=5)
    v = normalizar(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    for _ in range(2):  # solo 2 < min_rostros=5
        ident.agregar(4, _rostro(v))
    cat, pid, sim = ident.decidir(4)
    assert cat == NO_IDENTIFICADO


# ---------------------------------------------------------------------------
# test_05: sin perfiles cargados, track con suficientes rostros es PAGANTE
# ---------------------------------------------------------------------------

def test_05_sin_perfiles():
    """Sin perfiles, un track con suficientes rostros es PAGANTE."""
    ident = Identificador({}, umbral_alto=0.8, umbral_bajo=0.3, min_rostros=3)
    v = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    for _ in range(3):
        ident.agregar(5, _rostro(v))
    cat, pid, sim = ident.decidir(5)
    assert cat == PAGANTE
    assert pid is None


# ---------------------------------------------------------------------------
# test_06: umbrales invertidos lanzan ValueError
# ---------------------------------------------------------------------------

def test_06_umbrales_invertidos():
    """umbral_bajo > umbral_alto debe lanzar ValueError."""
    with pytest.raises(ValueError):
        Identificador({}, umbral_alto=0.3, umbral_bajo=0.8)


# ---------------------------------------------------------------------------
# test_07: descartar elimina todos los vectores del track
# ---------------------------------------------------------------------------

def test_07_descartar():
    """Después de descartar, decidir devuelve NO_IDENTIFICADO sin vectores."""
    ident = Identificador({}, umbral_alto=0.8, umbral_bajo=0.3, min_rostros=1)
    v = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    ident.agregar(6, _rostro(v))
    ident.descartar(6)
    cat, pid, sim = ident.decidir(6)
    assert cat == NO_IDENTIFICADO
    # Confirma que no quedan vectores en memoria
    assert 6 not in ident._tracks


# ---------------------------------------------------------------------------
# test_08: max_rostros conserva solo los de mayor puntaje
# ---------------------------------------------------------------------------

def test_08_max_rostros():
    """max_rostros=3: se conservan los 3 de mayor puntaje; el bajo se descarta."""
    ident = Identificador({}, umbral_alto=0.8, umbral_bajo=0.3, min_rostros=1,
                          max_rostros=3)
    v = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # 4 rostros con puntajes distintos
    ident.agregar(7, _rostro(v, puntaje=0.5))
    ident.agregar(7, _rostro(v, puntaje=0.9))
    ident.agregar(7, _rostro(v, puntaje=0.7))
    ident.agregar(7, _rostro(v, puntaje=0.6))  # este desplaza al 0.5

    datos = ident._tracks[7]
    assert datos.cantidad == 3
    puntajes = sorted([item[0] for item in datos._heap])
    # Los 3 mejores: 0.6, 0.7, 0.9
    assert puntajes == pytest.approx([0.6, 0.7, 0.9])
