"""Tests de vision.camaras con VideoCapture falso."""

from __future__ import annotations

import numpy as np
import pytest

import vision.camaras as camaras_mod


class _CapFalsa:
    """VideoCapture falso para pruebas: simula apertura y lectura de un cuadro."""

    def __init__(self, abre: bool = True, lee_ok: bool = True):
        self._abre = abre
        self._lee_ok = lee_ok
        self.liberada = False

    def isOpened(self) -> bool:
        return self._abre

    def read(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8) if self._lee_ok else None
        return self._lee_ok, frame

    def release(self):
        self.liberada = True


def test_01_listar_dos_camaras(monkeypatch):
    """listar() devuelve las cámaras que abren y leen correctamente."""
    instancias: list[_CapFalsa] = []

    def cap_falsa(idx, *args, **kwargs):
        # solo los índices 0 y 2 'abren'
        abre = idx in (0, 2)
        inst = _CapFalsa(abre=abre, lee_ok=abre)
        instancias.append(inst)
        return inst

    monkeypatch.setattr(camaras_mod.cv2, "VideoCapture", cap_falsa)
    resultado = camaras_mod.listar(max_indice=4)
    indices = [r[0] for r in resultado]
    assert indices == [0, 2]
    # Todas las cámaras probadas deben liberarse
    assert all(inst.liberada for inst in instancias)


def test_02_listar_ninguna(monkeypatch):
    """listar() devuelve lista vacía si ninguna cámara abre."""
    monkeypatch.setattr(
        camaras_mod.cv2,
        "VideoCapture",
        lambda *a, **kw: _CapFalsa(abre=False),
    )
    resultado = camaras_mod.listar(max_indice=3)
    assert resultado == []


def test_03_elegir_una_sola(monkeypatch):
    """Con una sola cámara disponible, elegir() la devuelve sin mostrar ventana."""
    monkeypatch.setattr(
        camaras_mod,
        "listar",
        lambda *a, **kw: [(0, np.zeros((120, 160, 3), dtype=np.uint8))],
    )
    assert camaras_mod.elegir() == 0


def test_04_elegir_sin_camaras(monkeypatch):
    """elegir() lanza RuntimeError si no hay cámaras."""
    monkeypatch.setattr(camaras_mod, "listar", lambda *a, **kw: [])
    with pytest.raises(RuntimeError):
        camaras_mod.elegir()
