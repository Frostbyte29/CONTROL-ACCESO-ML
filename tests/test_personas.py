"""Tests de vision.personas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from vision.personas import DetectorPersonas, Persona, a_personas


def _hacer_resultado(datos: list, orig_shape: tuple = (480, 640)) -> Results:
    """Construye un Results mock con los datos de cajas dados."""
    orig_img = np.zeros((*orig_shape, 3), dtype=np.uint8)
    return Results(
        orig_img=orig_img,
        path="test",
        names={0: "person"},
        boxes=torch.tensor(datos, dtype=torch.float32),
    )


def test_06_a_personas_con_ids():
    """a_personas con 2 cajas con IDs: track_ids y pies correctos."""
    # Formato Boxes con track: [x1, y1, x2, y2, track_id, conf, cls]
    resultado = _hacer_resultado([
        [10.0, 20.0, 100.0, 200.0, 1.0, 0.9, 0.0],
        [150.0, 30.0, 250.0, 300.0, 2.0, 0.8, 0.0],
    ])
    personas = a_personas(resultado)

    assert len(personas) == 2
    assert personas[0].track_id == 1
    assert personas[0].pies == (55.0, 200.0)   # ((10+100)/2, 200)
    assert personas[1].track_id == 2
    assert personas[1].pies == (200.0, 300.0)  # ((150+250)/2, 300)


def test_07_a_personas_sin_ids():
    """a_personas con resultado sin IDs (6 columnas) devuelve []."""
    # Formato Boxes sin track: [x1, y1, x2, y2, conf, cls]
    resultado = _hacer_resultado([
        [10.0, 20.0, 100.0, 200.0, 0.9, 0.0],
    ])
    assert a_personas(resultado) == []


@pytest.mark.skipif(
    not Path("yolov8n.pt").exists(),
    reason="yolov8n.pt no está disponible en el directorio de trabajo",
)
def test_08_detector_imagen_negra():
    """DetectorPersonas sobre imagen negra (sin personas) devuelve []."""
    detector = DetectorPersonas()
    imagen = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.procesar(imagen) == []
