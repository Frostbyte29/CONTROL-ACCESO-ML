"""Detección y seguimiento de personas con YOLOv8 + ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Persona:
    """Una persona detectada y seguida en un cuadro."""

    track_id: int
    caja: tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles del cuadro original
    confianza: float
    pies: tuple[float, float]                # centro inferior de la caja: ((x1+x2)/2, y2)


def a_personas(resultado) -> list[Persona]:
    """Convierte un resultado de Ultralytics en lista de Persona; pura, sin modelo."""
    cajas = resultado.boxes
    if cajas is None or cajas.id is None:
        return []
    personas = []
    for xyxy, tid, conf in zip(cajas.xyxy, cajas.id, cajas.conf):
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        personas.append(
            Persona(
                track_id=int(tid),
                caja=(x1, y1, x2, y2),
                confianza=float(conf),
                pies=((x1 + x2) / 2, y2),
            )
        )
    return personas


class DetectorPersonas:
    """Detecta y sigue personas en cuadros sucesivos con YOLOv8 + ByteTrack."""

    def __init__(self, pesos: str = "yolov8n.pt", confianza: float = 0.35) -> None:
        from ultralytics import YOLO
        self._modelo = YOLO(pesos)
        self._confianza = confianza

    def procesar(self, imagen: np.ndarray) -> list[Persona]:
        """Detecta personas en el cuadro y retorna la lista con su track_id."""
        resultados = self._modelo.track(
            imagen,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
            conf=self._confianza,
        )
        return a_personas(resultados[0])
