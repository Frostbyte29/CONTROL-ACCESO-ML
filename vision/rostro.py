"""Detección y extracción de vectores de rostro con YuNet y SFace."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rostro:
    """Rostro detectado con su vector normalizado, puntaje y ancho en píxeles."""

    vector: np.ndarray   # float32, 128 dimensiones, normalizado en L2
    puntaje: float
    ancho_px: float


def normalizar(v: np.ndarray) -> np.ndarray:
    """Normaliza un vector en L2; devuelve el mismo vector si la norma es cero."""
    norma = np.linalg.norm(v)
    return v / norma if norma > 0 else v


def centroide(vectores: list[np.ndarray]) -> np.ndarray:
    """Promedia los vectores y normaliza el resultado en L2."""
    media = np.mean(np.stack(vectores, axis=0), axis=0)
    return normalizar(media)


class ExtractorRostro:
    """Carga los modelos una vez y extrae vectores de rostros en cajas de personas."""

    def __init__(
        self,
        ruta_yunet: str,
        ruta_sface: str,
        confianza_min: float = 0.9,
        lado_min_px: int = 40,
    ) -> None:
        # Los warnings de cv2 DNN sobre targets son normales con la versión 5.x
        self._detector = cv2.FaceDetectorYN.create(ruta_yunet, "", (320, 320))
        self._reconocedor = cv2.FaceRecognizerSF.create(ruta_sface, "")
        self._confianza_min = confianza_min
        self._lado_min_px = lado_min_px

    def buscar(
        self,
        imagen: np.ndarray,
        caja: tuple[float, float, float, float],
    ) -> Rostro | None:
        """Busca el mejor rostro en la mitad superior de la caja de persona."""
        recorte, offset = _recortar_zona_rostro(imagen, caja)
        if recorte is None:
            return None

        h, w = recorte.shape[:2]
        self._detector.setInputSize((w, h))
        _, caras = self._detector.detect(recorte)
        if caras is None or len(caras) == 0:
            return None

        # Mejor cara por puntaje (columna 14)
        mejor = caras[int(np.argmax(caras[:, 14]))]
        puntaje = float(mejor[14])
        ancho = float(mejor[2])

        if puntaje < self._confianza_min or ancho < self._lado_min_px:
            return None

        recortado = self._reconocedor.alignCrop(recorte, mejor)
        raw = self._reconocedor.feature(recortado)
        vector = normalizar(raw.flatten().astype(np.float32))

        return Rostro(vector=vector, puntaje=puntaje, ancho_px=ancho)

    def buscar_en_cuadro(self, imagen: np.ndarray) -> list[Rostro]:
        """Detecta todos los rostros en el cuadro completo (para enrolamiento)."""
        h, w = imagen.shape[:2]
        self._detector.setInputSize((w, h))
        _, caras = self._detector.detect(imagen)
        if caras is None or len(caras) == 0:
            return []

        rostros = []
        for cara in caras:
            puntaje = float(cara[14])
            ancho = float(cara[2])
            if puntaje < self._confianza_min or ancho < self._lado_min_px:
                continue
            recortado = self._reconocedor.alignCrop(imagen, cara)
            raw = self._reconocedor.feature(recortado)
            vector = normalizar(raw.flatten().astype(np.float32))
            rostros.append(Rostro(vector=vector, puntaje=puntaje, ancho_px=ancho))

        return rostros


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _recortar_zona_rostro(
    imagen: np.ndarray,
    caja: tuple[float, float, float, float],
) -> tuple[np.ndarray | None, tuple[int, int]]:
    """Recorta la mitad superior de la caja con un margen del 10 %, respetando bordes."""
    img_h, img_w = imagen.shape[:2]
    x1, y1, x2, y2 = caja

    alto_caja = y2 - y1
    mitad_y = y1 + alto_caja / 2.0

    margen_x = (x2 - x1) * 0.10
    margen_y = alto_caja * 0.10

    rx1 = max(0, int(x1 - margen_x))
    ry1 = max(0, int(y1 - margen_y))
    rx2 = min(img_w, int(x2 + margen_x))
    ry2 = min(img_h, int(mitad_y + margen_y))

    if rx2 <= rx1 or ry2 <= ry1:
        return None, (0, 0)

    return imagen[ry1:ry2, rx1:rx2], (rx1, ry1)
