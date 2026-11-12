"""Identificación de personas con dos umbrales y tres categorías."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from vision.rostro import Rostro, centroide

# Categorías de persona
PERSONAL = "PERSONAL"
PAGANTE = "PAGANTE"
NO_IDENTIFICADO = "NO_IDENTIFICADO"


@dataclass
class _DatosTrack:
    """Acumula los mejores rostros de un track para decidir su categoría."""

    # heap de (puntaje, contador, vector) — min-heap; contador garantiza unicidad en el desempate
    _heap: list = field(default_factory=list)
    _contador: int = field(default=0)
    max_rostros: int = 10

    def agregar(self, rostro: Rostro) -> None:
        """Agrega el rostro manteniendo solo los max_rostros de mayor puntaje."""
        heapq.heappush(self._heap, (rostro.puntaje, self._contador, rostro.vector))
        self._contador += 1
        if len(self._heap) > self.max_rostros:
            heapq.heappop(self._heap)  # saca el de menor puntaje

    @property
    def vectores(self) -> list[np.ndarray]:
        """Lista de vectores acumulados."""
        return [item[2] for item in self._heap]

    @property
    def cantidad(self) -> int:
        return len(self._heap)


class Identificador:
    """Decide la categoría de cada track comparando contra perfiles enrolados."""

    def __init__(
        self,
        perfiles: dict[int, tuple[str, np.ndarray]],
        umbral_alto: float,
        umbral_bajo: float,
        min_rostros: int = 3,
        max_rostros: int = 10,
    ) -> None:
        if umbral_bajo > umbral_alto:
            raise ValueError(
                f"umbral_bajo ({umbral_bajo}) no puede ser mayor que umbral_alto ({umbral_alto})"
            )
        self._perfiles = perfiles  # {id: (nombre, centroide)}
        self._umbral_alto = umbral_alto
        self._umbral_bajo = umbral_bajo
        self._min_rostros = min_rostros
        self._max_rostros = max_rostros
        self._tracks: dict[int, _DatosTrack] = {}

    def agregar(self, track_id: int, rostro: Rostro) -> None:
        """Acumula el rostro para el track indicado."""
        if track_id not in self._tracks:
            self._tracks[track_id] = _DatosTrack(max_rostros=self._max_rostros)
        self._tracks[track_id].agregar(rostro)

    def decidir(
        self, track_id: int
    ) -> tuple[str, int | None, float]:
        """Decide la categoría del track; devuelve (categoria, perfil_id, similitud)."""
        datos = self._tracks.get(track_id)
        vectores = datos.vectores if datos else []

        if not vectores:
            return NO_IDENTIFICADO, None, 0.0

        c = centroide(vectores)
        mejor_id, mejor_sim = _mejor_perfil(c, self._perfiles)

        if mejor_sim >= self._umbral_alto:
            return PERSONAL, mejor_id, mejor_sim

        tiene_suficientes = len(vectores) >= self._min_rostros
        if tiene_suficientes and mejor_sim < self._umbral_bajo:
            return PAGANTE, None, mejor_sim

        return NO_IDENTIFICADO, None, mejor_sim

    def descartar(self, track_id: int) -> None:
        """Elimina de memoria todos los vectores del track."""
        self._tracks.pop(track_id, None)

    def nombre(self, perfil_id: int | None) -> str | None:
        """Devuelve el nombre del perfil o None si no existe o perfil_id es None."""
        if perfil_id is None:
            return None
        datos = self._perfiles.get(perfil_id)
        return datos[0] if datos else None


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _mejor_perfil(
    c: np.ndarray,
    perfiles: dict[int, tuple[str, np.ndarray]],
) -> tuple[int | None, float]:
    """Devuelve (perfil_id, similitud) del perfil más cercano, o (None, 0.0) si no hay."""
    if not perfiles:
        return None, 0.0

    mejor_id = None
    mejor_sim = -1.0
    for pid, (_nombre, cen) in perfiles.items():
        sim = float(np.dot(c, cen))  # similitud coseno (vectores normalizados)
        if sim > mejor_sim:
            mejor_sim = sim
            mejor_id = pid
    return mejor_id, mejor_sim
