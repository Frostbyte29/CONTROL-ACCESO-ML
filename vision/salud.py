"""Monitor de salud de la cámara: detecta imagen tapada o caída."""

from __future__ import annotations

import numpy as np


class MonitorSalud:
    """Emite eventos de cámara tapada, caída o recuperada según el estado de la imagen."""

    def __init__(
        self,
        brillo_min: float = 25.0,
        variacion_min: float = 8.0,
        segundos: float = 2.0,
    ) -> None:
        self._brillo_min = brillo_min
        self._variacion_min = variacion_min
        self._segundos = segundos

        self._t_mala: float | None = None      # momento en que la imagen empezó a ser mala
        self._t_buena: float | None = None     # momento en que la imagen empezó a recuperarse
        self._tapada_emitida = False
        self._caida_emitida = False

    def actualizar(self, imagen: np.ndarray, t: float) -> tuple[str, float] | None:
        """Evalúa la imagen; devuelve (tipo, t_evento) o None.

        Para CAMARA_TAPADA, t_evento es el instante en que la imagen empezó a estar mala.
        Para CAMARA_OK, t_evento es el instante en que la imagen empezó a estar buena.
        """
        gris = imagen if imagen.ndim == 2 else imagen.mean(axis=2)
        brillo = float(gris.mean())
        variacion = float(gris.std())

        imagen_mala = brillo < self._brillo_min or variacion < self._variacion_min

        if imagen_mala:
            self._t_buena = None
            if self._t_mala is None:
                self._t_mala = t
            elif not self._tapada_emitida and (t - self._t_mala) >= self._segundos:
                self._tapada_emitida = True
                return ("CAMARA_TAPADA", self._t_mala)
        else:
            self._t_mala = None
            if self._tapada_emitida or self._caida_emitida:
                if self._t_buena is None:
                    self._t_buena = t
                elif (t - self._t_buena) >= 1.0:
                    self._tapada_emitida = False
                    self._caida_emitida = False
                    t_ok = self._t_buena
                    self._t_buena = None
                    return ("CAMARA_OK", t_ok)

        return None

    def sin_senal(self, t: float) -> str | None:
        """Emite CAMARA_CAIDA una sola vez cuando no llegan cuadros."""
        if not self._caida_emitida:
            self._caida_emitida = True
            self._tapada_emitida = False
            self._t_mala = None
            self._t_buena = None
            return "CAMARA_CAIDA"
        return None
