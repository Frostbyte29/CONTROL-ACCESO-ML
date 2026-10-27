"""Conteo de personas que cruzan una banda imaginaria frente a una puerta."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Banda:
    """Define la banda de conteo: línea central, ancho y sentido del interior."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    ancho: float
    adentro_positivo: bool = True

    def __post_init__(self) -> None:
        """Verifica que la banda esté bien definida antes de usarla."""
        if self.p1 == self.p2:
            raise ValueError("p1 y p2 no pueden ser el mismo punto")
        if self.ancho <= 0:
            raise ValueError("ancho debe ser positivo")


@dataclass(frozen=True)
class Cruce:
    """Registra que un track cruzó la banda en un instante dado."""

    track_id: int
    tipo: str  # "ENTRADA" o "SALIDA"
    t: float   # segundos, mismo reloj que recibe actualizar()


@dataclass
class _EstadoTrack:
    """Estado interno de la máquina de conteo para un track individual."""

    ultimo_lado_firme: Optional[str] = None  # "AFUERA" | "ADENTRO" | None
    lado_candidato: Optional[str] = None
    frames_en_candidato: int = 0
    ultimo_cruce_t: float = float("-inf")  # -inf para no bloquear el primer cruce
    ultimo_visto_t: float = 0.0


class Contador:
    """Mantiene la máquina de estados de todos los tracks y emite Cruce cuando corresponde."""

    def __init__(
        self,
        banda: Banda,
        confirmar_cuadros: int = 3,
        enfriamiento_s: float = 2.0,
        perdido_s: float = 1.5,
    ) -> None:
        self.banda = banda
        self.confirmar_cuadros = confirmar_cuadros
        self.enfriamiento_s = enfriamiento_s
        self.perdido_s = perdido_s
        self._estados: dict[int, _EstadoTrack] = {}

    def distancia_a_banda(self, punto: tuple[float, float]) -> float:
        """Distancia con signo a la línea central; positiva hacia el lado adentro."""
        vx = self.banda.p2[0] - self.banda.p1[0]
        vy = self.banda.p2[1] - self.banda.p1[1]
        mag = math.sqrt(vx * vx + vy * vy)
        nx, ny = -vy / mag, vx / mag          # normal a la derecha de p1→p2 en coordenadas de imagen
        dx = punto[0] - self.banda.p1[0]
        dy = punto[1] - self.banda.p1[1]
        d = dx * nx + dy * ny
        return d if self.banda.adentro_positivo else -d

    def actualizar(
        self,
        t: float,
        puntos: dict[int, tuple[float, float]],
    ) -> list[Cruce]:
        """Procesa un cuadro y retorna la lista de cruces detectados en él."""
        perdidos = [
            tid
            for tid, est in self._estados.items()
            if t - est.ultimo_visto_t > self.perdido_s
        ]
        for tid in perdidos:
            del self._estados[tid]

        cruces: list[Cruce] = []
        m = self.banda.ancho / 2

        for track_id, punto in puntos.items():
            if track_id not in self._estados:
                self._estados[track_id] = _EstadoTrack(ultimo_visto_t=t)
            est = self._estados[track_id]
            est.ultimo_visto_t = t

            d = self.distancia_a_banda(punto)
            if d < -m:
                lado = "AFUERA"
            elif d > m:
                lado = "ADENTRO"
            else:
                lado = "BANDA"

            if lado == "BANDA":
                est.lado_candidato = None
                est.frames_en_candidato = 0
                continue

            if lado == est.lado_candidato:
                est.frames_en_candidato += 1
            else:
                est.lado_candidato = lado
                est.frames_en_candidato = 1

            if (
                est.frames_en_candidato >= self.confirmar_cuadros
                and est.lado_candidato != est.ultimo_lado_firme
            ):
                if (
                    est.ultimo_lado_firme is not None
                    and t - est.ultimo_cruce_t > self.enfriamiento_s
                ):
                    tipo = "ENTRADA" if est.ultimo_lado_firme == "AFUERA" else "SALIDA"
                    cruces.append(Cruce(track_id=track_id, tipo=tipo, t=t))
                    est.ultimo_cruce_t = t
                est.ultimo_lado_firme = est.lado_candidato

        return cruces
