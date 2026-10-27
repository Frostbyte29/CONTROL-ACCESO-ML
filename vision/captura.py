"""Captura de video en hilo propio: índice, URL RTSP o archivo."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Cuadro:
    """Un cuadro de video con su marca de tiempo e índice de lectura."""

    imagen: np.ndarray
    t: float      # segundos; mismo reloj que recibe Contador.actualizar()
    indice: int   # 0, 1, 2... en orden de lectura


class Captura:
    """Lee video en un hilo propio y expone los cuadros a través de una cola."""

    _TAMANIO_COLA = 16

    def __init__(
        self,
        fuente: int | str,
        ancho: int | None = None,
        alto: int | None = None,
    ) -> None:
        self._fuente = fuente
        self._ancho = ancho
        self._alto = alto
        self._cola: queue.Queue[Cuadro] = queue.Queue(maxsize=self._TAMANIO_COLA)
        self._detener = threading.Event()
        self._terminado = False
        self._hilo: threading.Thread | None = None

    @property
    def es_archivo(self) -> bool:
        """True si la fuente es una ruta de archivo (no índice ni URL de red)."""
        if isinstance(self._fuente, int):
            return False
        s = str(self._fuente)
        return not (
            s.startswith("http://")
            or s.startswith("https://")
            or s.startswith("rtsp://")
        )

    @property
    def terminado(self) -> bool:
        """True cuando la fuente se agotó o cerró y la cola está vacía."""
        return self._terminado and self._cola.empty()

    def iniciar(self) -> None:
        """Abre la fuente y arranca el hilo de lectura."""
        cap = cv2.VideoCapture(self._fuente)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir la fuente: {self._fuente!r}")
        if not self.es_archivo and self._ancho and self._alto:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._alto)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t_inicio = time.monotonic()
        self._detener.clear()
        self._terminado = False
        self._hilo = threading.Thread(
            target=self._leer_hilo,
            args=(cap, fps, t_inicio),
            daemon=True,
        )
        self._hilo.start()

    def _leer_hilo(self, cap: cv2.VideoCapture, fps: float, t_inicio: float) -> None:
        """Lee cuadros y los coloca en la cola hasta que la fuente se agote o se detenga."""
        indice = 0
        while not self._detener.is_set():
            ok, imagen = cap.read()
            if not ok:
                break
            t = indice / fps if self.es_archivo else time.monotonic() - t_inicio
            cuadro = Cuadro(imagen=imagen, t=t, indice=indice)
            indice += 1
            if self.es_archivo:
                # Pone con timeout corto para que detener() funcione aunque la cola esté llena
                while not self._detener.is_set():
                    try:
                        self._cola.put(cuadro, timeout=0.1)
                        break
                    except queue.Full:
                        pass
            else:
                # Fuente en vivo: solo conserva el cuadro más reciente
                while not self._cola.empty():
                    try:
                        self._cola.get_nowait()
                    except queue.Empty:
                        break
                self._cola.put(cuadro)
        cap.release()
        self._terminado = True

    def leer(self, timeout_s: float = 1.0) -> Cuadro | None:
        """Devuelve el siguiente cuadro o None si pasa timeout_s sin uno nuevo."""
        if self.terminado:
            return None
        try:
            return self._cola.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def detener(self) -> None:
        """Señala al hilo que se detenga y espera hasta 2 s a que termine."""
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=2.0)
