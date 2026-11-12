"""Dibuja el estado del sistema sobre la imagen de video (sin modificar el estado)."""

from __future__ import annotations

import cv2
import numpy as np

from vision.motor import EstadoVista

# Colores BGR
_BLANCO = (255, 255, 255)
_NEGRO = (0, 0, 0)
_BANDA_COLOR = (0, 200, 255)
_PANEL_BG = (30, 30, 30)


def dibujar(
    imagen: np.ndarray,
    estado: EstadoVista,
    banda: tuple[float, float, float, float, float],
    mostrar_monto: bool,
) -> np.ndarray:
    """Dibuja todo el HUD sobre una copia de la imagen; no modifica el estado."""
    salida = imagen.copy()

    _dibujar_banda(salida, banda)
    _dibujar_personas(salida, estado)
    _dibujar_panel(salida, estado, mostrar_monto)

    if estado.avisos:
        _dibujar_aviso(salida, estado.avisos[0])

    _dibujar_fps(salida, estado.fps)

    return salida


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _dibujar_banda(imagen: np.ndarray, banda: tuple) -> None:
    """Dibuja la línea central y los dos bordes de la banda."""
    x1, y1, x2, y2, ancho = banda
    # Vector normal para los bordes
    dx, dy = x2 - x1, y2 - y1
    largo = (dx * dx + dy * dy) ** 0.5
    if largo == 0:
        return
    nx, ny = -dy / largo, dx / largo
    mitad = ancho / 2

    def p_offset(factor: float):
        return (
            int(x1 + nx * factor), int(y1 + ny * factor),
            int(x2 + nx * factor), int(y2 + ny * factor),
        )

    # Línea central
    cv2.line(imagen, (int(x1), int(y1)), (int(x2), int(y2)), _BANDA_COLOR, 2)
    # Bordes
    bx1, by1, bx2, by2 = p_offset(mitad)
    cv2.line(imagen, (bx1, by1), (bx2, by2), _BANDA_COLOR, 1)
    bx1, by1, bx2, by2 = p_offset(-mitad)
    cv2.line(imagen, (bx1, by1), (bx2, by2), _BANDA_COLOR, 1)


def _dibujar_personas(imagen: np.ndarray, estado: EstadoVista) -> None:
    """Dibuja recuadros, etiquetas y recorridos de cada persona."""
    for p in estado.personas:
        x1, y1, x2, y2 = (int(v) for v in p.caja)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), p.color, 2)
        if p.etiqueta:
            cv2.putText(imagen, p.etiqueta, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, p.color, 2)
        # Recorrido
        pts = p.recorrido
        for i in range(1, len(pts)):
            cv2.line(imagen,
                     (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])),
                     p.color, 1)


def _dibujar_panel(
    imagen: np.ndarray,
    estado: EstadoVista,
    mostrar_monto: bool,
) -> None:
    """Dibuja el panel de contadores en la esquina superior izquierda."""
    lineas = [
        f"Entradas: {estado.entradas}",
        f"Salidas:  {estado.salidas}",
        f"Personal: {estado.personal}",
        f"Pagantes: {estado.pagantes}",
        f"No ident: {estado.no_identificados}",
    ]
    if mostrar_monto:
        confirmada = estado.recaudacion_confirmada / 100
        maxima = estado.recaudacion_maxima / 100
        lineas.append(f"Conf: S/{confirmada:.2f}")
        lineas.append(f"Max:  S/{maxima:.2f}")

    ancho_panel = 220
    alto_panel = len(lineas) * 22 + 12
    overlay = imagen.copy()
    cv2.rectangle(overlay, (5, 5), (ancho_panel, alto_panel), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, imagen, 0.4, 0, imagen)

    for i, linea in enumerate(lineas):
        cv2.putText(imagen, linea, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BLANCO, 1)


def _dibujar_aviso(imagen: np.ndarray, aviso: str) -> None:
    """Dibuja un aviso grande centrado sobre la imagen."""
    texto = "Camara tapada" if aviso == "CAMARA_TAPADA" else "Sin senal"
    h, w = imagen.shape[:2]
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    x = (w - tw) // 2
    y = h // 2
    cv2.putText(imagen, texto, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


def _dibujar_fps(imagen: np.ndarray, fps: float) -> None:
    """Dibuja los FPS en la esquina superior derecha."""
    h, w = imagen.shape[:2]
    texto = f"FPS: {fps:.1f}"
    cv2.putText(imagen, texto, (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BLANCO, 2)
