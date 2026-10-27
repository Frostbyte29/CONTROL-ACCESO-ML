"""Herramienta de verificación: muestra detecciones, banda y conteo en tiempo real."""

from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from vision.captura import Captura
from vision.conteo import Banda, Contador
from vision.personas import DetectorPersonas


def _dibujar(
    img: np.ndarray,
    personas,
    banda: Banda,
    entradas: int,
    salidas: int,
    fps_real: float,
) -> None:
    """Dibuja cajas, banda y contadores sobre img (en el lugar)."""
    for p in personas:
        x1, y1, x2, y2 = map(int, p.caja)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(p.track_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Banda: línea central y dos bordes paralelos
    vx = banda.p2[0] - banda.p1[0]
    vy = banda.p2[1] - banda.p1[1]
    mag = math.sqrt(vx * vx + vy * vy)
    nx, ny = -vy / mag * banda.ancho / 2, vx / mag * banda.ancho / 2
    for dx, dy, color in [
        (0, 0, (0, 0, 255)),
        (nx, ny, (0, 180, 255)),
        (-nx, -ny, (0, 180, 255)),
    ]:
        pt1 = (int(banda.p1[0] + dx), int(banda.p1[1] + dy))
        pt2 = (int(banda.p2[0] + dx), int(banda.p2[1] + dy))
        cv2.line(img, pt1, pt2, color, 2)

    cv2.putText(img, f"E:{entradas}  S:{salidas}  {fps_real:.0f} fps",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifica captura y conteo sobre un clip.")
    ap.add_argument("fuente", help="Índice de cámara, URL o ruta de archivo")
    ap.add_argument("--banda", required=True,
                    help="x1,y1,x2,y2,ancho — línea central y ancho de la banda")
    ap.add_argument("--adentro-negativo", action="store_true",
                    help="Invierte el lado adentro de la banda")
    args = ap.parse_args()

    try:
        fuente: int | str = int(args.fuente)
    except ValueError:
        fuente = args.fuente

    partes = [float(v) for v in args.banda.split(",")]
    banda = Banda(
        p1=(partes[0], partes[1]),
        p2=(partes[2], partes[3]),
        ancho=partes[4],
        adentro_positivo=not args.adentro_negativo,
    )

    captura = Captura(fuente)
    captura.iniciar()
    detector = DetectorPersonas()
    contador = Contador(banda)

    entradas = salidas = n_cuadros = 0
    t0 = time.monotonic()

    while True:
        cuadro = captura.leer(timeout_s=2.0)
        if cuadro is None:
            if captura.terminado:
                break
            continue

        personas = detector.procesar(cuadro.imagen)
        cruces = contador.actualizar(cuadro.t, {p.track_id: p.pies for p in personas})
        for c in cruces:
            if c.tipo == "ENTRADA":
                entradas += 1
            else:
                salidas += 1

        n_cuadros += 1
        transcurrido = time.monotonic() - t0
        fps_real = n_cuadros / transcurrido if transcurrido > 0 else 0.0

        img = cuadro.imagen.copy()
        _dibujar(img, personas, banda, entradas, salidas, fps_real)
        cv2.imshow("ver_clip", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.detener()
    cv2.destroyAllWindows()

    transcurrido = time.monotonic() - t0
    fps_medidos = n_cuadros / transcurrido if transcurrido > 0 else 0.0
    print(f"Cuadros procesados: {n_cuadros}")
    print(f"Tiempo transcurrido: {transcurrido:.2f} s")
    print(f"FPS medidos: {fps_medidos:.1f}")


if __name__ == "__main__":
    main()
