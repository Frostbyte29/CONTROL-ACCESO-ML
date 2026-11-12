"""Demo del sistema: bucle captura → motor → ventana.

Uso: python -m vision.demo [--config config.yaml] [--fuente X]
                           [--precio 150] [--ocultar-monto] [--sin-ventana]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import cv2

from datos.db import abrir_jornada, cargar_perfiles, conectar
from vision.camaras import elegir
from vision.captura import Captura
from vision.config import cargar as cargar_config
from vision.conteo import Banda, Contador
from vision.identidad import Identificador
from vision.motor import Motor
from vision.personas import DetectorPersonas
from vision.rostro import ExtractorRostro, centroide
from vision.salud import MonitorSalud
from vision.ventana import dibujar

_TZ = timezone(timedelta(hours=-5))

_MODELO_FACIAL = "sface_2021dec"


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada del demo."""
    args = _parsear_args(argv)
    cfg = cargar_config(args.config)

    # Sobrescribir desde la línea de comandos
    if args.fuente is not None:
        raw = args.fuente
        cfg.fuente = int(raw) if str(raw).isdigit() else raw
    if args.precio is not None:
        cfg.precio_centimos = int(args.precio)
    if args.ocultar_monto:
        cfg.mostrar_monto = False
    sin_ventana = args.sin_ventana

    # Elegir cámara si hace falta
    fuente = cfg.fuente
    if str(fuente) == "elegir":
        fuente = elegir()

    # Base de datos y jornada
    con = conectar(cfg.ruta_db)
    inicio = datetime.now(tz=_TZ)
    jornada_id, _ = abrir_jornada(
        con, str(fuente), cfg.precio_centimos, cfg.mostrar_monto,
        _MODELO_FACIAL, cfg.umbral_alto, cfg.umbral_bajo,
    )
    # Ajustar inicio real en la tabla (ya se guardó el ts en abrir_jornada)

    # Perfiles enrolados
    perfiles_db = cargar_perfiles(con, _MODELO_FACIAL)
    perfiles_ident: dict[int, tuple[str, object]] = {}
    for pid, (nombre, vectores) in perfiles_db.items():
        if vectores:
            cen = centroide(vectores)
            perfiles_ident[pid] = (nombre, cen)

    # Piezas del sistema
    detector = DetectorPersonas(cfg.pesos, cfg.confianza)
    extractor = ExtractorRostro(cfg.ruta_yunet, cfg.ruta_sface,
                                cfg.confianza_rostro, cfg.lado_min_px)
    identificador = Identificador(perfiles_ident, cfg.umbral_alto, cfg.umbral_bajo,
                                  cfg.min_rostros, cfg.max_rostros)
    banda_obj = Banda(
        p1=(cfg.banda[0], cfg.banda[1]),
        p2=(cfg.banda[2], cfg.banda[3]),
        ancho=cfg.banda[4],
        adentro_positivo=not cfg.adentro_negativo,
    )
    contador = Contador(banda_obj, cfg.confirmar_cuadros,
                        cfg.enfriamiento_s, cfg.perdido_s)
    monitor = MonitorSalud(cfg.brillo_min, cfg.variacion_min, cfg.segundos_tapada)
    motor = Motor(cfg, detector, extractor, identificador,
                  contador, monitor, con, jornada_id, inicio)

    # Captura
    captura = Captura(fuente, cfg.ancho, cfg.alto)
    captura.iniciar()

    n_cuadros = 0
    t0 = time.monotonic()
    t_ultimo_fps = t0
    fps_actual = 0.0

    # Fuente en vivo: índice numérico o URL con ://
    es_fuente_en_vivo = str(fuente).isdigit() or "://" in str(fuente)

    print("Sistema iniciado. Presiona 'q' para detener.")

    try:
        while True:
            cuadro = captura.leer(timeout_s=2.0)
            if cuadro is None:
                if not es_fuente_en_vivo and captura.terminado:
                    break  # archivo agotado
                # fuente en vivo sin señal
                t_ahora = captura.tiempo_actual()
                ev_caida = monitor.sin_senal(t_ahora)
                if ev_caida:
                    motor.registrar_camara("CAMARA_CAIDA", t_ahora)
                captura.detener()
                # Esperar 2 s respondiendo a q si hay ventana
                q_pulsado = False
                if not sin_ventana:
                    for _ in range(20):  # 20 × 100 ms = 2 s
                        if cv2.waitKey(100) & 0xFF == ord("q"):
                            q_pulsado = True
                            break
                else:
                    time.sleep(2.0)
                if q_pulsado:
                    break
                if captura.reabrir():
                    t_ok = captura.tiempo_actual()
                    motor.registrar_camara("CAMARA_OK", t_ok)
                continue

            estado = motor.procesar(cuadro)
            n_cuadros += 1

            # FPS medidos
            ahora = time.monotonic()
            if ahora - t_ultimo_fps >= 1.0:
                fps_actual = n_cuadros / (ahora - t0) if ahora > t0 else 0.0
                t_ultimo_fps = ahora
            estado.fps = fps_actual

            if not sin_ventana:
                img_hud = dibujar(cuadro.imagen, estado, cfg.banda, cfg.mostrar_monto)
                cv2.imshow("Control de acceso", img_hud)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
    finally:
        if not sin_ventana:
            cv2.destroyAllWindows()
        captura.detener()

    t_total = time.monotonic() - t0
    resumen = motor.detener(t_total)
    _mostrar_resumen(resumen, n_cuadros, t_total, sin_ventana)


def _mostrar_resumen(resumen: dict, n_cuadros: int, t_total: float, sin_ventana: bool) -> None:
    """Imprime el resumen de la jornada en consola."""
    print("\n--- Resumen de jornada ---")
    print(f"  Entradas:          {resumen['entradas']}")
    print(f"  Salidas:           {resumen['salidas']}")
    print(f"  Personal:          {resumen['personal']}")
    print(f"  Pagantes:          {resumen['pagantes']}")
    print(f"  No identificados:  {resumen['no_identificados']}")
    conf = resumen["recaudacion_confirmada_centimos"] / 100
    maxi = resumen["recaudacion_maxima_centimos"] / 100
    print(f"  Recaud. confirmada: S/{conf:.2f}")
    print(f"  Recaud. maxima:     S/{maxi:.2f}")
    sin_vig = resumen["sin_vigilancia_s"]
    print(f"  Sin vigilancia:    {sin_vig:.1f} s")
    print(f"  Cuadros procesados: {n_cuadros}")
    print(f"  Tiempo total:       {t_total:.1f} s")
    fps_med = n_cuadros / t_total if t_total > 0 else 0.0
    print(f"  FPS medios:         {fps_med:.1f}")


def _parsear_args(argv):
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Demo de control de acceso")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fuente", default=None)
    parser.add_argument("--precio", type=int, default=None)
    parser.add_argument("--ocultar-monto", action="store_true")
    parser.add_argument("--sin-ventana", action="store_true",
                        help="Solo para pruebas: no abre ventana de cv2")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
