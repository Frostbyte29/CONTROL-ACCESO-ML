"""Carga y validación de config.yaml con un dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """Parámetros de operación del sistema, validados al cargar."""

    # Fuente de video
    fuente: str | int
    ancho: int
    alto: int

    # Banda de conteo (x1, y1, x2, y2, ancho_px)
    banda: tuple[float, float, float, float, float]
    adentro_negativo: bool

    # Conteo
    confirmar_cuadros: int
    enfriamiento_s: float
    perdido_s: float

    # YOLO
    pesos: str
    imgsz: int
    confianza: float

    # Modelos de rostro
    ruta_yunet: str
    ruta_sface: str
    confianza_rostro: float
    lado_min_px: int

    # Identificación
    umbral_alto: float
    umbral_bajo: float
    min_rostros: int
    max_rostros: int

    # Ventana de identificación
    cada_n_cuadros: int
    distancia_ventana_px: float
    ventana_post_s: float

    # Salud
    brillo_min: float
    variacion_min: float
    segundos_tapada: float
    latido_s: float

    # Jornada
    precio_centimos: int
    mostrar_monto: bool

    # Base de datos
    ruta_db: str


def cargar(ruta: str | Path) -> Config:
    """Lee config.yaml y devuelve un Config validado; lanza ValueError si algo es incorrecto."""
    datos = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))

    umbral_alto = float(datos["umbral_alto"])
    umbral_bajo = float(datos["umbral_bajo"])
    if umbral_bajo > umbral_alto:
        raise ValueError(
            f"umbral_bajo ({umbral_bajo}) no puede ser mayor que umbral_alto ({umbral_alto})"
        )

    precio = datos["precio_centimos"]
    if not isinstance(precio, int) or precio < 0:
        raise ValueError(f"precio_centimos debe ser entero >= 0, recibido: {precio!r}")

    banda_raw = datos["banda"]
    banda = (
        float(banda_raw["x1"]),
        float(banda_raw["y1"]),
        float(banda_raw["x2"]),
        float(banda_raw["y2"]),
        float(banda_raw["ancho"]),
    )
    if len(banda) != 5:
        raise ValueError("banda debe tener exactamente 5 números (x1,y1,x2,y2,ancho)")

    fuente_raw = datos.get("fuente", "elegir")
    fuente: str | int = int(fuente_raw) if str(fuente_raw).isdigit() else str(fuente_raw)

    return Config(
        fuente=fuente,
        ancho=int(datos["ancho"]),
        alto=int(datos["alto"]),
        banda=banda,
        adentro_negativo=bool(datos.get("adentro_negativo", False)),
        confirmar_cuadros=int(datos["confirmar_cuadros"]),
        enfriamiento_s=float(datos["enfriamiento_s"]),
        perdido_s=float(datos["perdido_s"]),
        pesos=str(datos["pesos"]),
        imgsz=int(datos["imgsz"]),
        confianza=float(datos["confianza"]),
        ruta_yunet=str(datos["ruta_yunet"]),
        ruta_sface=str(datos["ruta_sface"]),
        confianza_rostro=float(datos["confianza_rostro"]),
        lado_min_px=int(datos["lado_min_px"]),
        umbral_alto=umbral_alto,
        umbral_bajo=umbral_bajo,
        min_rostros=int(datos["min_rostros"]),
        max_rostros=int(datos["max_rostros"]),
        cada_n_cuadros=int(datos["cada_n_cuadros"]),
        distancia_ventana_px=float(datos["distancia_ventana_px"]),
        ventana_post_s=float(datos["ventana_post_s"]),
        brillo_min=float(datos["brillo_min"]),
        variacion_min=float(datos["variacion_min"]),
        segundos_tapada=float(datos["segundos_tapada"]),
        latido_s=float(datos["latido_s"]),
        precio_centimos=int(precio),
        mostrar_monto=bool(datos.get("mostrar_monto", True)),
        ruta_db=str(datos["ruta_db"]),
    )
