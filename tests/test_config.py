"""Tests de vision.config: carga y validación de config.yaml."""

from __future__ import annotations

import pytest

from vision.config import cargar


def _config_base() -> dict:
    """Diccionario mínimo válido para armar un config.yaml de prueba."""
    return {
        "fuente": "elegir",
        "ancho": 1280,
        "alto": 720,
        "banda": {"x1": 0, "y1": 360, "x2": 1280, "y2": 360, "ancho": 60},
        "adentro_negativo": False,
        "confirmar_cuadros": 3,
        "enfriamiento_s": 2.0,
        "perdido_s": 1.5,
        "pesos": "yolov8n.pt",
        "imgsz": 640,
        "confianza": 0.35,
        "ruta_yunet": "data/modelos/yunet.onnx",
        "ruta_sface": "data/modelos/sface.onnx",
        "confianza_rostro": 0.9,
        "lado_min_px": 40,
        "umbral_alto": 0.45,
        "umbral_bajo": 0.30,
        "min_rostros": 3,
        "max_rostros": 10,
        "cada_n_cuadros": 3,
        "distancia_ventana_px": 150,
        "ventana_post_s": 1.0,
        "brillo_min": 25,
        "variacion_min": 8,
        "segundos_tapada": 2.0,
        "latido_s": 10,
        "precio_centimos": 150,
        "mostrar_monto": True,
        "ruta_db": "data/control.db",
    }


def _escribir_yaml(tmp_path, datos: dict) -> str:
    """Escribe los datos como YAML y devuelve la ruta."""
    import yaml
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.dump(datos), encoding="utf-8")
    return str(ruta)


def test_01_carga_valida(tmp_path):
    """Un config.yaml válido se carga sin excepción."""
    cfg = cargar(_escribir_yaml(tmp_path, _config_base()))
    assert cfg.precio_centimos == 150
    assert cfg.umbral_alto == 0.45
    assert cfg.banda[4] == 60.0  # ancho


def test_02_umbrales_invertidos(tmp_path):
    """umbral_bajo > umbral_alto lanza ValueError."""
    datos = _config_base()
    datos["umbral_bajo"] = 0.9
    datos["umbral_alto"] = 0.3
    with pytest.raises(ValueError, match="umbral_bajo"):
        cargar(_escribir_yaml(tmp_path, datos))


def test_03_precio_negativo(tmp_path):
    """precio_centimos negativo lanza ValueError."""
    datos = _config_base()
    datos["precio_centimos"] = -1
    with pytest.raises(ValueError, match="precio_centimos"):
        cargar(_escribir_yaml(tmp_path, datos))


def test_04_fuente_indice(tmp_path):
    """fuente numérica se convierte a int."""
    datos = _config_base()
    datos["fuente"] = "0"
    cfg = cargar(_escribir_yaml(tmp_path, datos))
    assert cfg.fuente == 0


def test_05_config_real(tmp_path):
    """Carga el config.yaml real del proyecto sin excepción."""
    cfg = cargar("config.yaml")
    assert cfg.precio_centimos >= 0
