"""Tests de datos.db: resumen, perfiles, validaciones y tiempo sin vigilancia."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from datos.db import (
    abrir_jornada,
    cargar_perfiles,
    cerrar_jornada,
    conectar,
    crear_perfil,
    insertar_evento,
    latido,
    resumen,
)

_TZ = timezone(timedelta(hours=-5))


def _ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _jornada_base(con, precio: int = 100):
    """Abre una jornada de prueba y devuelve (jornada_id, inicio_dt)."""
    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", precio, True, "sface_2021dec", 0.45, 0.30)
    # Forzamos el inicio a un valor conocido para poder calcular duraciones
    con.execute("UPDATE jornada SET inicio = ? WHERE id = ?", (_ts(inicio), jid))
    con.commit()
    return jid, inicio


# ---------------------------------------------------------------------------
# test_01: resumen con eventos escritos a mano
# ---------------------------------------------------------------------------

def test_01_resumen_completo(tmp_path):
    """Resumen con 2 entradas (1 personal, 1 pagante) y 1 salida."""
    con = conectar(str(tmp_path / "ctrl.db"))
    jid, inicio = _jornada_base(con, precio=150)
    t = inicio

    t += timedelta(minutes=1)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t), "tipo": "ENTRADA",
                          "track_id": 1, "categoria": "PERSONAL",
                          "precio_centimos": 150, "uuid": "u1"})
    t += timedelta(minutes=2)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t), "tipo": "ENTRADA",
                          "track_id": 2, "categoria": "PAGANTE",
                          "precio_centimos": 150, "uuid": "u2"})
    t += timedelta(minutes=1)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t), "tipo": "SALIDA",
                          "track_id": 1, "uuid": "u3"})

    cerrar_jornada(con, jid, _ts(t + timedelta(minutes=5)))
    r = resumen(con, jid)

    assert r["entradas"] == 2
    assert r["salidas"] == 1
    assert r["personal"] == 1
    assert r["pagantes"] == 1
    assert r["no_identificados"] == 0
    assert r["recaudacion_confirmada_centimos"] == 150
    assert r["recaudacion_maxima_centimos"] == 150
    assert r["sin_vigilancia_s"] == 0.0


# ---------------------------------------------------------------------------
# test_02: cambio de precio a mitad de jornada no reescribe recaudación anterior
# ---------------------------------------------------------------------------

def test_02_precio_no_se_reescribe(tmp_path):
    """Precio cambia entre eventos: cada entrada guarda el precio vigente en su momento."""
    con = conectar(str(tmp_path / "ctrl.db"))
    jid, inicio = _jornada_base(con, precio=100)
    t = inicio

    t += timedelta(minutes=1)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t), "tipo": "ENTRADA",
                          "track_id": 1, "categoria": "PAGANTE",
                          "precio_centimos": 100, "uuid": "u1"})
    # Precio sube a 200 — el primer evento ya está guardado con 100
    t += timedelta(minutes=2)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t), "tipo": "ENTRADA",
                          "track_id": 2, "categoria": "PAGANTE",
                          "precio_centimos": 200, "uuid": "u2"})

    cerrar_jornada(con, jid, _ts(t + timedelta(minutes=1)))
    r = resumen(con, jid)

    assert r["recaudacion_confirmada_centimos"] == 300  # 100 + 200
    assert r["recaudacion_maxima_centimos"] == 300


# ---------------------------------------------------------------------------
# test_03: perfil sin vectores no deja rastro
# ---------------------------------------------------------------------------

def test_03_perfil_sin_vectores(tmp_path):
    """crear_perfil sin vectores lanza ValueError y no deja perfil a medias."""
    con = conectar(str(tmp_path / "ctrl.db"))
    with pytest.raises(ValueError):
        crear_perfil(con, "Fantasma", [], "sface_2021dec", [])
    count = con.execute("SELECT COUNT(*) FROM perfil").fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# test_04: rechazos de insertar_evento
# ---------------------------------------------------------------------------

def test_04_rechazos_insertar_evento(tmp_path):
    """insertar_evento rechaza tipo inválido, categoría fuera de lugar y ts sin zona."""
    con = conectar(str(tmp_path / "ctrl.db"))
    jid, inicio = _jornada_base(con)
    ts = _ts(inicio + timedelta(minutes=1))

    with pytest.raises(ValueError, match="tipo inválido"):
        insertar_evento(con, {"jornada_id": jid, "ts": ts, "tipo": "DESCONOCIDO", "uuid": "x1"})

    with pytest.raises(ValueError, match="categoria solo puede aparecer en ENTRADA"):
        insertar_evento(con, {"jornada_id": jid, "ts": ts, "tipo": "SALIDA",
                               "categoria": "PAGANTE", "track_id": 1, "uuid": "x2"})

    with pytest.raises(ValueError, match="categoria inválida"):
        insertar_evento(con, {"jornada_id": jid, "ts": ts, "tipo": "ENTRADA",
                               "categoria": "LIBRE", "uuid": "x3"})

    with pytest.raises(ValueError, match="precio_centimos debe ser int"):
        insertar_evento(con, {"jornada_id": jid, "ts": ts, "tipo": "ENTRADA",
                               "categoria": "PAGANTE", "precio_centimos": 1.5, "uuid": "x4"})

    ts_sin_zona = "2025-01-10T08:00:00"
    with pytest.raises(ValueError, match="-05:00"):
        insertar_evento(con, {"jornada_id": jid, "ts": ts_sin_zona, "tipo": "SALIDA",
                               "track_id": 1, "uuid": "x5"})

    with pytest.raises(ValueError, match="ENTRADA requiere categoria"):
        insertar_evento(con, {"jornada_id": jid, "ts": ts, "tipo": "ENTRADA",
                               "track_id": 1, "uuid": "x6"})


# ---------------------------------------------------------------------------
# test_05: tiempo sin vigilancia — tapada con OK, caída sin OK, abierta hasta latido
# ---------------------------------------------------------------------------

def test_05_sin_vigilancia_tapada_y_ok(tmp_path):
    """Tapada 10 s con OK al final: sin_vigilancia_s == 10."""
    con = conectar(str(tmp_path / "ctrl.db"))
    jid, inicio = _jornada_base(con)
    t = inicio

    t += timedelta(seconds=10)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t),
                          "tipo": "CAMARA_TAPADA", "uuid": "v1"})
    t += timedelta(seconds=10)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t),
                          "tipo": "CAMARA_OK", "uuid": "v2"})

    cerrar_jornada(con, jid, _ts(t + timedelta(seconds=5)))
    r = resumen(con, jid)
    assert r["sin_vigilancia_s"] == pytest.approx(10.0)


def test_06_sin_vigilancia_caida_sin_ok(tmp_path):
    """Caída sin OK: el intervalo cuenta hasta el fin de la jornada."""
    con = conectar(str(tmp_path / "ctrl.db"))
    jid, inicio = _jornada_base(con)
    t = inicio

    t += timedelta(seconds=5)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t),
                          "tipo": "CAMARA_CAIDA", "uuid": "w1"})

    fin = t + timedelta(seconds=15)
    cerrar_jornada(con, jid, _ts(fin))
    r = resumen(con, jid)
    assert r["sin_vigilancia_s"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# test_08: columnas exactas de cada tabla (H2)
# ---------------------------------------------------------------------------

def test_08_esquema_columnas(tmp_path):
    """PRAGMA table_info verifica las columnas exactas del esquema PLAN §4."""
    con = conectar(str(tmp_path / "ctrl.db"))

    def columnas(tabla):
        rows = con.execute(f"PRAGMA table_info({tabla})").fetchall()
        return [r["name"] for r in rows]

    assert columnas("perfil") == ["id", "nombre", "rol", "activo", "creado_en"]
    assert columnas("vector_rostro") == [
        "id", "perfil_id", "modelo", "vector", "calidad", "creado_en",
    ]
    assert columnas("jornada") == [
        "id", "uuid", "inicio", "fin", "fuente",
        "precio_centimos", "mostrar_monto", "modelo_facial",
        "umbral_alto", "umbral_bajo", "ultimo_latido",
    ]
    assert columnas("evento") == [
        "id", "uuid", "jornada_id", "ts", "tipo", "track_id",
        "categoria", "perfil_id", "similitud", "precio_centimos", "sincronizado_en",
    ]


def test_07_sin_vigilancia_jornada_abierta_hasta_latido(tmp_path):
    """Jornada abierta (sin fin) con latido: el intervalo tapada→latido."""
    con = conectar(str(tmp_path / "ctrl.db"))
    jid, inicio = _jornada_base(con)
    t = inicio

    t += timedelta(seconds=5)
    insertar_evento(con, {"jornada_id": jid, "ts": _ts(t),
                          "tipo": "CAMARA_TAPADA", "uuid": "z1"})

    lat = t + timedelta(seconds=20)
    latido(con, jid, _ts(lat))

    # jornada sin fin
    r = resumen(con, jid)
    assert r["sin_vigilancia_s"] == pytest.approx(20.0)
