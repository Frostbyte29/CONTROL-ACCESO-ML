"""Tests de datos.reportes: agrupación por día, mes y año."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from datos.db import abrir_jornada, cerrar_jornada, conectar, insertar_evento
from datos.reportes import reporte

_TZ = timezone(timedelta(hours=-5))


def _ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _poblar(con, jornadas: list[tuple[str, int, list[str]]]) -> None:
    """
    Inserta jornadas y eventos de prueba.
    jornadas = [(fecha_inicio_iso, precio, [categorias_de_entrada])]
    """
    for idx, (inicio_iso, precio, categorias) in enumerate(jornadas):
        inicio = datetime.fromisoformat(inicio_iso)
        jid, _ = abrir_jornada(con, f"fuente_{idx}", precio, True,
                               "sface_2021dec", 0.45, 0.30)
        con.execute("UPDATE jornada SET inicio = ? WHERE id = ?", (_ts(inicio), jid))
        con.commit()

        t = inicio
        for cat in categorias:
            t += timedelta(minutes=1)
            insertar_evento(con, {
                "jornada_id": jid,
                "ts": _ts(t),
                "tipo": "ENTRADA",
                "track_id": idx * 100 + categorias.index(cat),
                "categoria": cat,
                "precio_centimos": precio,
            })

        t += timedelta(minutes=1)
        insertar_evento(con, {
            "jornada_id": jid,
            "ts": _ts(t),
            "tipo": "SALIDA",
            "track_id": 0,
        })
        cerrar_jornada(con, jid, _ts(t + timedelta(minutes=1)))


def test_01_reporte_por_dia(tmp_path):
    """Dos jornadas en días distintos: 2 filas en el reporte por día."""
    con = conectar(str(tmp_path / "ctrl.db"))
    _poblar(con, [
        ("2025-01-10T08:00:00-05:00", 100, ["PAGANTE", "PERSONAL"]),
        ("2025-01-11T08:00:00-05:00", 100, ["NO_IDENTIFICADO"]),
    ])
    filas = reporte(con, "dia")
    assert len(filas) == 2
    dias = [f["periodo"] for f in filas]
    assert "2025-01-10" in dias
    assert "2025-01-11" in dias


def test_02_reporte_por_mes(tmp_path):
    """Jornadas en meses distintos: agrupadas por mes."""
    con = conectar(str(tmp_path / "ctrl.db"))
    _poblar(con, [
        ("2025-01-10T08:00:00-05:00", 150, ["PAGANTE"]),
        ("2025-01-20T08:00:00-05:00", 150, ["PAGANTE"]),
        ("2025-02-05T08:00:00-05:00", 150, ["PERSONAL"]),
    ])
    filas = reporte(con, "mes")
    assert len(filas) == 2
    meses = [f["periodo"] for f in filas]
    assert "2025-01" in meses
    assert "2025-02" in meses

    enero = next(f for f in filas if f["periodo"] == "2025-01")
    assert enero["jornadas"] == 2
    assert enero["pagantes"] == 2
    assert enero["confirmada_centimos"] == 300


def test_03_reporte_por_anio(tmp_path):
    """Jornadas en años distintos: agrupadas por año."""
    con = conectar(str(tmp_path / "ctrl.db"))
    _poblar(con, [
        ("2024-12-31T08:00:00-05:00", 200, ["PAGANTE", "NO_IDENTIFICADO"]),
        ("2025-01-01T08:00:00-05:00", 200, ["PERSONAL"]),
    ])
    filas = reporte(con, "anio")
    assert len(filas) == 2
    anios = [f["periodo"] for f in filas]
    assert "2024" in anios
    assert "2025" in anios

    anio_2024 = next(f for f in filas if f["periodo"] == "2024")
    assert anio_2024["confirmada_centimos"] == 200   # solo el PAGANTE
    assert anio_2024["maxima_centimos"] == 400       # PAGANTE + NO_IDENTIFICADO


def test_04_reporte_vacio(tmp_path):
    """Base sin datos: lista vacía."""
    con = conectar(str(tmp_path / "ctrl.db"))
    filas = reporte(con, "dia")
    assert filas == []


def test_05_periodo_invalido(tmp_path):
    """--por inválido lanza ValueError."""
    con = conectar(str(tmp_path / "ctrl.db"))
    with pytest.raises(ValueError):
        reporte(con, "semana")
