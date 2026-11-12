"""Acceso a SQLite para jornadas, eventos y perfiles (sin ORM)."""

from __future__ import annotations

import sqlite3
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TZ = timezone(timedelta(hours=-5))

_TIPOS_VALIDOS = {"ENTRADA", "SALIDA", "CAMARA_TAPADA", "CAMARA_CAIDA", "CAMARA_OK"}
_CATEGORIAS_VALIDAS = {"PERSONAL", "PAGANTE", "NO_IDENTIFICADO"}

_SQL_ESQUEMA = (Path(__file__).parent / "esquema.sql").read_text(encoding="utf-8")


def conectar(ruta: str) -> sqlite3.Connection:
    """Abre (o crea) la base, aplica el esquema y activa claves foráneas."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    con.row_factory = sqlite3.Row
    con.executescript(_SQL_ESQUEMA)
    con.execute("PRAGMA foreign_keys = ON")
    con.commit()
    return con


def crear_perfil(
    con: sqlite3.Connection,
    nombre: str,
    vectores: list[np.ndarray],
    modelo: str,
    calidades: list[float],
) -> int:
    """Guarda perfil y vectores en una transacción; sin vectores lanza ValueError."""
    if not vectores:
        raise ValueError("Se necesita al menos un vector para crear un perfil")
    ts = _ahora_iso()
    with con:
        cur = con.execute(
            "INSERT INTO perfil (nombre, rol, activo, creado_en) VALUES (?, 'personal', 1, ?)",
            (nombre, ts),
        )
        perfil_id = cur.lastrowid
        for v, cal in zip(vectores, calidades):
            con.execute(
                "INSERT INTO vector_rostro (perfil_id, modelo, vector, calidad, creado_en)"
                " VALUES (?, ?, ?, ?, ?)",
                (perfil_id, modelo, v.astype(np.float32).tobytes(), float(cal), ts),
            )
    return perfil_id


def cargar_perfiles(
    con: sqlite3.Connection,
    modelo: str,
) -> dict[int, tuple[str, list[np.ndarray]]]:
    """Devuelve {perfil_id: (nombre, [vectores])} solo para perfiles activos y ese modelo."""
    filas_perfil = con.execute(
        "SELECT id, nombre FROM perfil WHERE activo = 1",
    ).fetchall()
    resultado: dict[int, tuple[str, list[np.ndarray]]] = {}
    for fila in filas_perfil:
        pid = fila["id"]
        vectores_filas = con.execute(
            "SELECT vector FROM vector_rostro WHERE perfil_id = ? AND modelo = ?",
            (pid, modelo),
        ).fetchall()
        vectores = [np.frombuffer(r["vector"], dtype=np.float32) for r in vectores_filas]
        resultado[pid] = (fila["nombre"], vectores)
    return resultado


def abrir_jornada(
    con: sqlite3.Connection,
    fuente: str,
    precio_centimos: int,
    mostrar_monto: bool,
    modelo_facial: str,
    umbral_alto: float,
    umbral_bajo: float,
) -> tuple[int, str]:
    """Registra el inicio de una jornada y devuelve (id, uuid)."""
    jornada_uuid = str(_uuid.uuid4())
    ts = _ahora_iso()
    with con:
        cur = con.execute(
            """INSERT INTO jornada
               (uuid, fuente, precio_centimos, mostrar_monto, modelo_facial,
                umbral_alto, umbral_bajo, inicio)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                jornada_uuid,
                str(fuente),
                int(precio_centimos),
                1 if mostrar_monto else 0,
                modelo_facial,
                umbral_alto,
                umbral_bajo,
                ts,
            ),
        )
    return cur.lastrowid, jornada_uuid


def insertar_evento(con: sqlite3.Connection, evento: dict[str, Any]) -> int:
    """Valida y guarda un evento; lanza ValueError si algo es incorrecto."""
    tipo = evento.get("tipo")
    if tipo not in _TIPOS_VALIDOS:
        raise ValueError(f"tipo inválido: {tipo!r}")

    categoria = evento.get("categoria")
    if categoria is not None:
        if tipo != "ENTRADA":
            raise ValueError("categoria solo puede aparecer en ENTRADA")
        if categoria not in _CATEGORIAS_VALIDAS:
            raise ValueError(f"categoria inválida: {categoria!r}")
    if tipo == "ENTRADA" and categoria is None:
        raise ValueError("ENTRADA requiere categoria")

    precio = evento.get("precio_centimos")
    if precio is not None and not isinstance(precio, int):
        raise ValueError("precio_centimos debe ser int")

    ts = evento.get("ts", "")
    if "-05:00" not in ts:
        raise ValueError(f"ts debe contener zona -05:00, recibido: {ts!r}")

    evento_uuid = evento.get("uuid") or str(_uuid.uuid4())
    with con:
        cur = con.execute(
            """INSERT INTO evento
               (uuid, jornada_id, ts, tipo, track_id, categoria,
                perfil_id, similitud, precio_centimos, sincronizado_en)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evento_uuid,
                evento["jornada_id"],
                ts,
                tipo,
                evento.get("track_id"),
                categoria,
                evento.get("perfil_id"),
                evento.get("similitud"),
                precio,
                None,
            ),
        )
    return cur.lastrowid


def latido(con: sqlite3.Connection, jornada_id: int, ts: str) -> None:
    """Actualiza el timestamp del último latido de la jornada."""
    with con:
        con.execute(
            "UPDATE jornada SET ultimo_latido = ? WHERE id = ?",
            (ts, jornada_id),
        )


def cerrar_jornada(con: sqlite3.Connection, jornada_id: int, ts: str) -> None:
    """Marca la jornada como cerrada con su timestamp de fin."""
    with con:
        con.execute(
            "UPDATE jornada SET fin = ? WHERE id = ?",
            (ts, jornada_id),
        )


def resumen(con: sqlite3.Connection, jornada_id: int) -> dict[str, Any]:
    """Calcula el resumen de una jornada exclusivamente desde los eventos."""
    jornada = con.execute(
        "SELECT inicio, fin, ultimo_latido, precio_centimos FROM jornada WHERE id = ?",
        (jornada_id,),
    ).fetchone()

    eventos = con.execute(
        """SELECT tipo, categoria, precio_centimos, ts
           FROM evento WHERE jornada_id = ? ORDER BY ts""",
        (jornada_id,),
    ).fetchall()

    entradas = salidas = personal = pagantes = no_identificados = 0
    confirmada = maxima = 0

    for ev in eventos:
        if ev["tipo"] == "ENTRADA":
            entradas += 1
            cat = ev["categoria"]
            precio_ev = ev["precio_centimos"] or 0
            if cat == "PERSONAL":
                personal += 1
            elif cat == "PAGANTE":
                pagantes += 1
                confirmada += precio_ev
                maxima += precio_ev
            elif cat == "NO_IDENTIFICADO":
                no_identificados += 1
                maxima += precio_ev
        elif ev["tipo"] == "SALIDA":
            salidas += 1

    # Tiempo sin vigilancia
    sin_vigilancia_s = _calcular_sin_vigilancia(eventos, jornada)

    # Duración
    inicio_dt = _parse_ts(jornada["inicio"])
    fin_str = jornada["fin"] or jornada["ultimo_latido"]
    duracion_s = (
        (_parse_ts(fin_str) - inicio_dt).total_seconds()
        if fin_str
        else 0.0
    )

    return {
        "duracion_s": duracion_s,
        "entradas": entradas,
        "salidas": salidas,
        "personal": personal,
        "pagantes": pagantes,
        "no_identificados": no_identificados,
        "recaudacion_confirmada_centimos": confirmada,
        "recaudacion_maxima_centimos": maxima,
        "sin_vigilancia_s": sin_vigilancia_s,
    }


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _ahora_iso() -> str:
    """Devuelve la hora actual en ISO 8601 con zona -05:00."""
    return datetime.now(tz=_TZ).isoformat(timespec="milliseconds")


def _parse_ts(ts: str) -> datetime:
    """Convierte un string ISO 8601 con zona a datetime."""
    return datetime.fromisoformat(ts)


def _calcular_sin_vigilancia(eventos, jornada) -> float:
    """Suma los segundos sin vigilancia (cámara tapada o caída) a partir de los eventos."""
    fin_ref_str = jornada["fin"] or jornada["ultimo_latido"]
    fin_ref = _parse_ts(fin_ref_str) if fin_ref_str else None

    total = 0.0
    inicio_incidente: datetime | None = None

    for ev in eventos:
        tipo = ev["tipo"]
        ts = _parse_ts(ev["ts"])
        if tipo in ("CAMARA_TAPADA", "CAMARA_CAIDA"):
            if inicio_incidente is None:
                inicio_incidente = ts
        elif tipo == "CAMARA_OK":
            if inicio_incidente is not None:
                total += (ts - inicio_incidente).total_seconds()
                inicio_incidente = None

    if inicio_incidente is not None and fin_ref is not None:
        total += (fin_ref - inicio_incidente).total_seconds()

    return total
