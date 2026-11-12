"""Reportes por día, mes y año desde la línea de comandos.

Uso: python -m datos.reportes --por dia|mes|anio [--db data/control.db]
"""

from __future__ import annotations

import argparse
import sys

from datos.db import conectar

_PERIODOS = ("dia", "mes", "anio")


def reporte(con, por: str) -> list[dict]:
    """Calcula el resumen agrupado por período desde los eventos."""
    if por not in _PERIODOS:
        raise ValueError(f"--por debe ser uno de {_PERIODOS}, recibido: {por!r}")

    if por == "dia":
        trunc = "substr(e.ts, 1, 10)"        # YYYY-MM-DD
    elif por == "mes":
        trunc = "substr(e.ts, 1, 7)"         # YYYY-MM
    else:
        trunc = "substr(e.ts, 1, 4)"         # YYYY

    sql = f"""
    SELECT
        {trunc}                               AS periodo,
        COUNT(DISTINCT j.id)                  AS jornadas,
        SUM(CASE WHEN e.tipo='ENTRADA' THEN 1 ELSE 0 END) AS entradas,
        SUM(CASE WHEN e.tipo='SALIDA'  THEN 1 ELSE 0 END) AS salidas,
        SUM(CASE WHEN e.tipo='ENTRADA' AND e.categoria='PERSONAL'        THEN 1 ELSE 0 END) AS personal,
        SUM(CASE WHEN e.tipo='ENTRADA' AND e.categoria='PAGANTE'         THEN 1 ELSE 0 END) AS pagantes,
        SUM(CASE WHEN e.tipo='ENTRADA' AND e.categoria='NO_IDENTIFICADO' THEN 1 ELSE 0 END) AS no_identificados,
        SUM(CASE WHEN e.tipo='ENTRADA' AND e.categoria='PAGANTE'
                 THEN COALESCE(e.precio_centimos, 0) ELSE 0 END)         AS confirmada_centimos,
        SUM(CASE WHEN e.tipo='ENTRADA' AND e.categoria IN ('PAGANTE','NO_IDENTIFICADO')
                 THEN COALESCE(e.precio_centimos, 0) ELSE 0 END)         AS maxima_centimos
    FROM evento e
    JOIN jornada j ON e.jornada_id = j.id
    GROUP BY periodo
    ORDER BY periodo
    """
    filas = con.execute(sql).fetchall()
    return [dict(f) for f in filas]


def imprimir(filas: list[dict]) -> None:
    """Imprime una tabla ASCII con los resultados."""
    if not filas:
        print("Sin datos.")
        return

    encabezado = (
        f"{'Periodo':<12} {'Jornadas':>8} {'Entradas':>9} {'Salidas':>8} "
        f"{'Personal':>9} {'Pagantes':>9} {'NoIdent':>8} "
        f"{'Confirmada':>12} {'Maxima':>12}"
    )
    separador = "-" * len(encabezado)
    print(encabezado)
    print(separador)

    for f in filas:
        conf = f["confirmada_centimos"] / 100
        maxi = f["maxima_centimos"] / 100
        print(
            f"{f['periodo']:<12} {f['jornadas']:>8} {f['entradas']:>9} {f['salidas']:>8} "
            f"{f['personal']:>9} {f['pagantes']:>9} {f['no_identificados']:>8} "
            f"S/{conf:>10.2f} S/{maxi:>10.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada del reporte."""
    parser = argparse.ArgumentParser(description="Reporte de jornadas")
    parser.add_argument("--por", choices=_PERIODOS, required=True)
    parser.add_argument("--db", default="data/control.db")
    args = parser.parse_args(argv)

    con = conectar(args.db)
    filas = reporte(con, args.por)
    imprimir(filas)


if __name__ == "__main__":
    main()
