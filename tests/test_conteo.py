"""Tests del módulo vision.conteo."""

from __future__ import annotations

import math

import numpy as np

from vision.conteo import Banda, Contador, Cruce

FPS = 15

# Banda horizontal: centro en y=300, mitad de ancho = 30 px
# AFUERA: y < 270 | BANDA: 270 ≤ y ≤ 330 | ADENTRO: y > 330
BANDA_H = Banda(p1=(0.0, 300.0), p2=(640.0, 300.0), ancho=60.0, adentro_positivo=True)


def lineal(
    y0: float,
    y1: float,
    n_frames: int,
    x: float = 320.0,
    ruido_std: float = 0.0,
    seed: int = 42,
) -> list:
    """Genera n_frames puntos (x, y) en línea recta con ruido gaussiano opcional."""
    ys = np.linspace(y0, y1, n_frames)
    if ruido_std > 0.0:
        rng = np.random.default_rng(seed)
        ys = ys + rng.normal(0.0, ruido_std, n_frames)
    return [(x, float(y)) for y in ys]


def simular(
    banda: Banda,
    secuencias: dict,
    confirmar_cuadros: int = 3,
    enfriamiento_s: float = 2.0,
    perdido_s: float = 1.5,
) -> list[Cruce]:
    """Corre Contador cuadro a cuadro y acumula todos los cruces.

    Cada lista en secuencias puede contener None para indicar ausencia del track
    en ese frame (útil para simular oclusiones).
    """
    contador = Contador(banda, confirmar_cuadros, enfriamiento_s, perdido_s)
    n_frames = max(len(pts) for pts in secuencias.values())
    cruces: list[Cruce] = []
    for i in range(n_frames):
        t = i / FPS
        puntos = {
            tid: pts[i]
            for tid, pts in secuencias.items()
            if i < len(pts) and pts[i] is not None
        }
        cruces.extend(contador.actualizar(t, puntos))
    return cruces


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------

def test_01_entrada_simple():
    """Un track cruza de afuera hacia adentro: debe contar 1 ENTRADA."""
    pts = lineal(150.0, 450.0, 60)
    cruces = simular(BANDA_H, {1: pts})
    assert [c.tipo for c in cruces] == ["ENTRADA"]


def test_02_salida_simple():
    """Un track cruza de adentro hacia afuera: debe contar 1 SALIDA."""
    pts = lineal(450.0, 150.0, 60)
    cruces = simular(BANDA_H, {1: pts})
    assert [c.tipo for c in cruces] == ["SALIDA"]


def test_03_se_queda_en_banda():
    """Track que llega a la banda, permanece 5 s y vuelve: 0 cruces."""
    pts = (
        [(320.0, 150.0)] * 5          # firma AFUERA
        + lineal(150.0, 300.0, 10)    # avanza hacia BANDA
        + [(320.0, 300.0)] * (5 * FPS)  # 5 s quieto en BANDA
        + lineal(300.0, 150.0, 10)    # regresa a AFUERA
    )
    cruces = simular(BANDA_H, {1: pts})
    assert cruces == []


def test_04_retrocede_y_cruza():
    """Track que entra a la banda, retrocede y luego cruza de verdad: 1 ENTRADA."""
    pts = (
        [(320.0, 150.0)] * 5          # firma AFUERA
        + lineal(150.0, 300.0, 8)     # entra a BANDA
        + lineal(300.0, 150.0, 8)     # retrocede a AFUERA
        + [(320.0, 150.0)] * 5        # confirma AFUERA de nuevo
        + lineal(150.0, 450.0, 30)    # cruza hasta ADENTRO
    )
    cruces = simular(BANDA_H, {1: pts})
    assert len(cruces) == 1
    assert cruces[0].tipo == "ENTRADA"


def test_05_oscila_en_borde():
    """Track que oscila entre BANDA y ADENTRO: a lo más 1 ENTRADA, 0 SALIDA."""
    # y=290 → BANDA (d=−10); y=365 → ADENTRO (d=65). Alterna cada frame.
    firma = [(320.0, 150.0)] * 5
    oscilacion = [(320.0, 290.0 if i % 2 == 0 else 365.0) for i in range(3 * FPS)]
    pts = firma + oscilacion
    cruces = simular(BANDA_H, {1: pts})
    entradas = [c for c in cruces if c.tipo == "ENTRADA"]
    salidas = [c for c in cruces if c.tipo == "SALIDA"]
    assert len(entradas) <= 1
    assert len(salidas) == 0


def test_06_enfriamiento():
    """Entra, vuelve antes de 1 s (sin SALIDA), re-entra 3 s después (2ª ENTRADA)."""
    firma = [(320.0, 150.0)] * 5
    cruce1 = lineal(150.0, 450.0, 21)   # ENTRADA alrededor de t ≈ 1.33 s
    retorno = lineal(450.0, 150.0, 10)  # vuelve en < 1 s → dentro del enfriamiento
    pausa = [(320.0, 150.0)] * 40       # espera hasta superar los 3 s pedidos
    cruce2 = lineal(150.0, 450.0, 21)   # 2ª ENTRADA bien después del enfriamiento
    pts = firma + cruce1 + retorno + pausa + cruce2
    cruces = simular(BANDA_H, {1: pts})
    entradas = [c for c in cruces if c.tipo == "ENTRADA"]
    salidas = [c for c in cruces if c.tipo == "SALIDA"]
    assert len(entradas) == 2
    assert len(salidas) == 0


def test_07_dos_tracks_opuestos():
    """Track 1 entra y Track 2 sale al mismo tiempo: ids correctos."""
    pts1 = lineal(150.0, 450.0, 60)
    pts2 = lineal(450.0, 150.0, 60)
    cruces = simular(BANDA_H, {1: pts1, 2: pts2})
    entradas = [c for c in cruces if c.tipo == "ENTRADA"]
    salidas = [c for c in cruces if c.tipo == "SALIDA"]
    assert len(entradas) == 1 and entradas[0].track_id == 1
    assert len(salidas) == 1 and salidas[0].track_id == 2


def test_08_nace_adentro_sin_cruzar():
    """Track que nace en ADENTRO y no cruza la banda: 0 cruces."""
    pts = [(320.0, 450.0)] * (2 * FPS)
    cruces = simular(BANDA_H, {1: pts})
    assert cruces == []


def test_09_desaparece_en_banda():
    """Track que llega a la banda y desaparece sin cruzar: 0 cruces."""
    pts = [(320.0, 150.0)] * 5 + lineal(150.0, 300.0, 10)
    cruces = simular(BANDA_H, {1: pts})
    assert cruces == []


def test_10_reaparece_con_mismo_id():
    """Track que desaparece 1 s en la banda y reaparece al otro lado: 1 ENTRADA."""
    firma = [(320.0, 150.0)] * 5
    hacia_banda = lineal(150.0, 300.0, 8)
    # Ausente 1 s (15 frames). perdido_s=1.5 s → estado se conserva.
    ausente: list = [None] * FPS
    reaparece = lineal(335.0, 450.0, 15)
    pts = firma + hacia_banda + ausente + reaparece
    cruces = simular(BANDA_H, {1: pts})
    assert len(cruces) == 1
    assert cruces[0].tipo == "ENTRADA"


def test_11_banda_diagonal():
    """Banda diagonal con adentro_positivo=False: 1 ENTRADA y 1 SALIDA correctas."""
    banda_d = Banda(
        p1=(0.0, 0.0), p2=(600.0, 600.0), ancho=60.0, adentro_positivo=False
    )
    # Con adentro_positivo=False: d = (Px − Py) / √2
    # AFUERA cuando Px < Py; ADENTRO cuando Px > Py (diferencia > 30√2 ≈ 42.4)
    r2 = math.sqrt(2.0)
    inicio = (300.0 - 100.0 / r2, 300.0 + 100.0 / r2)   # AFUERA  (Px−Py ≈ −141)
    fin = (300.0 + 100.0 / r2, 300.0 - 100.0 / r2)       # ADENTRO (Px−Py ≈  141)

    ida = [
        (float(x), float(y))
        for x, y in zip(
            np.linspace(inicio[0], fin[0], 40),
            np.linspace(inicio[1], fin[1], 40),
        )
    ]
    # 3 s de pausa en ADENTRO para que el enfriamiento (2 s) quede claramente superado
    pausa = [fin] * (3 * FPS)
    vuelta = [
        (float(x), float(y))
        for x, y in zip(
            np.linspace(fin[0], inicio[0], 40),
            np.linspace(fin[1], inicio[1], 40),
        )
    ]

    cruces = simular(banda_d, {1: ida + pausa + vuelta})
    assert [c.tipo for c in cruces] == ["ENTRADA", "SALIDA"]


def test_12_entrada_con_ruido():
    """Entrada limpia con ruido gaussiano de 5 px: sigue siendo 1 ENTRADA."""
    pts = lineal(150.0, 450.0, 60, ruido_std=5.0)
    cruces = simular(BANDA_H, {1: pts})
    assert len([c for c in cruces if c.tipo == "ENTRADA"]) == 1


def test_13_primer_cruce_antes_de_t2():
    """El primer cruce ocurre antes de t=2 s: valida que ultimo_cruce_t=-inf no lo bloquea."""
    # 30 frames = 2.0 s exactos; la ENTRADA se confirma antes de t=2 s
    pts = lineal(150.0, 450.0, 30)
    cruces = simular(BANDA_H, {1: pts})
    assert len([c for c in cruces if c.tipo == "ENTRADA"]) == 1


def test_14_reaparece_tras_perdido():
    """Track ausente más de perdido_s: estado borrado, no genera cruce al reaparecer."""
    firma = [(320.0, 150.0)] * 5
    # 2 s de ausencia (30 frames) > perdido_s=1.5 s → estado se borra
    ausente: list = [None] * (2 * FPS)
    adentro = [(320.0, 450.0)] * (2 * FPS)  # reaparece firme en ADENTRO, sin historial
    pts = firma + ausente + adentro
    cruces = simular(BANDA_H, {1: pts})
    assert cruces == []
