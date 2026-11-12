"""Tests de vision.motor con detector y extractor falsos."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from datos.db import abrir_jornada, conectar, crear_perfil
from vision.conteo import Banda, Contador
from vision.identidad import Identificador
from vision.motor import Motor
from vision.personas import Persona
from vision.rostro import Rostro, normalizar
from vision.salud import MonitorSalud

_TZ = timezone(timedelta(hours=-5))

# ---------------------------------------------------------------------------
# Parámetros de la banda de prueba
# ---------------------------------------------------------------------------
# Banda horizontal en y=300. AFUERA: y < 270, ADENTRO: y > 330.
_BANDA = Banda(p1=(0.0, 300.0), p2=(640.0, 300.0), ancho=60.0, adentro_positivo=True)
_FPS = 15


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _ConfigFalso:
    """Configuración mínima para el motor."""
    banda = (0.0, 300.0, 640.0, 300.0, 60.0)
    adentro_negativo = False
    ventana_post_s = 1.0
    cada_n_cuadros = 1
    distancia_ventana_px = 200.0
    precio_centimos = 100
    mostrar_monto = True
    latido_s = 60.0  # no latidos durante pruebas cortas
    perdido_s = 1.5


class _DetectorFalso:
    """Devuelve la lista de personas guionada para cada cuadro."""

    def __init__(self, guion: dict[int, list[Persona]]):
        # guion: {indice_cuadro: [Persona, ...]}
        self._guion = guion

    def procesar(self, imagen: np.ndarray) -> list[Persona]:
        idx = getattr(self, "_idx", 0)
        self._idx = idx + 1
        return self._guion.get(idx, [])


class _ExtractorFalso:
    """Devuelve vectores guionados por track_id; None si el track no está en el guion."""

    def __init__(self, vectores: dict[int, np.ndarray]):
        self._vectores = vectores

    def buscar(self, imagen: np.ndarray, caja) -> Rostro | None:
        # El extractor falso no tiene contexto de track; lo llama el motor por persona
        # Devuelve None siempre; los vectores se inyectan directamente en tests
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona(track_id: int, x: float, y: float) -> Persona:
    """Crea una Persona con pies en (x, y)."""
    return Persona(track_id=track_id, caja=(x - 20, y - 60, x + 20, y),
                   confianza=0.9, pies=(x, y))


def _imagen(h=480, w=640) -> np.ndarray:
    """Imagen gris con variación para no activar el monitor de salud."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h//2, :w//2] = 200
    img[h//2:, w//2:] = 50
    return img


def _cuadro(imagen: np.ndarray, indice: int, fps: float = _FPS):
    """Crea un cuadro falso con t calculado desde el índice."""
    class C:
        pass
    c = C()
    c.imagen = imagen
    c.t = indice / fps
    c.indice = indice
    return c


def _motor_con(tmp_path, perfiles=None, precio=100):
    """Construye un Motor con todas las dependencias."""
    con = conectar(str(tmp_path / "ctrl.db"))
    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", precio, True, "sface_2021dec", 0.45, 0.30)
    cfg = _ConfigFalso()
    cfg.precio_centimos = precio
    contador = Contador(_BANDA, confirmar_cuadros=3, enfriamiento_s=2.0, perdido_s=1.5)
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    identificador = Identificador(
        perfiles or {}, umbral_alto=0.45, umbral_bajo=0.30, min_rostros=1
    )
    motor = Motor(cfg, None, None, identificador, contador, monitor, con, jid, inicio)
    return motor, con, jid


# ---------------------------------------------------------------------------
# test_01: guion del PLAN §2
# ---------------------------------------------------------------------------

def test_01_guion_plan(tmp_path):
    """
    Reproduce el guion de PLAN §2:
    - Gerardo entra y sale como PERSONAL.
    - El profesor entra y sale como PAGANTE.
    - Carlos sale y vuelve a entrar como PERSONAL.
    - Gerardo sale y vuelve sin rostro → NO_IDENTIFICADO.
    - Cámara tapada 3 s.
    Resumen esperado: 4 entradas, 2 salidas, 2 personal, 1 pagante, 1 no_ident.
    """
    precio = 150

    # Vectores de identidad (dimensión 4 para pruebas)
    v_gerardo = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    v_carlos = normalizar(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    v_profesor = normalizar(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    con = conectar(str(tmp_path / "ctrl.db"))
    # Crear perfiles en la DB para que la FK funcione al insertar eventos
    id_gerardo = crear_perfil(con, "Gerardo", [v_gerardo], "sface_2021dec", [0.95])
    id_carlos = crear_perfil(con, "Carlos", [v_carlos], "sface_2021dec", [0.95])

    perfiles = {
        id_gerardo: ("Gerardo", v_gerardo),
        id_carlos: ("Carlos", v_carlos),
    }

    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", precio, True, "sface_2021dec", 0.45, 0.30)

    cfg = _ConfigFalso()
    cfg.precio_centimos = precio
    contador = Contador(_BANDA, confirmar_cuadros=3, enfriamiento_s=2.0, perdido_s=1.5)
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    identificador = Identificador(perfiles, umbral_alto=0.45, umbral_bajo=0.10, min_rostros=1)
    motor = Motor(cfg, None, None, identificador, contador, monitor, con, jid, inicio)

    img_normal = _imagen()

    # Función auxiliar: inyecta vector directamente al identificador antes del cruce
    def _inyectar(track_id: int, vector: np.ndarray, n: int = 3):
        for _ in range(n):
            identificador.agregar(track_id, Rostro(vector=vector, puntaje=0.95, ancho_px=60))

    # --- Gerardo (track 1): entra desde AFUERA (y=150) hacia ADENTRO (y=450) ---
    # Necesitamos que cruce la banda: 60 cuadros de y=150→450
    fps = _FPS
    idx = 0

    def avanza(track_id, y0, y1, n, inyectar_v=None):
        nonlocal idx
        ys = np.linspace(y0, y1, n)
        for y in ys:
            personas = [_persona(track_id, 320.0, float(y))]
            motor._detector = type("D", (), {"procesar": lambda s, i, p=personas: p})()
            if inyectar_v is not None and idx == 0:
                _inyectar(track_id, inyectar_v)
            cuadro = _cuadro(img_normal, idx)
            motor.procesar(cuadro)
            idx += 1

    # Gerardo entra (AFUERA→ADENTRO)
    _inyectar(1, v_gerardo)
    avanza(1, 150.0, 450.0, 60)
    # Gerardo sale (ADENTRO→AFUERA) después de > 2 s de enfriamiento
    pausa_frames = int(2.5 * fps)
    for _ in range(pausa_frames):
        motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(1, 320.0, 450.0)]})()
        motor.procesar(_cuadro(img_normal, idx)); idx += 1

    avanza(1, 450.0, 150.0, 60)

    # Profesor entra y sale (PAGANTE: sin perfil registrado, vector distante)
    _inyectar(3, v_profesor)
    avanza(3, 150.0, 450.0, 60)
    for _ in range(pausa_frames):
        motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(3, 320.0, 450.0)]})()
        motor.procesar(_cuadro(img_normal, idx)); idx += 1
    avanza(3, 450.0, 150.0, 60)

    # Carlos (track 2) sale primero y luego entra
    # Carlos ya estaba "adentro": primero firmamos ADENTRO, luego sale
    for _ in range(5):
        motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(2, 320.0, 450.0)]})()
        motor.procesar(_cuadro(img_normal, idx)); idx += 1
    avanza(2, 450.0, 150.0, 60)
    for _ in range(pausa_frames):
        motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(2, 320.0, 150.0)]})()
        motor.procesar(_cuadro(img_normal, idx)); idx += 1
    _inyectar(2, v_carlos)
    avanza(2, 150.0, 450.0, 60)

    # Gerardo (track 1) sale y vuelve sin rostro
    for _ in range(pausa_frames):
        motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(1, 320.0, 150.0)]})()
        motor.procesar(_cuadro(img_normal, idx)); idx += 1
    # No inyectamos vector → NO_IDENTIFICADO
    avanza(1, 150.0, 450.0, 60)

    # Cámara tapada ~3 s: CAMARA_TAPADA emitida con ts = t_mala (inicio oscuridad)
    img_oscura = np.zeros((480, 640, 3), dtype=np.uint8)
    frames_tapada = int(3.0 * fps)
    for _ in range(frames_tapada):
        motor._detector = type("D", (), {"procesar": lambda s, i: []})()
        motor.procesar(_cuadro(img_oscura, idx)); idx += 1

    t_total = idx / fps
    resumen = motor.detener(t_total)

    assert resumen["entradas"] == 4, f"entradas={resumen['entradas']}"
    assert resumen["salidas"] >= 2, f"salidas={resumen['salidas']}"
    assert resumen["personal"] == 2, f"personal={resumen['personal']}"
    assert resumen["pagantes"] == 1, f"pagantes={resumen['pagantes']}"
    assert resumen["no_identificados"] == 1, f"no_ident={resumen['no_identificados']}"
    assert resumen["recaudacion_confirmada_centimos"] == precio
    assert resumen["recaudacion_maxima_centimos"] == 2 * precio
    # ts de CAMARA_TAPADA = t_mala; sin_vigilancia ≈ duración real de oscuridad ± 1 cuadro
    duracion_oscuridad = frames_tapada / fps
    un_cuadro = 1.0 / fps
    assert abs(resumen["sin_vigilancia_s"] - duracion_oscuridad) <= un_cuadro, (
        f"sin_vigilancia={resumen['sin_vigilancia_s']}, esperado≈{duracion_oscuridad}"
    )


# ---------------------------------------------------------------------------
# test_02: cada evento tiene las claves del contrato §10, uuid único y ts con zona
# ---------------------------------------------------------------------------

def test_02_contrato_eventos(tmp_path):
    """Cada evento guardado tiene uuid, ts con -05:00 y solo los campos permitidos."""
    motor, con, jid = _motor_con(tmp_path)
    img = _imagen()
    fps = _FPS

    # Track 10: cruza de AFUERA a ADENTRO
    ys = np.linspace(150.0, 450.0, 60)
    for idx, y in enumerate(ys):
        motor._detector = type("D", (), {"procesar": lambda s, i, y=y: [_persona(10, 320.0, y)]})()
        motor.procesar(_cuadro(img, idx))

    motor.detener(60 / fps)

    eventos = con.execute(
        "SELECT * FROM evento WHERE jornada_id = ?", (jid,)
    ).fetchall()
    assert len(eventos) >= 1

    uuids = set()
    for ev in eventos:
        assert "-05:00" in ev["ts"], f"ts sin zona: {ev['ts']}"
        assert ev["uuid"] not in uuids, "uuid duplicado"
        uuids.add(ev["uuid"])
        if ev["tipo"] == "ENTRADA":
            assert ev["categoria"] is not None


# ---------------------------------------------------------------------------
# test_03: después de decidir, el identificador no conserva vectores del track
# ---------------------------------------------------------------------------

def test_03_descartar_tras_decidir(tmp_path):
    """Después de decidir una entrada, _tracks del track se elimina del identificador."""
    motor, con, jid = _motor_con(tmp_path)
    img = _imagen()
    fps = _FPS

    # Inyectar vectores manualmente para track 20
    v = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    for _ in range(3):
        motor._identificador.agregar(20, Rostro(vector=v, puntaje=0.9, ancho_px=60))

    ys = np.linspace(150.0, 450.0, 60)
    for idx, y in enumerate(ys):
        motor._detector = type("D", (), {"procesar": lambda s, i, y=y: [_persona(20, 320.0, y)]})()
        motor.procesar(_cuadro(img, idx))

    # Esperar vencimiento de ventana
    t_extra = 2.0
    frames_extra = int(t_extra * fps)
    for i in range(frames_extra):
        motor._detector = type("D", (), {"procesar": lambda s, i: []})()
        motor.procesar(_cuadro(img, 60 + i))

    assert 20 not in motor._identificador._tracks


# ---------------------------------------------------------------------------
# test_04: ENTRADA pendiente se decide como NO_IDENTIFICADO si el track se pierde
# ---------------------------------------------------------------------------

def test_04_track_perdido_sin_rostros(tmp_path):
    """Un track que cruza sin rostros y luego desaparece queda como NO_IDENTIFICADO."""
    motor, con, jid = _motor_con(tmp_path, precio=100)
    img = _imagen()
    fps = _FPS

    # Track 30 cruza sin que se inyecten vectores
    ys = np.linspace(150.0, 450.0, 60)
    for idx, y in enumerate(ys):
        motor._detector = type("D", (), {"procesar": lambda s, i, y=y: [_persona(30, 320.0, y)]})()
        motor.procesar(_cuadro(img, idx))

    # El track desaparece (frames vacíos)
    for i in range(45):
        motor._detector = type("D", (), {"procesar": lambda s, i: []})()
        motor.procesar(_cuadro(img, 60 + i))

    motor.detener(105 / fps)

    ev = con.execute(
        "SELECT categoria FROM evento WHERE tipo='ENTRADA' AND jornada_id=?",
        (jid,),
    ).fetchone()
    assert ev is not None
    assert ev["categoria"] == "NO_IDENTIFICADO"


# ---------------------------------------------------------------------------
# test_05: humo final con video sintético y --sin-ventana
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# test_06: vista muestra nombre y color correcto al decidir (H1)
# ---------------------------------------------------------------------------

def test_06_vista_etiqueta_personal(tmp_path):
    """Después de decidir una entrada PERSONAL, la vista muestra nombre y verde."""
    v_gerardo = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    v_carlos = normalizar(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    con = conectar(str(tmp_path / "ctrl.db"))
    id_gerardo = crear_perfil(con, "Gerardo", [v_gerardo], "sface_2021dec", [0.95])
    id_carlos = crear_perfil(con, "Carlos", [v_carlos], "sface_2021dec", [0.95])

    perfiles = {id_gerardo: ("Gerardo", v_gerardo), id_carlos: ("Carlos", v_carlos)}
    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", 150, True, "sface_2021dec", 0.45, 0.30)

    cfg = _ConfigFalso()
    cfg.precio_centimos = 150
    contador = Contador(_BANDA, confirmar_cuadros=3, enfriamiento_s=2.0, perdido_s=1.5)
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    identificador = Identificador(perfiles, umbral_alto=0.45, umbral_bajo=0.10, min_rostros=1)
    motor = Motor(cfg, None, None, identificador, contador, monitor, con, jid, inicio)

    img = _imagen()
    fps = _FPS
    idx = 0

    # Inyectar vectores de Gerardo (track 1) y Carlos (track 2)
    for _ in range(3):
        identificador.agregar(1, Rostro(vector=v_gerardo, puntaje=0.95, ancho_px=60))
        identificador.agregar(2, Rostro(vector=v_carlos, puntaje=0.95, ancho_px=60))

    # Ambos tracks cruzan la banda juntos
    ys = np.linspace(150.0, 450.0, 60)
    for y in ys:
        ps = [_persona(1, 200.0, float(y)), _persona(2, 400.0, float(y))]
        motor._detector = type("D", (), {"procesar": lambda s, i, ps=ps: ps})()
        motor.procesar(_cuadro(img, idx))
        idx += 1

    # Mantener visibles más allá de ventana_post_s (1.0 s)
    vista_final = None
    for _ in range(int(2.0 * fps)):
        ps = [_persona(1, 200.0, 450.0), _persona(2, 400.0, 450.0)]
        motor._detector = type("D", (), {"procesar": lambda s, i, ps=ps: ps})()
        vista_final = motor.procesar(_cuadro(img, idx)); idx += 1

    assert vista_final is not None
    por_track = {p.track_id: p for p in vista_final.personas}

    assert 1 in por_track, "Track 1 (Gerardo) no está en la vista"
    assert "Gerardo" in por_track[1].etiqueta, f"etiqueta={por_track[1].etiqueta!r}"
    assert por_track[1].color == (50, 200, 50), f"color={por_track[1].color}"

    assert 2 in por_track, "Track 2 (Carlos) no está en la vista"
    assert "Carlos" in por_track[2].etiqueta, f"etiqueta={por_track[2].etiqueta!r}"
    assert por_track[2].color == (50, 200, 50), f"color={por_track[2].color}"


# ---------------------------------------------------------------------------
# test_07: registrar_camara guarda eventos y suma sin_vigilancia (H4)
# ---------------------------------------------------------------------------

def test_07_registrar_camara(tmp_path):
    """registrar_camara guarda CAMARA_CAIDA y CAMARA_OK con contrato §10."""
    motor, con, jid = _motor_con(tmp_path)

    motor.registrar_camara("CAMARA_CAIDA", 10.0)
    motor.registrar_camara("CAMARA_OK", 20.0)

    t_total = 25.0
    resumen = motor.detener(t_total)

    eventos = con.execute(
        "SELECT tipo, ts, uuid FROM evento WHERE jornada_id = ? ORDER BY ts", (jid,)
    ).fetchall()

    tipos = [ev["tipo"] for ev in eventos]
    assert "CAMARA_CAIDA" in tipos
    assert "CAMARA_OK" in tipos

    for ev in eventos:
        assert "-05:00" in ev["ts"], f"ts sin zona: {ev['ts']}"
        assert ev["uuid"] is not None

    # CAIDA a t=10, OK a t=20 → 10 s sin vigilancia
    assert resumen["sin_vigilancia_s"] == pytest.approx(10.0, abs=0.001)


# ---------------------------------------------------------------------------
# test_08: extractor en el camino real de identificación (H5)
# ---------------------------------------------------------------------------

def test_08_extractor_camino_real(tmp_path):
    """Un track que cruza con extractor real llega a PERSONAL con su nombre."""
    v_gerardo = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    con = conectar(str(tmp_path / "ctrl.db"))
    id_gerardo = crear_perfil(con, "Gerardo", [v_gerardo], "sface_2021dec", [0.95])

    perfiles = {id_gerardo: ("Gerardo", v_gerardo)}
    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", 150, True, "sface_2021dec", 0.45, 0.30)

    cfg = _ConfigFalso()
    cfg.precio_centimos = 150
    cfg.distancia_ventana_px = 200.0
    cfg.cada_n_cuadros = 1

    contador = Contador(_BANDA, confirmar_cuadros=3, enfriamiento_s=2.0, perdido_s=1.5)
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    identificador = Identificador(perfiles, umbral_alto=0.45, umbral_bajo=0.10, min_rostros=1)

    # Extractor que siempre devuelve el vector de Gerardo
    extractor = type("E", (), {
        "buscar": staticmethod(
            lambda img, caja: Rostro(vector=v_gerardo, puntaje=0.95, ancho_px=60)
        ),
    })()

    motor = Motor(cfg, None, extractor, identificador, contador, monitor, con, jid, inicio)

    img = _imagen()
    fps = _FPS
    idx = 0

    # Track 1 cruza de AFUERA a ADENTRO
    ys = np.linspace(150.0, 450.0, 60)
    for y in ys:
        motor._detector = type("D", (), {"procesar": lambda s, i, y=y: [_persona(1, 320.0, y)]})()
        motor.procesar(_cuadro(img, idx)); idx += 1

    # Esperar que venza la ventana
    for _ in range(int(1.5 * fps)):
        motor._detector = type("D", (), {"procesar": lambda s, i: []})()
        motor.procesar(_cuadro(img, idx)); idx += 1

    resumen = motor.detener(idx / fps)

    assert resumen["personal"] == 1, f"personal={resumen['personal']}"

    ev = con.execute(
        "SELECT categoria, perfil_id FROM evento WHERE tipo='ENTRADA' AND jornada_id=?",
        (jid,),
    ).fetchone()
    assert ev["categoria"] == "PERSONAL"
    assert ev["perfil_id"] == id_gerardo


# ---------------------------------------------------------------------------
# test_09: track lejos de la banda no llama al extractor (H5)
# ---------------------------------------------------------------------------

def test_09_no_extractor_fuera_ventana(tmp_path):
    """Track que nunca se acerca a la banda no llama al extractor."""
    llamadas = []

    class _ExtractorContador:
        def buscar(self, img, caja):
            llamadas.append(1)
            return None

    con = conectar(str(tmp_path / "ctrl.db"))
    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", 100, True, "sface_2021dec", 0.45, 0.30)

    cfg = _ConfigFalso()
    cfg.distancia_ventana_px = 50.0  # ventana pequeña

    contador = Contador(_BANDA, confirmar_cuadros=3, enfriamiento_s=2.0, perdido_s=1.5)
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    identificador = Identificador({}, umbral_alto=0.45, umbral_bajo=0.10, min_rostros=1)
    extractor = _ExtractorContador()

    motor = Motor(cfg, None, extractor, identificador, contador, monitor, con, jid, inicio)

    img = _imagen()
    fps = _FPS

    # Track 1 se mantiene en y=150; banda en y=300 → distancia=150 > 50 px
    for idx in range(30):
        motor._detector = type("D", (), {
            "procesar": lambda s, i: [_persona(1, 320.0, 150.0)]
        })()
        motor.procesar(_cuadro(img, idx))

    assert len(llamadas) == 0, f"El extractor fue llamado {len(llamadas)} veces"


# ---------------------------------------------------------------------------
# test_10: etiqueta decidida persiste ante parpadeo (A2)
# ---------------------------------------------------------------------------

def test_10_etiqueta_persiste_tras_parpadeo(tmp_path):
    """Track decidido que falta <perdido_s conserva etiqueta; >perdido_s la pierde."""
    v = normalizar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    con = conectar(str(tmp_path / "ctrl.db"))
    id_ana = crear_perfil(con, "Ana", [v], "sface_2021dec", [0.95])

    perfiles = {id_ana: ("Ana", v)}
    inicio = datetime(2025, 1, 10, 8, 0, 0, tzinfo=_TZ)
    jid, _ = abrir_jornada(con, "test", 100, True, "sface_2021dec", 0.45, 0.30)

    cfg = _ConfigFalso()
    contador = Contador(_BANDA, confirmar_cuadros=3, enfriamiento_s=2.0, perdido_s=1.5)
    monitor = MonitorSalud(brillo_min=25, variacion_min=8, segundos=2.0)
    identificador = Identificador(perfiles, umbral_alto=0.45, umbral_bajo=0.10, min_rostros=1)
    motor = Motor(cfg, None, None, identificador, contador, monitor, con, jid, inicio)

    img = _imagen()
    fps = _FPS
    idx = 0

    # Inyectar vector de Ana para track 1
    for _ in range(3):
        identificador.agregar(1, Rostro(vector=v, puntaje=0.95, ancho_px=60))

    # Track 1 cruza la banda
    for y in np.linspace(150.0, 450.0, 60):
        motor._detector = type("D", (), {"procesar": lambda s, i, y=y: [_persona(1, 320.0, y)]})()
        motor.procesar(_cuadro(img, idx)); idx += 1

    # Mantener visible más allá de ventana_post_s para que se decida
    for _ in range(int(1.5 * fps)):
        motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(1, 320.0, 450.0)]})()
        motor.procesar(_cuadro(img, idx)); idx += 1

    assert 1 in motor._decididos, "Track 1 no fue decidido"

    # Falta 3 cuadros (< perdido_s = 1.5 s a 15 fps → < 22 cuadros)
    for _ in range(3):
        motor._detector = type("D", (), {"procesar": lambda s, i: []})()
        motor.procesar(_cuadro(img, idx)); idx += 1

    assert 1 in motor._decididos, "Etiqueta borrada prematuramente tras 3 cuadros"

    # Vuelve: debe mostrar etiqueta
    motor._detector = type("D", (), {"procesar": lambda s, i: [_persona(1, 320.0, 450.0)]})()
    vista = motor.procesar(_cuadro(img, idx)); idx += 1

    por_track = {p.track_id: p for p in vista.personas}
    assert 1 in por_track
    assert "Ana" in por_track[1].etiqueta, f"etiqueta={por_track[1].etiqueta!r}"

    # Falta más de perdido_s (1.5 s = 22 cuadros + margen)
    for _ in range(int(1.5 * fps) + 5):
        motor._detector = type("D", (), {"procesar": lambda s, i: []})()
        motor.procesar(_cuadro(img, idx)); idx += 1

    assert 1 not in motor._decididos, "Etiqueta no se borró tras perdido_s"


def test_05_humo_demo(tmp_path):
    """vision.demo procesa un video sintético sin tocar data/control.db."""
    import yaml
    from pathlib import Path

    # Crear video sintético
    video = tmp_path / "demo.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"XVID"),
        _FPS,
        (640, 480),
    )
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:240, :320] = 200  # variación para no activar tapada
        writer.write(frame)
    writer.release()

    # Crear config temporal con ruta_db y fuente dentro de tmp_path
    with open("config.yaml", encoding="utf-8") as f:
        cfg_datos = yaml.safe_load(f)
    cfg_datos["ruta_db"] = str(tmp_path / "ctrl.db")
    cfg_datos["fuente"] = str(video)
    cfg_tmp = tmp_path / "config_test.yaml"
    with open(str(cfg_tmp), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_datos, f)

    # Verificar estado de data/control.db antes
    db_real = Path("data/control.db")
    existia = db_real.exists()
    mtime_antes = db_real.stat().st_mtime if existia else None

    import vision.demo as demo_mod
    demo_mod.main([
        "--config", str(cfg_tmp),
        "--precio", "100",
        "--sin-ventana",
    ])

    # Verificar que data/control.db no fue tocado
    if existia:
        assert db_real.stat().st_mtime == mtime_antes, "data/control.db fue modificado"
    else:
        assert not db_real.exists(), "data/control.db fue creado"
