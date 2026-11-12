"""Motor principal: une captura, personas, conteo, identidad, salud y base de datos."""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from datos.db import insertar_evento, latido, cerrar_jornada, resumen as db_resumen
from vision.conteo import Contador, Cruce
from vision.identidad import Identificador, NO_IDENTIFICADO
from vision.personas import Persona
from vision.salud import MonitorSalud

_TZ = timezone(timedelta(hours=-5))


@dataclass
class InfoPersona:
    """Estado visible de una persona detectada en el cuadro actual."""

    track_id: int
    caja: tuple[float, float, float, float]
    etiqueta: str
    color: tuple[int, int, int]      # BGR
    recorrido: list[tuple[float, float]]  # últimos ~30 puntos de los pies


@dataclass
class EstadoVista:
    """Todo lo que necesita la ventana para dibujar un cuadro."""

    personas: list[InfoPersona] = field(default_factory=list)
    entradas: int = 0
    salidas: int = 0
    personal: int = 0
    pagantes: int = 0
    no_identificados: int = 0
    recaudacion_confirmada: int = 0
    recaudacion_maxima: int = 0
    avisos: list[str] = field(default_factory=list)
    fps: float = 0.0


# Colores BGR por categoría
_COLOR_GRIS = (160, 160, 160)
_COLOR_VERDE = (50, 200, 50)
_COLOR_ROJO = (50, 50, 200)
_COLOR_AMBAR = (50, 180, 220)

_MAX_RECORRIDO = 30


@dataclass
class _EntradaPendiente:
    """ENTRADA que espera decisión de identidad."""

    track_id: int
    ts_cruce: str                 # ISO 8601, momento del cruce
    vence_t: float                # tiempo monotónico de vencimiento de la ventana
    precio_centimos: int


class Motor:
    """Orquesta todas las piezas y emite eventos a la base de datos."""

    def __init__(
        self,
        cfg,                      # Config (duck-typing para facilitar pruebas)
        detector,                 # DetectorPersonas o fake
        extractor,                # ExtractorRostro o None o fake
        identificador: Identificador,
        contador: Contador,
        monitor: MonitorSalud,
        con,                      # conexión sqlite3
        jornada_id: int,
        inicio_jornada: datetime,
    ) -> None:
        self._cfg = cfg
        self._detector = detector
        self._extractor = extractor
        self._identificador = identificador
        self._contador = contador
        self._monitor = monitor
        self._con = con
        self._jornada_id = jornada_id
        self._inicio = inicio_jornada

        self._pendientes: dict[int, _EntradaPendiente] = {}
        self._decididos: dict[int, tuple[str, str | None]] = {}
        self._recorridos: dict[int, list[tuple[float, float]]] = {}
        self._contadores = dict(entradas=0, salidas=0, personal=0,
                                pagantes=0, no_identificados=0,
                                confirmada=0, maxima=0)
        self._avisos: list[str] = []
        self._n_cuadro = 0
        self._ultimo_latido_t = 0.0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def procesar(self, cuadro) -> EstadoVista:
        """Procesa un cuadro y devuelve el estado para dibujar."""
        t = cuadro.t
        imagen = cuadro.imagen
        self._n_cuadro += 1

        # 1. Salud
        ev_salud = self._monitor.actualizar(imagen, t)
        if ev_salud:
            tipo_salud, t_evento = ev_salud
            self._guardar_evento({"tipo": tipo_salud, "ts": self._ts(t_evento)})
            if tipo_salud in ("CAMARA_TAPADA", "CAMARA_CAIDA"):
                self._avisos = [tipo_salud]
            else:
                self._avisos = []

        # 2. Personas
        personas = self._detector.procesar(imagen)

        # 3. Ventana de identificación
        if self._extractor is not None:
            self._actualizar_identificacion(imagen, personas, t)

        # 4. Conteo
        puntos = {p.track_id: p.pies for p in personas}
        cruces = self._contador.actualizar(t, puntos)
        for cruce in cruces:
            if cruce.tipo == "SALIDA":
                self._registrar_salida(cruce, t)
            else:
                self._registrar_entrada_pendiente(cruce, t)

        # 5. Decisiones de entradas pendientes vencidas
        self._resolver_pendientes(t, tracks_activos={p.track_id for p in personas})

        # 6. Latido
        if t - self._ultimo_latido_t >= self._cfg.latido_s:
            latido(self._con, self._jornada_id, self._ts(t))
            self._ultimo_latido_t = t

        # Actualizar recorridos
        for p in personas:
            rec = self._recorridos.setdefault(p.track_id, [])
            rec.append(p.pies)
            if len(rec) > _MAX_RECORRIDO:
                rec.pop(0)

        vista = self._construir_vista(personas, t)

        # Actualizar t_ultima_vista para tracks visibles en este cuadro
        for p in personas:
            if p.track_id in self._decididos:
                cat, nombre, _ = self._decididos[p.track_id]
                self._decididos[p.track_id] = (cat, nombre, t)

        # Limpiar decididos para tracks perdidos hace más de perdido_s
        for tid in list(self._decididos):
            _, _, t_vista = self._decididos[tid]
            if (t - t_vista) > self._cfg.perdido_s:
                del self._decididos[tid]

        return vista

    def registrar_camara(self, tipo: str, t: float) -> None:
        """Registra un evento de cámara (CAIDA, TAPADA, OK) sin pasar por procesar."""
        self._guardar_evento({"tipo": tipo, "ts": self._ts(t)})
        if tipo in ("CAMARA_TAPADA", "CAMARA_CAIDA"):
            self._avisos = [tipo]
        else:
            self._avisos = []

    def detener(self, t: float) -> dict:
        """Decide pendientes, cierra la jornada y devuelve el resumen."""
        for track_id in list(self._pendientes):
            self._decidir_entrada(track_id, t)
        ts = self._ts(t)
        cerrar_jornada(self._con, self._jornada_id, ts)
        return db_resumen(self._con, self._jornada_id)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _ts(self, t: float) -> str:
        """Convierte t (segundos desde inicio de jornada) a ISO 8601 con zona -05:00."""
        dt = self._inicio + timedelta(seconds=t)
        return dt.isoformat(timespec="milliseconds")

    def _guardar_evento(self, ev: dict) -> None:
        """Inserta un evento asegurando uuid y jornada_id."""
        ev.setdefault("uuid", str(_uuid.uuid4()))
        ev["jornada_id"] = self._jornada_id
        insertar_evento(self._con, ev)

    def _registrar_salida(self, cruce: Cruce, t: float) -> None:
        """Guarda un evento SALIDA de inmediato."""
        self._guardar_evento({
            "tipo": "SALIDA",
            "ts": self._ts(cruce.t),
            "track_id": cruce.track_id,
        })
        self._contadores["salidas"] += 1

    def _registrar_entrada_pendiente(self, cruce: Cruce, t: float) -> None:
        """Deja la ENTRADA pendiente hasta que venza la ventana de identificación."""
        self._decididos.pop(cruce.track_id, None)
        self._pendientes[cruce.track_id] = _EntradaPendiente(
            track_id=cruce.track_id,
            ts_cruce=self._ts(cruce.t),
            vence_t=cruce.t + self._cfg.ventana_post_s,
            precio_centimos=self._cfg.precio_centimos,
        )
        self._contadores["entradas"] += 1

    def _resolver_pendientes(self, t: float, tracks_activos: set[int]) -> None:
        """Decide las entradas cuya ventana venció o cuyo track ya no está."""
        for track_id in list(self._pendientes):
            pendiente = self._pendientes[track_id]
            if t >= pendiente.vence_t or track_id not in tracks_activos:
                self._decidir_entrada(track_id, t)

    def _decidir_entrada(self, track_id: int, t: float) -> None:
        """Decide la categoría del track y guarda el evento ENTRADA."""
        pendiente = self._pendientes.pop(track_id, None)
        if pendiente is None:
            return

        cat, perfil_id, similitud = self._identificador.decidir(track_id)
        nombre = self._identificador.nombre(perfil_id)
        self._identificador.descartar(track_id)
        self._decididos[track_id] = (cat, nombre, t)

        ev: dict = {
            "tipo": "ENTRADA",
            "ts": pendiente.ts_cruce,
            "track_id": track_id,
            "categoria": cat,
            "perfil_id": perfil_id,
            "similitud": float(similitud),
            "precio_centimos": pendiente.precio_centimos,
        }
        self._guardar_evento(ev)

        if cat == "PERSONAL":
            self._contadores["personal"] += 1
        elif cat == "PAGANTE":
            self._contadores["pagantes"] += 1
            self._contadores["confirmada"] += pendiente.precio_centimos
            self._contadores["maxima"] += pendiente.precio_centimos
        else:
            self._contadores["no_identificados"] += 1
            self._contadores["maxima"] += pendiente.precio_centimos

    def _actualizar_identificacion(
        self,
        imagen: np.ndarray,
        personas: list[Persona],
        t: float,
    ) -> None:
        """Busca rostros en la ventana de identificación de cada track."""
        cx_banda = (self._cfg.banda[0] + self._cfg.banda[2]) / 2
        cy_banda = (self._cfg.banda[1] + self._cfg.banda[3]) / 2
        # Coordenada central de la banda (punto medio de la línea)
        _ = (cx_banda, cy_banda)

        for persona in personas:
            if not self._en_ventana(persona, t):
                continue
            if self._n_cuadro % self._cfg.cada_n_cuadros != 0:
                continue
            rostro = self._extractor.buscar(imagen, persona.caja)
            if rostro is not None:
                self._identificador.agregar(persona.track_id, rostro)

    def _en_ventana(self, persona: Persona, t: float) -> bool:
        """True si el track está dentro de la ventana de identificación."""
        # La ventana está abierta si el track tiene una entrada pendiente
        # cuya ventana aún no venció, o si está cerca de la banda
        if persona.track_id in self._pendientes:
            return t <= self._pendientes[persona.track_id].vence_t

        # Distancia de los pies a la línea central de la banda
        px, py = persona.pies
        bx1, by1, bx2, by2 = self._cfg.banda[:4]
        d = _distancia_punto_segmento(px, py, bx1, by1, bx2, by2)
        return d <= self._cfg.distancia_ventana_px

    def _construir_vista(
        self,
        personas: list[Persona],
        t: float,
    ) -> EstadoVista:
        """Arma el EstadoVista con la información de este cuadro."""
        infos = []
        for p in personas:
            if p.track_id in self._pendientes:
                etiqueta = "identificando..."
                color = _COLOR_GRIS
            elif p.track_id in self._decididos:
                cat, nombre, _ = self._decididos[p.track_id]
                etiqueta, color = _etiqueta_color(cat, p.track_id, nombre)
            else:
                etiqueta, color = "", _COLOR_GRIS

            rec = list(self._recorridos.get(p.track_id, []))
            infos.append(InfoPersona(
                track_id=p.track_id,
                caja=p.caja,
                etiqueta=etiqueta,
                color=color,
                recorrido=rec,
            ))

        return EstadoVista(
            personas=infos,
            entradas=self._contadores["entradas"],
            salidas=self._contadores["salidas"],
            personal=self._contadores["personal"],
            pagantes=self._contadores["pagantes"],
            no_identificados=self._contadores["no_identificados"],
            recaudacion_confirmada=self._contadores["confirmada"],
            recaudacion_maxima=self._contadores["maxima"],
            avisos=list(self._avisos),
            fps=0.0,  # calculado en demo.py
        )


# ---------------------------------------------------------------------------
# Helpers puros
# ---------------------------------------------------------------------------

def _distancia_punto_segmento(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Distancia del punto (px, py) al segmento (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    largo2 = dx * dx + dy * dy
    if largo2 == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / largo2))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _etiqueta_color(
    cat: str | None,
    track_id: int,
    nombre: str | None,
) -> tuple[str, tuple[int, int, int]]:
    """Devuelve (etiqueta, color_BGR) según la categoría."""
    if cat is None:
        return ("", _COLOR_GRIS)
    if cat == "PERSONAL":
        label = f"{nombre or 'Personal'} · personal"
        return label, _COLOR_VERDE
    if cat == "PAGANTE":
        return "Pagante", _COLOR_ROJO
    return "No identificado", _COLOR_AMBAR
