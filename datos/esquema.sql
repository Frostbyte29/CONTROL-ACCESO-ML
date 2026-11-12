-- Esquema SQLite del sistema de control de acceso (PLAN §4).
-- Nunca se almacena vector ni recorte del público.

CREATE TABLE IF NOT EXISTS perfil (
  id               INTEGER PRIMARY KEY,
  nombre           TEXT NOT NULL,
  rol              TEXT NOT NULL DEFAULT 'personal',  -- exento de pago
  activo           INTEGER NOT NULL DEFAULT 1,
  creado_en        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vector_rostro (
  id               INTEGER PRIMARY KEY,
  perfil_id        INTEGER NOT NULL REFERENCES perfil(id),
  modelo           TEXT NOT NULL,   -- vectores de modelos distintos no se comparan
  vector           BLOB NOT NULL,   -- float32
  calidad          REAL,
  creado_en        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jornada (
  id               INTEGER PRIMARY KEY,
  uuid             TEXT NOT NULL UNIQUE,
  inicio           TEXT NOT NULL,   -- ISO 8601 con zona (-05:00)
  fin              TEXT,
  fuente           TEXT NOT NULL,   -- índice de cámara, URL o archivo
  precio_centimos  INTEGER NOT NULL,
  mostrar_monto    INTEGER NOT NULL,
  modelo_facial    TEXT NOT NULL,
  umbral_alto      REAL NOT NULL,
  umbral_bajo      REAL NOT NULL,
  ultimo_latido    TEXT
);

CREATE TABLE IF NOT EXISTS evento (
  id               INTEGER PRIMARY KEY,
  uuid             TEXT NOT NULL UNIQUE,
  jornada_id       INTEGER NOT NULL REFERENCES jornada(id),
  ts               TEXT NOT NULL,
  tipo             TEXT NOT NULL,   -- ENTRADA, SALIDA, CAMARA_TAPADA, CAMARA_CAIDA, CAMARA_OK
  track_id         INTEGER,
  categoria        TEXT,            -- solo ENTRADA: PERSONAL, PAGANTE, NO_IDENTIFICADO
  perfil_id        INTEGER REFERENCES perfil(id),
  similitud        REAL,
  precio_centimos  INTEGER,
  sincronizado_en  TEXT
);
