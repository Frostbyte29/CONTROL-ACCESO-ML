from base_datos.conexion import conectar


PRECIO_CASUAL = 1.00


# Función: crear la tabla de atenciones
def crear_tabla_atenciones():
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS atenciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_cliente INTEGER,
            fecha_hora TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            monto REAL NOT NULL,
            FOREIGN KEY (id_cliente) REFERENCES clientes(id)
        )
    """)

    conexion.commit()
    conexion.close()


# Función: registrar una atención
def registrar_atencion(id_cliente=None):
    conexion = conectar()
    cursor = conexion.cursor()

    monto = 0.0 if id_cliente is not None else PRECIO_CASUAL

    cursor.execute("""
        INSERT INTO atenciones (id_cliente, monto)
        VALUES (?, ?)
    """, (id_cliente, monto))

    conexion.commit()
    conexion.close()


# Función: obtener la recaudación del día
def obtener_recaudacion_hoy():
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT COALESCE(SUM(monto), 0)
        FROM atenciones
        WHERE DATE(fecha_hora) = DATE('now', 'localtime')
    """)

    recaudacion = cursor.fetchone()[0]
    conexion.close()

    return recaudacion