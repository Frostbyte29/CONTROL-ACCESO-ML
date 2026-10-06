from base_datos.conexion import conectar


# Función: crear la tabla de clientes
def crear_tabla_clientes():
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            usuario_rostro TEXT NOT NULL,
            embedding BLOB,
            fecha_registro TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conexion.commit()
    conexion.close()


# Función: registrar un cliente
def registrar_cliente(nombre, usuario_rostro, embedding=None):
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO clientes (
            nombre,
            usuario_rostro,
            embedding
        )
        VALUES (?, ?, ?)
    """, (nombre, usuario_rostro, embedding))

    conexion.commit()
    conexion.close()


# Función: obtener todos los clientes
def obtener_clientes():
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT
            id,
            nombre,
            usuario_rostro,
            embedding,
            fecha_registro
        FROM clientes
        ORDER BY id
    """)

    clientes = cursor.fetchall()
    conexion.close()

    return clientes


# Función: buscar un cliente por su identificador facial
def obtener_cliente_por_rostro(usuario_rostro):
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT
            id,
            nombre,
            embedding
        FROM clientes
        WHERE usuario_rostro = ?
    """, (usuario_rostro,))

    cliente = cursor.fetchone()
    conexion.close()

    return cliente


# Función: actualizar el embedding de un cliente
def actualizar_embedding(id_cliente, embedding):
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        UPDATE clientes
        SET embedding = ?
        WHERE id = ?
    """, (embedding, id_cliente))

    conexion.commit()
    conexion.close()


# Función: eliminar un cliente por su ID
def eliminar_cliente(id_cliente):
    conexion = conectar()
    cursor = conexion.cursor()

    cursor.execute("""
        DELETE FROM clientes
        WHERE id = ?
    """, (id_cliente,))

    conexion.commit()
    conexion.close()