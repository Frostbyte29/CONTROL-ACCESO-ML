import sqlite3


# Función: conectar con la base de datos
def conectar():
    return sqlite3.connect("sistema_banos.db")