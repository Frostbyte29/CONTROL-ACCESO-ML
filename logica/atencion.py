PRECIO_CASUAL = 1.00


# Función: determinar el tipo de atención
def determinar_atencion(cliente):

    if cliente is not None:

        return {
            "tipo": "cliente",
            "id_cliente": cliente[0],
            "nombre": cliente[1],
            "monto": 0.0
        }

    return {
        "tipo": "casual",
        "id_cliente": None,
        "nombre": "Usuario casual",
        "monto": PRECIO_CASUAL
    }