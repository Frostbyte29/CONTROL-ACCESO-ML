# Función: calcular la zona de atención
def obtener_zona(frame):

    alto, ancho = frame.shape[:2]

    # La zona ocupa 1/4 del cuadrante inferior derecho
    ancho_zona = ancho//4

    # Coordenadas de la zona
    x1 = ancho - ancho_zona
    y1 = 0

    x2 = ancho
    y2 = alto

    return x1, y1, x2, y2


# Función: comprobar si una persona está dentro de la zona
def dentro_de_zona(centro, zona):

    x, y = centro
    x1, y1, x2, y2 = zona

    return (
        x1 <= x <= x2 and
        y1 <= y <= y2
    )