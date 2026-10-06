from base_datos.clientes import (
    crear_tabla_clientes,
    registrar_cliente,
    obtener_clientes,
    obtener_cliente_por_rostro
)

from base_datos.atenciones import (
    crear_tabla_atenciones,
    registrar_atencion,
    obtener_recaudacion_hoy
)

crear_tabla_clientes()
crear_tabla_atenciones()