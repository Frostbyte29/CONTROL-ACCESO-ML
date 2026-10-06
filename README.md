PROYECTO_BANOS

Sistema inteligente de reconocimiento facial y control de acceso para usuarios de servicios higiénicos.

Descripción

El proyecto implementa un sistema de control de acceso mediante reconocimiento facial. El sistema utiliza una cámara para detectar y realizar el seguimiento de las personas, generar embeddings faciales y reconocer usuarios registrados.

Cuando una persona ingresa a la zona de atención, el sistema determina si corresponde a un usuario registrado o a un usuario casual y registra la atención en una base de datos SQLite.

Tecnologías utilizadas
Python
OpenCV
InsightFace
SCRFD
ArcFace
NumPy
CustomTkinter
SQLite
Requisitos
Python 3.12
Cámara web
Windows
Instalación
1. Clonar el repositorio
git clone URL_DEL_REPOSITORIO
2. Ingresar al proyecto
cd PROYECTO_BANOS
3. Crear el entorno virtual
python -m venv .venv
4. Activar el entorno virtual

En Windows:

.venv\Scripts\activate
5. Instalar las dependencias
pip install -r requirements.txt
Ejecución

Para iniciar el sistema:

python main.py

Se abrirá la interfaz gráfica y se iniciará la captura de video mediante la cámara.

Estructura del proyecto
PROYECTO_BANOS/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── screens/
│   ├── main_screen.py
│   ├── registro_screen.py
│   └── actividad_screen.py
│
├── logica/
│   ├── sistema.py
│   ├── reconocimiento.py
│   ├── atencion.py
│   └── zona.py
│
├── reconocimiento/
│   ├── reconocimiento.py
│   └── tracking.py
│
├── base_datos/
│   ├── conexion.py
│   ├── clientes.py
│   └── atenciones.py
│
├── camara/
│   └── camara.py
│
├── modelos/
│
└── rostros/
Funcionamiento

El sistema sigue el siguiente flujo:

Captura imágenes mediante la cámara.
Detecta los rostros presentes en la imagen.
Realiza el seguimiento de las personas mediante IDs.
Genera varios embeddings faciales para una misma persona.
Combina los embeddings obtenidos.
Compara el embedding resultante con los embeddings almacenados en SQLite.
Identifica al usuario o lo clasifica como desconocido.
Comprueba si la persona se encuentra dentro de la zona de atención.
Registra la atención y el monto correspondiente en la base de datos.
Interfaz

Actualmente la interfaz cuenta con:

Registrar usuario: permite registrar un usuario mediante su nombre y 10 fotografías.
Actividad: muestra información relacionada con usuarios, desconocidos y monto recaudado.
Base de datos

El sistema utiliza SQLite para almacenar la información de los usuarios registrados y las atenciones realizadas.

Los embeddings faciales se almacenan directamente en la base de datos para realizar posteriormente la identificación de los usuarios.

Estado actual

El proyecto se encuentra en etapa de desarrollo académico. Actualmente se encuentran implementados:

Captura de video.
Detección facial mediante InsightFace.
Tracking de personas.
Generación de embeddings faciales.
Reconocimiento facial.
Registro de usuarios.
Registro de atenciones.
Control de zona de atención.
Interfaz gráfica básica.
Base de datos SQLite.
Visualización de actividad.