import customtkinter as ctk
import os
import shutil

from tkinter import filedialog, messagebox
from base_datos.clientes import (
    registrar_cliente,
    obtener_cliente_por_rostro,
    actualizar_embedding
)
from generar_embedding import generar_embedding


class RegistroScreen(ctk.CTkFrame):

    def __init__(self, parent, volver):

        super().__init__(
            parent,
            fg_color="transparent"
        )

        self.fotos = []

        # =========================
        # TÍTULO
        # =========================

        titulo = ctk.CTkLabel(
            self,
            text="Registrar usuario",
            font=("Arial", 22, "bold")
        )

        titulo.pack(
            pady=(10, 5)
        )

        # =========================
        # FILA 1
        # =========================

        fila_datos = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        fila_datos.pack(
            pady=5
        )

        # Nombre
        self.nombre_entry = ctk.CTkEntry(
            fila_datos,
            width=250,
            placeholder_text="Nombre del usuario"
        )

        self.nombre_entry.pack(
            side="left",
            padx=5
        )

        # Seleccionar fotos
        self.boton_fotos = ctk.CTkButton(
            fila_datos,
            text="Seleccionar fotos",
            command=self.seleccionar_fotos
        )

        self.boton_fotos.pack(
            side="left",
            padx=5
        )

        # Contador
        self.contador = ctk.CTkLabel(
            fila_datos,
            text="Fotos: 0/10"
        )

        self.contador.pack(
            side="left",
            padx=5
        )

        # =========================
        # FILA 2
        # =========================

        fila_botones = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        fila_botones.pack(
            pady=5
        )

        # Registrar
        self.boton_registrar = ctk.CTkButton(
            fila_botones,
            text="Registrar usuario",
            command=self.registrar_usuario
        )

        self.boton_registrar.pack(
            side="left",
            padx=5
        )

        # Volver
        self.boton_volver = ctk.CTkButton(
            fila_botones,
            text="Volver al menú",
            command=volver
        )

        self.boton_volver.pack(
            side="left",
            padx=5
        )

    # =========================
    # SELECCIONAR FOTOS
    # =========================

    def seleccionar_fotos(self):

        fotos = filedialog.askopenfilenames(
            title="Seleccionar fotografías",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png")
            ]
        )

        self.fotos = list(fotos)

        self.contador.configure(
            text=f"Fotos: {len(self.fotos)}/10"
        )

    # =========================
    # REGISTRAR USUARIO
    # =========================

    def registrar_usuario(self):

        nombre = self.nombre_entry.get().strip()

        if not nombre:

            messagebox.showwarning(
                "Aviso",
                "Ingrese el nombre del usuario."
            )

            return

        if len(self.fotos) != 10:

            messagebox.showwarning(
                "Aviso",
                "Debe seleccionar exactamente 10 fotos."
            )

            return

        # =========================
        # CARPETA ROSTROS
        # =========================

        carpeta_rostros = "rostros"

        if not os.path.exists(
            carpeta_rostros
        ):

            os.makedirs(
                carpeta_rostros
            )

        # =========================
        # SIGUIENTE USUARIO
        # =========================

        numero = 1

        while os.path.exists(
            os.path.join(
                carpeta_rostros,
                f"usuario{numero}"
            )
        ):

            numero += 1

        carpeta_usuario = os.path.join(
            carpeta_rostros,
            f"usuario{numero}"
        )

        os.makedirs(
            carpeta_usuario
        )

        # =========================
        # COPIAR FOTOS
        # =========================

        for i, foto in enumerate(
            self.fotos,
            start=1
        ):

            extension = os.path.splitext(
                foto
            )[1]

            destino = os.path.join(
                carpeta_usuario,
                f"foto{i}{extension}"
            )

            shutil.copy2(
                foto,
                destino
            )

        # =========================
        # REGISTRAR EN BD
        # =========================

        usuario_rostro = (
            f"usuario{numero}"
        )

        registrar_cliente(
            nombre,
            usuario_rostro
        )

        # =========================
        # GENERAR EMBEDDING
        # =========================

        embedding = generar_embedding(
            usuario_rostro
        )

        # =========================
        # OBTENER CLIENTE
        # =========================

        cliente = obtener_cliente_por_rostro(
            usuario_rostro
        )

        id_cliente = cliente[0]

        # =========================
        # GUARDAR EMBEDDING
        # =========================

        actualizar_embedding(
            id_cliente,
            embedding.tobytes()
        )

        # =========================
        # MENSAJE
        # =========================

        messagebox.showinfo(
            "Registro exitoso",
            f"Usuario registrado correctamente.\n\n"
            f"Nombre: {nombre}\n"
            f"Carpeta: usuario{numero}\n"
            f"Fotos: 10"
        )

        # =========================
        # LIMPIAR
        # =========================

        self.nombre_entry.delete(
            0,
            "end"
        )

        self.fotos = []

        self.contador.configure(
            text="Fotos: 0/10"
        )