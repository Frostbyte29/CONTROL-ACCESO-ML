import customtkinter as ctk


class ActividadScreen(ctk.CTkFrame):

    def __init__(self, parent, volver):

        super().__init__(
            parent,
            fg_color="transparent"
        )

        # =========================
        # TÍTULO
        # =========================

        titulo = ctk.CTkLabel(
            self,
            text="Actividad",
            font=("Arial", 22, "bold")
        )

        titulo.pack(
            pady=(10, 15)
        )

        # =========================
        # CONTENEDOR
        # =========================

        contenedor = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        contenedor.pack()

        # =========================
        # USUARIOS
        # =========================

        tarjeta_usuarios = ctk.CTkFrame(
            contenedor
        )

        tarjeta_usuarios.pack(
            side="left",
            padx=10
        )

        ctk.CTkLabel(
            tarjeta_usuarios,
            text="Usuarios",
            font=("Arial", 16, "bold")
        ).pack(
            padx=30,
            pady=(10, 5)
        )

        self.label_usuarios = ctk.CTkLabel(
            tarjeta_usuarios,
            text="0",
            font=("Arial", 25, "bold")
        )

        self.label_usuarios.pack(
            padx=30,
            pady=(0, 10)
        )

        # =========================
        # DESCONOCIDOS
        # =========================

        tarjeta_desconocidos = ctk.CTkFrame(
            contenedor
        )

        tarjeta_desconocidos.pack(
            side="left",
            padx=10
        )

        ctk.CTkLabel(
            tarjeta_desconocidos,
            text="Desconocidos",
            font=("Arial", 16, "bold")
        ).pack(
            padx=30,
            pady=(10, 5)
        )

        self.label_desconocidos = ctk.CTkLabel(
            tarjeta_desconocidos,
            text="0",
            font=("Arial", 25, "bold")
        )

        self.label_desconocidos.pack(
            padx=30,
            pady=(0, 10)
        )

        # =========================
        # RECAUDACIÓN
        # =========================

        tarjeta_monto = ctk.CTkFrame(
            contenedor
        )

        tarjeta_monto.pack(
            side="left",
            padx=10
        )

        ctk.CTkLabel(
            tarjeta_monto,
            text="Monto",
            font=("Arial", 16, "bold")
        ).pack(
            padx=30,
            pady=(10, 5)
        )

        self.label_monto = ctk.CTkLabel(
            tarjeta_monto,
            text="S/ 0.00",
            font=("Arial", 25, "bold")
        )

        self.label_monto.pack(
            padx=30,
            pady=(0, 10)
        )

        # =========================
        # VOLVER
        # =========================

        ctk.CTkButton(
            contenedor,
            text="Volver al menú",
            command=volver
        ).pack(
            side = "left",
            pady=15
        )