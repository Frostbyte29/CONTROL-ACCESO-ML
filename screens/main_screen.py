import customtkinter as ctk
import cv2

from PIL import Image, ImageTk

from screens.registro_screen import RegistroScreen
from screens.actividad_screen import ActividadScreen
from logica.sistema import Sistema


class MainScreen(ctk.CTk):

    def __init__(self):
        super().__init__()

        # =========================
        # VENTANA
        # =========================

        self.title("Sistema de Baños")
        self.geometry("1280x800")
        self.resizable(False, False)

        self.protocol(
            "WM_DELETE_WINDOW",
            self.cerrar
        )

        # =========================
        # SISTEMA CENTRAL
        # =========================

        self.sistema = Sistema()

        self.after_id = None

        # =========================
        # CONFIGURACIÓN
        # =========================

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)

        self.grid_columnconfigure(
            0,
            weight=1
        )

        # =========================
        # CÁMARA
        # =========================

        self.camara_frame = ctk.CTkFrame(
            self,
            width=1280,
            height=630
        )

        self.camara_frame.grid(
            row=0,
            column=0,
            padx=10,
            pady=(10, 5)
        )

        self.camara_frame.grid_propagate(False)

        self.label_camara = ctk.CTkLabel(
            self.camara_frame,
            text="CÁMARA",
            font=("Arial", 30, "bold")
        )

        self.label_camara.pack(
            fill="both",
            expand=True
        )

        # =========================
        # PANEL INFERIOR
        # =========================

        self.panel = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.panel.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        # =========================
        # CONTENIDO DEL PANEL
        # =========================

        self.contenido_panel = ctk.CTkFrame(
            self.panel,
            fg_color="transparent"
        )

        self.contenido_panel.pack(
            fill="both",
            expand=True
        )

        # =========================
        # MENÚ
        # =========================

        self.mostrar_menu()

        # =========================
        # INICIAR CÁMARA
        # =========================

        self.actualizar_camara()

    # =========================
    # ACTUALIZAR CÁMARA
    # =========================

    def actualizar_camara(self):

        frame = self.sistema.procesar_frame()

        if frame is not None:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            imagen = Image.fromarray(
                frame
            )

            imagen = imagen.resize(
                (1280, 630)
            )

            imagen = ImageTk.PhotoImage(
                imagen
            )

            self.label_camara.configure(
                image=imagen,
                text=""
            )

            self.label_camara.image = imagen

        self.after_id = self.after(
            30,
            self.actualizar_camara
        )

    # =========================
    # LIMPIAR CONTENIDO
    # =========================

    def limpiar_panel(self):

        for widget in self.contenido_panel.winfo_children():
            widget.destroy()

    # =========================
    # MENÚ
    # =========================

    def mostrar_menu(self):

        self.limpiar_panel()

        titulo = ctk.CTkLabel(
            self.contenido_panel,
            text="MENÚ",
            font=("Arial", 22, "bold")
        )

        titulo.pack(
            pady=(15, 10)
        )

        botones = ctk.CTkFrame(
            self.contenido_panel,
            fg_color="transparent"
        )

        botones.pack()

        ctk.CTkButton(
            botones,
            text="Actividad",
            command=self.mostrar_actividad
        ).pack(
            side="left",
            padx=10
        )

        ctk.CTkButton(
            botones,
            text="Registrar usuario",
            command=self.mostrar_registro
        ).pack(
            side="left",
            padx=10
        )

        ctk.CTkButton(
            botones,
            text="Usuarios",
            command=self.mostrar_usuarios
        ).pack(
            side="left",
            padx=10
        )

        ctk.CTkButton(
            botones,
            text="Historial",
            command=self.mostrar_historial
        ).pack(
            side="left",
            padx=10
        )
    # =========================
    # ACTIVIDD
    # =========================
    
    def mostrar_actividad(self):

        self.limpiar_panel()

        pantalla = ActividadScreen(
            self.contenido_panel,
            volver=self.mostrar_menu
        )

        pantalla.pack(
            fill="both",
            expand=True
        )
    
    # =========================
    # REGISTRO
    # =========================

    def mostrar_registro(self):

        self.limpiar_panel()

        pantalla = RegistroScreen(
            self.contenido_panel,
            volver=self.mostrar_menu
        )

        pantalla.pack(
            fill="both",
            expand=True
        )

    # =========================
    # USUARIOS
    # =========================

    def mostrar_usuarios(self):

        self.limpiar_panel()

        titulo = ctk.CTkLabel(
            self.contenido_panel,
            text="USUARIOS REGISTRADOS",
            font=("Arial", 22, "bold")
        )

        titulo.pack(
            pady=20
        )

        ctk.CTkButton(
            self.contenido_panel,
            text="Volver al menú",
            command=self.mostrar_menu
        ).pack()

    # =========================
    # HISTORIAL
    # =========================

    def mostrar_historial(self):

        self.limpiar_panel()

        titulo = ctk.CTkLabel(
            self.contenido_panel,
            text="HISTORIAL",
            font=("Arial", 22, "bold")
        )

        titulo.pack(
            pady=20
        )

        ctk.CTkButton(
            self.contenido_panel,
            text="Volver al menú",
            command=self.mostrar_menu
        ).pack()

    # =========================
    # CERRAR APLICACIÓN
    # =========================

    def cerrar(self):

        if self.after_id is not None:

            self.after_cancel(
                self.after_id
            )

            self.after_id = None

        self.sistema.cerrar()

        self.destroy()


if __name__ == "__main__":
    app = MainScreen()
    app.mainloop()