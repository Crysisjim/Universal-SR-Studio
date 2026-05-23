import customtkinter as ctk
from tkinter import Toplevel, Message

class ToolTip:
    def __init__(self, widget, text, delay=800):
        """
        :param widget: Le widget sur lequel l'infobulle s'affiche
        :param text: Le texte de l'infobulle
        :param delay: Temps en ms avant affichage (800ms = 0.8sec)
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.id = None
        self.tw = None
        
        # Bindings des événements souris
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)

    def on_enter(self, event=None):
        self.unschedule()
        # On programme l'affichage après le délai défini
        self.id = self.widget.after(self.delay, self.show_tip)

    def on_leave(self, event=None):
        self.unschedule()
        self.hide_tip()

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def show_tip(self):
        if self.tw or not self.text:
            return

        try:
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except Exception:
            return

        # Clamp tooltip to screen bounds so it never clips outside the window
        try:
            sw = self.widget.winfo_screenwidth()
            sh = self.widget.winfo_screenheight()
            tip_w, tip_h = 320, 140  # generous estimate
            if x + tip_w > sw:
                x = sw - tip_w - 8
            if y + tip_h > sh:
                y = self.widget.winfo_rooty() - tip_h - 4
            x = max(4, x)
            y = max(4, y)
        except Exception:
            pass

        # Création de la fenêtre flottante
        self.tw = Toplevel(self.widget)
        self.tw.wm_overrideredirect(True) # Pas de bordure fenêtre OS
        self.tw.wm_geometry(f"+{x}+{y}")
        self.tw.wm_attributes("-topmost", True)  # Toujours au-dessus
        self.tw.lift()                            # Force le z-order
        
        # Design "Dark Mode" propre
        label = Message(self.tw, text=self.text, justify='left',
                       background="#1e1e1e", # Fond très sombre
                       fg="#dde3e8",         # Texte gris clair lisible
                       relief='solid', borderwidth=1,
                       font=("Segoe UI Emoji", 10),
                       width=320) # Largeur max pour le wrap du texte
        label.pack(ipadx=8, ipady=5)

    def hide_tip(self):
        if self.tw:
            self.tw.destroy()
            self.tw = None