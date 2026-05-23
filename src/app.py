import customtkinter as ctk
import os
import sys
from tkinter import messagebox

# --- IMPORTS CORE (Le Cerveau) ---
from src.core.settings import SettingsManager
from src.core.config_handler import ConfigHandler

# --- TRADUCTION ---
try:
    from src.core.translations import t, set_language, get_translator
except ImportError:
    def t(key, **kw): return key
    def set_language(lang): pass
    def get_translator(): return None
    print("⚠️ translations.py non trouvé — interface en français par défaut")

# --- IMPORTS UI (Robustes — l'app ne crashe pas si un onglet optionnel manque) ---
try:
    from src.ui.tabs.tab_wizard import WizardTab
except ImportError:
    WizardTab = None
    print("⚠️ tab_wizard.py non trouvé (optionnel)")

try:
    from src.ui.tabs.tab_config import ConfigTab
except ImportError:
    ConfigTab = None
    print("❌ tab_config.py manquant (REQUIS)")

try:
    from src.ui.tabs.tab_run import RunTab
except ImportError:
    RunTab = None
    print("❌ tab_run.py manquant (REQUIS)")

try:
    from src.ui.tabs.tab_tools import ToolsTab
except ImportError:
    ToolsTab = None
    print("⚠️ tab_tools.py non trouvé (optionnel)")

try:
    from src.ui.tabs.tab_settings import SettingsTab
except ImportError:
    SettingsTab = None
    print("❌ tab_settings.py manquant (REQUIS)")


class App(ctk.CTk):
    def __init__(self):
        # 1. INITIALISATION DES DONNÉES AVANT L'INTERFACE
        self.settings = SettingsManager()
        self.config_handler = ConfigHandler()
        
        # 1b. LANGUE
        lang = self.settings.get("language", "fr")
        set_language(lang)
        
        # 2. CONFIGURATION DE L'APPARENCE
        mode = self.settings.get("appearance_mode", "System")
        ctk.set_appearance_mode(mode)
        
        theme_name = self.settings.get("theme_color", "green")
        standard_themes = ["blue", "green", "dark-blue"]

        if theme_name in standard_themes:
            ctk.set_default_color_theme(theme_name)
        else:
            theme_path = os.path.join(os.getcwd(), "assets", "themes", f"{theme_name}.json")
            if os.path.exists(theme_path):
                try:
                    ctk.set_default_color_theme(theme_path)
                    print(f"[App] Thème chargé : {theme_name}")
                except Exception as e:
                    print(f"[App] Thème invalide ({e}), fallback green")
                    ctk.set_default_color_theme("green")
            else:
                ctk.set_default_color_theme("green")

        # 3. CRÉATION DE LA FENÊTRE PRINCIPALE
        super().__init__()
        self.title(t("app_title"))
        self.geometry("1600x900")
        
        icon_path = os.path.join("assets", "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        # 4. GRILLE PRINCIPALE
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self,
                                        segmented_button_fg_color="#252535")
        self.tab_view.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # 5. CRÉATION CONDITIONNELLE DES ONGLETS
        created_tabs = []

        self.wizard_tab = None
        self.config_tab = None
        self.train_tab = None
        self.tools_tab = None
        self.settings_tab = None

        # Wizard (optionnel — icône smiley souriant)
        if WizardTab is not None:
            try:
                tab_name = "😊 Assistant"
                self.tab_view.add(tab_name)
                self.wizard_tab = WizardTab(self.tab_view.tab(tab_name))
                self.wizard_tab.pack(fill="both", expand=True)
                created_tabs.append(tab_name)
                print("[App] ✅ Onglet Wizard créé")
            except Exception as e:
                print(f"[App] ⚠️ Erreur Wizard: {e}")
                import traceback; traceback.print_exc()

        # Config (essentiel)
        if ConfigTab is not None:
            try:
                tab_name = f"📝 {t('tab_config')}"
                self.tab_view.add(tab_name)
                self.config_tab = ConfigTab(
                    self.tab_view.tab(tab_name), self.config_handler
                )
                self.config_tab.pack(fill="both", expand=True)
                created_tabs.append(tab_name)
                print("[App] ✅ Onglet Config créé")
            except Exception as e:
                print(f"[App] ❌ Erreur Config: {e}")
                import traceback; traceback.print_exc()

        # Run (essentiel)
        if RunTab is not None:
            try:
                tab_name = f"🚀 {t('tab_training')}"
                self.tab_view.add(tab_name)
                self.train_tab = RunTab(self.tab_view.tab(tab_name))
                self.train_tab.pack(fill="both", expand=True)
                created_tabs.append(tab_name)
                self._run_tab_name = tab_name
                print("[App] ✅ Onglet Run créé")
            except Exception as e:
                print(f"[App] ❌ Erreur Run: {e}")
                import traceback; traceback.print_exc()

        # Tools (optionnel)
        if ToolsTab is not None:
            try:
                tab_name = f"🔧 {t('tab_tools')}"
                self.tab_view.add(tab_name)
                self.tools_tab = ToolsTab(self.tab_view.tab(tab_name))
                self.tools_tab.pack(fill="both", expand=True)
                created_tabs.append(tab_name)
                print("[App] ✅ Onglet Tools créé")
            except Exception as e:
                print(f"[App] ⚠️ Erreur Tools: {e}")

        # Queue (entre Entrainement et Outils)
        try:
            from src.ui.tabs.tab_queue import QueueTab
            tab_name = "📋 File d'attente"
            self.tab_view.add(tab_name)
            self.queue_tab = QueueTab(self.tab_view.tab(tab_name))
            self.queue_tab.pack(fill="both", expand=True)
            created_tabs.append(tab_name)
            print("[App] ✅ Onglet Queue créé")
        except Exception as e:
            print(f"[App] ⚠️ Erreur Queue: {e}")

        # Settings (essentiel)
        if SettingsTab is not None:
            try:
                tab_name = f"⚙️ {t('tab_settings')}"
                self.tab_view.add(tab_name)
                self.settings_tab = SettingsTab(self.tab_view.tab(tab_name))
                self.settings_tab.pack(fill="both", expand=True)
                created_tabs.append(tab_name)
                print("[App] ✅ Onglet Settings créé")
            except Exception as e:
                print(f"[App] ❌ Erreur Settings: {e}")

        # Distributed Training (optionnel)
        try:
            from src.ui.tabs.tab_distributed import DistributedTab
            tab_name = "🌐 Partage"
            self.tab_view.add(tab_name)
            self.distributed_tab = DistributedTab(self.tab_view.tab(tab_name))
            self.distributed_tab.pack(fill="both", expand=True)
            created_tabs.append(tab_name)
            print("[App] ✅ Onglet Distribué créé")
        except Exception as e:
            print(f"[App] ⚠️ Erreur Distribué: {e}")

        # 5b. FONT EMOJI sur chaque bouton de l'onglet (après création de tous les onglets)
        try:
            _seg = self.tab_view._segmented_button
            _seg.configure(font=("Segoe UI Emoji", 13))
            # Appliquer aussi sur chaque bouton individuel déjà créé
            for _btn in getattr(_seg, "_buttons_dict", {}).values():
                try:
                    _btn.configure(font=("Segoe UI Emoji", 13))
                except Exception:
                    pass
        except Exception:
            pass

        # 6. CALLBACKS (CONNEXIONS ENTRE ONGLETS)
        if self.config_tab and self.train_tab:
            if hasattr(self.config_tab, 'set_run_callback'):
                self.config_tab.set_run_callback(self.switch_to_run_and_start)
            self.config_tab.run_tab_ref = self.train_tab

        # 7. ONGLET PAR DÉFAUT
        if created_tabs:
            self.tab_view.set(created_tabs[0])
            print(f"[App] Onglet par défaut : {created_tabs[0]}")
            print(f"[App] {len(created_tabs)} onglets créés avec succès")
        else:
            messagebox.showerror(
                "Erreur Critique",
                "Aucun onglet n'a pu être créé. L'application ne peut pas démarrer."
            )

    def switch_to_run_and_start(self, config_path):
        """Change d'onglet et lance l'entraînement."""
        if hasattr(self, '_run_tab_name'):
            self.tab_view.set(self._run_tab_name)
        else:
            self.tab_view.set("🚀 Entraînement")
        if self.train_tab:
            self.train_tab.external_start(config_path)

    def _cleanup_on_exit(self):
        """Stop background services (gallery server, training process) before quitting."""
        try:
            if self.train_tab and hasattr(self.train_tab, 'runner'):
                runner = self.train_tab.runner
                if runner and runner.process and runner.is_running:
                    try:
                        runner.process.kill()
                    except Exception:
                        pass
                runner.kill_monitoring_tools()
        except Exception:
            pass
        try:
            from src.core.gallery_server import get_server
            srv = get_server()
            if srv.is_running():
                srv.stop()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app._cleanup_on_exit)
    app.mainloop()
