"""
Système de traduction FR/EN pour Universal SR Studio
Permet de changer dynamiquement la langue de l'interface
"""

# Langue par défaut
DEFAULT_LANGUAGE = "fr"

# Toutes les traductions
TRANSLATIONS = {
    # ===== GÉNÉRAL =====
    "app_title": {
        "fr": "Universal SR Studio V2.5.6 - Interface NeoSR & TraiNNer",
        "en": "Universal SR Studio V2.5.6 - NeoSR & TraiNNer Interface"
    },
    
    # ===== ONGLETS =====
    "tab_guide": {"fr": "Guide", "en": "Guide"},
    "tab_config": {"fr": "Configuration", "en": "Configuration"},
    "tab_training": {"fr": "Entraînement", "en": "Training"},
    "tab_tools": {"fr": "Outils", "en": "Tools"},
    "tab_settings": {"fr": "Paramètres", "en": "Settings"},
    "tab_analyzer": {"fr": "📊 Analyzer", "en": "📊 Analyzer"},
    "tab_queue": {"fr": "📋 Queue", "en": "📋 Queue"},
    

    # ===== QUEUE =====
    "queue_title": {"fr": "File d'Attente d'Entrainements", "en": "Training Queue"},
    "queue_add": {"fr": "Ajouter Config", "en": "Add Config"},
    "queue_add_folder": {"fr": "Dossier", "en": "Folder"},
    "queue_remove": {"fr": "Retirer", "en": "Remove"},
    "queue_clear": {"fr": "Vider", "en": "Clear"},
    "queue_start": {"fr": "LANCER LA QUEUE", "en": "START QUEUE"},
    "queue_stop": {"fr": "STOP QUEUE", "en": "STOP QUEUE"},
    "queue_empty": {"fr": "Aucune configuration dans la queue.", "en": "No configurations in queue."},
    "queue_pending": {"fr": "en attente", "en": "pending"},
    "queue_running": {"fr": "en cours", "en": "running"},
    "queue_done": {"fr": "termine", "en": "done"},
    "queue_error": {"fr": "erreur", "en": "error"},

    # ===== NOTIFICATIONS =====
    "notif_title": {"fr": "Notifications & Son", "en": "Notifications & Sound"},
    "notif_sound_enabled": {"fr": "Jouer un son quand un entrainement se termine", "en": "Play sound when training finishes"},
    "notif_volume": {"fr": "Volume", "en": "Volume"},
    "notif_test": {"fr": "Tester le son", "en": "Test sound"},

    # ===== DISTRIBUTED =====
    "dist_title": {"fr": "Entrainement Partage", "en": "Shared Training"},
    "dist_discover": {"fr": "Rechercher Slaves", "en": "Discover Slaves"},
    "dist_benchmark": {"fr": "Benchmark", "en": "Benchmark"},
    "dist_sync": {"fr": "Test Sync", "en": "Sync Test"},
    "dist_launch": {"fr": "Lancer Entrainement Partage", "en": "Launch Shared Training"},
    "dist_empty": {"fr": "Aucun slave detecte.", "en": "No slaves detected."},

    # ===== TRAINING TAB =====
    "train_validation": {"fr": "Validation", "en": "Validation"},
    "train_metrics": {"fr": "Metriques", "en": "Metrics"},
    "train_tensorboard": {"fr": "TBoard", "en": "TBoard"},
    "train_lr_schedule": {"fr": "LR", "en": "LR"},
    "train_auto_resume": {"fr": "Auto-resume (reprendre si crash)", "en": "Auto-resume (resume on crash)"},
    "train_start": {"fr": "DEMARRER", "en": "START"},
    "train_stop": {"fr": "STOP & SAVE", "en": "STOP & SAVE"},
    "train_shutdown": {"fr": "Eteindre PC a la fin", "en": "Shutdown PC when done"},
    "train_progress": {"fr": "Progression", "en": "Progress"},
    "train_finished": {"fr": "Processus termine.", "en": "Process finished."},

    # ===== CONFIG TAB =====
    "config_ai_check": {"fr": "Verification de Configuration par IA", "en": "AI Configuration Check"},
    "config_ai_provider": {"fr": "Fournisseur AI", "en": "AI Provider"},
    "config_ai_key": {"fr": "Cle API", "en": "API Key"},
    "config_ai_model": {"fr": "Modele", "en": "Model"},
    "config_ai_analyze": {"fr": "Analyser ma Configuration", "en": "Analyze my Configuration"},
    "config_context": {"fr": "Contexte supplementaire", "en": "Additional context"},
    "config_dataset_type": {"fr": "Type de dataset", "en": "Dataset type"},
    "config_goal": {"fr": "But du modele", "en": "Model goal"},
    "config_time": {"fr": "Temps alloue", "en": "Time allocated"},
    "config_ds_size": {"fr": "Taille dataset", "en": "Dataset size"},
    "config_sysinfo": {"fr": "Inclure infos systeme", "en": "Include system info"},

    # ===== TOOLS =====
    "tools_comparator": {"fr": "Comparateur", "en": "Comparator"},
    "tools_upscale": {"fr": "Quick Upscale", "en": "Quick Upscale"},
    "tools_dataset": {"fr": "Generateur Dataset", "en": "Dataset Generator"},
    "tools_metrics": {"fr": "Metriques", "en": "Metrics"},

    # ===== ABOUT =====
    "about_title": {"fr": "Universal SR Studio", "en": "Universal SR Studio"},
    "about_credits": {"fr": "Auteurs & Credits", "en": "Authors & Credits"},
    "about_engines": {"fr": "Moteurs SR", "en": "SR Engines"},
    "about_license": {"fr": "Licence MIT", "en": "MIT License"},

    # ===== SETTINGS =====
    "settings_engines": {"fr": "Moteurs IA", "en": "AI Engines"},
    "settings_system": {"fr": "Systeme & Dependances", "en": "System & Dependencies"},
    "settings_language": {"fr": "Langue", "en": "Language"},
    "settings_apikeys": {"fr": "Cles API", "en": "API Keys"},
    "settings_notifications": {"fr": "Notifications", "en": "Notifications"},
    "settings_about": {"fr": "A Propos", "en": "About"},

    # ===== BOUTONS COMMUNS =====
    "btn_save": {"fr": "💾 Sauvegarder", "en": "💾 Save"},
    "btn_load": {"fr": "📂 Charger", "en": "📂 Load"},
    "btn_cancel": {"fr": "❌ Annuler", "en": "❌ Cancel"},
    "btn_start": {"fr": "▶️ Démarrer", "en": "▶️ Start"},
    "btn_stop": {"fr": "⏹️ Arrêter", "en": "⏹️ Stop"},
    "btn_pause": {"fr": "⏸️ Pause", "en": "⏸️ Pause"},
    "btn_resume": {"fr": "▶️ Reprendre", "en": "▶️ Resume"},
    "btn_browse": {"fr": "📁 Parcourir", "en": "📁 Browse"},
    "btn_export": {"fr": "📤 Exporter", "en": "📤 Export"},
    "btn_import": {"fr": "📥 Importer", "en": "📥 Import"},
    "btn_delete": {"fr": "🗑️ Supprimer", "en": "🗑️ Delete"},
    "btn_refresh": {"fr": "🔄 Actualiser", "en": "🔄 Refresh"},
    
    # ===== TAB CONFIG =====
    "config_title": {"fr": "Générateur de Configuration", "en": "Configuration Generator"},
    "config_engine": {"fr": "Moteur IA :", "en": "AI Engine:"},
    "config_template": {"fr": "📋 Charger un Template", "en": "📋 Load Template"},
    "config_general": {"fr": "Général", "en": "General"},
    "config_architecture": {"fr": "Architecture", "en": "Architecture"},
    "config_datasets": {"fr": "Datasets", "en": "Datasets"},
    "config_training": {"fr": "Entraînement", "en": "Training"},
    "config_losses": {"fr": "Fonctions de Perte", "en": "Loss Functions"},
    "config_system": {"fr": "Système / Avancé", "en": "System / Advanced"},
    
    # Champs généraux
    "field_name": {"fr": "Nom d'expérience", "en": "Experiment Name"},
    "field_scale": {"fr": "Facteur d'échelle", "en": "Scale Factor"},
    "field_architecture": {"fr": "Architecture", "en": "Architecture"},
    
    # Datasets
    "field_train_hq": {"fr": "Train HQ (Ground Truth)", "en": "Train HQ (Ground Truth)"},
    "field_train_lq": {"fr": "Train LQ (Optionnel)", "en": "Train LQ (Optional)"},
    "field_val_hq": {"fr": "Validation HQ", "en": "Validation HQ"},
    "field_val_lq": {"fr": "Validation LQ", "en": "Validation LQ"},
    
    # Training
    "field_batch_size": {"fr": "Batch Size", "en": "Batch Size"},
    "field_patch_size": {"fr": "Patch Size", "en": "Patch Size"},
    "field_total_iter": {"fr": "Total Itérations", "en": "Total Iterations"},
    "field_learning_rate": {"fr": "Learning Rate", "en": "Learning Rate"},
    "field_optimizer": {"fr": "Optimiseur", "en": "Optimizer"},
    "field_scheduler": {"fr": "Scheduler", "en": "Scheduler"},
    
    # Validation
    "validation_error": {"fr": "Erreurs de Validation", "en": "Validation Errors"},
    "validation_fix": {"fr": "Veuillez corriger :", "en": "Please fix:"},
    "validation_required": {"fr": "est requis", "en": "is required"},
    "validation_must_be": {"fr": "doit être", "en": "must be"},
    "validation_between": {"fr": "entre", "en": "between"},
    "validation_and": {"fr": "et", "en": "and"},
    
    # Messages
    "msg_saved": {"fr": "Configuration sauvegardée !", "en": "Configuration saved!"},
    "msg_loaded": {"fr": "Configuration chargée !", "en": "Configuration loaded!"},
    "msg_error": {"fr": "Erreur", "en": "Error"},
    "msg_success": {"fr": "Succès", "en": "Success"},
    "msg_warning": {"fr": "Attention", "en": "Warning"},
    
    # ===== TAB SETTINGS =====
    "settings_title": {"fr": "Paramètres de l'Application", "en": "Application Settings"},
    "settings_appearance": {"fr": "Apparence", "en": "Appearance"},
    "settings_language": {"fr": "Langue", "en": "Language"},
    "settings_theme": {"fr": "Thème", "en": "Theme"},
    "settings_paths": {"fr": "Chemins", "en": "Paths"},
    "settings_neosr_path": {"fr": "Chemin NeoSR", "en": "NeoSR Path"},
    "settings_redux_path": {"fr": "Chemin TraiNNer-Redux", "en": "TraiNNer-Redux Path"},
    
    # Appearance
    "appearance_mode": {"fr": "Mode d'apparence", "en": "Appearance Mode"},
    "appearance_system": {"fr": "Système", "en": "System"},
    "appearance_dark": {"fr": "Sombre", "en": "Dark"},
    "appearance_light": {"fr": "Clair", "en": "Light"},
    
    # Language
    "lang_french": {"fr": "🇫🇷 Français", "en": "🇫🇷 French"},
    "lang_english": {"fr": "🇬🇧 English", "en": "🇬🇧 English"},
    "lang_changed": {"fr": "Langue changée ! Redémarrez l'application.", "en": "Language changed! Restart the application."},
    
    # ===== TAB RUN =====
    "run_title": {"fr": "Console d'Entraînement", "en": "Training Console"},
    "run_status": {"fr": "Statut :", "en": "Status:"},
    "run_idle": {"fr": "En attente", "en": "Idle"},
    "run_running": {"fr": "En cours", "en": "Running"},
    "run_completed": {"fr": "Terminé", "en": "Completed"},
    "run_error": {"fr": "Erreur", "en": "Error"},
    "run_select_config": {"fr": "Sélectionner une config", "en": "Select a config"},
    "run_clear_logs": {"fr": "🗑️ Effacer les logs", "en": "🗑️ Clear logs"},
    
    # ===== TAB TOOLS =====
    "tools_title": {"fr": "Boîte à Outils", "en": "Toolbox"},
    "tools_lmdb": {"fr": "Créateur LMDB", "en": "LMDB Creator"},
    "tools_upscale": {"fr": "Upscale Rapide", "en": "Quick Upscale"},
    "tools_benchmark": {"fr": "Benchmark", "en": "Benchmark"},
    
    # ===== TAB ANALYZER =====
    "analyzer_title": {"fr": "Analyseur de Dataset", "en": "Dataset Analyzer"},
    "analyzer_select": {"fr": "Sélectionner un dossier", "en": "Select a folder"},
    "analyzer_analyze": {"fr": "🔍 Analyser", "en": "🔍 Analyze"},
    "analyzer_export": {"fr": "📄 Exporter HTML", "en": "📄 Export HTML"},
    "analyzer_stats": {"fr": "Statistiques", "en": "Statistics"},
    "analyzer_total": {"fr": "Total images", "en": "Total images"},
    "analyzer_formats": {"fr": "Formats", "en": "Formats"},
    "analyzer_resolutions": {"fr": "Résolutions", "en": "Resolutions"},
    "analyzer_duplicates": {"fr": "Doublons détectés", "en": "Duplicates detected"},
    
    # ===== TAB QUEUE =====
    "queue_title": {"fr": "File d'Attente d'Entraînements", "en": "Training Queue"},
    "queue_add": {"fr": "➕ Ajouter Config", "en": "➕ Add Config"},
    "queue_start": {"fr": "▶️ Démarrer la Queue", "en": "▶️ Start Queue"},
    "queue_clear": {"fr": "🗑️ Vider", "en": "🗑️ Clear"},
    "queue_save": {"fr": "💾 Sauvegarder Queue", "en": "💾 Save Queue"},
    "queue_load": {"fr": "📂 Charger Queue", "en": "📂 Load Queue"},
    "queue_waiting": {"fr": "En attente", "en": "Waiting"},
    "queue_completed": {"fr": "Terminé", "en": "Completed"},
    "queue_failed": {"fr": "Échoué", "en": "Failed"},
    
    # ===== TAB GUIDE =====
    "guide_title": {"fr": "Documentation & Aide", "en": "Documentation & Help"},
    "guide_welcome": {"fr": "Bienvenue", "en": "Welcome"},
    "guide_quickstart": {"fr": "Démarrage Rapide", "en": "Quick Start"},
    "guide_faq": {"fr": "FAQ", "en": "FAQ"},
    
    # ===== ARCHITECTURES =====
    "arch_span": {"fr": "SPAN - Léger, rapide (6-8GB)", "en": "SPAN - Lightweight, fast (6-8GB)"},
    "arch_realplksr": {"fr": "RealPLKSR - Photo réaliste (8-11GB)", "en": "RealPLKSR - Photo realistic (8-11GB)"},
    "arch_hat": {"fr": "HAT - Haute qualité (10-16GB)", "en": "HAT - High quality (10-16GB)"},
    "arch_dat": {"fr": "DAT - État de l'art (16-24GB)", "en": "DAT - State of the art (16-24GB)"},
    
    # ===== TEMPLATES =====
    "template_anime": {"fr": "🎬 Anime 4x (Léger)", "en": "🎬 Anime 4x (Light)"},
    "template_photo": {"fr": "📸 Photo Realistic 4x", "en": "📸 Photo Realistic 4x"},
    "template_texture": {"fr": "🎨 Texture/Detail 4x", "en": "🎨 Texture/Detail 4x"},
    "template_fast": {"fr": "⚡ Fast Test", "en": "⚡ Fast Test"},
    "template_highend": {"fr": "🚀 High-End 4x", "en": "🚀 High-End 4x"},
    
    # ===== LOSSES =====
    "loss_pixel": {"fr": "Pixel Loss (L1)", "en": "Pixel Loss (L1)"},
    "loss_percep": {"fr": "Perceptual (VGG)", "en": "Perceptual (VGG)"},
    "loss_gan": {"fr": "GAN Loss", "en": "GAN Loss"},
    "loss_dists": {"fr": "DISTS", "en": "DISTS"},
    "loss_ldl": {"fr": "LDL", "en": "LDL"},
    
    # ===== TOOLTIPS =====
    "tip_batch_size": {
        "fr": "Nombre d'images traitées simultanément.\nPlus élevé = plus rapide mais plus de VRAM.",
        "en": "Number of images processed simultaneously.\nHigher = faster but more VRAM."
    },
    "tip_patch_size": {
        "fr": "Taille des patchs extraits des images.\n64-128 recommandé.",
        "en": "Size of patches extracted from images.\n64-128 recommended."
    },
    "tip_learning_rate": {
        "fr": "Vitesse d'apprentissage.\n5e-5 à 1e-4 recommandé pour débuter.",
        "en": "Learning speed.\n5e-5 to 1e-4 recommended to start."
    },
    
    # ===== FORMATS D'EXPORT =====
    "export_pth": {"fr": "PyTorch (.pth)", "en": "PyTorch (.pth)"},
    "export_safetensors": {"fr": "SafeTensors (.safetensors)", "en": "SafeTensors (.safetensors)"},
    "export_onnx": {"fr": "ONNX (.onnx)", "en": "ONNX (.onnx)"},
    "export_torchscript": {"fr": "TorchScript (.pt)", "en": "TorchScript (.pt)"},
    
    # ===== ERREURS COURANTES =====
    "error_file_not_found": {
        "fr": "Fichier introuvable",
        "en": "File not found"
    },
    "error_invalid_config": {
        "fr": "Configuration invalide",
        "en": "Invalid configuration"
    },
    "error_gpu_memory": {
        "fr": "Mémoire GPU insuffisante",
        "en": "Insufficient GPU memory"
    },
    "error_dataset_empty": {
        "fr": "Dataset vide ou introuvable",
        "en": "Dataset empty or not found"
    },
}


class Translator:
    """Gestionnaire de traductions avec cache"""
    
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        """
        Initialise le traducteur.
        
        Args:
            language: Code langue ('fr' ou 'en')
        """
        self.language = language if language in ["fr", "en"] else DEFAULT_LANGUAGE
        self._cache = {}
    
    def set_language(self, language: str):
        """
        Change la langue active.
        
        Args:
            language: Code langue ('fr' ou 'en')
        """
        if language in ["fr", "en"]:
            self.language = language
            self._cache.clear()  # Clear cache
    
    def get(self, key: str, **kwargs) -> str:
        """
        Récupère une traduction.
        
        Args:
            key: Clé de traduction
            **kwargs: Arguments pour formatage (ex: {count})
            
        Returns:
            str: Texte traduit
        """
        # Cache lookup
        cache_key = f"{key}_{self.language}"
        if cache_key in self._cache and not kwargs:
            return self._cache[cache_key]
        
        # Get translation
        if key in TRANSLATIONS:
            text = TRANSLATIONS[key].get(self.language, TRANSLATIONS[key].get("en", key))
        else:
            # Fallback: return key if not found
            text = key
            print(f"[Translator] Warning: Missing translation for '{key}'")
        
        # Format with kwargs if provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                print(f"[Translator] Format error for '{key}': {e}")
        
        # Cache result
        if not kwargs:
            self._cache[cache_key] = text
        
        return text
    
    def __call__(self, key: str, **kwargs) -> str:
        """Alias pour get()"""
        return self.get(key, **kwargs)


# Instance globale
_translator = Translator()


def get_translator() -> Translator:
    """Retourne l'instance globale du traducteur"""
    return _translator


def t(key: str, **kwargs) -> str:
    """
    Fonction raccourci pour traduire.
    
    Args:
        key: Clé de traduction
        **kwargs: Arguments de formatage
        
    Returns:
        str: Texte traduit
        
    Example:
        >>> t("btn_save")
        "💾 Sauvegarder"
        >>> t("validation_between", min=1, max=64)
        "entre 1 et 64"
    """
    return _translator.get(key, **kwargs)


def set_language(language: str):
    """
    Change la langue globale.
    
    Args:
        language: 'fr' ou 'en'
    """
    _translator.set_language(language)


def get_available_languages() -> dict:
    """
    Retourne les langues disponibles.
    
    Returns:
        dict: {"fr": "🇫🇷 Français", "en": "🇬🇧 English"}
    """
    return {
        "fr": "🇫🇷 Français",
        "en": "🇬🇧 English"
    }
