import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class DatasetTools:
    def __init__(self):
        pass

    def log(self, message, callback=None):
        if callback:
            callback(message)
        else:
            print(message)

    def check_integrity(self, folder_path, callback=None):
        """Vérifie si les images s'ouvrent correctement"""
        if not os.path.exists(folder_path):
            self.log("[ERREUR] Le dossier n'existe pas.", callback)
            return

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        total = len(files)
        corrupted = []
        
        self.log(f"--- Début vérification de {total} images ---", callback)

        for i, file in enumerate(files):
            full_path = os.path.join(folder_path, file)
            try:
                with Image.open(full_path) as img:
                    img.verify() # Vérification rapide
            except Exception as e:
                corrupted.append(file)
                self.log(f"[CORROMPU] {file} : {e}", callback)

        if not corrupted:
            self.log(f"[OK] Aucune image corrompue détectée sur {total} fichiers.", callback)
        else:
            self.log(f"[ATTENTION] {len(corrupted)} images corrompues trouvées !", callback)

    def resize_images(self, input_folder, output_folder, max_size, callback=None):
        """Redimensionne les images pour qu'elles rentrent dans max_size x max_size"""
        if not os.path.exists(input_folder):
            self.log("[ERREUR] Dossier source introuvable.", callback)
            return
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            self.log(f"[INFO] Dossier destination créé : {output_folder}", callback)

        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        total = len(files)
        
        self.log(f"--- Début Redimensionnement (Max: {max_size}px) ---", callback)

        # Fonction interne pour le multithreading
        def process_one(filename):
            try:
                in_path = os.path.join(input_folder, filename)
                out_path = os.path.join(output_folder, filename)
                
                with Image.open(in_path) as img:
                    # Convertir en RGB si nécessaire (pour éviter crash PNG transparent vers JPG)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Calcul ratio
                    w, h = img.size
                    if w > max_size or h > max_size:
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        img.save(out_path, quality=95)
                        return f"[RESIZE] {filename} -> {img.size}"
                    else:
                        # Si déjà plus petit, on copie juste (ou on resave)
                        img.save(out_path, quality=95)
                        return f"[COPY] {filename} (Déjà ok)"
            except Exception as e:
                return f"[ERREUR] {filename} : {e}"

        # Utilisation de ThreadPool pour aller plus vite (IO bound)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(process_one, files):
                # On pourrait logger chaque fichier mais ça spamme trop l'UI
                pass 
        
        self.log(f"[TERMINÉ] Traitement fini. Images sauvegardées dans {output_folder}", callback)