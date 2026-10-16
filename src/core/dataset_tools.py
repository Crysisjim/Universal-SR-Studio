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

    _IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')

    @staticmethod
    def _t(fr, en):
        try:
            from qt_ui.i18n import tr
            return tr(fr, en)
        except Exception:
            return fr

    def _list_images(self, folder_path, collisions=None):
        """Recursive image listing → {relative_path_no_ext: full_path}. Two files with the
        same stem (a.png + a.jpg) used to hide one another silently: they are reported in
        `collisions` (the engines pair HR/LQ by stem, so this is a real dataset problem)."""
        out = {}
        for root, _dirs, filenames in os.walk(folder_path):
            for f in sorted(filenames):
                if f.lower().endswith(self._IMG_EXTS):
                    rel = os.path.relpath(os.path.join(root, f), folder_path)
                    stem = os.path.splitext(rel)[0].replace("\\", "/")
                    if stem in out and collisions is not None:
                        collisions.append(f"{os.path.basename(out[stem])} / {f}")
                    out[stem] = os.path.join(root, f)
        return out

    @staticmethod
    def _decode(path):
        """Full decode (verify() alone does not read the pixel data: a truncated file
        passed). Returns the size."""
        with Image.open(path) as img:
            img.load()
            return img.size

    def check_integrity(self, folder_path, callback=None, lq_folder_path=None):
        """Checks that every image decodes (recursive), and with lq_folder_path that every
        HR image has its LQ (same name, any extension) with an integer HR/LQ ratio."""
        t = self._t
        if not os.path.exists(folder_path):
            self.log(t("[ERREUR] Le dossier n'existe pas.", "[ERROR] The folder does not exist."), callback)
            return
        coll = []
        images = self._list_images(folder_path, coll)
        total = len(images)
        if total == 0:
            self.log(t("[ERREUR] Aucune image prise en charge (png, jpg, webp, bmp, tif) dans ce dossier.",
                       "[ERROR] No supported image (png, jpg, webp, bmp, tif) in this folder."), callback)
            return
        self.log(t(f"--- Vérification de {total} images (sous-dossiers inclus, décodage complet) ---",
                   f"--- Checking {total} images (subfolders included, full decode) ---"), callback)
        if coll:
            self.log(t(f"[NOMS EN DOUBLE] {len(coll)} fichier(s) au même nom avec une autre extension "
                       f"(un seul est utilisé) : {', '.join(coll[:20])}",
                       f"[DUPLICATE NAMES] {len(coll)} file(s) with the same name and another extension "
                       f"(only one is used): {', '.join(coll[:20])}"), callback)
        corrupted, dims = [], {}
        for stem, full_path in images.items():
            try:
                dims[stem] = self._decode(full_path)
            except Exception as e:
                corrupted.append(stem)
                self.log(t(f"[CORROMPU] {stem} : {e}", f"[CORRUPTED] {stem}: {e}"), callback)
        if not corrupted:
            self.log(t(f"[OK] Aucune image corrompue sur {total} fichiers.",
                       f"[OK] No corrupted image among {total} files."), callback)
        else:
            self.log(t(f"[ATTENTION] {len(corrupted)} image(s) corrompue(s) !",
                       f"[WARNING] {len(corrupted)} corrupted image(s)!"), callback)

        if not lq_folder_path:
            return
        if not os.path.exists(lq_folder_path):
            self.log(t("[ERREUR] Le dossier LQ n'existe pas.", "[ERROR] The LQ folder does not exist."), callback)
            return
        lq_coll = []
        lq_images = self._list_images(lq_folder_path, lq_coll)
        if lq_coll:
            self.log(t(f"[NOMS EN DOUBLE LQ] {', '.join(lq_coll[:20])}", f"[DUPLICATE LQ NAMES] {', '.join(lq_coll[:20])}"), callback)
        missing_lq = sorted(set(images) - set(lq_images))
        missing_hr = sorted(set(lq_images) - set(images))
        self.log(t(f"--- Vérification des paires HR/LQ ({len(images)} HR, {len(lq_images)} LQ) ---",
                   f"--- Checking HR/LQ pairs ({len(images)} HR, {len(lq_images)} LQ) ---"), callback)
        if missing_lq:
            self.log(t(f"[PAIRE MANQUANTE] {len(missing_lq)} HR sans LQ : ", f"[MISSING PAIR] {len(missing_lq)} HR without LQ: ")
                     + f"{', '.join(missing_lq[:20])}{' …' if len(missing_lq) > 20 else ''}", callback)
        if missing_hr:
            self.log(t(f"[PAIRE MANQUANTE] {len(missing_hr)} LQ sans HR : ", f"[MISSING PAIR] {len(missing_hr)} LQ without HR: ")
                     + f"{', '.join(missing_hr[:20])}{' …' if len(missing_hr) > 20 else ''}", callback)
        bad_ratio, lq_bad = [], []
        for stem in sorted(set(images) & set(lq_images)):
            try:
                lw, lh = self._decode(lq_images[stem])
            except Exception as e:
                lq_bad.append(stem)
                self.log(t(f"[LQ CORROMPU] {stem} : {e}", f"[LQ CORRUPTED] {stem}: {e}"), callback)
                continue
            if stem in corrupted:
                continue
            hw, hh = dims.get(stem, (0, 0))
            if lw <= 0 or lh <= 0 or hw % lw != 0 or hh % lh != 0 or (hw // lw) != (hh // lh):
                bad_ratio.append(f"{stem} (HR {hw}x{hh} / LQ {lw}x{lh})")
        if bad_ratio:
            self.log(t(f"[RÉSOLUTION INCOHÉRENTE] {len(bad_ratio)} paire(s) sans ratio HR/LQ entier : ",
                       f"[RESOLUTION MISMATCH] {len(bad_ratio)} pair(s) without an integer HR/LQ ratio: ")
                     + f"{', '.join(bad_ratio[:20])}{' …' if len(bad_ratio) > 20 else ''}", callback)
        if not (missing_lq or missing_hr or bad_ratio or lq_bad or corrupted or coll or lq_coll):
            self.log(t("[OK] Toutes les paires HR/LQ sont présentes, lisibles et cohérentes en résolution.",
                       "[OK] Every HR/LQ pair is present, readable and resolution-consistent."), callback)

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


def val_scale_mismatch(gt_dirs, lq_dirs, scale, sample=8):
    """Checks that validation LQ images are their GT divided by `scale`.

    A paired validation set whose LQ has the GT size (a 1x restoration set) is
    accepted silently by the engine: the x`scale` output gets resized back to the
    GT size before PSNR/SSIM, so the metrics (and the automatic GAN start) no
    longer measure the upscale. Returns up to `sample` mismatches as
    (name, (gt_w, gt_h), (lq_w, lq_h)); empty when the set is fine, scale <= 1,
    or the folders cannot be read."""
    try:
        scale = int(scale)
    except (TypeError, ValueError):
        return []
    if scale <= 1:
        return []
    gt_dirs = [gt_dirs] if isinstance(gt_dirs, str) else list(gt_dirs or [])
    lq_dirs = [lq_dirs] if isinstance(lq_dirs, str) else list(lq_dirs or [])
    tools = DatasetTools()
    gts, lqs = {}, {}
    for d in gt_dirs:
        if d and os.path.isdir(d):
            gts.update(tools._list_images(d))
    for d in lq_dirs:
        if d and os.path.isdir(d):
            lqs.update(tools._list_images(d))
    bad = []
    for stem in sorted(set(gts) & set(lqs))[:sample]:
        try:
            with Image.open(gts[stem]) as g, Image.open(lqs[stem]) as l:
                gw, gh = g.size
                lw, lh = l.size
        except Exception:
            continue
        if abs(lw * scale - gw) > scale or abs(lh * scale - gh) > scale:
            bad.append((os.path.basename(gts[stem]), (gw, gh), (lw, lh)))
    return bad
