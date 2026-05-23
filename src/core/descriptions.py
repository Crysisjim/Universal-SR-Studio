# --- LISTE DES PARAMÈTRES EXCLUSIFS À TRAINNER-REDUX ---
REDUX_ONLY_FIELDS = [
    "use_compile", "compile_mode", "fast_matmul", "amp_bf16", "adaptive_d", 
    "ema_decay", "ema_switch_iter", "use_moa", "moa_debug", "lq_usm", 
    "thicklines_prob", "format_input", "format_output"
]

# --- GROUPEMENT PAR FAMILLES ---
ARCH_FAMILIES = {
    "✨ Recommended (NeoSR)": ["omnisr", "swinir", "hat", "dat", "span"],
    "🚀 Lightweight / Fast": ["span", "span_s", "spanplus", "compact", "ultracompact", "realcugan", "safmn", "lmlt", "plksr", "realplksr"],
    "🤖 Transformers (Heavy)": ["hat", "hat_l", "hat_m", "hat_s", "dat", "dat_2", "dat_light", "dat_s", "swinir", "swinir_l", "swinir_m", "swinir_s", "srformer", "srformerv2", "drct", "drct_l", "drct_xl"],
    "🎨 GAN / Restoration": ["esrgan", "esrgan_lite", "rcan", "rcan_l", "rcan_unshuffle", "artcnn_r16f96", "artcnn_r8f48", "artcnn_r3f24", "cugan"],
    "🎞️ Video / Temporal": ["tscunet", "temporalspan", "temporalspanv2"],
    "📦 Others": ["atd", "atd_light", "cfsr", "craft", "dct", "dctlsa", "ditn", "ditn_real", "eimn", "eimn_a", "eimn_l", "elan", "elan_light", "emt", "esc", "escrealm", "escrealm_xl", "fdat", "fdat_light", "fdat_xl", "flexnet", "metaflexnet", "gaterv3", "grl_b", "grl_s", "grl_t", "hasn", "hit_sir", "hit_sng", "hit_srf", "hma", "krgn", "lkfmixer_b", "lkfmixer_l", "lkfmixer_t", "man", "man_light", "man_tiny", "metagan3", "moesr", "moesr2", "mosr", "mosr_t", "mosrv2", "msdan", "plainusr", "rgt", "rgt_s", "rtmosr", "rtmosr_l", "rtmosr_ul", "scunet_aaf6aa", "sebica", "sebica_mini", "seemore_t", "swin2sr_l", "swin2sr_m", "swin2sr_s"]
}

# --- INFOBULLES (TOOLTIPS) ---
TOOLTIPS = {
    # ================= GÉNÉRAL =================
    "name": "Nom de l'expérience.\nCrée un dossier dans 'experiments/'.\nConseil : Utilisez un préfixe comme '4x_MonModele'.",
    "engine": "Moteur d'entraînement.\n- NeoSR : Recommandé (Optimisé, moderne).\n- TraiNNer-Redux : Pour compatibilité ancienne.",
    "scale": "Facteur d'agrandissement.\nDoit correspondre exactement à la différence de taille entre vos dossiers LQ et HQ.",
    "use_gan": "Active le mode GAN (Adversarial).\n[+] Avantage : Textures réalistes, détails fins.\n[-] Risque : Hallucinations, artefacts, instabilité.",
    "manual_seed": "Graine aléatoire (Seed).\nFixe le hasard pour avoir des résultats reproductibles.\n- 10 : Standard.\n- Random : Change à chaque fois.",
    
    # ================= DATASETS =================
    "dataroot_gt": "Dossier HQ (Ground Truth).\nImages de référence parfaite (PNG/JPG).",
    "dataroot_lq": "Dossier LQ (Low Quality).\nSi vide, le logiciel générera le LQ à la volée (OTF) mais c'est plus lent.",
    "val_gt": "Dossier HQ pour la validation.\nUtilisé pour calculer le PSNR/SSIM pendant l'entraînement.",
    "val_lq": "Dossier LQ pour la validation.\nDoit correspondre aux fichiers du dossier Val HQ.",
    "val_freq": "Fréquence de validation (toutes les N itérations).\n[+] Plus haut (10000+) : Entraînement plus rapide (moins d'interruptions).\n[-] Plus bas (1000-5000) : Suivi fin des courbes PSNR/SSIM et détection précoce des problèmes.\n[Conseil] 5000 pour un suivi normal, 1000 pour déboguer.",
    "tile": "Taille de découpe (Tiling) pour la validation.\nImportant pour éviter les erreurs 'Out of Memory' sur les grandes images.\n- 0 : Image entière.\n- 200 : Recommandé.",
    "resume_state": "Fichier .state pour reprendre un entraînement.\nUtile après un crash ou pour continuer un modèle.",
    "pretrain_model": "Fichier .pth pour le Transfer Learning.\nCommencer avec un modèle déjà entraîné accélère grandement les résultats.",
    "dataset_mode": "Mode de Dataset.\n- OTF : Génère les défauts (bruit, flou) à la volée. Recommandé pour généraliser.\n- Paired : Dossiers HQ et LQ statiques déjà préparés.",

    # --- AUGMENTATIONS ---
    "aug_mixup": "MixUp : Mélange deux images par transparence.\nAide le réseau à comprendre les transitions douces.",
    "aug_cutmix": "CutMix : Colle un carré d'une image A sur une image B.\nForce le réseau à regarder l'image entière, pas juste des zones faciles.",
    "aug_resizemix": "ResizeMix : Redimensionne une image et la colle dans une autre.\nVariante plus stable de CutMix.",
    "aug_cutblur": "CutBlur : Colle une zone Basse Qualité (LQ) sur la Haute Qualité (HQ).\nApprend au réseau à gérer les zones mixtes net/flou. Très puissant.",



    # ================= HYPERPARAMÈTRES (ENTRAÎNEMENT) =================
    "batch_size": "Nombre d'images traitées simultanément par le GPU.\n[+] Plus haut : Entraînement plus stable, plus rapide.\n[-] Plus bas : Moins de VRAM requise, mais convergence plus chaotique.\n(1080Ti : 4 à 8 recommandé).",
    "patch_size": "Taille des morceaux d'images (Crops) vus par le réseau.\n[+] Plus haut (96, 128) : Meilleure cohérence globale, apprends les structures larges.\n[-] Plus bas (32, 48, 64) : Économise la VRAM, focalisé sur les textures locales.\n[!] Doit être multiple de la Window Size.",
    "total_iter": "Durée de vie de l'entraînement.\n- Finetuning : 50k - 150k.\n- From Scratch : 300k - 500k+.",
    "warmup_iter": "Itérations de chauffe.\nMonte progressivement le Learning Rate de 0 à la valeur cible.\n- -1 : Désactivé (Standard).\n- 5000 : Recommandé pour stabiliser les gros modèles.",
    "pixel_reduction": "Méthode de réduction de la Loss.\n- Mean : Moyenne des erreurs (Standard, Stable).\n- Sum : Somme des erreurs (Plus agressif, gradients plus forts).",
    "pixel_criterion": "Type de fonction de perte pixel.\n\n--- NeoSR ---\n* L1Loss : Erreur absolue. Donne des images légèrement floues mais stables.\n* MSELoss (L2) : Erreur quadratique. Encore plus lissé que L1. Peut créer des artefacts de saturation.\n* HuberLoss : Hybride L1/L2 (robuste aux outliers). Recommandé pour NeoSR.\n* chc : Clipped Huber + Cosine Similarity. Meilleure cohérence des couleurs. NeoSR uniquement.\n\n--- TraiNNer-Redux ---\n* charbonnierloss : L1 lissé autour de zéro. Recommandé Redux (défaut).\n* l1loss : Erreur absolue standard.\n* mseloss : Erreur quadratique standard.\n\n[Recommandation] NeoSR : HuberLoss ou chc. Redux : charbonnierloss.",
    "accumulate": "Gradient Accumulation.\nPermet de simuler un gros Batch Size.\nEx: Batch 4 + Accumulate 4 = Batch Effectif 16 (Très stable).",
    "match_lq_colors": "Correction colorimétrique.\nForce l'histograme du LQ à correspondre au GT avant l'entraînement.\nCorrige les sources aux couleurs délavées.",
    
    "optim_g": "Algorithme d'optimisation (Le cerveau de l'apprentissage).\n\n--- NeoSR ---\n* Adam / AdamW : Standard, robuste. lr=5e-4, betas=[0.9, 0.99].\n* NAdam : Adam + Nesterov momentum. Convergence plus rapide.\n* Adan : State-of-the-Art. 3 betas, lr=5e-4. Un peu plus lourd VRAM mais excellent.\n* AdamW_Win : Variante Winograd, mode 'win2'. Expérimental.\n* AdamW_SF / Adan_SF : Schedule-Free ! Pas besoin de scheduler.\n  [!] Mettre schedule_free=true OBLIGATOIREMENT.\n* SOAP_SF : Préconditionné (Gap-Aware). lr=1e-3 recommandé.\n\n--- TraiNNer-Redux ---\n* Adam / AdamW / NAdam / RAdam / SGD : Optimiseurs PyTorch standard.\n* Adadelta / Adagrad : Adaptatifs, taux d'apprentissage par paramètre. Rarement utilisés.\n\n[+] Plus haut le lr → convergence rapide mais instable.\n[-] Plus bas le lr → stable mais lent.",
    "scheduler": "Stratégie de réduction du Learning Rate.\n\n--- NeoSR ---\n* MultiStepLR : Baisse le LR par paliers (milestones). Classique.\n* CosineAnnealing : Descente en cosinus vers eta_min. Excellent pour le finetuning.\n* CosineAnnealingRestart : Cosinus + remontées périodiques.\n* CyclicLR / OneCycleLR : Oscillation entre bornes.\n[!] Inutile avec les optimiseurs _SF (Schedule-Free).\n\n--- TraiNNer-Redux ---\n* MultiStepLR / StepLR : Paliers fixes.\n* CosineAnnealingLR : Descente cosinus.\n* ExponentialLR : Décroissance exponentielle (gamma^epoch).\n* ReduceLROnPlateau : Baisse le LR quand la loss stagne.",
    "lr": "Vitesse d'apprentissage (Learning Rate).\n[+] Plus haut (1e-3, 5e-4) : Convergence rapide, risque de divergence (Loss qui explose).\n[-] Plus bas (1e-4, 5e-5) : Apprend lentement mais plus précis pour le finetuning.\n\nValeurs recommandées :\n- From Scratch : 5e-4\n- Finetuning depuis un pretrain : 1e-4 à 5e-5\n- Phase GAN (après PSNR) : 1e-4",
    "save_freq": "Fréquence de sauvegarde automatique (.pth).\nEx: 5000 = Sauvegarde toutes les 5000 itérations.\n[+] Plus fréquent : Reprises plus granulaires, moins de perte en cas de crash.\n[-] Plus fréquent : Disque plus sollicité, dossier experiments plus lourd.",
    "save_img": "Sauvegarde des images de validation sur le disque.\nPermet de voir visuellement les progrès dans le dossier 'visualization'.\n[+] True : Pratique pour auditer la qualité visuellement.\n[-] True : Peut générer beaucoup de fichiers (~100KB par image × freq).",
    "milestones": "Pour MultiStepLR : Itérations où le LR est divisé par gamma.\nEx: '75000, 112500' pour un entraînement de 150k.\nRègle générale : 50% et 75% du total_iter.",

    # ================= SYSTÈME / AVANCÉ =================
    "use_amp": "Automatic Mixed Precision.\n\n--- NeoSR ---\n[!] False OBLIGATOIRE sur GTX 1080 Ti (Pascal) — FP16 instable/crashs.\n[+] FP16 accélère sur RTX 2000/3000/4000.\n\n--- TraiNNer-Redux ---\n[+] AMP FP16 FONCTIONNE sur Pascal (sm_61) + PyTorch 2.7 (testé ✅ 9.4 it/s).\n[+] BF16 donne +30% boost sur RTX 3000+ (Tensor Cores).\n[+] BF16 fonctionne aussi sur Pascal sans compile (mode bf16_nocl).\n[!] Différence NeoSR vs Redux : Redux gère mieux l'AMP FP16 sur les vieilles cartes.",
    "bfloat16": "Format BF16 (Brain Float).\n[+] True : Meilleure stabilité que FP16.\n[+] RTX 3070 Ti Laptop bench : ultracompact 7.15 it/s (fp16) → 9.33 it/s (bf16) = +30% via Tensor Cores.\n[-] Ne fonctionne QUE sur RTX 3000/4000 (Ampere+) en mode natif.\n[i] Sur Pascal : bf16_nocl (sans compile) fonctionne à vitesse normale.",
    "grad_clip": "Gradient Clipping.\nCoupe les valeurs extrêmes pour éviter les erreurs NaN (Not a Number).\nIndispensable pour les GANs instables.",
    "deterministic": (
        "Mode Déterministe (torch.use_deterministic_algorithms).\n\n"
        "[+] Reproductibilité exacte : même seed → même résultat à chaque run.\n"
        "[-] Ralentit l'entraînement de 5 à 20% (certaines ops CUDA n'ont pas d'impl. déterministe).\n"
        "[-] Warnings 'does not have a deterministic implementation' dans les logs (warn_only=True).\n\n"
        "--- TraiNNer-Redux ---\n"
        "Contrôlé par le champ 'deterministic: true/false' dans le YAML.\n\n"
        "--- NeoSR ---\n"
        "Activé automatiquement quand manual_seed est défini dans le TOML.\n"
        "Pour désactiver : retirer manual_seed du fichier option.\n\n"
        "⚠️ Ne pas activer en production — réservé aux tests de reproductibilité."
    ),
    "ema": "Exponential Moving Average.\nGarde une version 'lissée' du modèle en parallèle.\nDonne souvent de meilleurs résultats finaux.",
    "use_tb_logger": "Active les logs TensorBoard.\nPermet de voir les courbes de Loss, PSNR et les images de validation.",
    "auto_tensorboard": "Lance automatiquement TensorBoard au démarrage de l'entraînement.\nOuvre http://localhost:6006 dans votre navigateur pour voir les courbes en temps réel.\n[+] Monitoring passif sans quitter l'application.\n[-] Consomme un peu de RAM (~100 MB).",
    "auto_ngrok": "Lance un tunnel Ngrok public pour TensorBoard.\nPermet de voir les graphiques depuis n'importe quel appareil (téléphone, tablette).\nNécessite que Ngrok soit installé et configuré (ngrok config add-authtoken...).\n[!] L'URL change à chaque démarrage.",
    "port_tb": "Port pour le serveur TensorBoard (Défaut : 6006).\nChangez-le si le port est déjà occupé par un autre service.",
    "port_ngrok": "Port local ciblé par Ngrok (Doit correspondre au port TB).",
    "num_gpu": "Nombre de cartes graphiques utilisées.\n- 1 : Standard.\n- auto : Tente d'utiliser toutes les cartes disponibles (Expérimental).",
    "fast_matmul": "Active la précision TF32 (Tensor Float 32) via torch.backends.cuda.matmul.fp32_precision.\n[+] Accélère l'entraînement sur RTX 3000/4000 (ampere+).\n[-] Perte infime de précision (invisible en SR).\n\n⚠️ INCOMPATIBLE — Pascal (GTX 1080 Ti, sm_61) + PyTorch 2.7 :\n   AttributeError: Unknown attribute fp32_precision\n   → Désactivé automatiquement sur votre GPU.\n⚠️ Crash aussi sur RTX 3070 Ti Laptop + PyTorch 2.7 (bug non résolu).",
    "compile": "Torch Compile (torch.compile).\n[+] Compilation JIT du modèle — gain de vitesse réel à l'inférence.\n[-] Requiert Triton — absent sur Windows natif.\n\n⚠️ Sur Windows : torch.compile échoue systématiquement :\n   'torch.compile requires triton'\n   → Désactivé automatiquement sur Windows.\n[i] Fonctionne sur Linux / WSL2 avec PyTorch + Triton installé.\n[i] Temps de démarrage +30-60s au premier run (compilation CUDA).",

    # ================= OPTIMISATION (TRAIN) =================
    "sam": "Sharpness-Aware Minimization (SAM).\nCherche les zones 'plates' de la Loss pour une généralisation parfaite.\n[+] Résultats souvent meilleurs sur le set de test.\n[-] Entraînement 2x plus lent (fait 2 calculs par itération).",
    "sam_init": "Itération à partir de laquelle activer SAM.\nConseil : Activer après 50% de l'entraînement.",
    "eco": "Efficient Computing Optimization.\n[+] Réduit la consommation VRAM/Calcul périodiquement.\n[-] Peut légèrement perturber la convergence finale.",
    "eco_init": "Itération de début pour le mode ECO.",
    "schedule_free": "Mode Schedule-Free (Adan/AdamW).\nL'optimiseur gère lui-même le Learning Rate.\n[+] Plus besoin de régler le Scheduler/Milestones.\n[-] Moins de contrôle sur la fin de l'entraînement.",
    "warmup_steps": "Pas de chauffe interne à l'optimiseur.\nDifférent du 'Warmup Iter' global. Laisser à -1 sauf expert.",
    
    # ================= DISCRIMINATEUR & GAN =================
    "net_d_type": "Architecture du Juge (Discriminateur).\n* UNet : Standard, équilibré.\n* PatchGAN : Focalisé sur la texture fine.\n* MetaGAN : Très puissant, lourd.\n* EA2FPN : Avancé, bonne détection des bords.\n* DUNet : UNet dense.",
    "gan_loss_weight": "Force du GAN.\n[+] Plus haut : Plus de détails, risque d'artefacts.\n[-] Plus bas : Plus stable, risque d'être trop flou.\nStandard : 0.05",
    "real_label_val": "Valeur cible pour les images réelles (1.0).\nParfois réduit à 0.9 (Label Smoothing) pour stabiliser.",
    "fake_label_val": "Valeur cible pour les images générées (0.0).",
    "gan_type": "Type de Loss Adversariale.\n\n--- NeoSR (gan_opt) ---\n* BCE : Binary Cross-Entropy. Standard, efficace, le plus utilisé.\n* MSE : Mean Squared Error. Plus stable que BCE, moins net.\n* Huber : Hybride L1/MSE. Robuste aux outliers.\n\n--- TraiNNer-Redux (ganloss) ---\n* vanilla : Équivalent à BCE. Standard.\n* LSGAN : Least Squares. Plus stable que vanilla.\n* Hinge : Géométrique. Donne des contours nets. Utilisé par StyleGAN.\n* WGAN : Wasserstein. Très stable (pas de crash), convergence lente.\n\n[!] Les types ne sont PAS interchangeables entre moteurs.",
    "lr_d": "Learning Rate du Discriminateur.\nConseil : Mettre une valeur plus faible que le Générateur (ex: 5e-5 vs 1e-4) pour éviter que le Juge ne domine trop.",

    # ================= CHARGEMENT =================
    "prefetch_mode": "Préchargement des données.\n* Cuda : Rapide, prend de la VRAM.\n* CPU : Lent, économise la VRAM.",
    "num_worker": "Threads CPU pour préparer les images.\n1080 Ti : Mettre entre 2 et 4.\nTrop haut = Surcharge CPU.",

    # ================= LOSSES (FONCTIONS DE PERTE) =================
    "loss_pixel": "Pixel Loss (L1 / L2 / Huber / CHC).\nLa base de tout training. Force l'image à être mathématiquement proche de la cible pixel par pixel.\n\n--- NeoSR ---\nTypes : L1Loss, MSELoss (L2), HuberLoss, chc (Clipped Huber + Cosine Similarity).\nchc améliore la cohérence des couleurs et réduit le bruit.\n\n--- Redux ---\nTypes : l1loss, mseloss, charbonnierloss (Charbonnier = L1 lissé, recommandé).\n\n[+] Augmenter le poids → image plus fidèle mathématiquement.\n[-] Trop de poids → image floue (perd les textures fines).",
    "loss_percep": "Perceptual Loss (VGG19).\nUtilise un réseau VGG pré-entraîné pour comparer les 'caractéristiques' visuelles.\nCrée la netteté structurelle et les textures réalistes.\n\n--- NeoSR ---\nOptions : criterion (l1/l2/huber/chc), patchloss (Patch Loss pour le focus local), ipk (Image Patch Kernel).\nlayer_weights : conv1_2=0.1, conv3_4=1.0, conv4_4=1.0, conv5_4=1.0.\n\n--- Redux ---\nType : perceptualloss. Inclut Focal Distribution (num_proj_fd) + FP16 variant.\ncriterion : charbonnier (défaut), l1.\n\n[+] Augmenter → textures plus nettes, détails fins.\n[-] Trop de poids → artefacts, hallucinations.\n[!] Consomme +1-2 GB VRAM (charge VGG19).",
    "percep_criterion": "Méthode de comparaison pour VGG Perceptual.\n- L1 : Strict, très net. Risque d'artefacts en damier.\n- L2/MSE : Plus doux, moins d'artefacts.\n- Huber : Hybride L1/L2, robuste aux outliers. Recommandé NeoSR.\n- Charbonnier : Comme L1 mais lissé. Recommandé Redux.\n- CHC : Clipped Huber + Cosine Similarity (NeoSR uniquement).",
    "percep_layer": "Couche VGG utilisée pour extraire les features.\n* conv1_2 (0.1) : Très local — bruit, grain, pixels.\n* conv2_2 (0.1) : Textures fines — lignes, bords.\n* conv3_4 (1.0) : Formes moyennes — standard.\n* conv4_4 (1.0) : Textures complexes — recommandé SR.\n* conv5_4 (1.0) : Sémantique/abstrait — recommandé GAN.\n\nPoids plus haut = cette couche influence plus le résultat.",
    "fdl_model": "Modèle pour la FDL (Frequency Distribution Loss).\n- vgg : VGG19, classique, bon pour la structure.\n- dinov2 : Transformer Facebook. Comprend mieux les textures et le sémantique.\n- resnet : ResNet101, alternative.\n- effnet : EfficientNet v1.\n\nnum_proj : nombre de projections (24 par défaut, 256 dans le papier original).\n[+] Plus de projections → meilleure qualité perceptuelle.\n[-] Plus lent (heavy hit sur la performance).",

    "loss_wavelet": "Wavelet Guided Loss (WGSR).\nSépare les hautes et basses fréquences via ondelettes.\nStabilise énormément les GANs en guidant chaque composante séparément.\n\n[!] Meilleur en finetuning (activer après ~40K iters via wavelet_init).\n[!] NeoSR uniquement.",
    "weight_loss_wavelet": "Poids de la Wavelet Loss.",
    "wavelet_init": "Délai d'activation de la Wavelet Loss (itérations).\nEx: 80000 = s'active seulement après 80K iters.\nRecommandé : entraîner au moins 40K avant d'activer.",
    "loss_fdl": "Frequency Distribution Loss (FDL).\nLoss perceptuelle basée sur la distribution fréquentielle.\nBacks : DINOv2, VGG19, ResNet101, EfficientNet.\n\n[+] Excellent pour restaurer le grain et les textures complexes.\n[+] Complémentaire à la Perceptual Loss.\n[-] Lourd en calcul (num_proj=24 vs 256 original).\n[!] NeoSR uniquement.",
    "loss_ldl": "LDL Loss (Local Discriminative Learning).\nPénalise les artefacts dans les zones à haute fréquence (détails).\ncriterion : l1, l2, huber. ksize : taille du kernel (7 par défaut).\n\n[+] Préserve les détails fins sans flouter.\n[+] Bon complément à L1.\n[!] Disponible NeoSR et Redux.",
    "loss_consistency": "Consistency Loss.\nForce la cohérence des couleurs et de la luminosité entre sortie et cible.\nUtilise les espaces Oklab et CIE L* + Cosine Similarity.\n\nOptions : blur (lissage), cosim (similarité cosinus), saturation/brightness.\nmatch_lq_colors : matcher les couleurs du LQ au lieu du GT.\n\n[!] NeoSR uniquement.",
    "loss_edge": "Edge Loss (Gradient-Weighted, GW Loss).\nForce le réseau à soigner les contours et les hautes fréquences.\ncriterion : l1, l2, huber, chc. corner : activer la détection de coins.\n\n[+] Lignes plus nettes, transitions plus propres.\n[!] NeoSR uniquement.",
    "loss_mssim": "MS-SSIM Loss (Multi-Scale SSIM).\nMesure la similarité structurelle à plusieurs échelles.\n\n--- NeoSR ---\nOptions : window_size=11, sigma=1.5, K1=0.01, K2=0.03.\n\n--- Redux ---\nType : mssimloss. Options : channels=3, downsample=false, is_prod=true, color_space=yiq.\nVariante sssiml1 disponible (combine SSIM + L1).\n\n[+] Meilleur que L1 pour la structure perçue.\n[-] Peut lisser légèrement les détails très fins.",
    "loss_dists": "DISTS Loss.\nMesure la distance texture/structure via VGG16.\n\n[+] Excellente tolérance aux textures (grain, herbe) contrairement à LPIPS.\n[+] Peut être utilisé seul comme perceptual loss.\n[-] Consomme de la VRAM (+VGG16).\n[!] Disponible NeoSR (dists_loss) et Redux (distsloss).",
    "loss_msswd": "Multiscale Sliced Wasserstein Distance.\nLoss de cohérence couleur basée sur la distance de Wasserstein.\nnum_scale=3, num_proj=24 (128 dans le papier).\n\n[+] Idéal pour les textures aléatoires (herbe, eau, asphalte).\n[+] Complémentaire à Consistency Loss.\n[-] Lourd en calcul.\n[!] NeoSR uniquement.",
    "loss_ff": "Focal Frequency Loss (FFL).\nForce le réseau à générer les fréquences manquantes dans le spectre.\nalpha=1.0, patch_factor=1, ave_spectrum=true.\n\n[+] Récupère les hautes fréquences (détails durs).\n[-] Peut causer des instabilités sans pretrain.\n[!] Disponible NeoSR (ff_loss) et Redux (ffloss).",
    "loss_ncc": "NCC Loss (Normalized Cross-Correlation).\nMesure la corrélation normalisée entre sortie et cible.\n\n[+] Robuste aux changements de contraste/luminosité.\n[!] NeoSR uniquement.",
    "loss_kl": "KL Loss (Kullback-Leibler Divergence).\nMesure la divergence statistique entre distributions.\n\n[!] Activer UNIQUEMENT avec un pretrain. Depuis zéro → NaN/résultats incorrects.\n[!] NeoSR uniquement.",
    "loss_gan": "GAN Loss.\nLe discriminateur force le générateur à produire des images réalistes.\n\n--- NeoSR ---\nTypes : bce (défaut), mse, huber.\n\n--- Redux ---\nType : ganloss. gan_type : vanilla (défaut).\nmultiscaleganloss : multi-échelle pour de meilleurs détails.\n\n[+] Textures nettes et réalistes.\n[-] +30% VRAM, training instable, risque d'artefacts.\nPoids recommandé : 0.1–0.3.",

    # Redux-specific losses
    "loss_hsluv": "HSLuv Loss (Redux uniquement).\nMesure la différence de couleur en espace HSLuv (perceptuellement uniforme).\nhue_weight=0.33, saturation_weight=0.33, lightness_weight=0.33.\n\n[+] Meilleure reproduction des couleurs que L1/L2.\n[+] Poids séparés pour teinte/saturation/luminosité.",
    "loss_cosim": "Cosine Similarity Loss (Redux uniquement).\nMesure l'angle entre vecteurs de pixels.\ncosim_lambda=5.\n\n[+] Bon pour la cohérence couleur globale.",
    "loss_color": "Color Loss (Redux uniquement).\nPénalise les décalages de couleur entre sortie et cible.\ncriterion : l1.\n\n[+] Empêche les dérives chromatiques.",
    "loss_gv": "Gradient Variance Loss (Redux uniquement).\nEncourage des gradients lisses dans l'image.\npatch_size=16, criterion=charbonnier.\n\n[+] Réduit les artefacts de bord.\n[+] Bon complément à Perceptual Loss.",
    "loss_contextual": "Contextual Loss (Redux uniquement).\nCompare les patches locaux via VGG19.\ndistance_type=cosine, band_width=0.5.\n\n[+] Tolère les petits décalages spatiaux.\n[-] Très lourd en calcul.",
    "loss_luma": "Luma Loss (Redux uniquement).\nPénalise les erreurs de luminance uniquement.\ncriterion : l1.\n\n[+] Utile si les couleurs sont bonnes mais la luminosité dévie.",

    # ================= METRICS =================
    "metric_psnr": "Peak Signal-to-Noise Ratio.\nMesure mathématique de la fidélité pixel par pixel.\n[+] Standard universel, rapide à calculer.\n[-] Ne voit pas le flou (une image floue peut avoir un bon PSNR).\n[-] Ne corrèle pas toujours avec la qualité perçue.",
    "metric_ssim": "Structural Similarity.\n[+] Mesure la structure, le contraste et la luminance.\n[+] Plus proche de la perception humaine que le PSNR.\n[-] Reste limité pour les textures fines.",
    "metric_dists": "DISTS Metric.\n[+] Mesure la qualité texture (grain, détails, réalisme).\n[+] Plus proche du jugement humain que PSNR/SSIM.\n[-] Plus lent à calculer.",
    "metric_lpips": "LPIPS (Perceptual).\nMesure la distance visuelle (Plus bas = Mieux). <0.10 est excellent.",
    "metric_niqe": "NIQE (Naturalness).\nScore 'No-Reference'. Évalue si l'image fait 'naturelle' sans la comparer.",

    # ================= DEGRADATIONS (OTF) =================
    "deg_level": "Profil de dégradation (Preset).\nSélectionner un niveau ajustera automatiquement les sliders ci-dessous.\n- Light : Nettoyage léger.\n- Medium : Standard.\n- Heavy : Restauration extrême.",
    "deg_shuffle_prob": "Probabilité de mélanger l'ordre des dégradations (Blur/Resize/Noise).\nPermet de couvrir plus de cas réels.",
    "final_sinc_prob": "Filtre Sinc (Ringing/Gibbs).\nSimule les échos autour des lignes (typique des vieux animes/DVD).\n[+] Indispensable pour nettoyer les vieux encodages.",
    
    # --- STAGE 1 ---
    "resize_prob": "Probabilités [Up, Down, Keep].\nEx: [0.2, 0.7, 0.1] = 20% Upscale, 70% Downscale, 10% Taille originale.",
    "resize_range": "Plage de redimensionnement [Min, Max].\nEx: [0.3, 1.5] = L'image peut être réduite à 30% ou agrandie à 150%.",
    "gaussian_noise_prob": "Probabilité d'appliquer un bruit Gaussien (Grain standard).",
    "noise_range": "Intensité du bruit [Min, Max] (Sigma).\nEx: [0, 15] = De propre à très bruité.",
    "poisson_scale_range": "Bruit de Poisson (Shot Noise).\nSimule le bruit des capteurs numériques (dépend de la luminosité).",
    "gray_noise_prob": "Probabilité que le bruit soit en noir & blanc (au lieu de couleur RGB).",
    "blur_prob": "Probabilité d'appliquer du flou.",
    "blur_kernel_size": "Taille physique du noyau de flou (Impair).\n[+] Grand (21) : Flou très large/doux.\n[-] Petit (7) : Flou léger/net.",
    "blur_sigma": "Écart-type du flou [Min, Max].\nContrôle l'intensité réelle du flou.",
    "kernel_list": "Types de flous possibles (Iso, Aniso, Plateau...).\nPlus la liste est longue, plus le modèle généralise.",
    "kernel_prob": "Probabilités associées à la liste ci-dessus.",
    "betag_range": "Forme du flou (Generalized Gaussian).\nContrôle si le flou est pointu ou plat.",
    "betap_range": "Forme du flou (Plateau).\nContrôle la largeur du plateau central du noyau.",
    "sinc_prob": "Probabilité d'appliquer un noyau Sinc (Ringing) dans l'étape de flou.",

    # --- STAGE 2 ---
    "second_blur_prob": "Probabilité d'activer la 2ème étape de dégradation.\nSimule une image déjà compressée qui est réencodée.",
    "resize_prob2": "Probabilités Resize (Passe 2).",
    "resize_range2": "Plage Resize (Passe 2).",
    "blur_kernel_size2": "Taille Noyau Flou (Passe 2).",
    "blur_sigma2": "Intensité Flou (Passe 2).",
    "compression_prob2": "Probabilité Compression (Passe 2).",
    "gaussian_noise_prob2": "Probabilité Bruit (Passe 2).",
    "noise_range2": "Intensité Bruit (Passe 2).",
    "poisson_scale_range2": "Bruit Poisson (Passe 2).",
    "gray_noise_prob2": "Bruit Gris (Passe 2).",
    "sinc_prob2": "Probabilité Sinc (Passe 2).",
    "betag_range2": "Beta G (Passe 2).",
    "betap_range2": "Beta P (Passe 2).",

    # --- COMPRESSION STAGE 1 ---
    "compression_prob": "Probabilité d'appliquer une compression (JPEG ou WebP) à la passe 1.\n[+] Simule la dégradation par compression de flux vidéo.",
    "compression_range": "Plage de qualité de compression [Min, Max].\nEx: [30, 95]. 30 = très compressé (blocs), 95 = propre.",

    # --- FINAL ---
    "jpeg_prob": "Probabilité d'appliquer une compression JPEG finale.",
    "jpeg_range": "Plage de qualité JPEG [Min, Max].\nEx: [30, 95]. 30 est très compressé (blocs), 95 est propre.",
    "jpeg_range2": "Plage pour le double JPEG (Simulation d'enregistrements successifs).",

    # --- QUANTIFICATION / BANDING (OTF Custom) ---
    "banding_prob": "Probabilité d'appliquer un effet de banding (gradients quantifiés).\n\nLe banding simule les sources anciennes (DVD, vieux encodages, screencaps).\nEn mode OTF, est injecté après les dégradations principales.\n\n[+] Améliore la robustesse aux sources avec bandes de couleur.\n[+] Indispensable pour les datasets anime / vidéo ancienne.\n[-] Ralentit légèrement le chargement des données.",
    "banding_levels_range": "Plage du nombre de niveaux de quantification pour le banding [Min, Max].\nEx: [16, 64].\n- Faible (8-16) : Banding très visible, comme les vieux GIF.\n- Moyen (32-64) : Banding subtil, typique d'un encodage H.264 ancien.\n- Haut (128+) : Effet très léger, presque imperceptible.",
    "posterize_prob": "Probabilité d'appliquer un effet de posterisation (réduction des niveaux par canal).\n\nLa posterisation est plus agressive que le banding — réduit les niveaux dans chaque canal R/G/B indépendamment.\n\n[+] Simule des sources avec une profondeur de couleur réduite.\n[+] Complémentaire au banding pour couvrir plus de défauts réels.\n[-] Peut créer des artefacts visuels forts si les bits sont trop bas.",
    "posterize_bits_range": "Plage du nombre de bits pour la posterisation [Min, Max].\nEx: [3, 6].\n- 2-3 bits : Poster 'cartoon' très visible (4-8 couleurs par canal).\n- 4-5 bits : Dégradation modérée, simule les encodages lossy.\n- 6-7 bits : Effet très subtil.",

    # --- OPTIQUE / ANALOGIQUE (OTF Custom) ---
    "chroma_prob": "Probabilité d'appliquer un sous-échantillonnage chromatique 4:2:0.\n\nSimule la compression vidéo YCbCr (JPEG, MPEG, DVD). Les contours colorés sont floutés horizontalement et verticalement.\n\n[+] Très réaliste pour les sources vidéo/screencaps.\n[+] Rapide — aucun paramètre à régler.\n[-] Aucun effet sur les images en niveaux de gris.",
    "ca_prob": "Probabilité d'appliquer une aberration chromatique.\n\nDécale les canaux R et B dans des directions opposées (horizontal), simulant un objectif de mauvaise qualité ou une ancienne caméra.\n\n[+] Ajoute un défaut optique réaliste.\n[+] Complémentaire à d'autres dégradations pour données CRT/VHS.\n[-] Peut créer des franges colorées visibles à fort décalage.",
    "ca_shift_range": "Plage du décalage en pixels pour l'aberration chromatique [Min, Max].\nEx: [1, 5].\n- 1-2 px : Effet subtil, quasi-imperceptible.\n- 3-5 px : Frange de couleur visible sur les bords.\n- 6+ px : Effet fort, type vieux scanner ou mauvaise optique.",
    "halation_prob": "Probabilité d'appliquer un effet de halation de film.\n\nLes zones très lumineuses 'saignent' une lueur chaude dans les pixels voisins, comme sur les pellicules argentiques ou les CRT.\n\n[+] Essentiel pour les datasets ciné/anime old-school.\n[+] Donne un caractère organique à l'image.\n[-] Peut saturer les hautes lumières si strength est trop fort.",
    "halation_strength_range": "Plage de l'intensité de la halation [Min, Max].\nEx: [0.05, 0.3].\n- 0.05-0.1 : Lueur très subtile, réaliste film.\n- 0.15-0.3 : Bloom visible, typique super-8 ou bobine ancienne.\n- 0.5+ : Effet stylisé, non réaliste.",
    "salt_pepper_prob": "Probabilité d'appliquer du bruit sel & poivre.\n\nPixels aléatoires blanc (sel) ou noir (poivre), typiques des capteurs analogiques dégradés, CCD anciens, ou transmissions corrompues.\n\n[+] Simule les vieux scanners et appareils photo numériques basiques.\n[+] Très différent du bruit gaussien — entraîne le modèle à ignorer les pixels isolés.\n[-] Visuellement intrusif si amount est trop élevé.",
    "salt_pepper_amount_range": "Plage de la proportion de pixels affectés [Min, Max].\nEx: [0.001, 0.05].\n- 0.001-0.005 : Bruit très discret (1 pixel sur 200-1000).\n- 0.01-0.03 : Bruit modéré, visible mais réaliste.\n- 0.05+ : Bruit fort, images très dégradées.",
    "vhs_prob": "Probabilité d'appliquer des artefacts VHS/analogiques.\n\nCombine : saignement chromatique (canal G/B décalés), dropouts de lignes horizontales simulant des têtes de lecture défaillantes.\n\n[+] Indispensable pour les datasets d'enregistrements VHS, cassettes, captures TV ancienne.\n[+] Couvre plusieurs défauts réels en un seul effet.\n[-] Peut interagir avec chroma_subsampling — les deux ensemble = effet très fort.",
    "vhs_strength_range": "Plage de l'intensité des artefacts VHS [Min, Max].\nEx: [0.1, 0.5].\n- 0.1-0.2 : Saignement léger, lignes rares.\n- 0.3-0.5 : Effets VHS clairement visibles.\n- 0.7+ : Cassette très endommagée, non réaliste.",
    "aliasing_prob": "Probabilité d'appliquer de l'aliasing sur les lignes.\n\nDownscale + upscale nearest-neighbor crée des artefacts en escalier (staircase) sur les bords diagonaux et les lignes fines — typique d'une résolution insuffisante ou d'un resize brutal.\n\n[+] Entraîne le modèle à restaurer les bords nets à partir d'artefacts scalaires.\n[+] Utile pour les sources basse résolution re-upscalées brutalement (pixel art, scans anciens, screenshots).\n[-] À forte probabilité peut masquer les structures de l'image.",
    "aliasing_scale_range": "Plage du facteur d'échelle pour l'aliasing [Min, Max].\nEx: [0.5, 0.85].\n- 0.85-0.95 : Aliasing subtil, bords légèrement pixelisés.\n- 0.65-0.80 : Artefacts escalier visibles sur les diagonales.\n- 0.5-0.6 : Aliasing fort, pixelisation marquée.",
    "interlace_weave_prob": "Probabilité d'appliquer l'entrelacement Weave.\n\nRemplace les lignes impaires par celles d'un champ décalé verticalement → crée des 'dents de peigne' sur les bords diagonaux. Artefact le plus identifiable de la vidéo entrelacée (VHS, MPEG-2 SD, DVD non-désentrelacé).\n\n[+] Indispensable pour les datasets issus de captures TV, VHS, ou DVD entrelacés.\n[+] Artefact très reconnaissable — entraîne efficacement le modèle.",
    "interlace_weave_strength_range": "Intensité de l'effet weave [Min, Max].\n1.0 = entrelacement complet, 0.5 = mélange 50/50.",
    "interlace_flicker_prob": "Probabilité d'appliquer le flicker de champ.\n\nAlterne lignes paires plus claires et lignes impaires plus sombres, simulant la variation de luminosité 50/60 Hz des téléviseurs CRT entrelacés.\n\n[+] Subtil mais réaliste — améliore la robustesse sur les sources TV PAL/NTSC.",
    "interlace_flicker_strength_range": "Intensité du flicker [Min, Max].\nEx: [0.1, 0.4]. 0.4 = ligne paire +40%, ligne impaire -40%.",
    "interlace_blend_prob": "Probabilité d'appliquer le field blending.\n\nMélange l'image avec une version décalée verticalement, simulant le ghosting entre deux champs interlacés ou un désentrelacement par moyennage (bob filter).\n\n[+] Simule le flou de mouvement typique des désentrelaceurs bas de gamme.",
    "interlace_blend_strength_range": "Intensité du blending [Min, Max].\n1.0 = mélange maximum (fantôme fort), 0.3 = léger ghosting.",
    "film_grain_prob": "Probabilité d'appliquer du grain cinéma.\n\nGrain luminance-dépendant (fort dans les tons moyens, faible dans les ombres noires et les hautes lumières) — reproduit le comportement réel de l'argentique. Peut utiliser un grain grossier (size > 1) pour les films 8mm ou 16mm.\n\n[+] Très différent du bruit gaussien — entraîne le modèle à distinguer grain et signal.",
    "film_grain_strength_range": "Intensité du grain cinéma [Min, Max].\nEx: [0.03, 0.12]. 0.12 = grain de film 16mm visible.",
    "film_grain_size_range": "Taille des grains en pixels [Min, Max].\nEx: [1, 2].\n- 1 : Grain fin (35mm, numérique).\n- 2-3 : Grain moyen (16mm).\n- 4+ : Grain très grossier (Super-8, télécine).",
    "oversharp_prob": "Probabilité d'appliquer de la sur-netteté (halos USM).\n\nSimule les halos de bord générés par un filtre Unsharp Mask trop agressif — artefact omniprésent dans les appareils photo grand public, les chaînes de compression vidéo, et les upscalers basiques.\n\n[+] Très utile pour les datasets issus de caméras consommateurs ou de vidéos YouTube re-encodées.\n[+] Force le modèle à reconnaître les halos comme des artefacts, non comme un signal.",
    "oversharp_strength_range": "Intensité des halos de sur-netteté [Min, Max].\nEx: [0.5, 2.0]. 2.0 = halos très visibles (caméra bas de gamme).",
    "scanlines_prob": "Probabilité d'appliquer des scanlines CRT.\n\nAssombrit une ligne sur N, simulant les lignes noires entre les rangées de phosphore d'un écran CRT. Artefact classique des captures de jeux retro, émulateurs, ou télévisions tubes.\n\n[+] Utile pour les datasets de jeux retro ou contenu CRT scanné.",
    "scanlines_strength_range": "Intensité de l'assombrissement des scanlines [Min, Max].\nEx: [0.2, 0.5]. 0.5 = ligne assombrie de 50%.",
    "scanlines_spacing_range": "Espacement entre les scanlines sombres en lignes [Min, Max].\nEx: [2, 4].\n- 2 : Une ligne sombre sur 2 (effet très prononcé).\n- 4 : Une ligne sombre sur 4 (subtil).",

    # ================= PARAMÈTRES ARCHITECTURES =================
    # --- COMMUNS ---
    "window_size": "Fenêtre d'attention (Transformer).\n[+] 16/32 : Voit plus large, mieux pour les motifs répétés.\n[-] 8 : Moins de VRAM, calcul plus rapide.",
    "num_feat": "Nombre de canaux internes (Neurones).\n[+] 64/128 : Modèle plus 'intelligent', plus lent.\n[-] 48 : Modèle 'Light', rapide.",
    "embed_dim": "Dimension de l'embedding (Similaire à num_feat).\nStandard : 180 (Lourd), 60 (Léger).",
    "upscale": "Facteur d'échelle (fixé par votre projet).",
    "upsampling": "Facteur d'échelle (fixé par votre projet).",
    
    # --- OMNISR & SPÉCIFIQUES ---
    # bias, pe, etc. sont gérés en interne pour éviter les doublons
    
    # --- AUTRES ARCHITECTURES ---
    "block_num": "Nombre de blocs d'attention.",
    "flash_attn": "Flash Attention.\n[+] True : Accélération massive.\n[-] False : Compatibilité maximale.",
    "squeeze_factor": "Facteur de compression interne (HAT/DRCT).\nInflue sur la complexité du calcul.",
    "compress_ratio": "Ratio de compression pour l'attention.",
    "split_size": "Découpage de l'attention (DAT/RGT).\nGère comment l'image est divisée pour le traitement.",
    "depth": "Profondeur du réseau (Nombre de couches).\n[+] Plus profond = Meilleure qualité, plus lent.",
    "depths": "Profondeur par étage (Liste).\nEx: [6, 6, 6, 6].",
    "n_blocks": "Nombre de blocs de traitement.",
    "expansion_ratio": "Facteur d'expansion du Feed-Forward Network (FFN).",
    "expansion_factor": "Facteur d'expansion du canal (Similaire à ci-dessus).",
    "use_ea": "Efficient Attention (Optimisation VRAM).",
    "use_dysample": "DySample Upscaling.\nPlus moderne que PixelShuffle, évite l'effet damier.",
    "unshuffle_mod": "PixelUnshuffle.\nRéduit la taille spatiale en augmentant les canaux.",
    "img_size": "Taille d'image interne pour l'entraînement (souvent égal à patch_size).",
    "norm": "Normalisation des couches (True/False).",
    "n_resgroups": "Groupes de résiduels (RCAN).",
    "n_resblocks": "Blocs par groupe (RCAN).",
    "reduction": "Facteur de réduction du canal (RCAN).",
    "qkv_bias": "Biais pour Query/Key/Value (Attention).",
    "drop_rate": "Taux de dropout (Oubli volontaire pour éviter le sur-apprentissage).",
    "attn_drop_rate": "Taux de dropout de l'attention.",
    "drop_path_rate": "Taux de dropout des chemins (Stochastic Depth).",
    "ape": "Absolute Positional Encoding.",
    "patch_norm": "Normalisation des patchs.",
    "mean_norm": "Normalisation par la moyenne.",
    "d8": "Utilisation de D8 (Rotation/Flip interne).",
    "num_heads": "Nombre de têtes d'attention (Multi-Head Attention).",
    "gc": "Growth Channel (DRCT).",
    "conv_scale": "Échelle de convolution.",
    "overlap_ratio": "Ratio de chevauchement des fenêtres.",
    "img_range": "Plage des valeurs de pixels (1.0 ou 255.0).",
    "resi_connection": "Type de connexion résiduelle (1conv, 3conv...).",
    "upsampler": (
        "Méthode d'upscaling pour le générateur :\n\n"
        "• dys (DySample) — Points d'échantillonnage appris dynamiquement. "
        "Meilleure qualité générale pour SPANPlus/SPAN. ✅ Recommandé.\n\n"
        "• pixelshuffle — Pixel Shuffle standard avec init ICNR. "
        "Bon équilibre qualité/vitesse.\n\n"
        "• pixelshuffledirect — Pixel Shuffle simplifié sans ICNR. "
        "Plus rapide, qualité légèrement inférieure. Défaut sur la plupart des archi.\n\n"
        "• nearest+conv — Nearest-neighbor + conv. Le plus rapide, pour edge/mobile.\n\n"
        "• conv — Conv simple. SPANPlus uniquement, scale=1 seulement (restauration/débruitage sans upscale).\n\n"
        "• ps — Alias pour pixelshuffle."
    ),
    "num_in_ch": "Nombre de canaux d'entrée. 3 = RGB (standard). 1 = niveaux de gris.",
    "num_out_ch": "Nombre de canaux de sortie. Doit correspondre à num_in_ch. Standard : 3.",
    
    # --- DISCRIMINATEURS ---
    "dims": "Dimensions des couches (Liste).\nEx: [48, 96, 192]. Contrôle la taille du réseau.",
    "blocks": "Nombre de blocs par étage (Liste).\nEx: [3, 3, 9, 3].",
    "attention": "Attention dans le Discriminateur (MetaGAN).\n[+] True : Meilleure qualité de texture.\n[-] False : Beaucoup moins de VRAM utilisée.",
    "head_dim": "Dimension de la tête d'attention.",
    "segmentation_channels": "Canaux pour la segmentation (EA2FPN).",
    "pyramid_channels": "Canaux pour la pyramide de features (EA2FPN).",
    "use_sn": "Spectral Normalization.\nStabilisatateur de GAN. Laisser sur True généralement.",
    "use_sigmoid": "Utilise une activation Sigmoid à la sortie (0-1).",
    "skip_connection": "Connexions de saut (Skip Connections) pour UNet.",
    "num_layers": "Nombre de couches (PatchGAN).",
    "act": "Fonction d'activation (ex: lrelu).",
    "category_size": "Taille de catégorie (ATD).",
    "dw_size": "Taille du noyau Depth-Wise.",
    "num_conv": "Nombre de convolutions (Compact).",
    "act_type": "Type d'activation.",
    "split_size_0": "Taille de split dimension 0.",
    "split_size_1": "Taille de split dimension 1.",
    "pro": "Mode Pro (CuGAN).",
    "nf": "Nombre de filtres.",
    "num_head": "Nombre de têtes.",
    "ITL_blocks": "Blocs ITL (DITN).",
    "SAL_blocks": "Blocs SAL (DITN).",
    "attn_type": "Type d'attention (SDPA...).",
    "hidden_rate": "Taux caché.",
    "base_win_size": "Taille de fenêtre de base.",
    "hier_win_ratios": "Ratios de fenêtre hiérarchique.",
    "interval_size": "Intervalle.",
    "n_feats": "Nombre de features.",
    "dilation": "Dilatation.",
    "res_scale": "Echelle résiduelle.",
    "channels": "Canaux.",
    "num_DFEB": "Nombre de blocs DFEB.",
    "im_feat": "Features image.",
    "attn_feat": "Features attention.",
    "kernel_size": "Taille du noyau.",
    "split_ratio": "Ratio de division.",
    "lk_type": "Type de grand noyau.",
    "c_ratio": "Ratio C.",
    "embed_dims": "Dimensions d'embedding.",
    "num_stages": "Nombre d'étapes.",
    "mlp_ratios": "Ratio MLP.",
    "num_grow_ch": "Canaux de croissance.",
    "feature_channels": "Nombre de canaux internes du réseau (largeur du modèle).\n\nPlus la valeur est élevée, plus le réseau est large : meilleure capacité de représentation, mais plus lourd en VRAM et plus lent à entraîner.\n\nValeurs courantes par architecture :\n• SPAN / SPANPlus standard : 48–52\n• SPANPlus léger (_s) : 32–48\n• CFSR / FlexNet : 64–96\n• MoESR / MoSRv2 : 48–64\n\n[↑] Augmenter → meilleure qualité, +VRAM\n[↓] Réduire → modèle plus rapide, plus léger",
    "num_modules": "Nombre de modules.",
}

# --- PARAMÈTRES DYNAMIQUES GÉNÉRATEURS (NEOSR) ---
ARCH_FIELDS = {
    "omnisr": [
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Upsampling", "key": "upsampling", "default": 4, "tip_key": "upscale"},
    ],
    "asid": [
        {"label": "Num Feat", "key": "num_feat", "default": 48, "tip_key": "num_feat"},
        {"label": "Res Num", "key": "res_num", "default": 3, "tip_key": "res_num"},
        {"label": "Block Num", "key": "block_num", "default": 1, "tip_key": "block_num"},
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "Flash Attn", "key": "flash_attn", "default": "true", "tip_key": "flash_attn"},
        {"label": "D8", "key": "d8", "default": "false", "tip_key": "d8"},
        {"label": "Bias", "key": "bias", "default": "true", "tip_key": "bias"},
        {"label": "Drop", "key": "drop", "default": 0.0, "tip_key": "drop_rate"},
    ],
    "atd": [
        {"label": "Img Size", "key": "img_size", "default": 96, "tip_key": "img_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 210, "tip_key": "embed_dim"},
        {"label": "Window Size", "key": "window_size", "default": 16, "tip_key": "window_size"},
        {"label": "Norm", "key": "norm", "default": "false", "tip_key": "norm"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "Category Size", "key": "category_size", "default": 256, "tip_key": "category_size"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
        {"label": "Patch Norm", "key": "patch_norm", "default": "true", "tip_key": "patch_norm"},
    ],
    "cfsr": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 48, "tip_key": "embed_dim"},
        {"label": "DW Size", "key": "dw_size", "default": 9, "tip_key": "dw_size"},
        {"label": "Depths", "key": "depths", "default": "[6, 6]", "tip_key": "depths"},
        {"label": "Mean Norm", "key": "mean_norm", "default": "false", "tip_key": "mean_norm"},
        {"label": "Upsampler", "key": "upsampler", "default": "pixelshuffledirect", "tip_key": "upsampler",
         "type": "combobox", "choices": ["pixelshuffledirect", "pixelshuffle", "dys", "nearest+conv"]},
    ],
    "compact": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Num Conv", "key": "num_conv", "default": 16, "tip_key": "num_conv"},
        {"label": "Act Type", "key": "act_type", "default": "prelu", "tip_key": "act_type",
         "type": "combobox", "choices": ["prelu", "relu", "leaky_relu", "gelu"]},
    ],
    "craft": [
        {"label": "Window Size", "key": "window_size", "default": 16, "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 48, "tip_key": "embed_dim"},
        {"label": "Split Size 0", "key": "split_size_0", "default": 4, "tip_key": "split_size_0"},
        {"label": "Split Size 1", "key": "split_size_1", "default": 16, "tip_key": "split_size_1"},
        {"label": "Depths", "key": "depths", "default": "[2, 2, 2, 2]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "Flash Attn", "key": "flash_attn", "default": "true", "tip_key": "flash_attn"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "cugan": [
        {"label": "Pro", "key": "pro", "default": "true", "tip_key": "pro"},
    ],
    "dat": [
        {"label": "Upscale", "key": "upscale", "default": 4, "tip_key": "upscale"},
        {"label": "Img Size", "key": "img_size", "default": 64, "tip_key": "img_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Split Size", "key": "split_size", "default": "[2, 4]", "tip_key": "split_size"},
        {"label": "Depth", "key": "depth", "default": "[2, 2, 2, 2]", "tip_key": "depths"},
        {"label": "Expansion", "key": "expansion_factor", "default": 4, "tip_key": "expansion_ratio"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
        {"label": "Drop Rate", "key": "drop_rate", "default": 0.0, "tip_key": "drop_rate"},
        {"label": "Attn Drop", "key": "attn_drop_rate", "default": 0.0, "tip_key": "attn_drop_rate"},
        {"label": "Drop Path", "key": "drop_path_rate", "default": 0.1, "tip_key": "drop_path_rate"},
    ],
    "dct": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 80, "tip_key": "embed_dim"},
        {"label": "Depth", "key": "depth", "default": "[20]", "tip_key": "depth"},
        {"label": "Num Heads", "key": "num_heads", "default": "[8]", "tip_key": "num_heads"},
        {"label": "Expansion", "key": "expansion_factor", "default": 4.0, "tip_key": "expansion_ratio"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
        {"label": "Drop Rate", "key": "drop_rate", "default": 0.0, "tip_key": "drop_rate"},
        {"label": "Attn Drop", "key": "attn_drop_rate", "default": 0.0, "tip_key": "attn_drop_rate"},
    ],
    "dctlsa": [
        {"label": "NF", "key": "nf", "default": 55, "tip_key": "nf"},
        {"label": "Num Modules", "key": "num_modules", "default": 6, "tip_key": "num_modules"},
        {"label": "Num Head", "key": "num_head", "default": 5, "tip_key": "num_head"},
    ],
    "ditn": [
        {"label": "Dim", "key": "dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "ITL Blocks", "key": "ITL_blocks", "default": 4, "tip_key": "ITL_blocks"},
        {"label": "SAL Blocks", "key": "SAL_blocks", "default": 4, "tip_key": "SAL_blocks"},
        {"label": "Patch Size", "key": "patch_size", "default": 8, "tip_key": "patch_size"},
        {"label": "Bias", "key": "bias", "default": "false", "tip_key": "bias"},
    ],
    "drct": [
        {"label": "Window Size", "key": "window_size", "default": 16, "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Compress", "key": "compress_ratio", "default": 3, "tip_key": "compress_ratio"},
        {"label": "Squeeze", "key": "squeeze_factor", "default": 30, "tip_key": "squeeze_factor"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "GC", "key": "gc", "default": 32, "tip_key": "gc"},
        {"label": "Conv Scale", "key": "conv_scale", "default": 0.01, "tip_key": "conv_scale"},
    ],
    "esc": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 5, "tip_key": "n_blocks"},
        {"label": "Window Size", "key": "window_size", "default": 32, "tip_key": "window_size"},
        {"label": "Attn Type", "key": "attn_type", "default": "sdpa", "tip_key": "attn_type",
         "type": "combobox", "choices": ["sdpa", "linear"]},
        {"label": "Dysample", "key": "use_dysample", "default": "true", "tip_key": "use_dysample"},
        {"label": "Exp Ratio", "key": "exp_ratio", "default": 1.25, "tip_key": "expansion_ratio"},
    ],
    "flexnet": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "Hidden Rate", "key": "hidden_rate", "default": 4, "tip_key": "hidden_rate"},
        {"label": "Flash Attn", "key": "flash_attn", "default": "true", "tip_key": "flash_attn"},
        {"label": "Num Blocks", "key": "num_blocks", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Upsampler", "key": "upsampler", "default": "ps", "tip_key": "upsampler",
         "type": "combobox", "choices": ["ps", "pixelshuffle", "pixelshuffledirect", "dys", "nearest+conv"]},
    ],
    "hasn": [
        {"label": "Feat Channels", "key": "feature_channels", "default": 52, "tip_key": "feature_channels"},
    ],
    "hat": [
        {"label": "Upscale", "key": "upscale", "default": 4, "tip_key": "upscale"},
        {"label": "Window Size", "key": "window_size", "default": 16, "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Compress", "key": "compress_ratio", "default": 3, "tip_key": "compress_ratio"},
        {"label": "Squeeze", "key": "squeeze_factor", "default": 30, "tip_key": "squeeze_factor"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "hit_srf": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Base Win", "key": "base_win_size", "default": "[8, 8]", "tip_key": "base_win_size"},
        {"label": "Hier Win", "key": "hier_win_ratios", "default": "[0.5, 1, 2, 4, 6, 8]", "tip_key": "hier_win_ratios"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6]", "tip_key": "num_heads"},
    ],
    "hma": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "Interval", "key": "interval_size", "default": 4, "tip_key": "interval_size"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "krgn": [
        {"label": "N Feats", "key": "n_feats", "default": 64, "tip_key": "num_feat"},
        {"label": "N ResGroups", "key": "n_resgroups", "default": 9, "tip_key": "n_resgroups"},
        {"label": "Dilation", "key": "dilation", "default": 3, "tip_key": "dilation"},
        {"label": "Act", "key": "act", "default": "lrelu", "tip_key": "act",
         "type": "combobox", "choices": ["lrelu", "relu", "gelu", "silu"]},
    ],
    "lmlt": [
        {"label": "Dim", "key": "dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 8, "tip_key": "n_blocks"},
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "FFN Scale", "key": "ffn_scale", "default": 2.0, "tip_key": "ffn_scale"},
    ],
    "man": [
        {"label": "N ResBlocks", "key": "n_resblocks", "default": 36, "tip_key": "n_resblocks"},
        {"label": "N Feats", "key": "n_feats", "default": 180, "tip_key": "num_feat"},
        {"label": "Res Scale", "key": "res_scale", "default": 1.0, "tip_key": "res_scale"},
    ],
    "mosrv2": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Block", "key": "n_block", "default": 24, "tip_key": "n_blocks"},
        {"label": "Exp Ratio", "key": "expansion_ratio", "default": 1.5, "tip_key": "expansion_ratio"},
        {"label": "Unshuffle", "key": "unshuffle_mod", "default": "true", "tip_key": "unshuffle_mod"},
        {"label": "Upsampler", "key": "upsampler", "default": "pixelshuffledirect", "tip_key": "upsampler",
         "type": "combobox", "choices": ["pixelshuffledirect", "pixelshuffle", "dys", "nearest+conv"]},
    ],
    "moesr": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 9, "tip_key": "n_blocks"},
        {"label": "Exp Factor", "key": "expansion_factor", "default": 2.6, "tip_key": "expansion_ratio"},
        {"label": "Upsampler", "key": "upsampler", "default": "pixelshuffledirect", "tip_key": "upsampler",
         "type": "combobox", "choices": ["pixelshuffledirect", "pixelshuffle", "dys", "nearest+conv"]},
    ],
    "msdan": [
        {"label": "Channels", "key": "channels", "default": 48, "tip_key": "num_feat"},
        {"label": "Num DFEB", "key": "num_DFEB", "default": 8, "tip_key": "num_DFEB"},
    ],
    "plainusr": [
        {"label": "N Feat", "key": "n_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Attn Feat", "key": "attn_feat", "default": 16, "tip_key": "attn_feat"},
        {"label": "Im Feat", "key": "im_feat", "default": "[64, 48, 32]", "tip_key": "im_feat"},
    ],
    "plksr": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 28, "tip_key": "n_blocks"},
        {"label": "Kernel Size", "key": "kernel_size", "default": 17, "tip_key": "kernel_size"},
        {"label": "Split Ratio", "key": "split_ratio", "default": 0.25, "tip_key": "split_ratio"},
        {"label": "LK Type", "key": "lk_type", "default": "PLK", "tip_key": "lk_type",
         "type": "combobox", "choices": ["PLK", "RFK", "DFK"]},
        {"label": "Use EA", "key": "use_ea", "default": "true", "tip_key": "use_ea"},
    ],
    "realplksr": [
        {"label": "Upscale", "key": "upscale", "default": 4, "tip_key": "upscale"},
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 28, "tip_key": "n_blocks"},
        {"label": "Kernel Size", "key": "kernel_size", "default": 17, "tip_key": "kernel_size"},
        {"label": "Use EA", "key": "use_ea", "default": "true", "tip_key": "use_ea"},
        {"label": "Dysample", "key": "dysample", "default": "false", "tip_key": "use_dysample"},
    ],
    "rcan": [
        {"label": "N ResGroups", "key": "n_resgroups", "default": 10, "tip_key": "n_resgroups"},
        {"label": "N ResBlocks", "key": "n_resblocks", "default": 20, "tip_key": "n_resblocks"},
        {"label": "N Feats", "key": "n_feats", "default": 64, "tip_key": "num_feat"},
        {"label": "Reduction", "key": "reduction", "default": 16, "tip_key": "reduction"},
    ],
    "rgt": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Split Size", "key": "split_size", "default": "[8, 32]", "tip_key": "split_size"},
        {"label": "C Ratio", "key": "c_ratio", "default": 0.5, "tip_key": "c_ratio"},
        {"label": "Depth", "key": "depth", "default": "[6, 6, 6, 6, 6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6, 6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "eimn": [
        {"label": "Embed Dims", "key": "embed_dims", "default": 64, "tip_key": "embed_dims"},
        {"label": "Num Stages", "key": "num_stages", "default": 16, "tip_key": "num_stages"},
        {"label": "MLP Ratios", "key": "mlp_ratios", "default": 2.66, "tip_key": "mlp_ratios"},
        {"label": "Depths", "key": "depths", "default": 1, "tip_key": "depths"},
    ],
    "esrgan": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Num Block", "key": "num_block", "default": 23, "tip_key": "n_blocks"},
        {"label": "Num Grow Ch", "key": "num_grow_ch", "default": 32, "tip_key": "num_grow_ch"},
    ],
    "grformer": [
        {"label": "Window Size", "key": "window_size", "default": "[8, 32]", "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[3, 3, 3, 3]", "tip_key": "num_heads"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "safmn": [
        {"label": "Dim", "key": "dim", "default": 36, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 8, "tip_key": "n_blocks"},
        {"label": "FFN Scale", "key": "ffn_scale", "default": 2.0, "tip_key": "ffn_scale"},
    ],
    "span": [
        {"label": "Num In Ch", "key": "num_in_ch", "default": 3, "tip_key": "num_in_ch"},
        {"label": "Num Out Ch", "key": "num_out_ch", "default": 3, "tip_key": "num_out_ch"},
        {"label": "Feat Channels", "key": "feature_channels", "default": 48, "tip_key": "feature_channels"},
        {"label": "Norm", "key": "norm", "default": "true", "tip_key": "norm"},
    ],
    "spanplus": [
        {"label": "Num In Ch", "key": "num_in_ch", "default": 3, "tip_key": "num_in_ch"},
        {"label": "Num Out Ch", "key": "num_out_ch", "default": 3, "tip_key": "num_out_ch"},
        {"label": "Feat Channels", "key": "feature_channels", "default": 48, "tip_key": "feature_channels"},
        {"label": "Upsampler", "key": "upsampler", "default": "dys", "tip_key": "upsampler",
         "type": "combobox", "choices": ["dys", "conv", "pixelshuffle", "pixelshuffledirect", "nearest+conv"]},
    ],
    "srformer": [
        {"label": "Window Size", "key": "window_size", "default": 16, "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "Patch Norm", "key": "patch_norm", "default": "true", "tip_key": "patch_norm"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "swinir": [
        {"label": "Img Size", "key": "img_size", "default": 64, "tip_key": "img_size"},
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]", "tip_key": "depths"},
        {"label": "Num Heads", "key": "num_heads", "default": "[6, 6, 6, 6]", "tip_key": "num_heads"},
        {"label": "Flash Attn", "key": "flash_attn", "default": "false", "tip_key": "flash_attn"},
        {"label": "QKV Bias", "key": "qkv_bias", "default": "true", "tip_key": "qkv_bias"},
    ],
    "swinir_medium": [
        {"label": "Img Size", "key": "img_size", "default": 64, "tip_key": "img_size"},
        {"label": "Window Size", "key": "window_size", "default": 8, "tip_key": "window_size"},
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]", "tip_key": "depths"},
    ],
    # ─── Entries for architectures from wiki — with real params where known ───
    "artcnn_r8f48": [
        {"label": "Filters", "key": "filters", "default": 48, "tip_key": "num_feat"},
        {"label": "N Blocks", "key": "n_block", "default": 8, "tip_key": "n_blocks"},
    ],
    "dat_2": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depth", "default": "[6,6,6,6,6,6]", "tip_key": "depths"},
        {"label": "Expansion", "key": "expansion_factor", "default": 2, "tip_key": "expansion_factor"},
    ],
    "ditn_real": [
        {"label": "Dim", "key": "dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "ITL Blocks", "key": "ITL_blocks", "default": 4, "tip_key": "n_blocks"},
    ],
    "drct_xl": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6]*14", "tip_key": "depths"},
    ],
    "eimn_l": [
        {"label": "Embed Dims", "key": "embed_dims", "default": 64, "tip_key": "embed_dims"},
        {"label": "Num Stages", "key": "num_stages", "default": 16, "tip_key": "num_stages"},
    ],
    "elan": [
        {"label": "C ELAN", "key": "c_elan", "default": 180, "tip_key": "embed_dim"},
        {"label": "M ELAN", "key": "m_elan", "default": 36, "tip_key": "n_blocks"},
    ],
    "elan_light": [
        {"label": "C ELAN", "key": "c_elan", "default": 60, "tip_key": "embed_dim"},
        {"label": "M ELAN", "key": "m_elan", "default": 24, "tip_key": "n_blocks"},
    ],
    "emt": [
        {"label": "Dim", "key": "dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 6, "tip_key": "n_blocks"},
    ],
    "escrealm": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 10, "tip_key": "n_blocks"},
    ],
    "escrealm_xl": [
        {"label": "Dim", "key": "dim", "default": 128, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 16, "tip_key": "n_blocks"},
    ],
    "esrgan_lite": [
        {"label": "Num Filters", "key": "num_filters", "default": 32, "tip_key": "num_feat"},
        {"label": "Num Blocks", "key": "num_blocks", "default": 12, "tip_key": "n_blocks"},
    ],
    "fdat": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 120, "tip_key": "embed_dim"},
        {"label": "Num Groups", "key": "num_groups", "default": 4, "tip_key": "n_blocks"},
    ],
    "fdat_light": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 108, "tip_key": "embed_dim"},
    ],
    "fdat_xl": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Num Groups", "key": "num_groups", "default": 6, "tip_key": "n_blocks"},
    ],
    "gaterv3": [
        {"label": "Dim", "key": "dim", "default": 32, "tip_key": "embed_dim"},
    ],
    "grl_s": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 128, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[4,4,4,4]", "tip_key": "depths"},
    ],
    "grl_t": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[4,4,4,4]", "tip_key": "depths"},
    ],
    "hat_m": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6,6,6,6,6,6]", "tip_key": "depths"},
        {"label": "Window Size", "key": "window_size", "default": 16, "tip_key": "window_size"},
    ],
    "hat_s": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 144, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6,6,6,6,6,6]", "tip_key": "depths"},
    ],
    "hit_sir": [],
    "hit_sng": [],
    "lkfmixer_b": [
        {"label": "Channels", "key": "channels", "default": 48, "tip_key": "num_feat"},
        {"label": "Num Blocks", "key": "num_block", "default": 8, "tip_key": "n_blocks"},
    ],
    "lkfmixer_l": [
        {"label": "Channels", "key": "channels", "default": 64, "tip_key": "num_feat"},
        {"label": "Num Blocks", "key": "num_block", "default": 12, "tip_key": "n_blocks"},
    ],
    "lkfmixer_t": [
        {"label": "Channels", "key": "channels", "default": 40, "tip_key": "num_feat"},
        {"label": "Num Blocks", "key": "num_block", "default": 6, "tip_key": "n_blocks"},
    ],
    "man_tiny": [
        {"label": "N Feats", "key": "n_feats", "default": 48, "tip_key": "num_feat"},
        {"label": "N ResBlocks", "key": "n_resblocks", "default": 5, "tip_key": "n_blocks"},
    ],
    "metaflexnet": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
    ],
    "metagan3": [
        {"label": "Dims", "key": "dims", "default": "[64,128,192,256]", "tip_key": "dims"},
    ],
    "moesr2": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 9, "tip_key": "n_blocks"},
    ],
    "mosr": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_block", "default": 24, "tip_key": "n_blocks"},
    ],
    "mosr_t": [
        {"label": "Dim", "key": "dim", "default": 48, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_block", "default": 5, "tip_key": "n_blocks"},
    ],
    "rcan_l": [
        {"label": "N Feats", "key": "n_feats", "default": 96, "tip_key": "num_feat"},
    ],
    "rcan_unshuffle": [
        {"label": "Unshuffle Mod", "key": "unshuffle_mod", "default": "true", "tip_key": "unshuffle_mod"},
    ],
    "realcugan": [],
    "rgt_s": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depth", "default": "[6,6,6,6,6,6]", "tip_key": "depths"},
    ],
    "rtmosr": [
        {"label": "Dim", "key": "dim", "default": 32, "tip_key": "embed_dim"},
    ],
    "rtmosr_l": [
        {"label": "Unshuffle Mod", "key": "unshuffle_mod", "default": "true", "tip_key": "unshuffle_mod"},
    ],
    "rtmosr_ul": [],
    "scunet_aaf6aa": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
    ],
    "sebica": [],
    "sebica_mini": [],
    "seemore_t": [
        {"label": "Embedding Dim", "key": "embedding_dim", "default": 36, "tip_key": "embed_dim"},
        {"label": "Num Layers", "key": "num_layers", "default": 6, "tip_key": "n_blocks"},
    ],
    "srformerv2": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 240, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[4,4,4,4,4,4]", "tip_key": "depths"},
    ],
    "swin2sr_l": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 240, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6]*9", "tip_key": "depths"},
    ],
    "swin2sr_m": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6,6,6,6,6,6]", "tip_key": "depths"},
    ],
    "swin2sr_s": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6,6,6,6]", "tip_key": "depths"},
    ],
    "swinir_m": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 180, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6,6,6,6,6,6]", "tip_key": "depths"},
    ],
    "swinir_s": [
        {"label": "Embed Dim", "key": "embed_dim", "default": 60, "tip_key": "embed_dim"},
        {"label": "Depths", "key": "depths", "default": "[6,6,6,6]", "tip_key": "depths"},
    ],
    "temporalspan": [
        {"label": "Feature Ch", "key": "feature_channels", "default": 48, "tip_key": "feature_channels"},
        {"label": "Num Frames", "key": "num_frames", "default": 5, "tip_key": "num_frames"},
    ],
    "temporalspanv2": [
        {"label": "Feature Ch", "key": "feature_channels", "default": 48, "tip_key": "feature_channels"},
        {"label": "Num Frames", "key": "num_frames", "default": 5, "tip_key": "num_frames"},
        {"label": "Num Blocks", "key": "num_blocks", "default": 6, "tip_key": "n_blocks"},
    ],
    "tscunet": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "Clip Size", "key": "clip_size", "default": 5, "tip_key": "num_frames"},
    ],
    "ultracompact": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Num Conv", "key": "num_conv", "default": 8, "tip_key": "num_conv"},
    ],
}

DISC_FIELDS = {
    # NeoSR discriminators
    "unet": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Skip Connection", "key": "skip_connection", "default": "true", "tip_key": "skip_connection"},
    ],
    "dunet": [
        {"label": "Dim", "key": "dim", "default": 64, "tip_key": "embed_dim"},
        {"label": "N Blocks", "key": "n_blocks", "default": 2, "tip_key": "n_blocks"},
    ],
    "metagan": [
        {"label": "Dims", "key": "dims", "default": "[48, 96, 192, 288]", "tip_key": "dims"},
        {"label": "Blocks", "key": "blocks", "default": "[3, 3, 9, 3]", "tip_key": "blocks"},
        {"label": "Attention", "key": "attention", "default": "true", "tip_key": "attention"},
        {"label": "Head Dim", "key": "head_dim", "default": 32, "tip_key": "head_dim"},
    ],
    "ea2fpn": [
        {"label": "Seg Channels", "key": "segmentation_channels", "default": 64, "tip_key": "segmentation_channels"},
        {"label": "Pyramid Channels", "key": "pyramid_channels", "default": 64, "tip_key": "pyramid_channels"},
        {"label": "Class Num", "key": "class_num", "default": 6, "tip_key": "class_num"},
    ],
    "patchgan": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Num Layers", "key": "num_layers", "default": 3, "tip_key": "num_layers"},
        {"label": "Use SN", "key": "use_sn", "default": "true", "tip_key": "use_sn"},
        {"label": "Use Sigmoid", "key": "use_sigmoid", "default": "false", "tip_key": "use_sigmoid"},
    ],
    # Redux discriminators
    "unetdiscriminatorsn": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Skip Connection", "key": "skip_connection", "default": "true", "tip_key": "skip_connection"},
    ],
    "metagan2": [
        {"label": "Dims", "key": "dims", "default": "[32, 64, 128, 192]", "tip_key": "dims"},
        {"label": "Blocks", "key": "blocks", "default": "[3, 3, 15, 3]", "tip_key": "blocks"},
        {"label": "Downs", "key": "downs", "default": "[4, 2, 2, 2]", "tip_key": "downs"},
    ],
    "patchgandiscriminatorsn": [
        {"label": "NDF", "key": "ndf", "default": 64, "tip_key": "num_feat"},
        {"label": "N Layers", "key": "n_layers", "default": 3, "tip_key": "num_layers"},
        {"label": "Use Sigmoid", "key": "use_sigmoid", "default": "false", "tip_key": "use_sigmoid"},
    ],
    "multiscalepatchgandiscriminatorsn": [
        {"label": "NDF", "key": "ndf", "default": 64, "tip_key": "num_feat"},
        {"label": "N Layers", "key": "n_layers", "default": 3, "tip_key": "num_layers"},
        {"label": "Num D", "key": "num_d", "default": 3, "tip_key": "num_d"},
    ],
    "vggstylediscriminator": [
        {"label": "Num Feat", "key": "num_feat", "default": 64, "tip_key": "num_feat"},
        {"label": "Input Size", "key": "input_size", "default": 128, "tip_key": "input_size"},
    ],
}

# Per-engine discriminator names
NEOSR_DISC_LIST = ["unet", "dunet", "patchgan", "metagan", "ea2fpn"]
REDUX_DISC_LIST = ["dunet", "unetdiscriminatorsn", "metagan2", "patchgandiscriminatorsn", "multiscalepatchgandiscriminatorsn", "vggstylediscriminator"]

# Friendly display names for long discriminator type names
DISC_DISPLAY_NAMES = {
    "unetdiscriminatorsn": "UNet-SN",
    "patchgandiscriminatorsn": "PatchGAN-SN",
    "multiscalepatchgandiscriminatorsn": "MultiPatchGAN-SN",
    "vggstylediscriminator": "VGG-Style",
    "dunet": "DUNet",
    "unet": "UNet",
    "patchgan": "PatchGAN",
    "metagan": "MetaGAN",
    "metagan2": "MetaGAN2",
    "ea2fpn": "EA2-FPN",
}

# Reverse mapping: display name → internal name
DISC_INTERNAL_NAMES = {v: k for k, v in DISC_DISPLAY_NAMES.items()}

# Per-engine GAN loss types
NEOSR_GAN_TYPES = ["bce", "mse", "huber"]
REDUX_GAN_TYPES = ["vanilla", "wgan", "hinge", "lsgan"]

OPTIMIZERS = [
    "AdamW", "Adam", "NAdam", "Adan",
    "AdamW_Win", "AdamW_SF", "Adan_SF", "SOAP_SF"
]

# Per-engine optimizer lists
NEOSR_OPTIMIZERS = [
    "Adam", "AdamW", "NAdam", "Adan",
    "AdamW_Win", "AdamW_SF", "Adan_SF", "SOAP_SF",
]

REDUX_OPTIMIZERS = [
    "Adam", "AdamW", "NAdam", "SGD", "RAdam", "Adadelta", "Adagrad",
]

SCHEDULERS = [
    "MultiStepLR", "CosineAnnealing", "CosineAnnealingRestart",
    "CyclicLR", "OneCycleLR"
]

# Per-engine scheduler lists
NEOSR_SCHEDULERS = [
    "MultiStepLR", "CosineAnnealing", "CosineAnnealingRestart",
    "CyclicLR", "OneCycleLR",
]

REDUX_SCHEDULERS = [
    "MultiStepLR", "CosineAnnealingLR", "StepLR",
    "ExponentialLR", "ReduceLROnPlateau",
]

# Per-engine scale options
NEOSR_SCALES = ["1", "2", "3", "4", "6", "8"]
REDUX_SCALES = ["1", "2", "3", "4", "8"]

# Per-engine architecture families
NEOSR_ARCH_FAMILIES = {
    "✨ Recommandé": ["omnisr", "span", "realplksr", "esrgan", "compact"],
    "🚀 Léger / Rapide": ["span", "spanplus", "compact", "ultracompact", "safmn", "lmlt", "plksr", "realplksr", "cugan"],
    "🤖 Transformers (Lourd)": ["hat", "swinir_small", "swinir_medium", "dat_s", "srformer_medium", "drct", "atd"],
    "🎨 GAN / Restauration": ["esrgan", "rcan", "artcnn_r16f96"],
    "📦 Autres": ["cfsr", "craft", "dct", "dctlsa", "ditn", "esc", "eimn", "flexnet", "grformer", "hasn", "hit_srf", "hma", "krgn", "man", "moesr", "mosrv2", "msdan", "plainusr", "rgt", "asid"],
}

REDUX_ARCH_FAMILIES = {
    # Catégorisation basée sur bench Redux 2026-05-19 (GTX 1080 Ti, scale=1)
    "✨ Recommandé": ["span_s", "artcnn_r16f96", "lkfmixer_t", "swinir_s", "seemore_t", "compact", "realplksr", "omnisr"],
    "🚀 Léger / Rapide": [
        "compact", "ultracompact", "superultracompact",
        "span", "span_s", "spanplus", "spanplus_s", "spanplus_st", "spanplus_sts",
        "rtmosr", "rtmosr_l", "rtmosr_ul", "mosr_t", "mosrv2",
        "artcnn_r16f96", "artcnn_r8f64", "artcnn_r8f48", "artcnn_r3f24",
        "esrgan_lite", "safmn", "sebica", "sebica_mini", "lmlt_tiny", "man_tiny",
        "plksr_tiny", "realplksr_tiny", "lkfmixer_t", "seemore_t", "gaterv3_s",
        "plksr", "realplksr", "esrgan", "realcugan",
    ],
    "🤖 Transformers / Attention": [
        "swinir_s", "swinir_m", "swinir_l", "swin2sr_s", "swin2sr_m", "swin2sr_l",
        "hat_s", "hat_m", "hat_l",
        "dat", "dat_2", "dat_light", "dat_s",
        "srformer", "srformerv2",
        "drct", "drct_l", "drct_xl", "drct_s",
        "atd", "atd_light",
        "elan", "elan_light",
        "omnisr",
    ],
    "🎨 GAN / Restauration": ["esrgan", "esrgan_lite", "rcan", "rcan_l", "rcan_unshuffle", "artcnn_r16f96", "artcnn_r8f64", "artcnn_r8f48", "artcnn_r3f24", "scunet_aaf6aa"],
    "🎞️ Vidéo / Temporal": ["temporalspan", "temporalspanv2", "tscunet"],
    "🧪 Expérimental / Lourd": [
        "man", "man_light", "safmn_l",
        "eimn_a", "eimn_l", "lmlt_base", "lmlt_large",
        "ditn_real", "gaterv3", "lkfmixer_b", "lkfmixer_l", "mosr",
        "moesr2", "rgt", "rgt_s", "fdat", "fdat_light", "fdat_xl",
        "grl_b", "grl_s", "grl_t", "hit_sir", "hit_sng",
        "cascadedgaze", "craft", "dctlsa", "dwt", "dwt_s",
        "emt", "escrealm", "escrealm_xl", "metaflexnet",
        "flexnet", "srformer_light",
    ],
}

VRAM_FACTORS = {
    "omnisr": 1.2, "dat": 1.6, "hat": 1.5, "span": 0.8, 
    "realplksr": 1.1, "esrgan": 1.0, "swinir": 1.4, "swinir_medium": 1.4,
    "rcan": 1.2, "compact": 0.5, "asid": 1.3,
    "drct": 1.5, "hit_srf": 1.3, "mosrv2": 1.2, "safmn": 0.8,
    "atd": 1.5, "cfsr": 1.0, "craft": 1.2, "cugan": 0.9, "dct": 1.3,
    "dctlsa": 1.2, "ditn": 1.0, "esc": 1.1, "flexnet": 1.2, "hma": 1.3,
    "krgn": 1.0, "lmlt": 1.0, "man": 1.4, "moesr": 1.2, "msdan": 1.0,
    "plainusr": 1.0, "plksr": 1.1, "rgt": 1.6, "eimn": 1.0, "grformer": 1.3,
    "spanplus": 0.9, "srformer": 1.3,
    "omnisr": 1.2, "dat": 1.6, "hat": 1.5, "span": 0.8,
    # neosr-specific aliases (correct registry names)
    "dat_s": 1.6, "dat_2": 1.8, "srformer_medium": 1.3, "swinir_small": 1.2,
}

DISC_VRAM_FACTORS = {
    "unet": 0.4, "dunet": 0.6, "patchgan": 0.2, 
    "metagan": 0.8, "ea2fpn": 0.7,
}

ARCH_PROFILES = {
    "omnisr":      [7, 6, 8, 6, 6, 8, 8],
    "dat":         [8, 7, 9, 4, 3, 7, 9],
    "hat":         [8, 8, 9, 3, 4, 7, 9],
    "span":        [6, 5, 7, 9, 8, 9, 5],
    "realplksr":   [7, 6, 8, 8, 7, 8, 7],
    "plksr":       [7, 6, 8, 8, 7, 8, 6],
    "esrgan":      [6, 5, 6, 7, 7, 7, 7],
    "swinir":      [7, 7, 8, 4, 5, 7, 8],
    "swinir_medium": [7, 7, 8, 4, 5, 7, 8],
    "rcan":        [6, 5, 9, 5, 6, 8, 6],
    "compact":     [5, 4, 6, 10, 10, 9, 4],
    "asid":        [7, 6, 8, 5, 5, 8, 7],
    "drct":        [9, 8, 8, 2, 2, 6, 9],
    "hit_srf":     [7, 7, 8, 5, 5, 7, 7],
    "mosrv2":      [6, 6, 7, 8, 7, 8, 6],
    "safmn":       [6, 5, 7, 9, 9, 8, 6],
    "spanplus":    [6, 5, 7, 8, 8, 9, 5],
    "srformer":         [8, 7, 8, 4, 4, 7, 8],
    "srformer_medium":  [8, 7, 8, 4, 4, 7, 8],
    "dat_s":            [8, 7, 9, 4, 3, 7, 9],
    "dat_2":            [8, 7, 9, 3, 3, 7, 9],
    "swinir_small":     [7, 7, 8, 5, 5, 7, 7],
    "cugan":       [7, 6, 7, 9, 8, 10, 3],
    "atd":         [8, 7, 8, 4, 4, 7, 8],
    "cfsr":        [7, 6, 7, 7, 7, 7, 7],
    "craft":       [7, 7, 8, 5, 5, 7, 7],
    "dct":         [7, 7, 8, 5, 5, 7, 7],
    "dctlsa":      [7, 6, 8, 6, 6, 7, 7],
    "ditn":        [6, 5, 7, 9, 9, 8, 6],
    "esc":         [7, 6, 8, 7, 7, 8, 6],
    "flexnet":     [7, 6, 8, 6, 6, 7, 7],
    "hasn":        [7, 6, 7, 6, 6, 7, 7],
    "hma":         [7, 7, 8, 5, 5, 7, 8],
    "krgn":        [6, 5, 7, 8, 8, 8, 6],
    "lmlt":        [6, 5, 7, 9, 9, 8, 6],
    "man":         [7, 6, 8, 5, 5, 7, 8],
    "moesr":       [7, 6, 7, 6, 6, 8, 6],
    "msdan":       [6, 5, 7, 8, 8, 8, 6],
    "plainusr":    [6, 5, 7, 8, 8, 7, 6],
    "rgt":         [9, 8, 9, 2, 2, 6, 9],
    "eimn":        [7, 6, 7, 8, 8, 7, 7],
    "grformer":    [8, 7, 8, 4, 4, 7, 8],
    "default":     [5, 5, 5, 5, 5, 5, 5],
    "artcnn_r16f96": [7, 6, 7, 7, 7, 7, 6],
    "artcnn_r3f24":  [5, 4, 5, 9, 9, 6, 4],
    "atd_light":     [7, 6, 7, 6, 6, 7, 7],
    "dat_light":     [7, 6, 8, 6, 5, 7, 8],
    "dat_s":         [8, 7, 9, 4, 4, 7, 9],
    "dis_balanced":  [6, 6, 6, 8, 8, 6, 6],
    "drct_l":        [9, 8, 8, 1, 1, 6, 9],
    "dwt_s":         [8, 7, 8, 4, 4, 7, 8],
    "eimn_a":        [7, 6, 7, 8, 8, 7, 7],
    "elan_light":    [6, 6, 7, 9, 9, 8, 6],
    "fdat_light":    [7, 6, 8, 6, 6, 7, 7],
    "fdat_xl":       [9, 8, 9, 2, 2, 7, 9],
    "grl_b":         [9, 8, 9, 2, 2, 7, 9],
    "hat_l":         [9, 8, 9, 2, 2, 7, 9],
    "man_light":     [7, 6, 8, 6, 6, 7, 7],
    "mosr_t":        [6, 5, 6, 9, 9, 7, 6],
    "plksr_tiny":    [6, 5, 7, 9, 9, 7, 6],
    "realplksr_large": [8, 7, 8, 7, 6, 8, 8],
    "rtmosr":        [6, 5, 6, 9, 9, 7, 6],
    "safmn_l":       [7, 6, 8, 7, 7, 7, 7],
    "sebica":        [7, 6, 7, 7, 7, 7, 6],
    "span_s":        [6, 5, 7, 9, 9, 8, 5],
    "swin2sr_l":     [8, 7, 8, 4, 4, 7, 8],
    "swinir_l":      [8, 7, 8, 4, 4, 7, 8],
    # --- Profils ajoutés d'après bench Redux 2026-05-19 + type d'architecture ---
    # Format : [Netteté, Texture, Fidélité/PSNR, Vitesse, Légèreté_VRAM, Efficacité_Anime, Efficacité_Réaliste]
    # Testés directement en bench (données mesurées)
    "artcnn_r8f64":    [7, 6, 8, 9, 9, 6, 7],   # PSNR=49.44, 9.09 it/s
    "ditn_real":       [7, 7, 8, 8, 6, 5, 9],   # PSNR=51.09, 9.36 it/s, VRAM SMI=4.13 GB
    "esrgan_lite":     [6, 5, 6, 9, 9, 5, 7],   # PSNR=35.03, 9.35 it/s — GAN-style
    "gaterv3_s":       [7, 6, 7, 9, 8, 7, 7],   # 9.25 it/s, arch attention efficiente
    "lkfmixer_t":      [6, 6, 7, 9, 9, 7, 6],   # PSNR=42.65, 9.34 it/s
    "lmlt_tiny":       [6, 5, 7, 9, 9, 7, 6],   # PSNR=41.84, 9.32 it/s
    "man_tiny":        [7, 6, 7, 9, 9, 7, 6],   # PSNR=46.28, 9.25 it/s — différent de man !
    "rtmosr_l":        [6, 5, 6, 9, 9, 7, 6],   # 9.45 it/s, ultra-léger
    "rtmosr_ul":       [5, 4, 5, 10, 10, 7, 5], # ultra-ultra-light
    "sebica":          [6, 5, 7, 8, 9, 7, 6],   # PSNR=44.14, 8.69 it/s — bench mesuré
    "seemore_t":       [6, 6, 7, 8, 8, 6, 7],   # PSNR=43.81, 9.22 it/s
    "spanplus_s":      [6, 5, 7, 9, 9, 8, 5],   # PSNR=40.24, 9.41 it/s
    "spanplus_st":     [6, 5, 7, 9, 9, 8, 5],
    "spanplus_sts":    [6, 5, 7, 9, 9, 8, 5],
    "superultracompact": [4, 3, 5, 10, 10, 7, 4], # 8.33 it/s, VRAM=1.56 GB — minimal
    "ultracompact":    [5, 4, 5, 10, 10, 7, 4], # 9.18 it/s — compact plus léger
    "realplksr_tiny":  [7, 6, 7, 9, 9, 7, 7],   # 9.34 it/s, PLKSR famille
    "plksr_tiny":      [6, 6, 7, 9, 9, 7, 6],   # 9.39 it/s
    "sebica_mini":     [5, 4, 6, 10, 10, 6, 5], # plus petit que sebica
    # Variants dérivés des testés
    "artcnn_r8f48":    [6, 5, 7, 9, 10, 6, 6],
    "artcnn_r3f24":    [5, 4, 5, 10, 10, 5, 5], # plus petit, moins précis
    "lmlt_base":       [7, 6, 8, 7, 7, 7, 6],
    "lmlt_large":      [7, 7, 8, 5, 5, 7, 7],
    "lkfmixer_b":      [7, 6, 7, 6, 6, 7, 6],
    "lkfmixer_l":      [7, 7, 8, 5, 5, 7, 6],
    "gaterv3":         [7, 6, 7, 7, 7, 7, 7],   # version pleine taille
    "mosr":            [7, 6, 7, 7, 7, 7, 6],   # heavier mosr_t
    "realplksr_large": [8, 7, 9, 5, 5, 8, 8],
    "rcan_l":          [7, 6, 9, 4, 4, 8, 7],   # bigger rcan
    "rcan_unshuffle":  [6, 5, 8, 5, 5, 8, 6],   # rcan + pixel unshuffle
    # Transformers non-testés — dérivés des types connus
    "hat_s":           [8, 8, 9, 4, 4, 7, 9],
    "hat_m":           [8, 8, 9, 2, 3, 7, 9],
    "drct_s":          [8, 7, 8, 4, 4, 6, 8],
    "drct_xl":         [9, 8, 9, 1, 1, 6, 9],
    "srformerv2":      [8, 7, 8, 4, 4, 7, 8],
    "srformer_light":  [7, 7, 7, 6, 6, 7, 7],  # ⚠️ crash Pascal sm_61 + PyTorch 2.7
    "swinir_s":        [7, 7, 8, 5, 5, 7, 7],  # alias swinir_small
    "swinir_m":        [7, 7, 8, 4, 5, 7, 8],  # alias swinir_medium
    "swin2sr_s":       [7, 7, 8, 6, 6, 7, 7],
    "swin2sr_m":       [7, 7, 8, 4, 4, 7, 8],
    "rgt_s":           [8, 8, 8, 3, 3, 7, 8],
    "moesr2":          [7, 6, 8, 5, 5, 7, 7],
    # Archs axés anime
    "realcugan":       [7, 6, 7, 8, 8, 9, 4],  # CUNet — excellente sur anime, limité réel
    # Archs restauration / vidéo
    "scunet_aaf6aa":   [8, 7, 7, 5, 5, 6, 8],  # SCUNet — débruitage/restauration
    "temporalspan":    [7, 6, 7, 7, 6, 7, 6],  # Vidéo — span temporel
    "temporalspanv2":  [7, 6, 8, 6, 6, 7, 6],
    "tscunet":         [7, 7, 8, 5, 5, 7, 7],
    # Archs expérimentaux (non benchmarkés)
    "cascadedgaze":    [7, 6, 8, 5, 5, 7, 7],
    "dwt":             [7, 6, 7, 7, 7, 7, 7],   # wavelet-guided
    "dwt_s":           [7, 6, 7, 8, 8, 7, 7],
    "eimn_l":          [7, 6, 7, 7, 7, 7, 7],
    "elan":            [7, 7, 8, 5, 5, 6, 8],
    "emt":             [7, 7, 8, 5, 5, 7, 8],
    "escrealm":        [7, 6, 8, 6, 6, 7, 7],
    "escrealm_xl":     [8, 7, 8, 4, 4, 7, 8],
    "fdat":            [8, 7, 8, 5, 5, 7, 8],
    "grl_s":           [8, 7, 8, 5, 5, 7, 8],
    "grl_t":           [7, 6, 7, 7, 7, 7, 7],
    "hit_sir":         [7, 7, 8, 5, 5, 7, 7],
    "hit_sng":         [7, 7, 8, 5, 5, 7, 7],
    "metaflexnet":     [7, 6, 8, 5, 5, 7, 7],
}

REDUX_LOSS_INFO = {
    "l1loss": {"loss_weight": 1.0, "reduction": "mean"},
    "mseloss": {"loss_weight": 1.0, "reduction": "mean"},
    "charbonnierloss": {"loss_weight": 1.0, "reduction": "mean", "eps": 1e-12},
    "perceptualloss": {"loss_weight": 1.0, "layer_weights": "dict(conv5_4=1)", "criterion": "l1"},
    "ganloss": {"loss_weight": 1.0, "gan_type": "vanilla", "real_label_val": 1.0, "fake_label_val": 0.0},
    "multiscaleganloss": {"loss_weight": 1.0, "gan_type": "vanilla", "real_label_val": 1.0, "fake_label_val": 0.0},
    "adistsloss": {"loss_weight": 1.0, "window_size": 21},
    "distsloss": {"loss_weight": 1.0, "as_loss": "true"},
    "mssimloss": {"loss_weight": 1.0, "channels": 3, "test_y_channel": "true"},
    "ssimloss": {"loss_weight": 1.0, "channels": 3, "test_y_channel": "true"},
    "ffloss": {"loss_weight": 1.0, "alpha": 1.0},
    "fftloss": {"loss_weight": 1.0},
    "colorloss": {"loss_weight": 1.0, "criterion": "l1"},
    "averageloss": {"loss_weight": 1.0, "criterion": "l1"},
    "bicubicloss": {"loss_weight": 1.0, "criterion": "l1"},
    "contextualloss": {"loss_weight": 1.0, "use_vgg": "true", "layer_weights": "dict(conv4_4=1)"},
    "ldlloss": {"loss_weight": 1.0, "criterion": "l1"},
    "lumaloss": {"loss_weight": 1.0, "criterion": "l1"},
    "gradientvarianceloss": {"loss_weight": 1.0, "patch_size": 16},
    "huberloss": {"loss_weight": 1.0, "delta": 1.0},
    "kldivloss": {"loss_weight": 1.0},
    "consistent_loss": {"loss_weight": 1.0, "criterion": "l1"},
    "palette_matching_loss": {"loss_weight": 1.0},
}

# --- LISTE COMPLETE ARCHITECTURES REDUX ---
REDUX_ARCH_FIELDS = {
    # ATD
    "atd": [{"label": "Embed Dim", "key": "embed_dim", "default": 210}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}, {"label": "Category Size", "key": "category_size", "default": 256}],
    "atd_light": [{"label": "Embed Dim", "key": "embed_dim", "default": 48}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]"}, {"label": "Category Size", "key": "category_size", "default": 128}],
    
    # ArtCNN
    "artcnn_r16f96": [{"label": "Filters", "key": "filters", "default": 96}, {"label": "Blocks", "key": "n_block", "default": 16}],
    "artcnn_r8f48": [{"label": "Filters", "key": "filters", "default": 48}, {"label": "Blocks", "key": "n_block", "default": 8}],
    "artcnn_r3f24": [{"label": "Filters", "key": "filters", "default": 24}, {"label": "Blocks", "key": "n_block", "default": 3}],
    
    # AutoEncoder / Misc
    "autoencoder": [{"label": "NF", "key": "nf", "default": 64}],
    "cascadedgaze": [{"label": "Width", "key": "width", "default": 60}, {"label": "Middle Blk", "key": "middle_blk_num", "default": 10}],
    
    # CRAFT
    "craft": [{"label": "Window", "key": "window_size", "default": 16}, {"label": "Embed Dim", "key": "embed_dim", "default": 48}, {"label": "Depths", "key": "depths", "default": "[2, 2, 2, 2]"}],
    
    # DAT
    "dat": [{"label": "Img Size", "key": "img_size", "default": 64}, {"label": "Split Size", "key": "split_size", "default": "[8, 32]"}, {"label": "Depth", "key": "depth", "default": "[6, 6, 6, 6, 6, 6]"}, {"label": "Embed Dim", "key": "embed_dim", "default": 180}],
    "dat_2": [{"label": "Split Size", "key": "split_size", "default": "[8, 32]"}, {"label": "Depth", "key": "depth", "default": "[6, 6, 6, 6, 6, 6]"}, {"label": "Expansion", "key": "expansion_factor", "default": 2}],
    "dat_light": [{"label": "Embed Dim", "key": "embed_dim", "default": 60}, {"label": "Depth", "key": "depth", "default": "[18]"}],
    "dat_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Split Size", "key": "split_size", "default": "[8, 16]"}],
    
    # DCTLSA & DIS
    "dctlsa": [{"label": "NF", "key": "nf", "default": 55}, {"label": "Num Modules", "key": "num_modules", "default": 6}],
    "dis_balanced": [{"label": "Num Feat", "key": "num_features", "default": 32}, {"label": "Num Blocks", "key": "num_blocks", "default": 12}],
    "dis_fast": [{"label": "Num Feat", "key": "num_features", "default": 32}, {"label": "Num Blocks", "key": "num_blocks", "default": 8}],
    
    # DITN
    "ditn_real": [{"label": "Dim", "key": "dim", "default": 60}, {"label": "ITL Blocks", "key": "ITL_blocks", "default": 4}],
    
    # DRCT
    "drct": [{"label": "Window", "key": "window_size", "default": 16}, {"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}],
    "drct_l": [{"label": "Window", "key": "window_size", "default": 16}, {"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]"}],
    "drct_xl": [{"label": "Window", "key": "window_size", "default": 16}, {"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]"}],
    
    # DWT
    "dwt": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}, {"label": "Window", "key": "window_size", "default": 16}],
    "dwt_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}, {"label": "Window", "key": "window_size", "default": 8}],
    
    # EIMN & ELAN
    "eimn_a": [{"label": "Embed Dims", "key": "embed_dims", "default": 64}, {"label": "Num Stages", "key": "num_stages", "default": 14}],
    "eimn_l": [{"label": "Embed Dims", "key": "embed_dims", "default": 64}, {"label": "Num Stages", "key": "num_stages", "default": 16}],
    "elan": [{"label": "M Elan", "key": "m_elan", "default": 36}, {"label": "C Elan", "key": "c_elan", "default": 180}],
    "elan_light": [{"label": "M Elan", "key": "m_elan", "default": 24}, {"label": "C Elan", "key": "c_elan", "default": 60}],
    
    # EMT & ESCRealM
    "emt": [{"label": "Dim", "key": "dim", "default": 60}, {"label": "N Blocks", "key": "n_blocks", "default": 6}, {"label": "N GTLs", "key": "n_GTLs", "default": 2}],
    "escrealm": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "Blocks", "key": "n_blocks", "default": 10}, {"label": "Window", "key": "window_size", "default": 32}],
    "escrealm_xl": [{"label": "Dim", "key": "dim", "default": 128}, {"label": "Blocks", "key": "n_blocks", "default": 16}, {"label": "Window", "key": "window_size", "default": 32}],
    
    # FDAT
    "fdat": [{"label": "Embed Dim", "key": "embed_dim", "default": 120}, {"label": "Groups", "key": "num_groups", "default": 4}],
    "fdat_light": [{"label": "Embed Dim", "key": "embed_dim", "default": 108}, {"label": "Groups", "key": "num_groups", "default": 3}],
    "fdat_xl": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Groups", "key": "num_groups", "default": 6}],
    
    # FlexNet
    "flexnet": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "Window", "key": "window_size", "default": 8}, {"label": "Blocks", "key": "num_blocks", "default": "[6, 6, 6, 6, 6, 6]"}],
    "metaflexnet": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "Blocks", "key": "num_blocks", "default": "[4, 6, 6, 8]"}],
    
    # GRL
    "grl_b": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[4, 4, 8, 8, 8, 4, 4]"}, {"label": "Window", "key": "window_size", "default": 32}],
    "grl_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 128}, {"label": "Depths", "key": "depths", "default": "[4, 4, 4, 4]"}],
    "grl_t": [{"label": "Embed Dim", "key": "embed_dim", "default": 64}, {"label": "Depths", "key": "depths", "default": "[4, 4, 4, 4]"}],
    
    # GateRV3
    "gaterv3": [{"label": "Dim", "key": "dim", "default": 32}, {"label": "Latent", "key": "num_latent", "default": 8}],
    
    # HAT
    "hat_l": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]"}],
    "hat_m": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}],
    "hat_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 144}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}],
    
    # HiT
    "hit_sir": [{"label": "Embed Dim", "key": "embed_dim", "default": 60}, {"label": "Base Win", "key": "base_win_size", "default": "[8, 8]"}],
    "hit_sng": [{"label": "Embed Dim", "key": "embed_dim", "default": 60}, {"label": "Base Win", "key": "base_win_size", "default": "[8, 8]"}],
    
    # LKF
    "lkfmixer_b": [{"label": "Channels", "key": "channels", "default": 48}, {"label": "Blocks", "key": "num_block", "default": 8}],
    "lkfmixer_l": [{"label": "Channels", "key": "channels", "default": 64}, {"label": "Blocks", "key": "num_block", "default": 12}],
    "lkfmixer_t": [{"label": "Channels", "key": "channels", "default": 40}, {"label": "Blocks", "key": "num_block", "default": 6}],
    
    # LMLT
    "lmlt_base": [{"label": "Dim", "key": "dim", "default": 60}, {"label": "Blocks", "key": "n_blocks", "default": 8}],
    "lmlt_large": [{"label": "Dim", "key": "dim", "default": 84}, {"label": "Blocks", "key": "n_blocks", "default": 8}],
    "lmlt_tiny": [{"label": "Dim", "key": "dim", "default": 36}, {"label": "Blocks", "key": "n_blocks", "default": 8}],
    
    # MAN
    "man": [{"label": "N Feats", "key": "n_feats", "default": 180}, {"label": "N ResBlocks", "key": "n_resblocks", "default": 36}],
    "man_light": [{"label": "N Feats", "key": "n_feats", "default": 60}, {"label": "N ResBlocks", "key": "n_resblocks", "default": 24}],
    "man_tiny": [{"label": "N Feats", "key": "n_feats", "default": 48}, {"label": "N ResBlocks", "key": "n_resblocks", "default": 5}],
    
    # MetaGAN & MoESR
    "metagan3": [{"label": "Dims", "key": "dims", "default": "[64, 128, 192, 256]"}, {"label": "Blocks", "key": "blocks", "default": "[2, 3, 5, 2]"}],
    "moesr2": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "N Blocks", "key": "n_blocks", "default": 9}],
    
    # MoSR & MoSRv2
    "mosr": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "N Block", "key": "n_block", "default": 24}],
    "mosr_t": [{"label": "Dim", "key": "dim", "default": 48}, {"label": "N Block", "key": "n_block", "default": 5}],
    "mosrv2": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "N Block", "key": "n_block", "default": 24}],
    
    # OmniSR & PLKSR
    "omnisr": [{"label": "Num Feat", "key": "num_feat", "default": 64}, {"label": "Window", "key": "window_size", "default": 8}],
    "plksr": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "N Blocks", "key": "n_blocks", "default": 28}, {"label": "Kernel", "key": "kernel_size", "default": 17}],
    "plksr_tiny": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "N Blocks", "key": "n_blocks", "default": 12}, {"label": "Kernel", "key": "kernel_size", "default": 13}],
    
    # RCAN
    "rcan": [{"label": "N ResGroups", "key": "n_resgroups", "default": 10}, {"label": "N ResBlocks", "key": "n_resblocks", "default": 20}, {"label": "N Feats", "key": "n_feats", "default": 64}],
    "rcan_l": [{"label": "N ResGroups", "key": "n_resgroups", "default": 10}, {"label": "N ResBlocks", "key": "n_resblocks", "default": 20}, {"label": "N Feats", "key": "n_feats", "default": 96}],
    "rcan_unshuffle": [{"label": "N ResGroups", "key": "n_resgroups", "default": 10}, {"label": "N Feats", "key": "n_feats", "default": 64}, {"label": "Unshuffle", "key": "unshuffle_mod", "default": "true"}],
    
    # RGT
    "rgt": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depth", "default": "[6, 6, 6, 6, 6, 6, 6, 6]"}],
    "rgt_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depth", "default": "[6, 6, 6, 6, 6, 6]"}],
    
    # ESRGAN (RRDB)
    "esrgan": [{"label": "Num Feat", "key": "num_filters", "default": 64}, {"label": "Num Blocks", "key": "num_blocks", "default": 23}, {"label": "Pixel Unshuffle", "key": "use_pixel_unshuffle", "default": "true"}],
    "esrgan_lite": [{"label": "Num Feat", "key": "num_filters", "default": 32}, {"label": "Num Blocks", "key": "num_blocks", "default": 12}],
    
    # RTMoSR
    "rtmosr": [{"label": "Dim", "key": "dim", "default": 32}, {"label": "N Blocks", "key": "n_blocks", "default": 2}, {"label": "Expansion", "key": "ffn_expansion", "default": 2}],
    "rtmosr_l": [{"label": "Dim", "key": "dim", "default": 32}, {"label": "N Blocks", "key": "n_blocks", "default": 2}, {"label": "Unshuffle", "key": "unshuffle_mod", "default": "true"}],
    
    # RealPLKSR
    "realplksr": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "N Blocks", "key": "n_blocks", "default": 28}, {"label": "Kernel", "key": "kernel_size", "default": 17}, {"label": "Layer Norm", "key": "layer_norm", "default": "true"}],
    "realplksr_large": [{"label": "Dim", "key": "dim", "default": 96}, {"label": "N Blocks", "key": "n_blocks", "default": 28}, {"label": "Kernel", "key": "kernel_size", "default": 17}],
    
    # SAFMN & SCUNet
    "safmn": [{"label": "Dim", "key": "dim", "default": 36}, {"label": "N Blocks", "key": "n_blocks", "default": 8}],
    "safmn_l": [{"label": "Dim", "key": "dim", "default": 128}, {"label": "N Blocks", "key": "n_blocks", "default": 16}],
    "scunet_aaf6aa": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "Residual", "key": "residual", "default": "true"}],
    
    # SPAN
    "span": [{"label": "Feat Channels", "key": "feature_channels", "default": 52}, {"label": "Norm", "key": "norm", "default": "false"}],
    "span_s": [{"label": "Feat Channels", "key": "feature_channels", "default": 48}, {"label": "Norm", "key": "norm", "default": "false"}],
    "span_f32": [{"label": "Feat Channels", "key": "feature_channels", "default": 32}],
    
    # SRFormer
    "srformer": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}],
    "srformerv2": [{"label": "Embed Dim", "key": "embed_dim", "default": 240}, {"label": "Depths", "key": "depths", "default": "[4, 4, 4, 4, 4, 4]"}],
    
    # SwinIR & Swin2SR
    "swinir_l": [{"label": "Embed Dim", "key": "embed_dim", "default": 240}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6, 6, 6, 6]"}],
    "swinir_m": [{"label": "Embed Dim", "key": "embed_dim", "default": 180}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6]"}],
    "swinir_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 60}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]"}],
    
    "swin2sr_l": [{"label": "Embed Dim", "key": "embed_dim", "default": 240}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6, 6, 6, 6, 6, 6]"}],
    "swin2sr_s": [{"label": "Embed Dim", "key": "embed_dim", "default": 60}, {"label": "Depths", "key": "depths", "default": "[6, 6, 6, 6]"}],
    
    # Compact & Others
    "compact": [{"label": "Num Feat", "key": "num_feat", "default": 64}, {"label": "Num Conv", "key": "num_conv", "default": 16}],
    "ultracompact": [{"label": "Num Feat", "key": "num_feat", "default": 64}, {"label": "Num Conv", "key": "num_conv", "default": 8}],
    "superultracompact": [{"label": "Num Feat", "key": "num_feat", "default": 24}, {"label": "Num Conv", "key": "num_conv", "default": 8}],
    "realcugan": [{"label": "Pro", "key": "pro", "default": "false"}, {"label": "Fast", "key": "fast", "default": "false"}],
    "sebica": [{"label": "N", "key": "N", "default": 16}],
    "sebica_mini": [{"label": "N", "key": "N", "default": 8}],
    "seemore_t": [{"label": "Embed Dim", "key": "embedding_dim", "default": 36}, {"label": "Num Experts", "key": "num_experts", "default": 3}],
    "spanplus": [
        {"label": "Feat Channels", "key": "feature_channels", "default": 48},
        {"label": "Upsampler", "key": "upsampler", "default": "dysample", "tip_key": "upsampler",
         "type": "combobox", "choices": ["dysample", "pixelshuffle", "pixelshuffledirect", "nearest+conv"]},
    ],
    "spanplus_s": [
        {"label": "Feat Channels", "key": "feature_channels", "default": 32},
        {"label": "Upsampler", "key": "upsampler", "default": "dysample", "tip_key": "upsampler",
         "type": "combobox", "choices": ["dysample", "pixelshuffle", "pixelshuffledirect", "nearest+conv"]},
    ],
    "tscunet": [{"label": "Dim", "key": "dim", "default": 64}, {"label": "Residual", "key": "residual", "default": "true"}],
    "temporalspan": [{"label": "Feat Channels", "key": "feature_channels", "default": 48}, {"label": "Num Frames", "key": "num_frames", "default": 5}],
    "temporalspanv2": [{"label": "Feat Channels", "key": "feature_channels", "default": 48}, {"label": "Num Frames", "key": "num_frames", "default": 5}],
    
    # Fallback
    "default": [{"label": "Num Feat", "key": "num_feat", "default": 64}]
}

# Estimations VRAM pour Redux — mis à jour avec bench 2026-05-19 (GTX 1080 Ti, scale=1, SMI normalisé)
# Facteurs relatifs à compact (2.09 GB SMI @ patch=96 → référence = 1.0)
REDUX_VRAM_FACTORS = {
    # Compact family — très léger
    "superultracompact": 0.75, "ultracompact": 0.86, "compact": 1.0,
    # SPAN family
    "span": 1.23, "span_s": 1.20, "spanplus": 1.10, "spanplus_s": 0.91,
    # RTMoSR / MoSR
    "rtmosr_l": 0.74, "rtmosr": 1.11, "rtmosr_ul": 0.60, "mosrv2": 0.77, "mosr_t": 0.95, "mosr": 1.5,
    # ArtCNN
    "artcnn_r16f96": 1.01, "artcnn_r8f64": 0.80, "artcnn_r8f48": 0.70, "artcnn_r3f24": 0.55,
    # PLKSR / RealPLKSR
    "plksr": 1.22, "plksr_tiny": 1.04, "realplksr": 1.38, "realplksr_tiny": 1.18, "realplksr_large": 2.5,
    # ESRGAN
    "esrgan": 0.88, "esrgan_lite": 0.78,
    # SwinIR — mesure scale=1 beaucoup plus légère que scale=4
    "swinir_s": 0.84, "swinir_m": 3.0, "swinir_l": 5.0,
    "swin2sr_s": 1.0, "swin2sr_m": 3.0, "swin2sr_l": 5.0,
    # HAT family
    "hat_s": 4.0, "hat_m": 5.0, "hat_l": 6.0,
    # DAT family
    "dat": 4.0, "dat_s": 3.5, "dat_2": 4.5, "dat_light": 2.0,
    # DRCT family
    "drct": 1.59, "drct_l": 4.0, "drct_xl": 5.0, "drct_s": 2.5,
    # Omni / Attention — VRAM élevée (5.33 GB SMI)
    "omnisr": 2.55,
    # ELan
    "elan_light": 1.95, "elan": 3.0,
    # Heavy / experimental
    "man": 5.07, "man_light": 1.60, "man_tiny": 1.00,
    "safmn": 1.02, "safmn_l": 2.62,
    "seemore_t": 1.40, "lkfmixer_t": 1.19, "lkfmixer_b": 2.5, "lkfmixer_l": 4.0,
    "lmlt_tiny": 1.06, "lmlt_base": 2.0, "lmlt_large": 3.5,
    "gaterv3_s": 1.35, "gaterv3": 2.5,
    "sebica": 0.75, "sebica_mini": 0.60,
    "ditn_real": 1.98, "ditn": 3.0,
    "eimn_a": 0.94, "eimn_l": 2.0,
    "realcugan": 0.80, "scunet_aaf6aa": 1.1,
    # GRL / FDAT / Heavy
    "fdat": 1.3, "fdat_xl": 1.7, "grl_b": 1.6, "grl_s": 1.3,
    "rgt": 1.6, "srformer": 1.3, "atd": 1.5, "atd_light": 1.0,
    "default": 1.2,
}