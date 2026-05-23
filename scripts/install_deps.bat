@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title Universal SR Studio — Installation des dépendances

echo ============================================================
echo  Universal SR Studio — Installation des dépendances moteurs
echo ============================================================
echo.

:: Vérification Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python introuvable dans le PATH.
    echo          Installe Python 3.10+ depuis https://www.python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] %PY_VER% detecte

:: Vérification pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] pip introuvable.
    pause
    exit /b 1
)
echo [OK] pip disponible
echo.

:: Mise à jour pip
echo [1/8] Mise a jour pip / wheel / setuptools ^<70...
python -m pip install --upgrade pip wheel "setuptools<70" --quiet
if errorlevel 1 (echo [WARN] Erreur mise a jour pip — continuation...) else echo [OK]
echo.

:: numpy >=2.0 (avant les autres pour éviter les conflits)
echo [2/8] numpy ^>=2.0 (numpy moderne requis par NeoSR^)...
python -m pip install "numpy>=2.0" --quiet
if errorlevel 1 (echo [WARN] Erreur numpy — verifie compatibilite PyTorch) else echo [OK]
echo.

:: safetensors
echo [3/8] safetensors (format modeles securise^)...
python -m pip install safetensors --quiet
if errorlevel 1 (echo [WARN] Erreur safetensors) else echo [OK]

:: msgspec (requis par TraiNNer-Redux)
python -m pip install msgspec --quiet
if errorlevel 1 (echo [WARN] Erreur msgspec) else echo [OK] msgspec

:: timm (architectures backbone)
echo [4/8] timm (architectures backbone^)...
python -m pip install timm --quiet
if errorlevel 1 (echo [WARN] Erreur timm) else echo [OK]
echo.

:: spandrel + spandrel-extra-arches
echo [5/8] spandrel + spandrel-extra-arches (chargement modeles SR^)...
python -m pip install spandrel spandrel-extra-arches --quiet
if errorlevel 1 (echo [WARN] Erreur spandrel) else echo [OK]
echo.

:: antialiased-cnns
echo [6/8] antialiased-cnns (anti-aliasing blur pool^)...
python -m pip install antialiased-cnns --quiet
if errorlevel 1 (echo [WARN] Erreur antialiased-cnns) else echo [OK]

:: pytorch-optimizer
python -m pip install pytorch-optimizer --quiet
if errorlevel 1 (echo [WARN] Erreur pytorch-optimizer) else echo [OK] pytorch-optimizer
echo.

:: pyvips (traitement image rapide, optionnel)
echo [7/8] pyvips ^(optionnel — traitement image haute performance^)...
python -m pip install pyvips --quiet
if errorlevel 1 (
    echo [INFO] pyvips non disponible - PIL utilise en fallback
    echo        Pour installer manuellement : https://www.libvips.org/install.html
) else (
    echo [OK] pyvips
)
echo.

:: TensorBoard nightly
echo [8/8] TensorBoard nightly...
python -m pip install tb-nightly --quiet
if errorlevel 1 (
    echo [WARN] tb-nightly echec — tentative avec tensorboard stable...
    python -m pip install tensorboard --quiet
    if errorlevel 1 (echo [WARN] TensorBoard non installe) else echo [OK] tensorboard ^(stable^)
) else (
    echo [OK] tb-nightly
)
echo.

echo ============================================================
echo  Verification de l'installation
echo ============================================================
python -c "import numpy; print(f'numpy       {numpy.__version__}')" 2>nul || echo numpy       MANQUANT
python -c "import safetensors; print(f'safetensors {safetensors.__version__}')" 2>nul || echo safetensors MANQUANT
python -c "import msgspec; print(f'msgspec     {msgspec.__version__}')" 2>nul || echo msgspec     MANQUANT
python -c "import timm; print(f'timm        {timm.__version__}')" 2>nul || echo timm        MANQUANT
python -c "import spandrel; print(f'spandrel    {spandrel.__version__}')" 2>nul || echo spandrel    MANQUANT
python -c "import antialiased_cnns; print('antialiased-cnns OK')" 2>nul || echo antialiased-cnns MANQUANT
python -c "import pytorch_optimizer; print(f'pytorch-opt {pytorch_optimizer.__version__}')" 2>nul || echo pytorch-optimizer MANQUANT
python -c "import pyvips; print(f'pyvips      {pyvips.__version__}')" 2>nul || echo pyvips      non installe ^(optionnel^)
python -c "import tensorboard; print(f'tensorboard {tensorboard.__version__}')" 2>nul || echo tensorboard MANQUANT
echo.
echo Installation terminee.
pause
