@echo off
chcp 65001 >nul 2>&1
:: Lance GFISRv2 Upscale avec le Python système.
:: Requiert : torch, einops, customtkinter, PIL, tkinterdnd2 installés.
:: Avantage : ~0 KB supplémentaire (utilise les dépendances déjà présentes).
cd /d "%~dp0"
python gfisrv2_app.py
if errorlevel 1 (
    echo.
    echo [ERREUR] Vérifiez que torch / customtkinter / einops / tkinterdnd2 sont installés.
    echo  pip install customtkinter tkinterdnd2 einops Pillow
    pause
)
