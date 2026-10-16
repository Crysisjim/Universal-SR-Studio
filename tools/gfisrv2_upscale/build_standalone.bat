@echo off
chcp 65001 >nul 2>&1
title GFISRv2 Upscale — Build standalone

echo ============================================================
echo  GFISRv2 Upscale — Build standalone exe
echo ============================================================
echo.

:: Verify PyInstaller
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] PyInstaller non trouve. Installe-le:
    echo   pip install pyinstaller
    pause
    exit /b 1
)
echo [OK] PyInstaller disponible
echo.

:: Output dir
set DIST_DIR=%~dp0..\..\dist\gfisrv2_upscale

echo [1/3] Nettoyage dist existant...
if exist "%DIST_DIR%" (
    rmdir /s /q "%DIST_DIR%"
    echo     Supprime: %DIST_DIR%
)
echo.

echo [2/3] Build PyInstaller...
cd /d "%~dp0"
pyinstaller gfisrv2_standalone.spec --distpath "%~dp0..\..\dist" --workpath "%~dp0..\..\build_tmp\gfisrv2" --noconfirm
if errorlevel 1 (
    echo [ERREUR] Build echoue.
    pause
    exit /b 1
)
echo.

echo [3/3] Verification...
if not exist "%DIST_DIR%\gfisrv2_upscale.exe" (
    echo [ERREUR] exe non trouve dans %DIST_DIR%
    pause
    exit /b 1
)
echo [OK] exe: %DIST_DIR%\gfisrv2_upscale.exe
echo.

echo ============================================================
echo  Build reussi !
echo  Livrable: %DIST_DIR%\
echo
echo  Contenu minimal a livrer avec le modele:
echo    gfisrv2_upscale.exe
echo    + tous les fichiers/dossiers du meme repertoire
echo ============================================================
echo.
pause
