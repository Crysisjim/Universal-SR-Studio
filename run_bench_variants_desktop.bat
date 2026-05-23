@echo off
:: Bench des architectures VARIANTS non testés dans le bench principal
:: GPU cible : GTX 1080 Ti (Pascal sm_61)
:: Mode : normal uniquement (tf32 incompatible Pascal, fp16/bf16 optionnels)
::
:: Ces archs n'étaient PAS dans le bench original du 2026-05-19 :
::   artcnn_r8f48, artcnn_r3f24, rtmosr_ul, mosr, sebica_mini,
::   spanplus, spanplus_st, spanplus_sts,
::   eimn, eimn_l, lmlt_base, lmlt_large,
::   lkfmixer_b, lkfmixer_l, gaterv3, ditn, elan,
::   swinir_m, srformerv2, drct_xl, realplksr_large,
::   swin2sr_s, swin2sr_m, dat_s, dat_light
::
:: Double-cliquer pour lancer. L'état est sauvegardé dans redux_bench_variants\
:: et le bench peut être repris si interrompu.

cd /d "%~dp0"

set VARIANTS=artcnn_r8f48,artcnn_r3f24,rtmosr_ul,mosr,sebica_mini,spanplus,spanplus_st,spanplus_sts,eimn,eimn_l,lmlt_base,lmlt_large,lkfmixer_b,lkfmixer_l,gaterv3,ditn,elan,swinir_m,srformerv2,drct_xl,realplksr_large,swin2sr_s,swin2sr_m,dat_s,dat_light

echo [Bench Variants] Architectures : %VARIANTS%
echo [Bench Variants] GPU : GTX 1080 Ti - Pascal sm_61
echo [Bench Variants] Mode : normal (TF32 desactive - incompatible Pascal+PyTorch2.7)
echo [Bench Variants] Resultats : redux_bench_variants\
echo.

:: Python du venv traiNNer sur le desktop
set REDUX_PY=%USERPROFILE%\IA_Engine\traiNNer-redux\.venv\Scripts\python.exe

if not exist "%REDUX_PY%" (
    echo [ERREUR] Python venv introuvable: %REDUX_PY%
    pause
    exit /b 1
)

"%REDUX_PY%" "src\core\redux_arch_benchmark.py" ^
    --output-dir "redux_bench_variants" ^
    --modes normal ^
    --tests "%VARIANTS%" ^
    %*

echo.
echo [Bench Variants] Termine. Code de sortie: %ERRORLEVEL%
pause
