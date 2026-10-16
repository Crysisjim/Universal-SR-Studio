@echo off
rem Universal SR Studio — build mamba_ssm (selective_scan_cuda) on Windows.
rem Needs: VS 2022 C++ workload, CUDA Toolkit 12.6, Python headers in the engine's base Python
rem (include\ + libs\ from the nuget "python" package of the same version), patched mamba_src
rem (patch_mamba_windows.py). Writes the wheel to %~dp0wheels for reuse on other machines.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "MAMBA_FORCE_BUILD=TRUE"
set "MAX_JOBS=12"
set "PY=%USERPROFILE%\USRS\runtimes\.venv\Scripts\python.exe"
"%PY%" -c "import sys, torch; print('EXE:', sys.executable, 'torch', torch.__version__)"
cd /d "%~dp0mamba_src"
rem Stale objects survive a setup.py change (gencode list...): "build_mamba.bat clean" after one.
if /i "%~1"=="clean" (
  if exist build rmdir /s /q build
  if exist mamba_ssm.egg-info rmdir /s /q mamba_ssm.egg-info
)
"%PY%" -m pip wheel . --no-build-isolation --no-deps -w "%~dp0wheels"
if errorlevel 1 (
  echo MAMBA_SSM_FAILED
  exit /b 1
)
echo MAMBA_SSM_WHEEL_OK
